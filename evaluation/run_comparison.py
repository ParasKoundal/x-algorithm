#!/usr/bin/env python3
# Copyright 2026 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script to run Phoenix V1 vs V2 comparison."""

import sys
sys.path.append('../')

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import haiku as hk

# Import Phoenix V1
from phoenix.grok import TransformerConfig
from phoenix.recsys_model import PhoenixModelConfig, PhoenixModel, RecsysBatch, RecsysEmbeddings, HashConfig

# Import Phoenix V2
from phoenix_v2.config import PhoenixV2Config
from phoenix_v2.recsys_model_v2 import PhoenixModelV2Config, PhoenixModelV2

# Import comparison framework
from evaluation.comparison_framework import ModelComparator, create_synthetic_test_data


def create_v1_model(emb_size: int = 256, num_layers: int = 6):
    """Create Phoenix V1 model."""

    transformer_config = TransformerConfig(
        emb_size=emb_size,
        key_size=64,
        num_q_heads=8,
        num_kv_heads=4,
        num_layers=num_layers,
        widening_factor=4.0,
        attn_output_multiplier=1.0,
    )

    model_config = PhoenixModelConfig(
        model=transformer_config,
        emb_size=emb_size,
        num_actions=14,
        history_seq_len=128,
        candidate_seq_len=32,
        hash_config=HashConfig(),
    ).initialize()

    def forward_fn(batch, embeddings):
        model = model_config.make()
        return model(batch, embeddings)

    forward = hk.transform(forward_fn)

    return forward, model_config


def create_v2_model(
    emb_size: int = 256,
    num_layers: int = 6,
    v2_config: Optional[PhoenixV2Config] = None,
):
    """Create Phoenix V2 model with improvements."""

    if v2_config is None:
        v2_config = PhoenixV2Config(
            emb_size=emb_size,
            num_layers=num_layers,
            use_temporal_attention=True,
            use_hierarchical_user_encoder=True,
            use_multimodal_encoder=False,  # Requires image data
            use_intelligent_hashing=True,
            use_multi_task_learning=True,
            use_context_features=True,
            use_social_gnn=False,  # Requires graph data
            use_causal_debiasing=True,
            use_exploration=True,
        )

    transformer_config = TransformerConfig(
        emb_size=emb_size,
        key_size=64,
        num_q_heads=8,
        num_kv_heads=4,
        num_layers=num_layers,
        widening_factor=4.0,
        attn_output_multiplier=1.0,
    )

    model_config = PhoenixModelV2Config(
        model=transformer_config,
        v2_config=v2_config,
        emb_size=emb_size,
        num_actions=14,
        history_seq_len=128,
        candidate_seq_len=32,
        hash_config=HashConfig(),
    ).initialize()

    def forward_fn(batch, embeddings, context=None, timestamps=None):
        model = model_config.make()
        return model(batch, embeddings, context, timestamps)

    forward = hk.transform(forward_fn)

    return forward, model_config


def initialize_models(key, batch_size=2, emb_size=256, num_layers=6):
    """Initialize both V1 and V2 models."""

    print("Initializing models...")

    # Create dummy batch for initialization
    dummy_batch = RecsysBatch(
        user_hashes=jnp.ones((batch_size, 2), dtype=jnp.int32),
        history_post_hashes=jnp.ones((batch_size, 128, 2), dtype=jnp.int32),
        history_author_hashes=jnp.ones((batch_size, 128, 2), dtype=jnp.int32),
        history_actions=jnp.ones((batch_size, 128, 14), dtype=jnp.float32),
        history_product_surface=jnp.ones((batch_size, 128), dtype=jnp.int32),
        candidate_post_hashes=jnp.ones((batch_size, 32, 2), dtype=jnp.int32),
        candidate_author_hashes=jnp.ones((batch_size, 32, 2), dtype=jnp.int32),
        candidate_product_surface=jnp.ones((batch_size, 32), dtype=jnp.int32),
    )

    dummy_embeddings = RecsysEmbeddings(
        user_embeddings=jnp.ones((batch_size, 2, emb_size)),
        history_post_embeddings=jnp.ones((batch_size, 128, 2, emb_size)),
        candidate_post_embeddings=jnp.ones((batch_size, 32, 2, emb_size)),
        history_author_embeddings=jnp.ones((batch_size, 128, 2, emb_size)),
        candidate_author_embeddings=jnp.ones((batch_size, 32, 2, emb_size)),
    )

    # Initialize V1
    print("  - Initializing V1...")
    v1_forward, v1_config = create_v1_model(emb_size=emb_size, num_layers=num_layers)
    v1_key, key = jax.random.split(key)
    v1_params = v1_forward.init(v1_key, dummy_batch, dummy_embeddings)

    # Initialize V2
    print("  - Initializing V2...")
    v2_forward, v2_config = create_v2_model(emb_size=emb_size, num_layers=num_layers)
    v2_key, key = jax.random.split(key)
    v2_params = v2_forward.init(v2_key, dummy_batch, dummy_embeddings, None, None)

    print(f"✓ Models initialized")
    print(f"  V1 params: {sum(p.size for p in jax.tree_leaves(v1_params)):,} parameters")
    print(f"  V2 params: {sum(p.size for p in jax.tree_leaves(v2_params)):,} parameters")

    return v1_forward, v1_params, v2_forward, v2_params


def ablation_study(key, num_examples: int = 100):
    """Run ablation study testing each V2 feature individually."""

    print("\n" + "=" * 80)
    print("ABLATION STUDY: Testing each V2 feature individually")
    print("=" * 80)

    # Generate test data
    test_data = create_synthetic_test_data(num_examples=num_examples)

    # Initialize V1 (baseline)
    v1_forward, v1_params, _, _ = initialize_models(key, emb_size=256, num_layers=4)

    results = {}

    # Feature combinations to test
    feature_configs = {
        'v1_baseline': PhoenixV2Config(
            use_temporal_attention=False,
            use_hierarchical_user_encoder=False,
            use_intelligent_hashing=False,
            use_multi_task_learning=False,
            use_context_features=False,
            use_causal_debiasing=False,
            use_exploration=False,
        ),
        'temporal_attention': PhoenixV2Config(
            use_temporal_attention=True,
            use_hierarchical_user_encoder=False,
            use_intelligent_hashing=False,
            use_multi_task_learning=False,
            use_context_features=False,
        ),
        'hierarchical_user': PhoenixV2Config(
            use_temporal_attention=False,
            use_hierarchical_user_encoder=True,
            use_intelligent_hashing=False,
            use_multi_task_learning=False,
            use_context_features=False,
        ),
        'intelligent_hashing': PhoenixV2Config(
            use_temporal_attention=False,
            use_hierarchical_user_encoder=False,
            use_intelligent_hashing=True,
            use_multi_task_learning=False,
            use_context_features=False,
        ),
        'multi_task': PhoenixV2Config(
            use_temporal_attention=False,
            use_hierarchical_user_encoder=False,
            use_intelligent_hashing=False,
            use_multi_task_learning=True,
            use_context_features=False,
        ),
        'context_features': PhoenixV2Config(
            use_temporal_attention=False,
            use_hierarchical_user_encoder=False,
            use_intelligent_hashing=False,
            use_multi_task_learning=False,
            use_context_features=True,
        ),
        'all_features': PhoenixV2Config(
            use_temporal_attention=True,
            use_hierarchical_user_encoder=True,
            use_intelligent_hashing=True,
            use_multi_task_learning=True,
            use_context_features=True,
            use_causal_debiasing=True,
            use_exploration=True,
        ),
    }

    for feature_name, config in feature_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Testing: {feature_name}")
        print(f"{'=' * 60}")

        # Create V2 model with this config
        v2_forward, v2_params = create_v2_model(emb_size=256, num_layers=4, v2_config=config)[0:2]
        v2_key = jax.random.PRNGKey(42)
        dummy_batch = create_dummy_batch(batch_size=2, emb_size=256)
        v2_params = v2_forward.init(v2_key, *dummy_batch)

        # Compare with V1
        comparator = ModelComparator(v1_forward, v2_forward, v1_params, v2_params)
        result = comparator.compare(test_data)

        results[feature_name] = result['improvements_pct']

    # Print ablation summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    print(f"\n{'Feature':<25} {'NDCG@10':>12} {'MRR':>12} {'Latency':>15}")
    print("-" * 80)

    for feature_name, improvements in results.items():
        ndcg = improvements.get('ndcg_at_10', 0.0)
        mrr = improvements.get('mrr', 0.0)
        latency = improvements.get('mean_latency_ms', 0.0)

        print(f"{feature_name:<25} {ndcg:>11.2f}% {mrr:>11.2f}% {latency:>14.2f}%")

    # Save ablation results
    output_dir = Path("./evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "ablation_study.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Ablation results saved to: {output_dir / 'ablation_study.json'}")


def create_dummy_batch(batch_size, emb_size):
    """Create dummy batch for initialization."""
    dummy_batch = RecsysBatch(
        user_hashes=jnp.ones((batch_size, 2), dtype=jnp.int32),
        history_post_hashes=jnp.ones((batch_size, 128, 2), dtype=jnp.int32),
        history_author_hashes=jnp.ones((batch_size, 128, 2), dtype=jnp.int32),
        history_actions=jnp.ones((batch_size, 128, 14), dtype=jnp.float32),
        history_product_surface=jnp.ones((batch_size, 128), dtype=jnp.int32),
        candidate_post_hashes=jnp.ones((batch_size, 32, 2), dtype=jnp.int32),
        candidate_author_hashes=jnp.ones((batch_size, 32, 2), dtype=jnp.int32),
        candidate_product_surface=jnp.ones((batch_size, 32), dtype=jnp.int32),
    )

    dummy_embeddings = RecsysEmbeddings(
        user_embeddings=jnp.ones((batch_size, 2, emb_size)),
        history_post_embeddings=jnp.ones((batch_size, 128, 2, emb_size)),
        candidate_post_embeddings=jnp.ones((batch_size, 32, 2, emb_size)),
        history_author_embeddings=jnp.ones((batch_size, 128, 2, emb_size)),
        candidate_author_embeddings=jnp.ones((batch_size, 32, 2, emb_size)),
    )

    return dummy_batch, dummy_embeddings, None, None


def main():
    parser = argparse.ArgumentParser(description="Run Phoenix V1 vs V2 comparison")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of test examples")
    parser.add_argument("--emb_size", type=int, default=256, help="Embedding size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)

    if args.ablation:
        ablation_study(key, num_examples=args.num_examples)
    else:
        # Initialize models
        v1_forward, v1_params, v2_forward, v2_params = initialize_models(
            key,
            emb_size=args.emb_size,
            num_layers=args.num_layers,
        )

        # Generate test data
        print(f"\nGenerating {args.num_examples} test examples...")
        test_data = create_synthetic_test_data(num_examples=args.num_examples)

        # Run comparison
        comparator = ModelComparator(v1_forward, v2_forward, v1_params, v2_params)
        results = comparator.compare(test_data)

        print("\n✓ Comparison complete!")
        print(f"Results saved to: ./evaluation_results/comparison_results.json")


if __name__ == "__main__":
    main()

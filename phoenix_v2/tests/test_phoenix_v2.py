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

"""Comprehensive tests for Phoenix V2 improvements."""

import sys
sys.path.append('../../')

import unittest
import jax
import jax.numpy as jnp
import haiku as hk

from phoenix_v2.config import PhoenixV2Config
from phoenix_v2.temporal_attention import (
    TemporalPositionalEncoding,
    apply_temporal_decay,
    compute_recency_weights,
)
from phoenix_v2.hierarchical_user_encoder import HierarchicalUserEncoder
from phoenix_v2.context_features import ContextEncoder, create_dummy_context
from phoenix_v2.intelligent_hashing import IntelligentHashEmbedding, block_reduce_v2
from phoenix_v2.multi_task_learning import MultiTaskHead, multi_task_loss
from phoenix_v2.exploration import ExplorationPolicy
from phoenix_v2.causal_debiasing import PropensityScoreEstimator, CausalRanker


class TestTemporalAttention(unittest.TestCase):
    """Test temporal attention mechanisms."""

    def test_temporal_positional_encoding(self):
        """Test temporal positional encoding."""

        def forward(embeddings, timestamps):
            encoder = TemporalPositionalEncoding(emb_size=64)
            return encoder(embeddings, timestamps)

        transform = hk.transform(forward)

        B, T, D = 2, 10, 64
        embeddings = jnp.ones((B, T, D))
        timestamps = jnp.arange(T)[None, :].repeat(B, axis=0).astype(jnp.float32)

        key = jax.random.PRNGKey(42)
        params = transform.init(key, embeddings, timestamps)
        output = transform.apply(params, key, embeddings, timestamps)

        self.assertEqual(output.shape, (B, T, D))
        print("✓ Temporal positional encoding test passed")

    def test_temporal_decay(self):
        """Test temporal decay on attention logits."""

        B, H, T = 2, 8, 10
        attention_logits = jnp.ones((B, H, T, T))
        timestamps = jnp.arange(T, dtype=jnp.float32)[None, :].repeat(B, axis=0)

        decayed_logits = apply_temporal_decay(attention_logits, timestamps, decay_rate=0.1)

        self.assertEqual(decayed_logits.shape, (B, H, T, T))

        # Check that recent items have higher weights
        # Last position should have higher attention to recent items
        self.assertGreater(
            float(decayed_logits[0, 0, -1, -1]),
            float(decayed_logits[0, 0, -1, 0])
        )

        print("✓ Temporal decay test passed")

    def test_recency_weights(self):
        """Test recency weight computation."""

        B, T = 2, 10
        timestamps = jnp.arange(T, dtype=jnp.float32)[None, :].repeat(B, axis=0)

        weights = compute_recency_weights(timestamps, decay_rate=0.1)

        self.assertEqual(weights.shape, (B, T))

        # Recent items should have higher weights
        self.assertGreater(float(weights[0, -1]), float(weights[0, 0]))

        print("✓ Recency weights test passed")


class TestHierarchicalUserEncoder(unittest.TestCase):
    """Test hierarchical user encoder."""

    def test_hierarchical_encoding(self):
        """Test multi-level user encoding."""

        def forward(user_emb, history_emb, mask, timestamps):
            encoder = HierarchicalUserEncoder(emb_size=64, recent_window=5, num_clusters=3)
            return encoder(user_emb, history_emb, mask, timestamps)

        transform = hk.transform(forward)

        B, S, D = 2, 20, 64
        user_emb = jnp.ones((B, 1, D))
        history_emb = jnp.ones((B, S, D))
        mask = jnp.ones((B, S), dtype=jnp.bool_)
        timestamps = jnp.arange(S, dtype=jnp.float32)[None, :].repeat(B, axis=0)

        key = jax.random.PRNGKey(42)
        params = transform.init(key, user_emb, history_emb, mask, timestamps)
        output = transform.apply(params, key, user_emb, history_emb, mask, timestamps)

        self.assertEqual(output.shape, (B, D))
        print("✓ Hierarchical user encoder test passed")


class TestContextFeatures(unittest.TestCase):
    """Test context feature encoding."""

    def test_context_encoder(self):
        """Test encoding of context features."""

        def forward(context):
            encoder = ContextEncoder(emb_size=64)
            return encoder(context)

        transform = hk.transform(forward)

        B, C = 2, 32
        context = create_dummy_context(batch_size=B, num_candidates=C)

        key = jax.random.PRNGKey(42)
        params = transform.init(key, context)
        user_context, candidate_context = transform.apply(params, key, context)

        self.assertEqual(user_context.shape, (B, 64))
        self.assertEqual(candidate_context.shape, (B, C, 64))
        print("✓ Context encoder test passed")


class TestIntelligentHashing(unittest.TestCase):
    """Test intelligent hash embeddings."""

    def test_intelligent_hash_embedding(self):
        """Test collision-aware hash embedding."""

        def forward(hashes, embeddings):
            hasher = IntelligentHashEmbedding(
                emb_size=64,
                num_hashes=2,
                vocab_size=1000,
            )
            return hasher(hashes, embeddings)

        transform = hk.transform(forward)

        B, num_hashes, D = 4, 2, 64
        hashes = jnp.array([[1, 2], [1, 2], [3, 4], [3, 4]], dtype=jnp.int32)  # Simulate collisions
        embeddings = jax.random.normal(jax.random.PRNGKey(42), (B, num_hashes, D))

        key = jax.random.PRNGKey(42)
        params = transform.init(key, hashes, embeddings)
        output = transform.apply(params, key, hashes, embeddings)

        self.assertEqual(output.shape, (B, D))
        print("✓ Intelligent hash embedding test passed")


class TestMultiTaskLearning(unittest.TestCase):
    """Test multi-task learning framework."""

    def test_multi_task_head(self):
        """Test multi-task prediction heads."""

        def forward(candidate_emb, user_emb, history_emb):
            head = MultiTaskHead(emb_size=64, num_actions=14, num_topics=100)
            return head(candidate_emb, user_emb, history_emb)

        transform = hk.transform(forward)

        B, C, S, D = 2, 32, 20, 64
        candidate_emb = jnp.ones((B, C, D))
        user_emb = jnp.ones((B, D))
        history_emb = jnp.ones((B, S, D))

        key = jax.random.PRNGKey(42)
        params = transform.init(key, candidate_emb, user_emb, history_emb)
        output = transform.apply(params, key, candidate_emb, user_emb, history_emb)

        self.assertEqual(output.engagement_logits.shape, (B, C, 14))
        self.assertEqual(output.dwell_time.shape, (B, C))
        self.assertEqual(output.topic_logits.shape, (B, C, 100))
        self.assertEqual(output.contrastive_scores.shape, (B, C))
        self.assertEqual(output.next_action_logits.shape, (B, S, 14))

        print("✓ Multi-task head test passed")


class TestExploration(unittest.TestCase):
    """Test exploration/exploitation strategies."""

    def test_epsilon_greedy(self):
        """Test epsilon-greedy exploration."""

        policy = ExplorationPolicy(epsilon=0.1)

        candidate_ids = [f"item_{i}" for i in range(10)]
        scores = jnp.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        key = jax.random.PRNGKey(42)
        indices, final_scores = policy.select_candidates(
            candidate_ids,
            scores,
            exploration_budget=2,
            strategy="epsilon_greedy",
            key=key,
        )

        self.assertEqual(len(indices), 10)
        print("✓ Epsilon-greedy test passed")

    def test_thompson_sampling(self):
        """Test Thompson sampling."""

        policy = ExplorationPolicy()

        # Simulate some feedback
        for i in range(5):
            policy.update_stats(f"item_{i}", was_engaged=(i % 2 == 0))

        candidate_ids = [f"item_{i}" for i in range(5)]
        scores = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])

        key = jax.random.PRNGKey(42)
        indices, final_scores = policy.select_candidates(
            candidate_ids,
            scores,
            strategy="thompson",
            key=key,
        )

        self.assertEqual(len(indices), 5)
        print("✓ Thompson sampling test passed")


class TestCausalDebiasing(unittest.TestCase):
    """Test causal debiasing methods."""

    def test_propensity_estimator(self):
        """Test propensity score estimation."""

        def forward(positions, popularities):
            estimator = PropensityScoreEstimator(hidden_size=64)
            return estimator(positions, popularities)

        transform = hk.transform(forward)

        B, C = 2, 32
        positions = jnp.arange(C, dtype=jnp.float32)[None, :].repeat(B, axis=0)
        popularities = jnp.ones((B, C)) * 0.1

        key = jax.random.PRNGKey(42)
        params = transform.init(key, positions, popularities)
        propensity = transform.apply(params, key, positions, popularities)

        self.assertEqual(propensity.shape, (B, C))
        self.assertTrue(jnp.all((propensity >= 0.01) & (propensity <= 1.0)))

        print("✓ Propensity estimator test passed")

    def test_causal_debiasing(self):
        """Test causal score debiasing."""

        ranker = CausalRanker()

        B, C = 2, 32
        raw_scores = jnp.ones((B, C))
        positions = jnp.arange(C, dtype=jnp.float32)[None, :].repeat(B, axis=0)
        popularities = jnp.ones((B, C)) * 0.1

        debiased_scores = ranker.debias_scores(
            raw_scores,
            positions=positions,
            popularities=popularities,
            use_ipw=False,  # Don't use IPW without estimator
        )

        self.assertEqual(debiased_scores.shape, (B, C))

        # Top position should be debiased (reduced)
        self.assertLess(
            float(debiased_scores[0, 0]),
            float(raw_scores[0, 0])
        )

        print("✓ Causal debiasing test passed")


def run_all_tests():
    """Run all Phoenix V2 tests."""

    print("\n" + "=" * 80)
    print("PHOENIX V2 TEST SUITE")
    print("=" * 80 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestHierarchicalUserEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestContextFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestIntelligentHashing))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiTaskLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestExploration))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalDebiasing))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

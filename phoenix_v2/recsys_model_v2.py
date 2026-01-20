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

"""Phoenix V2: Enhanced ranking model with all improvements."""

import sys
sys.path.append('../phoenix')

from dataclasses import dataclass
from typing import Any, Optional

import haiku as hk
import jax
import jax.numpy as jnp

# Import original Phoenix components
from phoenix.grok import Transformer, TransformerConfig, layer_norm
from phoenix.recsys_model import (
    RecsysBatch,
    RecsysEmbeddings,
    RecsysModelOutput,
    HashConfig,
    block_user_reduce,
    block_history_reduce,
    block_candidate_reduce,
)

# Import V2 improvements
from phoenix_v2.config import PhoenixV2Config
from phoenix_v2.hierarchical_user_encoder import HierarchicalUserEncoder
from phoenix_v2.context_features import ContextEncoder, ContextFeatures, create_dummy_context
from phoenix_v2.multi_task_learning import MultiTaskHead, MultiTaskOutput
from phoenix_v2.intelligent_hashing import block_reduce_v2


@dataclass
class PhoenixModelV2Config:
    """Configuration for Phoenix V2 ranking model."""

    model: TransformerConfig
    v2_config: PhoenixV2Config
    emb_size: int
    num_actions: int
    history_seq_len: int = 128
    candidate_seq_len: int = 32

    name: Optional[str] = None
    fprop_dtype: Any = jnp.bfloat16

    hash_config: HashConfig = None  # type: ignore
    product_surface_vocab_size: int = 16

    _initialized = False

    def __post_init__(self):
        if self.hash_config is None:
            self.hash_config = HashConfig()

    def initialize(self):
        self._initialized = True
        return self

    def make(self):
        if not self._initialized:
            self.initialize()

        return PhoenixModelV2(
            model=self.model.make(),
            config=self,
            fprop_dtype=self.fprop_dtype,
        )


@dataclass
class PhoenixModelV2(hk.Module):
    """Enhanced transformer-based recommendation model.

    Improvements over V1:
    - Hierarchical user encoder (multi-level attention)
    - Context feature integration
    - Multi-task learning heads
    - Intelligent hash embeddings
    - Support for temporal information
    """

    model: Transformer
    config: PhoenixModelV2Config
    fprop_dtype: Any = jnp.bfloat16
    name: Optional[str] = None

    def _get_action_embeddings(self, actions: jax.Array) -> jax.Array:
        """Convert multi-hot action vectors to embeddings."""
        config = self.config
        _, _, num_actions = actions.shape
        D = config.emb_size

        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        action_projection = hk.get_parameter(
            "action_projection",
            [num_actions, D],
            dtype=jnp.float32,
            init=embed_init,
        )

        actions_signed = (2 * actions - 1).astype(jnp.float32)
        action_emb = jnp.dot(actions_signed.astype(action_projection.dtype), action_projection)

        valid_mask = jnp.any(actions, axis=-1, keepdims=True)
        action_emb = action_emb * valid_mask

        return action_emb.astype(self.fprop_dtype)

    def _single_hot_to_embeddings(
        self, input: jax.Array, vocab_size: int, emb_size: int, name: str
    ) -> jax.Array:
        """Convert single-hot indices to embeddings via lookup table."""
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        embedding_table = hk.get_parameter(
            name, [vocab_size, emb_size], dtype=jnp.float32, init=embed_init
        )

        input_one_hot = jax.nn.one_hot(input, vocab_size)
        output = jnp.dot(input_one_hot, embedding_table)
        return output.astype(self.fprop_dtype)

    def _get_unembedding(self) -> jax.Array:
        """Get the unembedding matrix for decoding to logits."""
        config = self.config
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        unembed_mat = hk.get_parameter(
            "unembeddings",
            [config.emb_size, config.num_actions],
            dtype=jnp.float32,
            init=embed_init,
        )
        return unembed_mat

    def build_inputs(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
        context: Optional[ContextFeatures] = None,
        timestamps: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array, int]:
        """Build input embeddings with V2 improvements.

        Args:
            batch: RecsysBatch containing hashes, actions, product surfaces
            recsys_embeddings: Pre-looked-up embeddings
            context: Optional context features
            timestamps: Optional timestamps for temporal modeling

        Returns:
            embeddings: [B, 1 + history_len + num_candidates, D]
            padding_mask: [B, 1 + history_len + num_candidates]
            candidate_start_offset: Position where candidates start
        """
        config = self.config
        v2_config = config.v2_config
        hash_config = config.hash_config

        # Product surface embeddings
        history_product_surface_embeddings = self._single_hot_to_embeddings(
            batch.history_product_surface,  # type: ignore
            config.product_surface_vocab_size,
            config.emb_size,
            "product_surface_embedding_table",
        )
        candidate_product_surface_embeddings = self._single_hot_to_embeddings(
            batch.candidate_product_surface,  # type: ignore
            config.product_surface_vocab_size,
            config.emb_size,
            "product_surface_embedding_table",
        )

        # Action embeddings
        history_actions_embeddings = self._get_action_embeddings(batch.history_actions)  # type: ignore

        # User embeddings (V1 or V2 depending on config)
        if v2_config.use_intelligent_hashing:
            # Use intelligent hashing for user
            user_embeddings, user_padding_mask = block_reduce_v2(
                batch.user_hashes[:, None, :],  # type: ignore # Add sequence dim
                recsys_embeddings.user_embeddings[:, None, :, :],  # type: ignore
                hash_config.num_user_hashes,
                config.emb_size,
                "user"
            )
        else:
            # Use original V1 hash reduction
            user_embeddings, user_padding_mask = block_user_reduce(
                batch.user_hashes,  # type: ignore
                recsys_embeddings.user_embeddings,  # type: ignore
                hash_config.num_user_hashes,
                config.emb_size,
                1.0,
            )

        # History embeddings
        if v2_config.use_intelligent_hashing:
            history_embeddings_temp, history_padding_mask = block_reduce_v2(
                batch.history_post_hashes,  # type: ignore
                recsys_embeddings.history_post_embeddings,  # type: ignore
                hash_config.num_item_hashes,
                config.emb_size,
                "history_post"
            )
            history_author_embeddings_temp, _ = block_reduce_v2(
                batch.history_author_hashes,  # type: ignore
                recsys_embeddings.history_author_embeddings,  # type: ignore
                hash_config.num_author_hashes,
                config.emb_size,
                "history_author"
            )
            # Combine post, author, actions, product_surface
            history_embeddings = (
                history_embeddings_temp +
                history_author_embeddings_temp +
                history_actions_embeddings +
                history_product_surface_embeddings
            )
        else:
            history_embeddings, history_padding_mask = block_history_reduce(
                batch.history_post_hashes,  # type: ignore
                recsys_embeddings.history_post_embeddings,  # type: ignore
                recsys_embeddings.history_author_embeddings,  # type: ignore
                history_product_surface_embeddings,
                history_actions_embeddings,
                hash_config.num_item_hashes,
                hash_config.num_author_hashes,
                1.0,
            )

        # Candidate embeddings
        if v2_config.use_intelligent_hashing:
            candidate_embeddings_temp, candidate_padding_mask = block_reduce_v2(
                batch.candidate_post_hashes,  # type: ignore
                recsys_embeddings.candidate_post_embeddings,  # type: ignore
                hash_config.num_item_hashes,
                config.emb_size,
                "candidate_post"
            )
            candidate_author_embeddings_temp, _ = block_reduce_v2(
                batch.candidate_author_hashes,  # type: ignore
                recsys_embeddings.candidate_author_embeddings,  # type: ignore
                hash_config.num_author_hashes,
                config.emb_size,
                "candidate_author"
            )
            candidate_embeddings = (
                candidate_embeddings_temp +
                candidate_author_embeddings_temp +
                candidate_product_surface_embeddings
            )
        else:
            candidate_embeddings, candidate_padding_mask = block_candidate_reduce(
                batch.candidate_post_hashes,  # type: ignore
                recsys_embeddings.candidate_post_embeddings,  # type: ignore
                recsys_embeddings.candidate_author_embeddings,  # type: ignore
                candidate_product_surface_embeddings,
                hash_config.num_item_hashes,
                hash_config.num_author_hashes,
                1.0,
            )

        # Add context features if enabled
        if v2_config.use_context_features and context is not None:
            context_encoder = ContextEncoder(emb_size=config.emb_size, name="context_encoder")
            user_context, candidate_context = context_encoder(context)

            # Add context to embeddings
            user_embeddings = user_embeddings + user_context[:, None, :]
            candidate_embeddings = candidate_embeddings + candidate_context

        # Concatenate all embeddings
        embeddings = jnp.concatenate(
            [user_embeddings, history_embeddings, candidate_embeddings], axis=1
        )
        padding_mask = jnp.concatenate(
            [user_padding_mask, history_padding_mask, candidate_padding_mask], axis=1
        )

        candidate_start_offset = user_padding_mask.shape[1] + history_padding_mask.shape[1]

        return embeddings.astype(self.fprop_dtype), padding_mask, candidate_start_offset

    def __call__(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
        context: Optional[ContextFeatures] = None,
        timestamps: Optional[jax.Array] = None,
    ) -> MultiTaskOutput | RecsysModelOutput:
        """Forward pass for ranking candidates.

        Returns multi-task output if enabled, otherwise standard output.
        """
        config = self.config
        v2_config = config.v2_config

        embeddings, padding_mask, candidate_start_offset = self.build_inputs(
            batch, recsys_embeddings, context, timestamps
        )

        # Apply hierarchical user encoder if enabled
        if v2_config.use_hierarchical_user_encoder:
            hierarchical_encoder = HierarchicalUserEncoder(
                emb_size=config.emb_size,
                recent_window=10,
                num_clusters=5,
                name="hierarchical_user_encoder"
            )

            user_embedding = embeddings[:, 0, :]
            history_embeddings = embeddings[:, 1:candidate_start_offset, :]
            history_mask = padding_mask[:, 1:candidate_start_offset]

            enhanced_user = hierarchical_encoder(
                user_embedding[:, None, :],
                history_embeddings,
                history_mask,
                timestamps=timestamps,
            )

            # Replace user embedding
            embeddings = embeddings.at[:, 0, :].set(enhanced_user)

        # Transformer
        model_output = self.model(
            embeddings,
            padding_mask,
            candidate_start_offset=candidate_start_offset,
        )

        out_embeddings = model_output.embeddings
        out_embeddings = layer_norm(out_embeddings)

        user_embedding = out_embeddings[:, 0, :]
        candidate_embeddings = out_embeddings[:, candidate_start_offset:, :]

        # Multi-task learning if enabled
        if v2_config.use_multi_task_learning:
            multi_task_head = MultiTaskHead(
                emb_size=config.emb_size,
                num_actions=config.num_actions,
                num_topics=v2_config.num_topics,
                name="multi_task_head"
            )

            history_embeddings = out_embeddings[:, 1:candidate_start_offset, :]

            return multi_task_head(
                candidate_embeddings,
                user_embedding,
                history_embeddings,
            )
        else:
            # Standard engagement prediction only
            unembeddings = self._get_unembedding()
            logits = jnp.dot(candidate_embeddings.astype(unembeddings.dtype), unembeddings)
            logits = logits.astype(self.fprop_dtype)

            return RecsysModelOutput(logits=logits)

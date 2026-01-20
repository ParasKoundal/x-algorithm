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

"""Intelligent hash embeddings with collision awareness."""

from dataclasses import dataclass
from typing import Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp


@dataclass
class IntelligentHashEmbedding(hk.Module):
    """Collision-aware hash embedding with learned mixing."""

    emb_size: int
    num_hashes: int
    vocab_size: int
    name: Optional[str] = None

    def __call__(
        self,
        hash_values: jax.Array,  # [B, num_hashes]
        embeddings: jax.Array,  # [B, num_hashes, D]
    ) -> jax.Array:
        """Combine hash embeddings with collision detection.

        Args:
            hash_values: Hash values (0 = padding)
            embeddings: Pre-looked-up embeddings

        Returns:
            combined_embedding: [B, D]
        """

        B, num_hashes, D = embeddings.shape

        # Detect collisions (approximate by counting hash frequency)
        collision_weights = self._compute_collision_weights(hash_values)  # [B, num_hashes]

        # Attention-based mixing (learn which hashes are most informative)
        attention_weights = self._compute_attention_weights(embeddings)  # [B, num_hashes]

        # Combine collision and attention weights
        final_weights = collision_weights * attention_weights
        final_weights = final_weights / (jnp.sum(final_weights, axis=1, keepdims=True) + 1e-8)

        # Weighted combination
        combined = jnp.sum(
            embeddings * final_weights[:, :, None],  # [B, num_hashes, D]
            axis=1
        )  # [B, D]

        return combined

    def _compute_collision_weights(self, hash_values: jax.Array) -> jax.Array:
        """Estimate collision likelihood and downweight colliding hashes."""

        B, num_hashes = hash_values.shape

        # Simple heuristic: if hash value is small (common), it might collide more
        # In production, maintain actual collision statistics
        collision_score = jnp.where(
            hash_values == 0,
            0.0,  # Padding
            1.0 / (jnp.log1p(hash_values.astype(jnp.float32)) + 1.0)
        )

        # Normalize
        collision_weights = collision_score / (
            jnp.sum(collision_score, axis=1, keepdims=True) + 1e-8
        )

        return collision_weights

    def _compute_attention_weights(self, embeddings: jax.Array) -> jax.Array:
        """Learn which hash embeddings are most informative."""

        B, num_hashes, D = embeddings.shape

        # Query: average of all embeddings
        query = jnp.mean(embeddings, axis=1, keepdims=True)  # [B, 1, D]

        # Attention scores
        scores = jnp.matmul(query, embeddings.transpose(0, 2, 1)).squeeze(1)  # [B, num_hashes]

        # Softmax
        attention_weights = jax.nn.softmax(scores, axis=-1)

        return attention_weights


def block_reduce_v2(
    hashes: jax.Array,  # [B, S, num_hashes]
    embeddings: jax.Array,  # [B, S, num_hashes, D]
    num_hashes: int,
    emb_size: int,
    name_prefix: str,
) -> Tuple[jax.Array, jax.Array]:
    """Enhanced hash embedding reduction with collision handling.

    Args:
        hashes: Hash values
        embeddings: Pre-looked-up embeddings
        num_hashes: Number of hash functions
        emb_size: Embedding dimension
        name_prefix: Prefix for parameter names

    Returns:
        reduced_embeddings: [B, S, D]
        padding_mask: [B, S]
    """

    B, S, num_hashes_actual, D = embeddings.shape
    assert num_hashes == num_hashes_actual

    # Reshape for processing
    embeddings_flat = embeddings.reshape(B * S, num_hashes, D)
    hashes_flat = hashes.reshape(B * S, num_hashes)

    # Apply intelligent hashing
    intelligent_hasher = IntelligentHashEmbedding(
        emb_size=emb_size,
        num_hashes=num_hashes,
        vocab_size=100000,  # Approximate vocab size
        name=f"{name_prefix}_intelligent_hash"
    )

    reduced_flat = intelligent_hasher(hashes_flat, embeddings_flat)  # [B*S, D]
    reduced = reduced_flat.reshape(B, S, D)

    # Padding mask (hash 0 is padding)
    padding_mask = (hashes[:, :, 0] != 0).astype(jnp.bool_)

    return reduced, padding_mask

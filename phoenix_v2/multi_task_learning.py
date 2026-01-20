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

"""Multi-task learning framework with auxiliary objectives."""

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional

import haiku as hk
import jax
import jax.numpy as jnp


class MultiTaskOutput(NamedTuple):
    """Output from multi-task model."""

    engagement_logits: jax.Array  # [B, C, num_actions] - primary task
    dwell_time: jax.Array  # [B, C] - auxiliary task
    topic_logits: jax.Array  # [B, C, num_topics] - auxiliary task
    contrastive_scores: jax.Array  # [B, C] - auxiliary task
    next_action_logits: Optional[jax.Array] = None  # [B, S, num_actions] - auxiliary


@dataclass
class MultiTaskHead(hk.Module):
    """Multi-task learning heads."""

    emb_size: int
    num_actions: int
    num_topics: int = 100
    name: Optional[str] = None

    def __call__(
        self,
        candidate_embeddings: jax.Array,  # [B, C, D]
        user_embedding: jax.Array,  # [B, D]
        history_embeddings: Optional[jax.Array] = None,  # [B, S, D]
    ) -> MultiTaskOutput:
        """Compute all task outputs."""

        # Task 1: Engagement prediction (primary)
        engagement_logits = self._engagement_head(candidate_embeddings)

        # Task 2: Dwell time regression
        dwell_time = self._dwell_time_head(candidate_embeddings)

        # Task 3: Topic classification
        topic_logits = self._topic_head(candidate_embeddings)

        # Task 4: Contrastive learning (user-candidate similarity)
        contrastive_scores = self._contrastive_head(
            user_embedding,
            candidate_embeddings
        )

        # Task 5: Next action prediction (if history provided)
        next_action_logits = None
        if history_embeddings is not None:
            next_action_logits = self._next_action_head(history_embeddings)

        return MultiTaskOutput(
            engagement_logits=engagement_logits,
            dwell_time=dwell_time,
            topic_logits=topic_logits,
            contrastive_scores=contrastive_scores,
            next_action_logits=next_action_logits,
        )

    def _engagement_head(self, candidate_embeddings: jax.Array) -> jax.Array:
        """Predict multiple engagement types."""
        hidden = hk.Linear(self.emb_size * 2, name="engagement_hidden")(candidate_embeddings)
        hidden = jax.nn.gelu(hidden)
        logits = hk.Linear(self.num_actions, name="engagement_logits")(hidden)
        return logits

    def _dwell_time_head(self, candidate_embeddings: jax.Array) -> jax.Array:
        """Predict dwell time (seconds)."""
        hidden = hk.Linear(self.emb_size, name="dwell_hidden")(candidate_embeddings)
        hidden = jax.nn.relu(hidden)
        dwell_time = hk.Linear(1, name="dwell_output")(hidden)
        return dwell_time.squeeze(-1)

    def _topic_head(self, candidate_embeddings: jax.Array) -> jax.Array:
        """Predict topic distribution."""
        hidden = hk.Linear(self.emb_size, name="topic_hidden")(candidate_embeddings)
        hidden = jax.nn.gelu(hidden)
        logits = hk.Linear(self.num_topics, name="topic_logits")(hidden)
        return logits

    def _contrastive_head(
        self, user_embedding: jax.Array, candidate_embeddings: jax.Array
    ) -> jax.Array:
        """Compute user-candidate similarity scores."""
        user_proj = hk.Linear(self.emb_size, name="contrastive_user_proj")(user_embedding)
        candidate_proj = hk.Linear(self.emb_size, name="contrastive_candidate_proj")(
            candidate_embeddings
        )

        user_proj = user_proj / (jnp.linalg.norm(user_proj, axis=-1, keepdims=True) + 1e-8)
        candidate_proj = candidate_proj / (
            jnp.linalg.norm(candidate_proj, axis=-1, keepdims=True) + 1e-8
        )

        scores = jnp.matmul(
            user_proj[:, None, :], candidate_proj.transpose(0, 2, 1)
        ).squeeze(1)
        return scores

    def _next_action_head(self, history_embeddings: jax.Array) -> jax.Array:
        """Predict next action from history."""
        hidden = hk.Linear(self.emb_size, name="next_action_hidden")(history_embeddings)
        hidden = jax.nn.gelu(hidden)
        logits = hk.Linear(self.num_actions, name="next_action_logits")(hidden)
        return logits


def multi_task_loss(
    outputs: MultiTaskOutput,
    labels: Dict[str, jax.Array],
    task_weights: Dict[str, float],
) -> tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute weighted multi-task loss."""
    losses = {}

    if 'engagement' in labels:
        engagement_loss = optax_cross_entropy(
            outputs.engagement_logits.reshape(-1, outputs.engagement_logits.shape[-1]),
            labels['engagement'].flatten()
        ).mean()
        losses['engagement'] = engagement_loss

    if 'dwell_time' in labels:
        dwell_loss = jnp.mean((outputs.dwell_time - jnp.log1p(labels['dwell_time'])) ** 2)
        losses['dwell_time'] = dwell_loss

    if 'topic' in labels:
        topic_loss = optax_cross_entropy(
            outputs.topic_logits.reshape(-1, outputs.topic_logits.shape[-1]),
            labels['topic'].flatten()
        ).mean()
        losses['topic'] = topic_loss

    if 'positive_mask' in labels:
        contrastive_loss = compute_contrastive_loss(
            outputs.contrastive_scores, labels['positive_mask']
        )
        losses['contrastive'] = contrastive_loss

    if outputs.next_action_logits is not None and 'next_action' in labels:
        next_action_loss = optax_cross_entropy(
            outputs.next_action_logits.reshape(-1, outputs.next_action_logits.shape[-1]),
            labels['next_action'].flatten()
        ).mean()
        losses['next_action'] = next_action_loss

    total_loss = sum(task_weights.get(task, 1.0) * loss for task, loss in losses.items())
    return total_loss, losses


def optax_cross_entropy(logits, labels):
    """Simple cross-entropy implementation."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(one_hot * log_probs, axis=-1)


def compute_contrastive_loss(
    scores: jax.Array, positive_mask: jax.Array, temperature: float = 0.07
) -> jax.Array:
    """Compute contrastive loss (InfoNCE)."""
    scores = scores / temperature
    log_probs = jax.nn.log_softmax(scores, axis=-1)
    positive_log_probs = log_probs * positive_mask
    loss = -jnp.sum(positive_log_probs) / jnp.maximum(jnp.sum(positive_mask), 1.0)
    return loss

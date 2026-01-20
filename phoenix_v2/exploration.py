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

"""Exploration/exploitation strategies for recommendations."""

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp


class ExplorationPolicy:
    """Balance exploration and exploitation using various strategies."""

    def __init__(
        self,
        epsilon: float = 0.1,
        thompson_beta_prior: Tuple[float, float] = (1.0, 1.0),
        ucb_confidence: float = 2.0,
    ):
        self.epsilon = epsilon
        self.thompson_prior = thompson_beta_prior
        self.ucb_confidence = ucb_confidence
        self.item_stats = {}  # Track (alpha, beta, count) for each item

    def select_candidates(
        self,
        candidate_ids: List[str],
        predicted_scores: jax.Array,  # [C]
        exploration_budget: int = 5,
        strategy: str = "thompson",
        key: jax.random.PRNGKey = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Select candidates balancing exploration and exploitation.

        Args:
            candidate_ids: List of candidate IDs
            predicted_scores: Model predictions [C]
            exploration_budget: Number of slots for exploration
            strategy: "epsilon_greedy", "thompson", or "ucb"
            key: Random key for sampling

        Returns:
            indices: Selected candidate indices
            final_scores: Final scores used for selection
        """

        if key is None:
            key = jax.random.PRNGKey(0)

        C = len(candidate_ids)

        if strategy == "epsilon_greedy":
            return self._epsilon_greedy(
                candidate_ids, predicted_scores, exploration_budget, key
            )
        elif strategy == "thompson":
            return self._thompson_sampling(
                candidate_ids, predicted_scores, key
            )
        elif strategy == "ucb":
            return self._ucb(
                candidate_ids, predicted_scores
            )
        else:
            # Pure exploitation
            indices = jnp.argsort(predicted_scores)[::-1]
            return indices, predicted_scores

    def _epsilon_greedy(
        self,
        candidate_ids: List[str],
        predicted_scores: jax.Array,
        exploration_budget: int,
        key: jax.random.PRNGKey,
    ) -> Tuple[jax.Array, jax.Array]:
        """Epsilon-greedy exploration."""

        C = len(candidate_ids)

        # With probability epsilon, explore
        explore_key, select_key = jax.random.split(key)
        should_explore = jax.random.uniform(explore_key) < self.epsilon

        if should_explore:
            # Random exploration
            explore_indices = jax.random.choice(
                select_key,
                C,
                shape=(exploration_budget,),
                replace=False
            )

            # Exploitation for remaining slots
            exploit_scores = predicted_scores.at[explore_indices].set(-jnp.inf)
            exploit_indices = jnp.argsort(exploit_scores)[::-1][:(C - exploration_budget)]

            # Combine
            indices = jnp.concatenate([explore_indices, exploit_indices])
            final_scores = predicted_scores
        else:
            # Pure exploitation
            indices = jnp.argsort(predicted_scores)[::-1]
            final_scores = predicted_scores

        return indices, final_scores

    def _thompson_sampling(
        self,
        candidate_ids: List[str],
        predicted_scores: jax.Array,
        key: jax.random.PRNGKey,
    ) -> Tuple[jax.Array, jax.Array]:
        """Thompson sampling with Beta distribution."""

        C = len(candidate_ids)
        thompson_scores = []

        for i, candidate_id in enumerate(candidate_ids):
            # Get Beta parameters for this candidate
            alpha, beta, _ = self.item_stats.get(
                candidate_id,
                (*self.thompson_prior, 0)
            )

            # Sample from Beta distribution
            sample_key = jax.random.fold_in(key, i)
            sample = jax.random.beta(sample_key, alpha, beta)

            # Combine with model prediction
            thompson_score = 0.7 * predicted_scores[i] + 0.3 * sample
            thompson_scores.append(thompson_score)

        thompson_scores = jnp.array(thompson_scores)

        # Select by Thompson scores
        indices = jnp.argsort(thompson_scores)[::-1]

        return indices, thompson_scores

    def _ucb(
        self,
        candidate_ids: List[str],
        predicted_scores: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Upper Confidence Bound (UCB) exploration."""

        C = len(candidate_ids)
        ucb_scores = []

        total_count = sum(
            stats[2] for stats in self.item_stats.values()
        ) + 1

        for i, candidate_id in enumerate(candidate_ids):
            alpha, beta, count = self.item_stats.get(
                candidate_id,
                (*self.thompson_prior, 0)
            )

            # UCB1 formula
            mean_estimate = alpha / (alpha + beta)
            exploration_bonus = jnp.sqrt(
                self.ucb_confidence * jnp.log(total_count) / (count + 1)
            )

            # Combine with model prediction
            ucb_score = 0.7 * predicted_scores[i] + 0.3 * (mean_estimate + exploration_bonus)
            ucb_scores.append(ucb_score)

        ucb_scores = jnp.array(ucb_scores)

        # Select by UCB scores
        indices = jnp.argsort(ucb_scores)[::-1]

        return indices, ucb_scores

    def update_stats(self, candidate_id: str, was_engaged: bool):
        """Update Beta distribution parameters for Thompson sampling."""

        alpha, beta, count = self.item_stats.get(
            candidate_id,
            (*self.thompson_prior, 0)
        )

        if was_engaged:
            alpha += 1.0
        else:
            beta += 1.0

        count += 1

        self.item_stats[candidate_id] = (alpha, beta, count)

    def get_stats(self, candidate_id: str) -> Dict[str, float]:
        """Get statistics for a candidate."""

        alpha, beta, count = self.item_stats.get(
            candidate_id,
            (*self.thompson_prior, 0)
        )

        ctr = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.0

        return {
            'ctr': ctr,
            'count': count,
            'alpha': alpha,
            'beta': beta,
        }

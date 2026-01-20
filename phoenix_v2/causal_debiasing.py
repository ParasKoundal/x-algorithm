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

"""Causal inference for debiasing recommendations."""

from dataclasses import dataclass
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp


@dataclass
class PropensityScoreEstimator(hk.Module):
    """Estimate propensity scores for inverse propensity weighting."""

    hidden_size: int = 128
    name: Optional[str] = None

    def __call__(
        self,
        positions: jax.Array,  # [B, C] - position in feed
        popularities: jax.Array,  # [B, C] - historical CTR
        features: Optional[jax.Array] = None,  # [B, C, D] - additional features
    ) -> jax.Array:
        """Estimate probability of showing each candidate.

        Returns:
            propensity_scores: [B, C] - P(shown | position, popularity, features)
        """

        B, C = positions.shape

        # Position encoding (log scale)
        position_features = jnp.stack([
            jnp.log1p(positions),
            1.0 / (positions + 1.0),
        ], axis=-1)  # [B, C, 2]

        # Popularity features
        popularity_features = jnp.stack([
            popularities,
            jnp.log1p(popularities),
        ], axis=-1)  # [B, C, 2]

        # Combine features
        if features is not None:
            all_features = jnp.concatenate([
                position_features,
                popularity_features,
                features,
            ], axis=-1)
        else:
            all_features = jnp.concatenate([
                position_features,
                popularity_features,
            ], axis=-1)

        # MLP to predict propensity
        hidden = hk.Linear(self.hidden_size, name="propensity_hidden_1")(all_features)
        hidden = jax.nn.relu(hidden)

        hidden = hk.Linear(self.hidden_size // 2, name="propensity_hidden_2")(hidden)
        hidden = jax.nn.relu(hidden)

        logits = hk.Linear(1, name="propensity_logits")(hidden)
        propensity_scores = jax.nn.sigmoid(logits).squeeze(-1)  # [B, C]

        # Ensure minimum propensity to avoid division by zero
        propensity_scores = jnp.clip(propensity_scores, 0.01, 1.0)

        return propensity_scores


class CausalRanker:
    """Debias recommendations using causal inference."""

    def __init__(
        self,
        propensity_estimator: Optional[PropensityScoreEstimator] = None,
        position_bias_decay: float = 0.1,
        popularity_bias_strength: float = 0.5,
    ):
        self.propensity_estimator = propensity_estimator
        self.position_bias_decay = position_bias_decay
        self.popularity_bias_strength = popularity_bias_strength

    def debias_scores(
        self,
        raw_scores: jax.Array,  # [B, C]
        positions: Optional[jax.Array] = None,  # [B, C] - previous positions
        popularities: Optional[jax.Array] = None,  # [B, C] - historical CTR
        use_ipw: bool = True,
    ) -> jax.Array:
        """Debias scores using causal inference techniques.

        Args:
            raw_scores: Model predictions
            positions: Previous feed positions
            popularities: Historical engagement rates
            use_ipw: Whether to use inverse propensity weighting

        Returns:
            debiased_scores: Causally debiased scores
        """

        debiased_scores = raw_scores

        # Method 1: Position bias correction
        if positions is not None:
            position_effect = self._estimate_position_effect(positions)
            debiased_scores = debiased_scores - position_effect

        # Method 2: Popularity bias correction
        if popularities is not None:
            popularity_effect = self._estimate_popularity_effect(popularities)
            debiased_scores = debiased_scores - popularity_effect

        # Method 3: Inverse Propensity Weighting (IPW)
        if use_ipw and self.propensity_estimator is not None:
            propensity = self.propensity_estimator(
                positions if positions is not None else jnp.zeros_like(raw_scores),
                popularities if popularities is not None else jnp.zeros_like(raw_scores),
            )
            debiased_scores = debiased_scores / propensity

        return debiased_scores

    def _estimate_position_effect(self, positions: jax.Array) -> jax.Array:
        """Estimate causal effect of position on engagement.

        Position 0 (top) has highest CTR regardless of content quality.
        Model this bias and subtract it.
        """

        # Exponential decay: earlier positions have higher bias
        position_bias = jnp.exp(-self.position_bias_decay * positions)

        return position_bias

    def _estimate_popularity_effect(self, popularities: jax.Array) -> jax.Array:
        """Estimate causal effect of popularity on engagement.

        Popular items get more clicks due to social proof, not just quality.
        """

        # Log transform to handle skewed distribution
        popularity_bias = self.popularity_bias_strength * jnp.log1p(popularities)

        return popularity_bias

    def compute_ate(
        self,
        treatment_scores: jax.Array,  # [B, C] - scores with treatment
        control_scores: jax.Array,  # [B, C] - scores without treatment
        propensity: jax.Array,  # [B, C] - propensity scores
    ) -> jax.Array:
        """Compute Average Treatment Effect (ATE).

        Estimates the causal effect of a treatment (e.g., showing a candidate).
        """

        # Inverse propensity weighting for ATE
        treatment_effect = treatment_scores - control_scores

        # Weight by inverse propensity
        weighted_effect = treatment_effect / propensity

        # Average across batch and candidates
        ate = jnp.mean(weighted_effect)

        return ate


def doubly_robust_estimator(
    rewards: jax.Array,  # [B, C] - observed rewards
    predicted_rewards: jax.Array,  # [B, C] - model predictions
    propensity: jax.Array,  # [B, C] - propensity scores
    treatment_mask: jax.Array,  # [B, C] - 1 if shown, 0 otherwise
) -> jax.Array:
    """Doubly robust estimator for offline policy evaluation.

    Combines model-based and importance sampling estimates.
    """

    # Direct method (model-based)
    direct_estimate = predicted_rewards

    # Importance sampling correction
    residual = (rewards - predicted_rewards) * treatment_mask / propensity

    # Combine
    doubly_robust = direct_estimate + residual

    return jnp.mean(doubly_robust)

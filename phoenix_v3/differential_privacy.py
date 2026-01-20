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

"""Differential Privacy for GDPR/CCPA Compliance.

Implements (ε, δ)-differential privacy using:
- DP-SGD for training (gradient clipping + noise)
- Noisy predictions for inference
- Privacy budget tracking
- Privacy accounting (Rényi DP)

Reference: Abadi et al. "Deep Learning with Differential Privacy" (2016)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize


@dataclass
class PrivacyAccountant:
    """Track cumulative privacy loss.

    Uses Rényi Differential Privacy for tight composition.
    """

    epsilon: float = 1.0
    delta: float = 1e-5
    orders: tuple = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10.0, 20.0, 50.0, 100.0)

    def __post_init__(self):
        self.rdp_spent = {order: 0.0 for order in self.orders}
        self.steps = 0

    def add_step(
        self,
        noise_multiplier: float,
        batch_size: int,
        dataset_size: int,
        sampling_probability: Optional[float] = None,
    ):
        """Add privacy cost of one training step."""

        if sampling_probability is None:
            sampling_probability = batch_size / dataset_size

        # Compute RDP for each order
        for order in self.orders:
            rdp = self._compute_rdp(noise_multiplier, sampling_probability, order)
            self.rdp_spent[order] += rdp

        self.steps += 1

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Convert RDP to (ε, δ)-DP."""

        if delta is None:
            delta = self.delta

        # Convert each RDP bound to (ε, δ)
        epsilons = []
        for order, rdp in self.rdp_spent.items():
            if order == 1:
                continue  # Skip α=1
            eps = rdp + np.log(1 / delta) / (order - 1)
            epsilons.append(eps)

        return min(epsilons) if epsilons else float('inf')

    def _compute_rdp(
        self,
        noise_multiplier: float,
        sampling_prob: float,
        alpha: float,
    ) -> float:
        """Compute Rényi DP for one step.

        RDP with order α for Gaussian mechanism.
        """

        if alpha <= 1:
            return 0.0

        # Gaussian mechanism RDP
        return alpha * sampling_prob**2 / (2 * noise_multiplier**2)

    def is_budget_exceeded(self) -> bool:
        """Check if privacy budget is exceeded."""
        current_epsilon = self.get_epsilon()
        return current_epsilon > self.epsilon

    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        current_epsilon = self.get_epsilon()
        return max(0, self.epsilon - current_epsilon)


class DPOptimizer:
    """Differentially Private SGD optimizer."""

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        noise_multiplier: float = 1.1,
    ):
        """Initialize DP optimizer.

        Args:
            epsilon: Privacy parameter (smaller = more private)
            delta: Privacy parameter (probability of privacy breach)
            clip_norm: L2 norm for gradient clipping
            noise_multiplier: Noise scale (larger = more private)
        """

        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.accountant = PrivacyAccountant(epsilon=epsilon, delta=delta)

    def dp_gradient_step(
        self,
        params: Any,
        grads: Any,
        batch_size: int,
        dataset_size: int,
        key: jax.random.PRNGKey,
    ) -> Tuple[Any, Dict[str, float]]:
        """Apply DP to gradients (clip + noise).

        Args:
            params: Model parameters
            grads: Gradients (per-example or aggregated)
            batch_size: Batch size
            dataset_size: Total dataset size
            key: Random key for noise generation

        Returns:
            dp_grads: DP gradients
            info: Dict with privacy metrics
        """

        # Step 1: Clip gradients
        clipped_grads = self._clip_gradients(grads, self.clip_norm)

        # Step 2: Add Gaussian noise
        noise_scale = self.noise_multiplier * self.clip_norm
        noisy_grads = self._add_noise(clipped_grads, noise_scale, key)

        # Step 3: Update privacy accountant
        self.accountant.add_step(
            noise_multiplier=self.noise_multiplier,
            batch_size=batch_size,
            dataset_size=dataset_size,
        )

        # Return info
        info = {
            'epsilon_spent': self.accountant.get_epsilon(),
            'epsilon_remaining': self.accountant.get_remaining_budget(),
            'steps': self.accountant.steps,
            'budget_exceeded': self.accountant.is_budget_exceeded(),
        }

        return noisy_grads, info

    def _clip_gradients(self, grads: Any, max_norm: float) -> Any:
        """Clip gradients to max L2 norm."""

        # Compute L2 norm
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree_leaves(grads))
        )

        # Clip
        clip_factor = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))

        clipped = jax.tree_map(
            lambda g: g * clip_factor,
            grads
        )

        return clipped

    def _add_noise(
        self,
        grads: Any,
        noise_scale: float,
        key: jax.random.PRNGKey,
    ) -> Any:
        """Add Gaussian noise to gradients."""

        # Generate noise for each parameter
        noise_tree = jax.tree_map(
            lambda g: jax.random.normal(key, g.shape) * noise_scale,
            grads
        )

        # Add noise
        noisy_grads = jax.tree_map(
            lambda g, n: g + n,
            grads,
            noise_tree
        )

        return noisy_grads


class DPPredictor:
    """Differentially private predictions at inference time."""

    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0):
        """Initialize DP predictor.

        Args:
            epsilon: Privacy budget per prediction
            sensitivity: Sensitivity of prediction function
        """

        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def private_predict(
        self,
        logits: jax.Array,
        key: jax.random.PRNGKey,
        mechanism: str = "laplace",
    ) -> jax.Array:
        """Add calibrated noise to predictions.

        Args:
            logits: Model predictions
            key: Random key
            mechanism: "laplace" or "gaussian"

        Returns:
            noisy_logits: Private predictions
        """

        if mechanism == "laplace":
            # Laplace mechanism: scale = sensitivity / epsilon
            scale = self.sensitivity / self.epsilon
            noise = jax.random.laplace(key, logits.shape) * scale

        elif mechanism == "gaussian":
            # Gaussian mechanism: scale = sensitivity * sqrt(2 * log(1.25/delta)) / epsilon
            delta = 1e-5
            scale = self.sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / self.epsilon
            noise = jax.random.normal(key, logits.shape) * scale

        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        noisy_logits = logits + noise

        return noisy_logits

    def report_noisy_max(
        self,
        values: jax.Array,
        key: jax.random.PRNGKey,
    ) -> int:
        """Report argmax with differential privacy (exponential mechanism)."""

        # Add Gumbel noise (equivalent to exponential mechanism)
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(key, values.shape)))

        # Scale by epsilon
        noisy_values = values + gumbel_noise / self.epsilon

        # Return noisy argmax
        return jnp.argmax(noisy_values)


def compute_noise_multiplier(
    epsilon: float,
    delta: float,
    num_steps: int,
    batch_size: int,
    dataset_size: int,
) -> float:
    """Compute required noise multiplier for target (ε, δ).

    Uses binary search to find noise_multiplier that achieves target privacy.
    """

    def privacy_cost(noise_mult):
        accountant = PrivacyAccountant(epsilon=epsilon, delta=delta)
        for _ in range(num_steps):
            accountant.add_step(
                noise_multiplier=noise_mult,
                batch_size=batch_size,
                dataset_size=dataset_size,
            )
        return accountant.get_epsilon() - epsilon

    # Binary search for noise multiplier
    result = optimize.brentq(privacy_cost, 0.1, 10.0)

    return result


# Example usage
if __name__ == "__main__":
    # Test privacy accountant
    accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5)

    # Simulate 1000 training steps
    for i in range(1000):
        accountant.add_step(
            noise_multiplier=1.1,
            batch_size=256,
            dataset_size=100000,
        )

    print(f"Privacy spent after 1000 steps: ε = {accountant.get_epsilon():.4f}")
    print(f"Budget exceeded: {accountant.is_budget_exceeded()}")

    # Test DP optimizer
    dp_opt = DPOptimizer(epsilon=1.0, delta=1e-5, clip_norm=1.0, noise_multiplier=1.1)

    # Dummy gradients
    params = {'w': jnp.ones((10, 10))}
    grads = {'w': jnp.ones((10, 10)) * 0.1}

    key = jax.random.PRNGKey(0)
    dp_grads, info = dp_opt.dp_gradient_step(
        params, grads, batch_size=256, dataset_size=100000, key=key
    )

    print(f"\nDP Gradient Step:")
    print(f"  Epsilon spent: {info['epsilon_spent']:.4f}")
    print(f"  Epsilon remaining: {info['epsilon_remaining']:.4f}")

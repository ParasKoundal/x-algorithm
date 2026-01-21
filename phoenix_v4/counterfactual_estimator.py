"""Counterfactual Fairness Evaluator for Phoenix V4

Tests counterfactual fairness: outcomes should be similar under demographic counterfactuals.

Key idea: If we change protected attributes (age, gender, etc.) but keep behavior
the same, the model's predictions should remain similar.

CRITICAL: Model should NOT use demographics for scoring.
This tool TESTS that it doesn't, not enables it to do so.

Focus: Individual fairness - similar individuals treated similarly.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualResult:
    """Result of counterfactual fairness evaluation."""

    original_score: float
    """Original prediction score."""

    counterfactual_scores: List[float]
    """Scores under counterfactual demographics."""

    max_gap: float
    """Maximum gap between original and counterfactuals."""

    mean_gap: float
    """Mean gap between original and counterfactuals."""

    is_fair: bool
    """Whether counterfactual fairness holds (gap < threshold)."""

    threshold: float
    """Fairness threshold used."""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'original_score': float(self.original_score),
            'counterfactual_scores': [float(s) for s in self.counterfactual_scores],
            'max_gap': float(self.max_gap),
            'mean_gap': float(self.mean_gap),
            'is_fair': bool(self.is_fair),
            'threshold': float(self.threshold),
        }


class CounterfactualFairnessEvaluator:
    """Evaluates counterfactual fairness.

    Tests whether changing protected attributes (while keeping behavior constant)
    leads to similar predictions.

    This is a TEST of fairness, not a mechanism to USE protected attributes.
    """

    def __init__(
        self,
        fairness_threshold: float = 0.05,
        protected_attributes: Optional[List[str]] = None
    ):
        """Initialize counterfactual fairness evaluator.

        Args:
            fairness_threshold: Max allowed gap (5% default)
            protected_attributes: List of protected attributes for testing
        """
        self.fairness_threshold = fairness_threshold

        if protected_attributes is None:
            protected_attributes = ['age', 'gender', 'race', 'ethnicity']

        self.protected_attributes = protected_attributes

        logger.info(
            f"CounterfactualFairnessEvaluator initialized: "
            f"threshold={fairness_threshold}, "
            f"protected_attrs={protected_attributes}"
        )
        logger.warning(
            "Protected attributes used for TESTING only, NEVER for scoring!"
        )

    def evaluate_counterfactual_fairness(
        self,
        model: Callable,
        test_data: List[Dict],
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate counterfactual fairness across test dataset.

        For each user, compute:
        1. Original score with real demographics
        2. Counterfactual scores with altered demographics
        3. Compare scores - should be similar

        Args:
            model: Recommendation model (function: user_dict -> score)
            test_data: Test dataset with protected attributes and behavior
            verbose: Whether to print progress

        Returns:
            fairness_metrics: {
                'counterfactual_parity': 0-1 (closer to 1 = more fair),
                'max_counterfactual_gap': 0-1 (smaller = more fair),
                'mean_gap': 0-1,
                'pass_rate': 0-1 (fraction passing fairness test),
                'num_tested': int
            }
        """
        results = []
        num_passed = 0

        for i, user in enumerate(test_data):
            if verbose and i % 100 == 0:
                logger.info(f"Testing counterfactual fairness: {i}/{len(test_data)}")

            # Test this user
            result = self.evaluate_user(model, user)
            results.append(result)

            if result.is_fair:
                num_passed += 1

        # Aggregate results
        all_gaps = [r.max_gap for r in results]
        mean_gaps = [r.mean_gap for r in results]

        fairness_metrics = {
            'counterfactual_parity': 1.0 - np.mean(all_gaps),
            'max_counterfactual_gap': float(np.max(all_gaps)),
            'mean_gap': float(np.mean(mean_gaps)),
            'pass_rate': num_passed / len(results),
            'num_tested': len(results)
        }

        if verbose:
            logger.info("Counterfactual Fairness Results:")
            logger.info(f"  Parity: {fairness_metrics['counterfactual_parity']:.3f}")
            logger.info(f"  Max gap: {fairness_metrics['max_counterfactual_gap']:.3f}")
            logger.info(f"  Mean gap: {fairness_metrics['mean_gap']:.3f}")
            logger.info(f"  Pass rate: {fairness_metrics['pass_rate']:.1%}")

        return fairness_metrics

    def evaluate_user(
        self,
        model: Callable,
        user: Dict
    ) -> CounterfactualResult:
        """Evaluate counterfactual fairness for a single user.

        Args:
            model: Recommendation model
            user: User data with demographics and behavior

        Returns:
            result: CounterfactualResult
        """
        # Original score
        original_score = model(user)

        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(user)

        # Score counterfactuals
        counterfactual_scores = []
        for cf_user in counterfactuals:
            # CRITICAL: Model should NOT use demographics for scoring
            # This test validates that changing demographics doesn't change scores
            cf_score = model(cf_user)
            counterfactual_scores.append(cf_score)

        # Compute gaps
        gaps = [abs(cf_score - original_score) for cf_score in counterfactual_scores]
        max_gap = max(gaps) if gaps else 0.0
        mean_gap = np.mean(gaps) if gaps else 0.0

        # Check fairness
        is_fair = max_gap < self.fairness_threshold

        return CounterfactualResult(
            original_score=float(original_score),
            counterfactual_scores=counterfactual_scores,
            max_gap=float(max_gap),
            mean_gap=float(mean_gap),
            is_fair=is_fair,
            threshold=self.fairness_threshold
        )

    def _generate_counterfactuals(self, user: Dict) -> List[Dict]:
        """Generate counterfactual versions of user.

        Changes protected attributes while keeping behavior constant.

        Args:
            user: Original user data

        Returns:
            counterfactuals: List of counterfactual user dicts
        """
        counterfactuals = []

        # For each protected attribute, generate counterfactuals
        for attr in self.protected_attributes:
            if attr not in user:
                continue

            # Generate alternative values for this attribute
            alternatives = self._get_alternative_values(attr, user[attr])

            for alt_value in alternatives:
                # Create counterfactual
                cf_user = user.copy()
                cf_user[attr] = alt_value
                counterfactuals.append(cf_user)

        return counterfactuals

    def _get_alternative_values(self, attribute: str, current_value: any) -> List[any]:
        """Get alternative values for protected attribute.

        Args:
            attribute: Attribute name
            current_value: Current value

        Returns:
            alternatives: List of alternative values
        """
        if attribute == 'age':
            # Test different age groups
            age_groups = [18, 25, 35, 45, 55, 65]
            return [age for age in age_groups if age != current_value]

        elif attribute == 'gender':
            # Test different genders
            genders = ['M', 'F', 'Non-binary', 'Prefer not to say']
            return [g for g in genders if g != current_value]

        elif attribute == 'race' or attribute == 'ethnicity':
            # Test different races/ethnicities
            categories = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
            return [cat for cat in categories if cat != current_value]

        else:
            # For other attributes, just flip binary or rotate categorical
            if isinstance(current_value, bool):
                return [not current_value]
            elif isinstance(current_value, (int, float)):
                return [current_value + 10, current_value - 10]
            else:
                return []

    def test_model_fairness(
        self,
        model: Callable,
        test_users: List[Dict],
        fail_fast: bool = False
    ) -> Tuple[bool, Dict]:
        """Test whether model satisfies counterfactual fairness.

        Args:
            model: Model to test
            test_users: Test users with demographics
            fail_fast: If True, stop at first failure

        Returns:
            passes: Whether model passes test
            metrics: Detailed metrics
        """
        metrics = self.evaluate_counterfactual_fairness(
            model,
            test_users,
            verbose=True
        )

        # Check if model passes
        passes = metrics['pass_rate'] > 0.95  # 95% pass rate required

        if not passes:
            logger.warning(
                f"Model FAILS counterfactual fairness test! "
                f"Pass rate: {metrics['pass_rate']:.1%} (need 95%+)"
            )
        else:
            logger.info(
                f"Model PASSES counterfactual fairness test. "
                f"Pass rate: {metrics['pass_rate']:.1%}"
            )

        return passes, metrics


def create_synthetic_test_data(
    num_users: int = 100,
    include_demographics: bool = True
) -> List[Dict]:
    """Create synthetic test data for counterfactual fairness testing.

    Args:
        num_users: Number of users to generate
        include_demographics: Whether to include protected attributes

    Returns:
        test_data: List of user dicts
    """
    test_data = []

    for i in range(num_users):
        user = {
            # Behavioral features (used by model)
            'user_id': i,
            'engagement_history': np.random.rand(10).tolist(),
            'following_count': int(np.random.exponential(100)),
            'follower_count': int(np.random.exponential(150)),
            'account_age_days': int(np.random.uniform(30, 365 * 5)),
            'avg_engagement_rate': float(np.random.beta(2, 5)),
        }

        # Protected attributes (NOT used by model, only for testing)
        if include_demographics:
            user['age'] = int(np.random.choice([18, 25, 35, 45, 55, 65]))
            user['gender'] = np.random.choice(['M', 'F', 'Non-binary'])
            user['race'] = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'])

        test_data.append(user)

    return test_data


def test_counterfactual_evaluator():
    """Test counterfactual fairness evaluator."""

    # Create a FAIR model (only uses behavior, not demographics)
    def fair_model(user: Dict) -> float:
        """Fair model: only uses behavioral features."""
        score = (
            user.get('avg_engagement_rate', 0.0) * 0.5 +
            np.log1p(user.get('follower_count', 1)) / 10.0 * 0.3 +
            user.get('account_age_days', 0) / 1000.0 * 0.2
        )
        return float(score)

    # Create an UNFAIR model (uses demographics)
    def unfair_model(user: Dict) -> float:
        """Unfair model: uses demographics (BAD!)."""
        base_score = fair_model(user)

        # Add bias based on demographics (THIS IS WRONG!)
        if user.get('age', 30) < 30:
            base_score *= 1.2  # Bias toward young users
        if user.get('gender') == 'M':
            base_score *= 1.1  # Bias toward one gender

        return base_score

    # Create test data
    test_data = create_synthetic_test_data(num_users=50)

    # Create evaluator
    evaluator = CounterfactualFairnessEvaluator(fairness_threshold=0.05)

    print("Testing Counterfactual Fairness")
    print("=" * 50)

    # Test FAIR model
    print("\n1. Testing FAIR model (behavior only):")
    fair_passes, fair_metrics = evaluator.test_model_fairness(fair_model, test_data)
    print(f"   Result: {'PASS' if fair_passes else 'FAIL'}")
    print(f"   Pass rate: {fair_metrics['pass_rate']:.1%}")

    # Test UNFAIR model
    print("\n2. Testing UNFAIR model (uses demographics):")
    unfair_passes, unfair_metrics = evaluator.test_model_fairness(unfair_model, test_data)
    print(f"   Result: {'PASS' if unfair_passes else 'FAIL'}")
    print(f"   Pass rate: {unfair_metrics['pass_rate']:.1%}")

    print("\n" + "=" * 50)

    # Assertions
    assert fair_passes, "Fair model should pass counterfactual fairness!"
    assert not unfair_passes, "Unfair model should fail counterfactual fairness!"

    print("✓ Counterfactual fairness evaluator test passed!")


if __name__ == "__main__":
    test_counterfactual_evaluator()

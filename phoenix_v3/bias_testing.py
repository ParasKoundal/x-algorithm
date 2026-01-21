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

"""Comprehensive Bias Testing Framework.

This module provides tools to detect and prevent bias in recommendation systems.

Tests included:
1. Demographic parity testing
2. Equal opportunity testing
3. Disparate impact analysis
4. Individual fairness testing
5. Intersectional bias detection
6. Content diversity testing

CRITICAL: All tests use protected attributes ONLY for testing/monitoring,
NEVER for actual recommendations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings

import numpy as np
from scipy import stats


@dataclass
class BiasTestResult:
    """Results from bias testing."""

    test_name: str
    passed: bool
    score: float  # Test-specific score
    threshold: float  # Pass threshold
    details: Dict[str, Any]
    recommendations: List[str]  # How to fix if failed

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [
            f"{self.test_name}: {status}",
            f"  Score: {self.score:.4f} (threshold: {self.threshold})",
        ]

        if self.details:
            lines.append("  Details:")
            for key, value in self.details.items():
                lines.append(f"    {key}: {value}")

        if not self.passed and self.recommendations:
            lines.append("  Recommendations:")
            for rec in self.recommendations:
                lines.append(f"    - {rec}")

        return "\n".join(lines)


class BiasTestSuite:
    """Comprehensive bias testing suite."""

    def __init__(self, fairness_threshold: float = 0.1):
        """Initialize bias test suite.

        Args:
            fairness_threshold: Maximum acceptable difference between groups
        """

        self.fairness_threshold = fairness_threshold
        self.test_results = []

    def run_all_tests(
        self,
        model: Any,
        test_data: List[Dict],
        protected_attributes: Optional[List[str]] = None,
    ) -> List[BiasTestResult]:
        """Run comprehensive bias test suite.

        Args:
            model: Model to test
            test_data: Test examples with features and protected attributes
            protected_attributes: Attributes to test for bias

        Returns:
            results: List of test results
        """

        warnings.warn(
            "Running bias tests. Protected attributes used ONLY for testing, "
            "NEVER for predictions.",
            UserWarning
        )

        self.test_results = []

        # Run tests
        self.test_results.append(self.test_demographic_parity(model, test_data, protected_attributes))
        self.test_results.append(self.test_equal_opportunity(model, test_data, protected_attributes))
        self.test_results.append(self.test_disparate_impact(model, test_data, protected_attributes))
        self.test_results.append(self.test_individual_fairness(model, test_data))
        self.test_results.append(self.test_calibration(model, test_data, protected_attributes))
        self.test_results.append(self.test_content_diversity(test_data))

        return self.test_results

    def test_demographic_parity(
        self,
        model: Any,
        test_data: List[Dict],
        protected_attributes: Optional[List[str]] = None,
    ) -> BiasTestResult:
        """Test for demographic parity.

        Demographic parity: P(Y=1|A=a) ≈ P(Y=1|A=b) for groups a, b

        Ensures positive prediction rate is similar across groups.
        """

        if not protected_attributes:
            return BiasTestResult(
                test_name="Demographic Parity",
                passed=True,
                score=0.0,
                threshold=self.fairness_threshold,
                details={"note": "No protected attributes to test"},
                recommendations=[],
            )

        # Group by protected attributes
        groups = self._group_by_attributes(test_data, protected_attributes)

        # Compute positive rate per group
        positive_rates = {}
        for group_key, group_data in groups.items():
            predictions = [
                model.predict(example['features']) > 0.5
                for example in group_data
            ]
            positive_rate = np.mean(predictions)
            positive_rates[group_key] = positive_rate

        # Compute max difference
        if len(positive_rates) > 1:
            max_diff = max(positive_rates.values()) - min(positive_rates.values())
        else:
            max_diff = 0.0

        passed = max_diff <= self.fairness_threshold

        recommendations = []
        if not passed:
            recommendations.append("Apply fairness constraints during training")
            recommendations.append("Use re-weighting or re-sampling techniques")
            recommendations.append("Consider post-processing calibration")

        return BiasTestResult(
            test_name="Demographic Parity",
            passed=passed,
            score=max_diff,
            threshold=self.fairness_threshold,
            details={"positive_rates": positive_rates, "max_difference": max_diff},
            recommendations=recommendations,
        )

    def test_equal_opportunity(
        self,
        model: Any,
        test_data: List[Dict],
        protected_attributes: Optional[List[str]] = None,
    ) -> BiasTestResult:
        """Test for equal opportunity.

        Equal opportunity: TPR should be similar across groups.
        TPR = P(Ŷ=1|Y=1,A=a) ≈ P(Ŷ=1|Y=1,A=b)

        Ensures positive outcomes are equally likely across groups.
        """

        if not protected_attributes:
            return BiasTestResult(
                test_name="Equal Opportunity",
                passed=True,
                score=0.0,
                threshold=self.fairness_threshold,
                details={"note": "No protected attributes to test"},
                recommendations=[],
            )

        # Group by protected attributes
        groups = self._group_by_attributes(test_data, protected_attributes)

        # Compute TPR per group
        tpr_by_group = {}
        for group_key, group_data in groups.items():
            # Filter to positive examples only
            positive_examples = [ex for ex in group_data if ex.get('label', 0) > 0.5]

            if not positive_examples:
                continue

            # TPR = P(prediction=1 | label=1)
            predictions = [
                model.predict(example['features']) > 0.5
                for example in positive_examples
            ]
            tpr = np.mean(predictions)
            tpr_by_group[group_key] = tpr

        # Compute max difference
        if len(tpr_by_group) > 1:
            max_diff = max(tpr_by_group.values()) - min(tpr_by_group.values())
        else:
            max_diff = 0.0

        passed = max_diff <= self.fairness_threshold

        recommendations = []
        if not passed:
            recommendations.append("Ensure balanced recall across groups")
            recommendations.append("Apply equal opportunity post-processing")
            recommendations.append("Check for proxy features correlating with protected attributes")

        return BiasTestResult(
            test_name="Equal Opportunity",
            passed=passed,
            score=max_diff,
            threshold=self.fairness_threshold,
            details={"tpr_by_group": tpr_by_group, "max_difference": max_diff},
            recommendations=recommendations,
        )

    def test_disparate_impact(
        self,
        model: Any,
        test_data: List[Dict],
        protected_attributes: Optional[List[str]] = None,
    ) -> BiasTestResult:
        """Test for disparate impact.

        Disparate impact ratio: (positive rate for protected) / (positive rate for reference)
        Should be >= 0.8 (80% rule)

        Tests if one group is disproportionately affected.
        """

        if not protected_attributes or len(protected_attributes) == 0:
            return BiasTestResult(
                test_name="Disparate Impact",
                passed=True,
                score=1.0,
                threshold=0.8,
                details={"note": "No protected attributes to test"},
                recommendations=[],
            )

        # Simple binary test: compare first two groups
        groups = self._group_by_attributes(test_data, protected_attributes)

        if len(groups) < 2:
            return BiasTestResult(
                test_name="Disparate Impact",
                passed=True,
                score=1.0,
                threshold=0.8,
                details={"note": "Need at least 2 groups for comparison"},
                recommendations=[],
            )

        # Get positive rates
        group_keys = list(groups.keys())[:2]
        positive_rates = []

        for group_key in group_keys:
            group_data = groups[group_key]
            predictions = [
                model.predict(example['features']) > 0.5
                for example in group_data
            ]
            positive_rate = np.mean(predictions)
            positive_rates.append(positive_rate)

        # Disparate impact ratio
        if positive_rates[1] > 0:
            ratio = positive_rates[0] / positive_rates[1]
        else:
            ratio = 1.0

        passed = ratio >= 0.8 and ratio <= 1.25  # Within 80%-125%

        recommendations = []
        if not passed:
            recommendations.append(f"Disparate impact ratio: {ratio:.2f} (should be 0.8-1.25)")
            recommendations.append("Review feature selection for proxies of protected attributes")
            recommendations.append("Consider re-balancing training data")

        return BiasTestResult(
            test_name="Disparate Impact",
            passed=passed,
            score=ratio,
            threshold=0.8,
            details={
                "group_1": group_keys[0],
                "group_2": group_keys[1],
                "positive_rate_1": positive_rates[0],
                "positive_rate_2": positive_rates[1],
                "ratio": ratio,
            },
            recommendations=recommendations,
        )

    def test_individual_fairness(
        self,
        model: Any,
        test_data: List[Dict],
        num_pairs: int = 100,
    ) -> BiasTestResult:
        """Test for individual fairness.

        Individual fairness: Similar users should receive similar predictions.
        Measures Lipschitz continuity.

        Does NOT use protected attributes, only feature similarity.
        """

        if len(test_data) < 2:
            return BiasTestResult(
                test_name="Individual Fairness",
                passed=True,
                score=0.0,
                threshold=0.1,
                details={"note": "Insufficient data"},
                recommendations=[],
            )

        # Sample pairs of similar users
        lipschitz_constants = []

        for _ in range(min(num_pairs, len(test_data) // 2)):
            # Sample two random examples
            idx1, idx2 = np.random.choice(len(test_data), size=2, replace=False)
            ex1 = test_data[idx1]
            ex2 = test_data[idx2]

            # Compute feature distance
            feat1 = np.array(ex1['features'])
            feat2 = np.array(ex2['features'])
            feature_dist = np.linalg.norm(feat1 - feat2)

            if feature_dist < 1e-6:
                continue

            # Compute prediction difference
            pred1 = model.predict(ex1['features'])
            pred2 = model.predict(ex2['features'])
            pred_diff = abs(pred1 - pred2)

            # Lipschitz constant
            lipschitz = pred_diff / feature_dist
            lipschitz_constants.append(lipschitz)

        if not lipschitz_constants:
            avg_lipschitz = 0.0
        else:
            avg_lipschitz = np.mean(lipschitz_constants)

        # Lower Lipschitz = more fair (similar users get similar predictions)
        passed = avg_lipschitz <= 0.5

        recommendations = []
        if not passed:
            recommendations.append("Model may be too sensitive to small feature changes")
            recommendations.append("Consider regularization to encourage smoothness")
            recommendations.append("Review features for irrelevant or noisy attributes")

        return BiasTestResult(
            test_name="Individual Fairness",
            passed=passed,
            score=avg_lipschitz,
            threshold=0.5,
            details={
                "avg_lipschitz_constant": avg_lipschitz,
                "interpretation": "Lower = more fair (similar users get similar predictions)",
            },
            recommendations=recommendations,
        )

    def test_calibration(
        self,
        model: Any,
        test_data: List[Dict],
        protected_attributes: Optional[List[str]] = None,
    ) -> BiasTestResult:
        """Test for calibration across groups.

        Calibration: Predictions should match observed outcomes across groups.
        E.g., if model predicts 70% probability, 70% should be positive.
        """

        if not protected_attributes:
            # Test overall calibration
            predictions = [model.predict(ex['features']) for ex in test_data]
            labels = [ex.get('label', 0) for ex in test_data]

            ece = self._compute_ece(np.array(predictions), np.array(labels))

            passed = ece <= 0.1

            return BiasTestResult(
                test_name="Calibration",
                passed=passed,
                score=ece,
                threshold=0.1,
                details={"overall_ece": ece},
                recommendations=["Apply temperature scaling" if not passed else ""],
            )

        # Test calibration per group
        groups = self._group_by_attributes(test_data, protected_attributes)

        ece_by_group = {}
        for group_key, group_data in groups.items():
            predictions = [model.predict(ex['features']) for ex in group_data]
            labels = [ex.get('label', 0) for ex in group_data]

            if len(predictions) > 0:
                ece = self._compute_ece(np.array(predictions), np.array(labels))
                ece_by_group[group_key] = ece

        # Check if calibration is similar across groups
        if len(ece_by_group) > 1:
            max_ece = max(ece_by_group.values())
            ece_diff = max(ece_by_group.values()) - min(ece_by_group.values())
        else:
            max_ece = list(ece_by_group.values())[0] if ece_by_group else 0.0
            ece_diff = 0.0

        passed = max_ece <= 0.1 and ece_diff <= 0.05

        recommendations = []
        if not passed:
            recommendations.append("Apply group-specific calibration")
            recommendations.append("Use isotonic regression per group")
            recommendations.append("Ensure sufficient training data per group")

        return BiasTestResult(
            test_name="Calibration",
            passed=passed,
            score=max_ece,
            threshold=0.1,
            details={"ece_by_group": ece_by_group, "max_ece": max_ece, "ece_difference": ece_diff},
            recommendations=recommendations,
        )

    def test_content_diversity(
        self,
        test_data: List[Dict],
        min_diversity: float = 0.3,
    ) -> BiasTestResult:
        """Test for content diversity.

        Ensures recommendations are diverse across topics/categories.
        Prevents filter bubbles and echo chambers.
        """

        # Extract content categories
        categories = []
        for example in test_data:
            category = example.get('category', 'unknown')
            categories.append(category)

        if not categories:
            return BiasTestResult(
                test_name="Content Diversity",
                passed=True,
                score=0.0,
                threshold=min_diversity,
                details={"note": "No category information"},
                recommendations=[],
            )

        # Compute diversity (normalized entropy)
        unique_categories = len(set(categories))
        total_categories = len(categories)

        diversity = unique_categories / total_categories if total_categories > 0 else 0.0

        passed = diversity >= min_diversity

        recommendations = []
        if not passed:
            recommendations.append("Increase content diversity in recommendations")
            recommendations.append("Apply MMR (Maximal Marginal Relevance) ranking")
            recommendations.append("Inject serendipitous content from different categories")

        return BiasTestResult(
            test_name="Content Diversity",
            passed=passed,
            score=diversity,
            threshold=min_diversity,
            details={
                "unique_categories": unique_categories,
                "total_items": total_categories,
                "diversity_score": diversity,
            },
            recommendations=recommendations,
        )

    def _group_by_attributes(
        self,
        data: List[Dict],
        attributes: List[str],
    ) -> Dict[str, List[Dict]]:
        """Group data by protected attributes (for testing only)."""

        groups = {}

        for example in data:
            # Create group key
            group_parts = []
            for attr in attributes:
                value = example.get('attributes', {}).get(attr, 'unknown')
                group_parts.append(f"{attr}={value}")

            group_key = ":".join(group_parts) if group_parts else "all"

            if group_key not in groups:
                groups[group_key] = []

            groups[group_key].append(example)

        return groups

    def _compute_ece(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""

        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_pred = np.mean(predictions[mask])
                bin_actual = np.mean(labels[mask])
                bin_weight = np.sum(mask) / len(predictions)
                ece += bin_weight * np.abs(bin_pred - bin_actual)

        return ece

    def generate_report(self) -> str:
        """Generate comprehensive bias test report."""

        lines = [
            "=" * 70,
            "BIAS TEST REPORT",
            "=" * 70,
            "",
        ]

        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)

        lines.append(f"Total Tests: {total_tests}")
        lines.append(f"Passed: {passed_tests}")
        lines.append(f"Failed: {total_tests - passed_tests}")
        lines.append("")

        # Overall status
        all_passed = passed_tests == total_tests
        lines.append(f"Overall Status: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("")

        # Individual test results
        for result in self.test_results:
            lines.append(str(result))
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("Bias Testing Framework Demo")
    print("=" * 60)
    print()

    # Mock model
    class MockModel:
        def predict(self, features):
            # Completely unbiased model (random)
            return np.random.rand()

    # Create test suite
    suite = BiasTestSuite(fairness_threshold=0.1)

    # Mock test data (NO actual demographic information)
    test_data = [
        {
            'features': np.random.rand(10),
            'label': np.random.rand() > 0.5,
            'category': np.random.choice(['tech', 'sports', 'news']),
            'attributes': {},  # NO demographic information
        }
        for _ in range(100)
    ]

    # Run tests
    model = MockModel()
    results = suite.run_all_tests(model, test_data, protected_attributes=[])

    # Generate report
    report = suite.generate_report()
    print(report)

    print("\n✓ All tests designed to PREVENT bias, not introduce it")
    print("✓ Protected attributes used ONLY for testing, NEVER for predictions")

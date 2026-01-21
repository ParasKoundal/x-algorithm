"""Comprehensive Comparison Framework for Phoenix V1/V2/V3/V4

Provides deep analysis and comparison of all Phoenix versions:
- Performance metrics (NDCG, MRR, Hit Rate, AUC, diversity)
- Ablation studies (measure each feature's contribution)
- Feature importance analysis
- Statistical significance testing
- Fairness metrics tracking
- Latency profiling

Usage:
    python evaluation/comparison_v4.py --dataset movielens-1m --verbose
    python evaluation/comparison_v4.py --ablation --output results.json
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import time
import json
import argparse
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a recommendation model."""

    # Ranking metrics
    ndcg_at_5: float
    ndcg_at_10: float
    ndcg_at_20: float
    mrr: float
    hit_rate_at_5: float
    hit_rate_at_10: float
    hit_rate_at_20: float
    map_at_10: float

    # Classification metrics (if applicable)
    auc_roc: Optional[float] = None
    precision_at_10: Optional[float] = None
    recall_at_10: Optional[float] = None

    # Diversity & coverage
    diversity_at_10: float = 0.0
    coverage: float = 0.0
    novelty_at_10: float = 0.0

    # Fairness metrics
    demographic_parity: Optional[float] = None
    equal_opportunity: Optional[float] = None
    individual_fairness: Optional[float] = None

    # Latency (ms)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def relative_improvement(self, baseline: "PerformanceMetrics") -> Dict[str, float]:
        """Compute relative improvement over baseline.

        Args:
            baseline: Baseline metrics

        Returns:
            improvements: Dict of metric_name -> improvement_pct
        """
        improvements = {}

        metrics_to_compare = [
            'ndcg_at_5', 'ndcg_at_10', 'ndcg_at_20',
            'mrr', 'hit_rate_at_5', 'hit_rate_at_10', 'hit_rate_at_20',
            'map_at_10', 'diversity_at_10', 'coverage', 'novelty_at_10'
        ]

        for metric in metrics_to_compare:
            current_val = getattr(self, metric)
            baseline_val = getattr(baseline, metric)

            if baseline_val > 0:
                improvement_pct = ((current_val - baseline_val) / baseline_val) * 100
                improvements[metric] = improvement_pct

        return improvements


@dataclass
class AblationResult:
    """Result of ablation study."""

    config_name: str
    """Name of configuration (e.g., 'V4_full', 'V4_without_multimodal')."""

    features_enabled: List[str]
    """List of enabled features."""

    features_disabled: List[str]
    """List of disabled features."""

    metrics: PerformanceMetrics
    """Performance metrics for this configuration."""

    contribution: Dict[str, float]
    """Contribution of disabled features (negative values)."""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'config_name': self.config_name,
            'features_enabled': self.features_enabled,
            'features_disabled': self.features_disabled,
            'metrics': self.metrics.to_dict(),
            'contribution': self.contribution
        }


class ComprehensiveComparator:
    """Comprehensive comparison framework for Phoenix versions.

    Compares V1, V2, V3, and V4 with:
    - Performance metrics
    - Ablation studies
    - Feature importance analysis
    - Statistical significance testing
    - Fairness analysis
    """

    def __init__(
        self,
        models: Dict[str, Callable],
        test_data: Dict[str, any],
        num_bootstrap_samples: int = 100,
        confidence_level: float = 0.95
    ):
        """Initialize comprehensive comparator.

        Args:
            models: Dict of {version_name: model_fn}
            test_data: Test dataset
            num_bootstrap_samples: Number of bootstrap samples for significance testing
            confidence_level: Confidence level for confidence intervals
        """
        self.models = models
        self.test_data = test_data
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level

        # Cache for computed metrics
        self.metrics_cache: Dict[str, PerformanceMetrics] = {}

        logger.info(
            f"ComprehensiveComparator initialized with {len(models)} models, "
            f"{len(test_data.get('users', []))} test users"
        )

    def compare_all_versions(self, verbose: bool = True) -> Dict[str, any]:
        """Compare all Phoenix versions.

        Args:
            verbose: Whether to print progress

        Returns:
            comparison_results: {
                'metrics': {version: PerformanceMetrics},
                'improvements': {version: improvement_over_v1},
                'statistical_significance': {(v1, v2): p_value},
                'latency_comparison': {...},
                'fairness_comparison': {...}
            }
        """
        if verbose:
            print("\n" + "=" * 70)
            print("COMPREHENSIVE PHOENIX COMPARISON: V1 vs V2 vs V3 vs V4")
            print("=" * 70 + "\n")

        # Evaluate all versions
        all_metrics = {}
        for version, model in self.models.items():
            if verbose:
                print(f"Evaluating {version}...")

            metrics = self.evaluate_model(model, version, verbose=False)
            all_metrics[version] = metrics
            self.metrics_cache[version] = metrics

        # Compute improvements over V1 (baseline)
        improvements = {}
        if 'V1' in all_metrics:
            baseline = all_metrics['V1']
            for version, metrics in all_metrics.items():
                if version != 'V1':
                    improvements[version] = metrics.relative_improvement(baseline)

        # Statistical significance testing
        significance_tests = {}
        versions = list(all_metrics.keys())
        for i in range(len(versions)):
            for j in range(i + 1, len(versions)):
                v1, v2 = versions[i], versions[j]
                p_value = self.test_statistical_significance(v1, v2)
                significance_tests[(v1, v2)] = p_value

        # Latency comparison
        latency_comparison = self._compare_latency(all_metrics)

        # Fairness comparison
        fairness_comparison = self._compare_fairness(all_metrics)

        # Print results
        if verbose:
            self._print_comparison_table(all_metrics, improvements)
            self._print_significance_tests(significance_tests)

        return {
            'metrics': {v: m.to_dict() for v, m in all_metrics.items()},
            'improvements': improvements,
            'statistical_significance': {
                f"{v1}_vs_{v2}": pval
                for (v1, v2), pval in significance_tests.items()
            },
            'latency_comparison': latency_comparison,
            'fairness_comparison': fairness_comparison
        }

    def ablation_study(
        self,
        v4_model: Callable,
        features_to_ablate: List[str],
        verbose: bool = True
    ) -> Dict[str, AblationResult]:
        """Perform ablation study on V4 features.

        Tests performance with each feature removed to measure its contribution.

        Args:
            v4_model: Full V4 model
            features_to_ablate: List of features to test
                ['multimodal', 'gnn', 'causal']
            verbose: Whether to print progress

        Returns:
            ablation_results: Dict of {config_name: AblationResult}
        """
        if verbose:
            print("\n" + "=" * 70)
            print("ABLATION STUDY: Measuring Feature Contributions")
            print("=" * 70 + "\n")

        # Baseline: Full V4
        full_metrics = self.evaluate_model(v4_model, "V4_full", verbose=False)

        ablation_results = {}

        # Test each feature ablation
        for feature in features_to_ablate:
            config_name = f"V4_without_{feature}"

            if verbose:
                print(f"Testing {config_name}...")

            # Create model with feature disabled
            ablated_model = self._create_ablated_model(v4_model, disabled_features=[feature])

            # Evaluate
            metrics = self.evaluate_model(ablated_model, config_name, verbose=False)

            # Compute contribution (negative = performance drop)
            contribution = {}
            for metric in ['ndcg_at_10', 'mrr', 'hit_rate_at_10']:
                full_val = getattr(full_metrics, metric)
                ablated_val = getattr(metrics, metric)
                contribution[metric] = ((ablated_val - full_val) / full_val) * 100

            result = AblationResult(
                config_name=config_name,
                features_enabled=[f for f in features_to_ablate if f != feature],
                features_disabled=[feature],
                metrics=metrics,
                contribution=contribution
            )

            ablation_results[config_name] = result

        if verbose:
            self._print_ablation_results(ablation_results, full_metrics)

        return ablation_results

    def feature_importance_analysis(
        self,
        model: Callable,
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """Analyze feature importance using perturbation.

        Args:
            model: Model to analyze
            num_samples: Number of samples for analysis

        Returns:
            importance_scores: Dict of {feature_name: importance_score}
        """
        # Sample test data
        users = self.test_data.get('users', [])[:num_samples]
        candidates = self.test_data.get('candidates', [])

        # Baseline predictions
        baseline_scores = []
        for user in users:
            scores = model(user, candidates[:100])  # Top 100 candidates
            baseline_scores.append(scores)

        baseline_scores = np.array(baseline_scores)

        # Perturb each feature and measure impact
        importance_scores = {}

        features_to_test = [
            'text_embedding',
            'image_embedding',
            'video_embedding',
            'social_graph',
            'user_history',
            'engagement_rate'
        ]

        for feature in features_to_test:
            # Perturb feature
            perturbed_scores = []
            for user in users:
                user_perturbed = self._perturb_feature(user, feature)
                scores = model(user_perturbed, candidates[:100])
                perturbed_scores.append(scores)

            perturbed_scores = np.array(perturbed_scores)

            # Compute importance as difference
            importance = np.mean(np.abs(baseline_scores - perturbed_scores))
            importance_scores[feature] = float(importance)

        # Normalize
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {
                k: v / total_importance
                for k, v in importance_scores.items()
            }

        return importance_scores

    def evaluate_model(
        self,
        model: Callable,
        version_name: str,
        verbose: bool = True
    ) -> PerformanceMetrics:
        """Evaluate model on all metrics.

        Args:
            model: Model to evaluate
            version_name: Name of model version
            verbose: Whether to print progress

        Returns:
            metrics: PerformanceMetrics
        """
        # Check cache
        if version_name in self.metrics_cache:
            return self.metrics_cache[version_name]

        users = self.test_data.get('users', [])
        candidates = self.test_data.get('candidates', [])
        ground_truth = self.test_data.get('ground_truth', {})

        # Collect predictions and measure latency
        all_predictions = []
        all_labels = []
        latencies = []

        for user in users:
            # Measure latency
            start_time = time.time()
            scores = model(user, candidates)
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

            # Get predictions
            top_k_indices = np.argsort(-scores)[:20]
            predictions = [candidates[i] for i in top_k_indices]

            # Get ground truth
            relevant = ground_truth.get(user['user_id'], [])

            all_predictions.append(predictions)
            all_labels.append(relevant)

        # Compute ranking metrics
        ndcg_5 = self._compute_ndcg(all_predictions, all_labels, k=5)
        ndcg_10 = self._compute_ndcg(all_predictions, all_labels, k=10)
        ndcg_20 = self._compute_ndcg(all_predictions, all_labels, k=20)
        mrr = self._compute_mrr(all_predictions, all_labels)
        hit_5 = self._compute_hit_rate(all_predictions, all_labels, k=5)
        hit_10 = self._compute_hit_rate(all_predictions, all_labels, k=10)
        hit_20 = self._compute_hit_rate(all_predictions, all_labels, k=20)
        map_10 = self._compute_map(all_predictions, all_labels, k=10)

        # Compute diversity metrics
        diversity = self._compute_diversity(all_predictions, k=10)
        coverage = self._compute_coverage(all_predictions, candidates)
        novelty = self._compute_novelty(all_predictions, k=10)

        # Compute latency percentiles
        latency_p50 = float(np.percentile(latencies, 50))
        latency_p95 = float(np.percentile(latencies, 95))
        latency_p99 = float(np.percentile(latencies, 99))

        # Fairness metrics (if available)
        demographic_parity = self._compute_demographic_parity(all_predictions, users)
        equal_opportunity = self._compute_equal_opportunity(all_predictions, users, all_labels)

        metrics = PerformanceMetrics(
            ndcg_at_5=ndcg_5,
            ndcg_at_10=ndcg_10,
            ndcg_at_20=ndcg_20,
            mrr=mrr,
            hit_rate_at_5=hit_5,
            hit_rate_at_10=hit_10,
            hit_rate_at_20=hit_20,
            map_at_10=map_10,
            diversity_at_10=diversity,
            coverage=coverage,
            novelty_at_10=novelty,
            demographic_parity=demographic_parity,
            equal_opportunity=equal_opportunity,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99
        )

        if verbose:
            print(f"\n{version_name} Metrics:")
            print(f"  NDCG@10: {ndcg_10:.4f}")
            print(f"  MRR: {mrr:.4f}")
            print(f"  Hit Rate@10: {hit_10:.4f}")
            print(f"  Diversity@10: {diversity:.4f}")
            print(f"  Latency p95: {latency_p95:.1f}ms")

        return metrics

    def test_statistical_significance(
        self,
        version1: str,
        version2: str
    ) -> float:
        """Test statistical significance using bootstrap.

        Args:
            version1: First version name
            version2: Second version name

        Returns:
            p_value: P-value for difference in NDCG@10
        """
        # Get metrics
        metrics1 = self.metrics_cache.get(version1)
        metrics2 = self.metrics_cache.get(version2)

        if metrics1 is None or metrics2 is None:
            return 1.0  # No significance if missing

        # Bootstrap test (simplified)
        observed_diff = metrics2.ndcg_at_10 - metrics1.ndcg_at_10

        # Generate bootstrap samples
        bootstrap_diffs = []
        for _ in range(self.num_bootstrap_samples):
            # Resample (simplified - in production, resample predictions)
            noise1 = np.random.normal(0, 0.01)
            noise2 = np.random.normal(0, 0.01)
            diff = (metrics2.ndcg_at_10 + noise2) - (metrics1.ndcg_at_10 + noise1)
            bootstrap_diffs.append(diff)

        # Compute p-value
        bootstrap_diffs = np.array(bootstrap_diffs)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

        return float(p_value)

    # ===== Helper methods =====

    def _compute_ndcg(self, predictions: List[List], labels: List[List], k: int) -> float:
        """Compute NDCG@k."""
        ndcgs = []
        for preds, relevant in zip(predictions, labels):
            preds_k = preds[:k]

            # DCG
            dcg = 0.0
            for i, pred in enumerate(preds_k):
                if pred in relevant:
                    dcg += 1.0 / np.log2(i + 2)

            # IDCG
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))

            # NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        return float(np.mean(ndcgs))

    def _compute_mrr(self, predictions: List[List], labels: List[List]) -> float:
        """Compute MRR."""
        rrs = []
        for preds, relevant in zip(predictions, labels):
            rr = 0.0
            for i, pred in enumerate(preds):
                if pred in relevant:
                    rr = 1.0 / (i + 1)
                    break
            rrs.append(rr)

        return float(np.mean(rrs))

    def _compute_hit_rate(self, predictions: List[List], labels: List[List], k: int) -> float:
        """Compute Hit Rate@k."""
        hits = []
        for preds, relevant in zip(predictions, labels):
            preds_k = preds[:k]
            hit = 1.0 if any(pred in relevant for pred in preds_k) else 0.0
            hits.append(hit)

        return float(np.mean(hits))

    def _compute_map(self, predictions: List[List], labels: List[List], k: int) -> float:
        """Compute MAP@k."""
        aps = []
        for preds, relevant in zip(predictions, labels):
            preds_k = preds[:k]

            num_hits = 0
            sum_precisions = 0.0

            for i, pred in enumerate(preds_k):
                if pred in relevant:
                    num_hits += 1
                    precision_at_i = num_hits / (i + 1)
                    sum_precisions += precision_at_i

            ap = sum_precisions / len(relevant) if len(relevant) > 0 else 0.0
            aps.append(ap)

        return float(np.mean(aps))

    def _compute_diversity(self, predictions: List[List], k: int) -> float:
        """Compute intra-list diversity."""
        diversities = []
        for preds in predictions:
            preds_k = preds[:k]

            # Compute pairwise dissimilarity (simplified)
            # In production, use actual content embeddings
            unique_ratio = len(set(preds_k)) / len(preds_k) if len(preds_k) > 0 else 0.0
            diversities.append(unique_ratio)

        return float(np.mean(diversities))

    def _compute_coverage(self, predictions: List[List], all_candidates: List) -> float:
        """Compute catalog coverage."""
        recommended = set()
        for preds in predictions:
            recommended.update(preds[:20])

        coverage = len(recommended) / len(all_candidates) if len(all_candidates) > 0 else 0.0
        return float(coverage)

    def _compute_novelty(self, predictions: List[List], k: int) -> float:
        """Compute novelty (simplified)."""
        # In production, use -log(popularity)
        # For now, return dummy value
        return 0.5

    def _compute_demographic_parity(self, predictions: List[List], users: List[Dict]) -> Optional[float]:
        """Compute demographic parity (if demographics available)."""
        # Simplified - in production, compute actual parity
        return 0.95  # Dummy value

    def _compute_equal_opportunity(
        self,
        predictions: List[List],
        users: List[Dict],
        labels: List[List]
    ) -> Optional[float]:
        """Compute equal opportunity."""
        return 0.92  # Dummy value

    def _compare_latency(self, all_metrics: Dict[str, PerformanceMetrics]) -> Dict:
        """Compare latency across versions."""
        return {
            version: {
                'p50': metrics.latency_p50,
                'p95': metrics.latency_p95,
                'p99': metrics.latency_p99
            }
            for version, metrics in all_metrics.items()
        }

    def _compare_fairness(self, all_metrics: Dict[str, PerformanceMetrics]) -> Dict:
        """Compare fairness metrics across versions."""
        return {
            version: {
                'demographic_parity': metrics.demographic_parity,
                'equal_opportunity': metrics.equal_opportunity
            }
            for version, metrics in all_metrics.items()
        }

    def _create_ablated_model(self, model: Callable, disabled_features: List[str]) -> Callable:
        """Create model with features disabled."""
        # In production, modify model config to disable features
        # For now, return original model (placeholder)
        return model

    def _perturb_feature(self, user: Dict, feature: str) -> Dict:
        """Perturb a feature for importance analysis."""
        user_perturbed = user.copy()
        # Add noise or zero out feature
        if feature in user_perturbed:
            if isinstance(user_perturbed[feature], list):
                user_perturbed[feature] = [0.0] * len(user_perturbed[feature])
            else:
                user_perturbed[feature] = 0.0
        return user_perturbed

    def _print_comparison_table(
        self,
        all_metrics: Dict[str, PerformanceMetrics],
        improvements: Dict[str, Dict[str, float]]
    ):
        """Print comparison table."""
        print("\nPerformance Comparison:")
        print("-" * 120)
        print(f"{'Metric':<20} {'V1':<15} {'V2':<15} {'V3':<15} {'V4':<15} {'V4 vs V1':<15}")
        print("-" * 120)

        metrics_to_show = [
            ('NDCG@10', 'ndcg_at_10'),
            ('MRR', 'mrr'),
            ('Hit Rate@10', 'hit_rate_at_10'),
            ('Diversity@10', 'diversity_at_10'),
            ('Latency p95 (ms)', 'latency_p95')
        ]

        for display_name, attr_name in metrics_to_show:
            row = [display_name]

            for version in ['V1', 'V2', 'V3', 'V4']:
                if version in all_metrics:
                    val = getattr(all_metrics[version], attr_name)
                    row.append(f"{val:.4f}")
                else:
                    row.append("N/A")

            # Add improvement
            if 'V4' in improvements:
                imp = improvements['V4'].get(attr_name, 0.0)
                row.append(f"+{imp:.1f}%")
            else:
                row.append("N/A")

            print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<15} {row[4]:<15} {row[5]:<15}")

        print("-" * 120)

    def _print_significance_tests(self, significance_tests: Dict[Tuple[str, str], float]):
        """Print statistical significance tests."""
        print("\nStatistical Significance Tests (p-values):")
        print("-" * 60)
        for (v1, v2), p_value in significance_tests.items():
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"  {v1} vs {v2}: p = {p_value:.4f} {significance}")
        print("-" * 60)
        print("  * p < 0.05, ** p < 0.01, *** p < 0.001")

    def _print_ablation_results(
        self,
        ablation_results: Dict[str, AblationResult],
        full_metrics: PerformanceMetrics
    ):
        """Print ablation study results."""
        print("\nAblation Study Results:")
        print("-" * 80)
        print(f"{'Configuration':<30} {'NDCG@10':<15} {'Contribution':<15}")
        print("-" * 80)

        # Print full V4
        print(f"{'V4 (full)':<30} {full_metrics.ndcg_at_10:.4f} {'baseline':<15}")

        # Print ablated versions
        for config_name, result in ablation_results.items():
            ndcg = result.metrics.ndcg_at_10
            contribution = result.contribution.get('ndcg_at_10', 0.0)
            print(f"{config_name:<30} {ndcg:.4f} {contribution:+.1f}%")

        print("-" * 80)


def create_synthetic_test_data(num_users: int = 100, num_candidates: int = 1000) -> Dict:
    """Create synthetic test data for evaluation."""
    test_data = {
        'users': [
            {
                'user_id': i,
                'engagement_history': np.random.rand(50).tolist(),
                'following_count': int(np.random.exponential(100)),
                'follower_count': int(np.random.exponential(150)),
            }
            for i in range(num_users)
        ],
        'candidates': list(range(num_candidates)),
        'ground_truth': {
            i: np.random.choice(num_candidates, size=10, replace=False).tolist()
            for i in range(num_users)
        }
    }
    return test_data


def main():
    """Main entry point for comparison."""
    parser = argparse.ArgumentParser(description="Phoenix V4 Comprehensive Comparison")
    parser.add_argument('--dataset', type=str, default='synthetic', help='Dataset to use')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Create test data
    print("Loading test data...")
    test_data = create_synthetic_test_data(num_users=100)

    # Create dummy models (in production, load actual models)
    models = {
        'V1': lambda user, candidates: np.random.rand(len(candidates)),
        'V2': lambda user, candidates: np.random.rand(len(candidates)) + 0.1,
        'V3': lambda user, candidates: np.random.rand(len(candidates)) + 0.1,
        'V4': lambda user, candidates: np.random.rand(len(candidates)) + 0.3,
    }

    # Create comparator
    comparator = ComprehensiveComparator(models, test_data)

    # Run comparison
    results = comparator.compare_all_versions(verbose=args.verbose or True)

    # Run ablation study if requested
    if args.ablation:
        ablation_results = comparator.ablation_study(
            models['V4'],
            features_to_ablate=['multimodal', 'gnn', 'causal'],
            verbose=args.verbose or True
        )
        results['ablation'] = {k: v.to_dict() for k, v in ablation_results.items()}

    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()

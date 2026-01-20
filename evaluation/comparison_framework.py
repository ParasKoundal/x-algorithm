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

"""Comparison framework for Phoenix V1 vs V2.

Provides comprehensive evaluation and comparison metrics.
"""

import sys
sys.path.append('../')

import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for recommendations."""

    # Ranking metrics
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    hit_rate_at_5: float = 0.0
    hit_rate_at_10: float = 0.0

    # CTR/Engagement metrics
    auc: float = 0.0
    log_loss: float = 0.0
    precision_at_5: float = 0.0
    recall_at_10: float = 0.0

    # Diversity metrics
    topic_diversity: float = 0.0
    author_diversity: float = 0.0
    gini_coefficient: float = 0.0  # Lower = more diverse

    # Latency metrics
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_qps: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)

    def __sub__(self, other: 'EvaluationMetrics') -> 'EvaluationMetrics':
        """Compute improvement (self - other)."""
        return EvaluationMetrics(**{
            k: getattr(self, k) - getattr(other, k)
            for k in asdict(self).keys()
        })

    def relative_improvement(self, baseline: 'EvaluationMetrics') -> Dict[str, float]:
        """Compute relative improvement percentage."""
        improvements = {}
        for metric in asdict(self).keys():
            baseline_val = getattr(baseline, metric)
            current_val = getattr(self, metric)

            if baseline_val == 0:
                improvements[metric] = 0.0
            else:
                # For latency metrics, lower is better
                if 'latency' in metric:
                    improvements[metric] = (baseline_val - current_val) / baseline_val * 100
                else:
                    improvements[metric] = (current_val - baseline_val) / baseline_val * 100

        return improvements


class ModelComparator:
    """Compare Phoenix V1 and V2 models."""

    def __init__(
        self,
        v1_model: Any,
        v2_model: Any,
        v1_params: Any,
        v2_params: Any,
        output_dir: str = "./evaluation_results",
    ):
        self.v1_model = v1_model
        self.v2_model = v2_model
        self.v1_params = v1_params
        self.v2_params = v2_params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model: Any,
        params: Any,
        test_data: List[Dict],
        model_name: str,
    ) -> EvaluationMetrics:
        """Evaluate a single model on test data.

        Args:
            model: Model to evaluate
            params: Model parameters
            test_data: List of test examples
            model_name: Name for logging

        Returns:
            EvaluationMetrics with all computed metrics
        """

        print(f"\nEvaluating {model_name}...")
        print("=" * 60)

        metrics = EvaluationMetrics()

        # Ranking metrics
        ndcg_scores = []
        mrr_scores = []
        hit_rates_5 = []
        hit_rates_10 = []

        # CTR metrics
        predictions = []
        labels = []

        # Latency tracking
        latencies = []

        # Diversity tracking
        recommended_authors = []
        recommended_topics = []

        for example in tqdm(test_data, desc=f"Evaluating {model_name}"):
            # Time inference
            start_time = time.time()

            # Run model (simplified - adapt to actual model interface)
            try:
                # This is a simplified version - adapt to your actual model interface
                batch = example['batch']
                embeddings = example['embeddings']

                output = model.apply(params, batch, embeddings)

                if hasattr(output, 'engagement_logits'):
                    scores = output.engagement_logits[:, :, 0]  # First action type
                else:
                    scores = output.logits[:, :, 0]

                scores = jax.device_get(scores)[0]  # [C]

            except Exception as e:
                print(f"Error during inference: {e}")
                continue

            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

            # Ranking metrics
            ground_truth = example.get('ground_truth', [])
            if ground_truth:
                ndcg_5 = self._compute_ndcg(scores, ground_truth, k=5)
                ndcg_10 = self._compute_ndcg(scores, ground_truth, k=10)
                ndcg_20 = self._compute_ndcg(scores, ground_truth, k=20)
                ndcg_scores.append((ndcg_5, ndcg_10, ndcg_20))

                mrr = self._compute_mrr(scores, ground_truth)
                mrr_scores.append(mrr)

                hit_5 = self._compute_hit_rate(scores, ground_truth, k=5)
                hit_10 = self._compute_hit_rate(scores, ground_truth, k=10)
                hit_rates_5.append(hit_5)
                hit_rates_10.append(hit_10)

            # CTR metrics
            if 'labels' in example:
                predictions.extend(scores.tolist())
                labels.extend(example['labels'])

            # Diversity metrics
            if 'candidate_authors' in example:
                top_k_indices = np.argsort(scores)[::-1][:10]
                recommended_authors.extend([
                    example['candidate_authors'][i] for i in top_k_indices
                ])

            if 'candidate_topics' in example:
                top_k_indices = np.argsort(scores)[::-1][:10]
                for i in top_k_indices:
                    if i < len(example['candidate_topics']):
                        recommended_topics.append(example['candidate_topics'][i])

        # Aggregate metrics
        if ndcg_scores:
            ndcg_5_mean = np.mean([s[0] for s in ndcg_scores])
            ndcg_10_mean = np.mean([s[1] for s in ndcg_scores])
            ndcg_20_mean = np.mean([s[2] for s in ndcg_scores])
            metrics.ndcg_at_5 = ndcg_5_mean
            metrics.ndcg_at_10 = ndcg_10_mean
            metrics.ndcg_at_20 = ndcg_20_mean

        if mrr_scores:
            metrics.mrr = np.mean(mrr_scores)

        if hit_rates_5:
            metrics.hit_rate_at_5 = np.mean(hit_rates_5)

        if hit_rates_10:
            metrics.hit_rate_at_10 = np.mean(hit_rates_10)

        # CTR metrics
        if predictions and labels:
            metrics.auc = self._compute_auc(predictions, labels)
            metrics.log_loss = self._compute_log_loss(predictions, labels)

        # Diversity metrics
        if recommended_authors:
            metrics.author_diversity = len(set(recommended_authors)) / len(recommended_authors)

        if recommended_topics:
            metrics.topic_diversity = len(set(recommended_topics)) / len(recommended_topics)
            metrics.gini_coefficient = self._compute_gini(recommended_topics)

        # Latency metrics
        if latencies:
            metrics.mean_latency_ms = np.mean(latencies)
            metrics.p50_latency_ms = np.percentile(latencies, 50)
            metrics.p95_latency_ms = np.percentile(latencies, 95)
            metrics.p99_latency_ms = np.percentile(latencies, 99)
            metrics.throughput_qps = 1000.0 / metrics.mean_latency_ms

        return metrics

    def compare(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Compare V1 and V2 models.

        Returns:
            Comparison results with metrics and improvements
        """

        print("\n" + "=" * 60)
        print("Phoenix V1 vs V2 Comparison")
        print("=" * 60)

        # Evaluate V1
        v1_metrics = self.evaluate_model(
            self.v1_model,
            self.v1_params,
            test_data,
            "Phoenix V1"
        )

        # Evaluate V2
        v2_metrics = self.evaluate_model(
            self.v2_model,
            self.v2_params,
            test_data,
            "Phoenix V2"
        )

        # Compute improvements
        improvements = v2_metrics.relative_improvement(v1_metrics)

        # Create comparison report
        results = {
            'v1_metrics': v1_metrics.to_dict(),
            'v2_metrics': v2_metrics.to_dict(),
            'improvements_pct': improvements,
            'absolute_diff': (v2_metrics - v1_metrics).to_dict(),
        }

        # Print results
        self._print_comparison(v1_metrics, v2_metrics, improvements)

        # Save results
        self._save_results(results)

        return results

    def _print_comparison(
        self,
        v1_metrics: EvaluationMetrics,
        v2_metrics: EvaluationMetrics,
        improvements: Dict[str, float],
    ):
        """Print formatted comparison table."""

        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        print(f"\n{'Metric':<25} {'V1':>12} {'V2':>12} {'Improvement':>15}")
        print("-" * 80)

        for metric in ['ndcg_at_10', 'mrr', 'hit_rate_at_10', 'auc',
                       'topic_diversity', 'author_diversity', 'mean_latency_ms', 'throughput_qps']:
            v1_val = getattr(v1_metrics, metric)
            v2_val = getattr(v2_metrics, metric)
            improvement = improvements.get(metric, 0.0)

            improvement_str = f"{improvement:+.2f}%"
            if improvement > 0:
                improvement_str = f"✓ {improvement_str}"
            elif improvement < -5:  # Significant regression
                improvement_str = f"✗ {improvement_str}"

            print(f"{metric:<25} {v1_val:>12.4f} {v2_val:>12.4f} {improvement_str:>15}")

        print("=" * 80)

        # Summary
        print("\n" + "Summary:".upper())
        print("-" * 80)

        significant_improvements = [k for k, v in improvements.items() if v > 5]
        if significant_improvements:
            print(f"✓ Significant improvements in: {', '.join(significant_improvements)}")

        regressions = [k for k, v in improvements.items() if v < -5]
        if regressions:
            print(f"✗ Regressions in: {', '.join(regressions)}")

        overall_improvement = np.mean([
            improvements.get('ndcg_at_10', 0),
            improvements.get('mrr', 0),
            improvements.get('hit_rate_at_10', 0),
        ])
        print(f"\nOverall ranking improvement: {overall_improvement:+.2f}%")

        print("=" * 80)

    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""

        output_file = self.output_dir / "comparison_results.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def _compute_ndcg(self, scores: np.ndarray, ground_truth: List[int], k: int) -> float:
        """Compute NDCG@k."""
        # Get top-k predictions
        top_k_indices = np.argsort(scores)[::-1][:k]

        # Compute DCG
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            if idx in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # +2 because position starts at 1

        # Compute IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_mrr(self, scores: np.ndarray, ground_truth: List[int]) -> float:
        """Compute Mean Reciprocal Rank."""
        sorted_indices = np.argsort(scores)[::-1]

        for i, idx in enumerate(sorted_indices):
            if idx in ground_truth:
                return 1.0 / (i + 1)

        return 0.0

    def _compute_hit_rate(self, scores: np.ndarray, ground_truth: List[int], k: int) -> float:
        """Compute Hit Rate@k."""
        top_k_indices = np.argsort(scores)[::-1][:k]
        return float(any(idx in ground_truth for idx in top_k_indices))

    def _compute_auc(self, predictions: List[float], labels: List[int]) -> float:
        """Compute AUC-ROC."""
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(labels, predictions)
        except:
            return 0.0

    def _compute_log_loss(self, predictions: List[float], labels: List[int]) -> float:
        """Compute log loss."""
        from sklearn.metrics import log_loss
        try:
            return log_loss(labels, predictions)
        except:
            return 0.0

    def _compute_gini(self, items: List[Any]) -> float:
        """Compute Gini coefficient (inequality measure)."""
        from collections import Counter

        counts = list(Counter(items).values())
        counts = sorted(counts)

        n = len(counts)
        if n == 0:
            return 0.0

        cum_counts = np.cumsum(counts)
        return (2 * np.sum((np.arange(1, n + 1)) * counts) - (n + 1) * cum_counts[-1]) / (
            n * cum_counts[-1]
        )


def create_synthetic_test_data(num_examples: int = 100, num_candidates: int = 32) -> List[Dict]:
    """Create synthetic test data for quick evaluation."""

    test_data = []

    for i in range(num_examples):
        example = {
            'batch': None,  # Placeholder - replace with actual batch
            'embeddings': None,  # Placeholder - replace with actual embeddings
            'ground_truth': [0, 1, 2],  # First 3 candidates are relevant
            'labels': np.random.randint(0, 2, num_candidates).tolist(),
            'candidate_authors': [f"author_{j % 20}" for j in range(num_candidates)],
            'candidate_topics': [f"topic_{j % 10}" for j in range(num_candidates)],
        }
        test_data.append(example)

    return test_data

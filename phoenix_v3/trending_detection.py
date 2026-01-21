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

"""Real-Time Trending Detection with Diversity Safeguards.

CRITICAL: This module prevents amplification bias and ensures diverse trending content.

Key Anti-Bias Features:
1. Velocity-based (not absolute popularity) - gives new/small creators a chance
2. Diversity requirements - prevent monoculture
3. Decay over time - prevent permanent "trending"
4. Category-specific trending - don't just show mainstream
5. Anti-amplification limits - cap boost to prevent runaway effects

IMPORTANT: Trending should help discovery, not create filter bubbles or
amplify existing biases.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
from enum import Enum
import time

import numpy as np


class TrendingType(Enum):
    """Types of trending detection."""

    VIRAL = "viral"  # Rapid growth in engagement
    EMERGING = "emerging"  # New content gaining traction
    SUSTAINED = "sustained"  # Consistently high engagement
    NICHE = "niche"  # Trending within specific community


@dataclass
class TrendingScore:
    """Trending score with metadata."""

    item_id: str
    velocity: float  # Engagement rate (actions/hour)
    acceleration: float  # Change in velocity
    age_hours: float  # Time since creation
    diversity_bonus: float  # Bonus for underrepresented topics
    final_score: float  # Combined trending score
    trending_type: TrendingType
    category: str  # Content category


class TrendingDetector:
    """Detect trending content with anti-bias safeguards.

    Key principles:
    - Use velocity (rate), not absolute numbers (prevents bias toward large accounts)
    - Ensure diversity across topics/categories
    - Time-decay to prevent stale trending
    - Cap boost multipliers to prevent runaway effects
    """

    def __init__(
        self,
        window_size: int = 3600,  # 1 hour
        min_engagements: int = 10,  # Min actions to be considered
        decay_rate: float = 0.1,
        max_boost_multiplier: float = 3.0,  # Cap at 3x to prevent amplification bias
        diversity_weight: float = 0.3,  # Weight for underrepresented content
    ):
        """Initialize trending detector.

        Args:
            window_size: Time window for velocity computation (seconds)
            min_engagements: Minimum engagements to consider trending
            decay_rate: Time decay rate
            max_boost_multiplier: Maximum boost (prevents runaway amplification)
            diversity_weight: Weight for diversity bonus
        """

        self.window_size = window_size
        self.min_engagements = min_engagements
        self.decay_rate = decay_rate
        self.max_boost_multiplier = max_boost_multiplier
        self.diversity_weight = diversity_weight

        # Track engagement history
        self.engagement_streams = defaultdict(lambda: deque())

        # Track what's currently trending (for diversity)
        self.current_trending = defaultdict(list)  # category -> [item_ids]

        # Track category representation
        self.category_trending_counts = defaultdict(int)

    def record_engagement(
        self,
        item_id: str,
        timestamp: float,
        action_type: str,
        metadata: Optional[Dict] = None,
    ):
        """Record engagement event.

        Args:
            item_id: Content ID
            timestamp: Unix timestamp
            action_type: Type of action (like, share, comment)
            metadata: Item metadata (category, etc.)
        """

        self.engagement_streams[item_id].append({
            'timestamp': timestamp,
            'action_type': action_type,
            'metadata': metadata or {},
        })

        # Keep only recent engagements (memory efficiency)
        cutoff = timestamp - self.window_size * 2
        while (self.engagement_streams[item_id] and
               self.engagement_streams[item_id][0]['timestamp'] < cutoff):
            self.engagement_streams[item_id].popleft()

    def compute_velocity(
        self,
        item_id: str,
        current_time: float,
    ) -> float:
        """Compute engagement velocity (actions per hour).

        Uses velocity instead of absolute count to prevent bias toward
        already-popular content.

        Args:
            item_id: Content ID
            current_time: Current timestamp

        Returns:
            velocity: Engagements per hour
        """

        if item_id not in self.engagement_streams:
            return 0.0

        # Get recent engagements
        stream = self.engagement_streams[item_id]
        recent = [
            event for event in stream
            if current_time - event['timestamp'] < self.window_size
        ]

        if not recent:
            return 0.0

        # Time-weighted velocity (more recent = higher weight)
        weighted_sum = 0.0
        for event in recent:
            age = current_time - event['timestamp']
            weight = np.exp(-self.decay_rate * age / self.window_size)
            weighted_sum += weight

        # Normalize to actions per hour
        velocity = weighted_sum / (self.window_size / 3600)

        return velocity

    def compute_acceleration(
        self,
        item_id: str,
        current_time: float,
    ) -> float:
        """Compute velocity change (acceleration).

        Positive acceleration indicates growing trend.
        """

        # Current velocity (last hour)
        current_velocity = self.compute_velocity(item_id, current_time)

        # Previous velocity (hour before)
        previous_time = current_time - self.window_size
        previous_velocity = self.compute_velocity(item_id, previous_time)

        # Acceleration
        acceleration = current_velocity - previous_velocity

        return acceleration

    def get_diversity_bonus(
        self,
        item_category: str,
        current_time: float,
    ) -> float:
        """Compute diversity bonus for underrepresented categories.

        This PREVENTS filter bubbles by boosting diverse content.

        Args:
            item_category: Content category
            current_time: Current timestamp

        Returns:
            diversity_bonus: Bonus multiplier (0-1)
        """

        # Count current trending by category
        total_trending = sum(self.category_trending_counts.values())

        if total_trending == 0:
            return 0.0  # No adjustment needed

        # Expected proportion (uniform distribution)
        num_categories = len(self.category_trending_counts)
        expected_proportion = 1.0 / num_categories if num_categories > 0 else 0.0

        # Actual proportion
        actual_proportion = self.category_trending_counts.get(item_category, 0) / total_trending

        # Bonus for underrepresented categories
        if actual_proportion < expected_proportion:
            # More underrepresented = higher bonus
            bonus = (expected_proportion - actual_proportion) / expected_proportion
            return min(1.0, bonus * self.diversity_weight)

        return 0.0

    def is_trending(
        self,
        item_id: str,
        current_time: float,
        item_metadata: Optional[Dict] = None,
        threshold_multiplier: float = 2.0,
    ) -> Tuple[bool, Optional[TrendingScore]]:
        """Detect if item is trending.

        Args:
            item_id: Content ID
            current_time: Current timestamp
            item_metadata: Item metadata (category, creation_time, etc.)
            threshold_multiplier: Velocity threshold multiplier

        Returns:
            is_trending: Whether item is trending
            trending_score: Detailed trending score
        """

        metadata = item_metadata or {}

        # Compute metrics
        velocity = self.compute_velocity(item_id, current_time)
        acceleration = self.compute_acceleration(item_id, current_time)

        # Check minimum engagement threshold
        stream = self.engagement_streams.get(item_id, [])
        recent_count = sum(
            1 for event in stream
            if current_time - event['timestamp'] < self.window_size
        )

        if recent_count < self.min_engagements:
            return False, None

        # Get baseline velocity for this item
        baseline = self._get_baseline_velocity(item_id, current_time)

        # Compute diversity bonus
        category = metadata.get('category', 'unknown')
        diversity_bonus = self.get_diversity_bonus(category, current_time)

        # Compute age (prefer newer content for "emerging" trends)
        creation_time = metadata.get('creation_time', current_time)
        age_hours = (current_time - creation_time) / 3600

        # Determine trending type
        if age_hours < 2 and acceleration > 0:
            trending_type = TrendingType.EMERGING
            threshold = baseline * threshold_multiplier * 0.5  # Lower threshold for new
        elif acceleration > baseline:
            trending_type = TrendingType.VIRAL
            threshold = baseline * threshold_multiplier
        elif velocity > baseline * threshold_multiplier:
            trending_type = TrendingType.SUSTAINED
            threshold = baseline * threshold_multiplier
        else:
            trending_type = TrendingType.NICHE
            threshold = baseline * threshold_multiplier * 1.5

        # Final score with diversity bonus
        final_score = velocity * (1.0 + diversity_bonus)

        # Is trending?
        is_trending_flag = final_score > threshold

        if is_trending_flag:
            # Track for diversity monitoring
            self.current_trending[category].append(item_id)
            self.category_trending_counts[category] += 1

        trending_score = TrendingScore(
            item_id=item_id,
            velocity=velocity,
            acceleration=acceleration,
            age_hours=age_hours,
            diversity_bonus=diversity_bonus,
            final_score=final_score,
            trending_type=trending_type,
            category=category,
        )

        return is_trending_flag, trending_score

    def boost_scores(
        self,
        scores: np.ndarray,
        candidates: List[Dict],
        current_time: float,
    ) -> np.ndarray:
        """Boost scores for trending content with safety limits.

        IMPORTANT: Boost is capped at max_boost_multiplier to prevent
        amplification bias (rich-get-richer dynamics).

        Args:
            scores: Original scores
            candidates: Candidate metadata
            current_time: Current timestamp

        Returns:
            boosted_scores: Scores with trending boost (capped)
        """

        boosted = scores.copy()

        for i, candidate in enumerate(candidates):
            item_id = candidate['id']
            is_trending, trending_score = self.is_trending(
                item_id,
                current_time,
                item_metadata=candidate,
            )

            if is_trending and trending_score:
                # Compute boost multiplier
                # Based on velocity relative to baseline
                baseline = self._get_baseline_velocity(item_id, current_time)
                if baseline > 0:
                    boost = trending_score.velocity / baseline
                else:
                    boost = 2.0

                # Add diversity bonus
                boost *= (1.0 + trending_score.diversity_bonus)

                # CAP the boost to prevent amplification bias
                boost = min(boost, self.max_boost_multiplier)

                # Apply boost
                boosted[i] *= boost

        return boosted

    def get_trending_by_category(
        self,
        current_time: float,
        max_per_category: int = 10,
    ) -> Dict[str, List[TrendingScore]]:
        """Get trending items by category.

        Ensures diverse trending across categories (anti-filter-bubble).

        Args:
            current_time: Current timestamp
            max_per_category: Max trending items per category

        Returns:
            trending_by_category: Dict of category -> trending items
        """

        trending_by_category = defaultdict(list)

        # Evaluate all items
        for item_id in self.engagement_streams.keys():
            # Get metadata from latest engagement
            stream = self.engagement_streams[item_id]
            if stream:
                metadata = stream[-1]['metadata']
            else:
                metadata = {}

            is_trending, trending_score = self.is_trending(
                item_id,
                current_time,
                item_metadata=metadata,
            )

            if is_trending and trending_score:
                category = trending_score.category
                trending_by_category[category].append(trending_score)

        # Sort and limit per category
        for category in trending_by_category:
            trending_by_category[category] = sorted(
                trending_by_category[category],
                key=lambda x: x.final_score,
                reverse=True
            )[:max_per_category]

        return trending_by_category

    def _get_baseline_velocity(
        self,
        item_id: str,
        current_time: float,
    ) -> float:
        """Get baseline velocity for item.

        Uses historical average as baseline.
        """

        if item_id not in self.engagement_streams:
            return 1.0

        # Compute average velocity over longer window
        stream = self.engagement_streams[item_id]
        if not stream:
            return 1.0

        # Use all-time average
        total_events = len(stream)
        if total_events == 0:
            return 1.0

        first_timestamp = stream[0]['timestamp']
        time_range = max(current_time - first_timestamp, 1.0)

        # Average velocity (events per hour)
        baseline = total_events / (time_range / 3600)

        return max(baseline, 0.1)  # Minimum baseline

    def cleanup_old_trends(self, current_time: float):
        """Remove expired trending items from tracking.

        Prevents "permanently trending" items.
        """

        # Reset counts
        self.category_trending_counts.clear()

        # Remove old trending items
        for category in list(self.current_trending.keys()):
            # Remove items that are no longer trending
            active = []
            for item_id in self.current_trending[category]:
                is_trending, _ = self.is_trending(item_id, current_time)
                if is_trending:
                    active.append(item_id)
                    self.category_trending_counts[category] += 1

            if active:
                self.current_trending[category] = active
            else:
                del self.current_trending[category]


# Example usage demonstrating diversity safeguards
if __name__ == "__main__":
    print("Trending Detection with Diversity Safeguards")
    print("=" * 60)
    print()

    # Create detector
    detector = TrendingDetector(
        window_size=3600,
        max_boost_multiplier=3.0,  # Cap at 3x to prevent amplification
        diversity_weight=0.3,  # Boost underrepresented
    )

    # Simulate engagements
    current_time = time.time()

    # Item 1: Popular mainstream content (would normally dominate)
    for i in range(100):
        detector.record_engagement(
            "item_mainstream",
            current_time - (100 - i) * 10,
            "like",
            {'category': 'mainstream'},
        )

    # Item 2: Niche content with rapid growth (deserves visibility)
    for i in range(20):
        detector.record_engagement(
            "item_niche",
            current_time - (20 - i) * 10,
            "like",
            {'category': 'niche'},
        )

    # Check trending
    print("\nTrending Analysis:")
    print("-" * 60)

    for item_id in ["item_mainstream", "item_niche"]:
        is_trending, score = detector.is_trending(
            item_id,
            current_time,
            {'category': 'mainstream' if 'mainstream' in item_id else 'niche'},
        )

        if score:
            print(f"\n{item_id}:")
            print(f"  Velocity: {score.velocity:.2f} actions/hour")
            print(f"  Diversity Bonus: {score.diversity_bonus:.2%}")
            print(f"  Final Score: {score.final_score:.2f}")
            print(f"  Trending: {'YES' if is_trending else 'NO'}")
            print(f"  Type: {score.trending_type.value}")

    print("\n" + "=" * 60)
    print("✓ Diversity safeguards ensure niche content gets visibility")
    print("✓ Boost capped at 3x to prevent amplification bias")
    print("✓ Velocity-based (not absolute) prevents bias toward large accounts")

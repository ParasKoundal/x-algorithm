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

"""Content Safety & Moderation Filters.

Multi-layer safety filtering for:
- Toxicity detection
- Hate speech filtering
- Misinformation detection
- NSFW content filtering
- Violence detection

Provides user-configurable safety levels.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re

import jax
import jax.numpy as jnp
import numpy as np


class SafetyLevel(Enum):
    """User-configurable safety levels."""

    STRICT = "strict"  # Filter aggressively (0.9 threshold)
    MODERATE = "moderate"  # Balanced filtering (0.7 threshold)
    RELAXED = "relaxed"  # Minimal filtering (0.5 threshold)
    CUSTOM = "custom"  # User-defined thresholds


@dataclass
class SafetyScore:
    """Comprehensive safety scoring."""

    overall_safety: float  # 0-1 (1 = completely safe)
    toxicity: float  # 0-1 (1 = very toxic)
    hate_speech: float  # 0-1 (1 = hate speech)
    misinformation: float  # 0-1 (1 = likely misinfo)
    nsfw: float  # 0-1 (1 = NSFW)
    violence: float  # 0-1 (1 = violent)
    is_safe: bool  # Whether content passes filter
    reasons: List[str]  # Reasons for filtering


class ToxicityDetector:
    """Toxicity detection using keyword + ML approach."""

    def __init__(self):
        # Load toxic keywords (simplified - in production use larger list)
        self.toxic_patterns = [
            r'\b(hate|kill|die|stupid|idiot|dumb)\b',
            r'\b(racist|sexist|homophobic)\b',
            r'\b(attack|violence|harm)\b',
        ]

        # Compile patterns
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.toxic_patterns
        ]

    def score(self, text: str) -> float:
        """Compute toxicity score (0-1).

        Args:
            text: Input text

        Returns:
            toxicity: 0 = not toxic, 1 = very toxic
        """

        if not text:
            return 0.0

        # Rule-based component
        matches = sum(
            1 for pattern in self.compiled_patterns
            if pattern.search(text)
        )
        rule_score = min(1.0, matches / 3.0)

        # ML component (placeholder - replace with actual model)
        ml_score = self._ml_toxicity_score(text)

        # Combine (max for safety)
        toxicity = max(rule_score, ml_score)

        return toxicity

    def _ml_toxicity_score(self, text: str) -> float:
        """ML-based toxicity scoring (placeholder).

        In production, use models like:
        - Perspective API
        - Detoxify
        - Custom BERT-based classifier
        """

        # Placeholder: simple heuristic
        # Real implementation would use trained model
        text_lower = text.lower()

        # Length penalty (very short/long texts are suspicious)
        length_penalty = 0.0
        if len(text) < 10 or len(text) > 1000:
            length_penalty = 0.1

        # Caps lock penalty
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        caps_penalty = min(0.3, caps_ratio * 0.5) if caps_ratio > 0.5 else 0.0

        # Exclamation/question marks
        punctuation_count = text.count('!') + text.count('?')
        punctuation_penalty = min(0.2, punctuation_count / 10.0)

        return min(1.0, length_penalty + caps_penalty + punctuation_penalty)


class HateSpeechDetector:
    """Hate speech detection."""

    def __init__(self):
        # Hate speech indicators (simplified)
        self.hate_keywords = {
            'racial': ['racist', 'racial slur', 'discrimination'],
            'gender': ['sexist', 'misogyny', 'misandry'],
            'lgbtq': ['homophobic', 'transphobic'],
            'religious': ['antisemitic', 'islamophobic'],
            'general': ['hate group', 'supremacy'],
        }

    def score(self, text: str) -> float:
        """Compute hate speech score (0-1)."""

        if not text:
            return 0.0

        text_lower = text.lower()

        # Check for hate keywords
        matches = 0
        for category, keywords in self.hate_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matches += 1

        # Score based on matches
        hate_score = min(1.0, matches / 3.0)

        return hate_score


class MisinformationDetector:
    """Misinformation and fact-checking."""

    def __init__(self):
        # Indicators of potential misinformation
        self.misinfo_indicators = [
            r'\b(fake news|hoax|conspiracy|cover-?up)\b',
            r'\b(they don\'t want you to know|hidden truth|wake up)\b',
            r'\b(miracle cure|doctors hate|one weird trick)\b',
        ]

        self.compiled_indicators = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.misinfo_indicators
        ]

    def score(self, text: str, metadata: Optional[Dict] = None) -> float:
        """Compute misinformation score (0-1).

        Args:
            text: Content text
            metadata: Optional metadata (source credibility, etc.)

        Returns:
            misinfo_score: 0 = credible, 1 = likely misinformation
        """

        if not text:
            return 0.0

        # Check for misinfo indicators
        matches = sum(
            1 for pattern in self.compiled_indicators
            if pattern.search(text)
        )
        indicator_score = min(1.0, matches / 2.0)

        # Source credibility (if available)
        source_score = 0.0
        if metadata and 'source_credibility' in metadata:
            # Lower credibility = higher misinfo score
            source_score = 1.0 - metadata['source_credibility']

        # Fact-check status (if available)
        factcheck_score = 0.0
        if metadata and 'factcheck_rating' in metadata:
            # 0 = false, 1 = true
            factcheck_score = 1.0 - metadata['factcheck_rating']

        # Combine scores (weighted average)
        weights = {'indicator': 0.3, 'source': 0.3, 'factcheck': 0.4}

        if not metadata:
            # If no metadata, use only indicators
            return indicator_score

        misinfo_score = (
            weights['indicator'] * indicator_score +
            weights['source'] * source_score +
            weights['factcheck'] * factcheck_score
        )

        return min(1.0, misinfo_score)


class NSFWDetector:
    """NSFW (Not Safe For Work) content detection."""

    def __init__(self):
        # NSFW keywords (heavily simplified)
        self.nsfw_patterns = [
            r'\b(nsfw|explicit|adult)\b',
            # Add more patterns in production
        ]

        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.nsfw_patterns
        ]

    def score(self, text: str, has_images: bool = False) -> float:
        """Compute NSFW score (0-1).

        Args:
            text: Content text
            has_images: Whether content contains images

        Returns:
            nsfw_score: 0 = safe, 1 = NSFW
        """

        if not text:
            return 0.0

        # Text-based detection
        matches = sum(
            1 for pattern in self.compiled_patterns
            if pattern.search(text)
        )
        text_score = min(1.0, matches / 2.0)

        # Image-based detection (placeholder)
        image_score = 0.0
        if has_images:
            # In production, use image classification model
            image_score = 0.0  # Placeholder

        return max(text_score, image_score)


class SafetyFilter:
    """Comprehensive safety filtering system."""

    def __init__(
        self,
        default_level: SafetyLevel = SafetyLevel.MODERATE,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ):
        """Initialize safety filter.

        Args:
            default_level: Default safety level
            custom_thresholds: Custom thresholds for each category
        """

        self.default_level = default_level
        self.custom_thresholds = custom_thresholds or {}

        # Initialize detectors
        self.toxicity_detector = ToxicityDetector()
        self.hate_speech_detector = HateSpeechDetector()
        self.misinfo_detector = MisinformationDetector()
        self.nsfw_detector = NSFWDetector()

        # Logging
        self.filtered_count = 0
        self.total_count = 0

    def compute_safety_score(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        has_images: bool = False,
    ) -> SafetyScore:
        """Compute comprehensive safety score.

        Args:
            text: Content text
            metadata: Optional metadata
            has_images: Whether content has images

        Returns:
            SafetyScore with all metrics
        """

        # Compute individual scores
        toxicity = self.toxicity_detector.score(text)
        hate_speech = self.hate_speech_detector.score(text)
        misinformation = self.misinfo_detector.score(text, metadata)
        nsfw = self.nsfw_detector.score(text, has_images)
        violence = 0.0  # Placeholder - implement violence detection

        # Overall safety (1 - max unsafe score)
        overall_safety = 1.0 - max(toxicity, hate_speech, misinformation, nsfw, violence)

        # Get thresholds
        thresholds = self._get_thresholds(self.default_level)

        # Check if safe
        is_safe = True
        reasons = []

        if toxicity > thresholds['toxicity']:
            is_safe = False
            reasons.append(f"Toxicity: {toxicity:.2f}")

        if hate_speech > thresholds['hate_speech']:
            is_safe = False
            reasons.append(f"Hate speech: {hate_speech:.2f}")

        if misinformation > thresholds['misinformation']:
            is_safe = False
            reasons.append(f"Misinformation: {misinformation:.2f}")

        if nsfw > thresholds['nsfw']:
            is_safe = False
            reasons.append(f"NSFW: {nsfw:.2f}")

        if violence > thresholds.get('violence', 0.7):
            is_safe = False
            reasons.append(f"Violence: {violence:.2f}")

        return SafetyScore(
            overall_safety=overall_safety,
            toxicity=toxicity,
            hate_speech=hate_speech,
            misinformation=misinformation,
            nsfw=nsfw,
            violence=violence,
            is_safe=is_safe,
            reasons=reasons,
        )

    def filter_batch(
        self,
        candidates: List[Dict],
        user_preferences: Optional[Dict] = None,
    ) -> Tuple[List[Dict], List[SafetyScore]]:
        """Filter a batch of candidates.

        Args:
            candidates: List of candidate content
            user_preferences: User safety preferences

        Returns:
            filtered_candidates: Safe candidates
            safety_scores: Safety scores for all candidates
        """

        # Get user's safety level
        safety_level = SafetyLevel.MODERATE
        if user_preferences and 'safety_level' in user_preferences:
            safety_level = SafetyLevel(user_preferences['safety_level'])

        filtered = []
        scores = []

        for candidate in candidates:
            # Compute safety score
            score = self.compute_safety_score(
                text=candidate.get('text', ''),
                metadata=candidate.get('metadata'),
                has_images=candidate.get('has_images', False),
            )

            scores.append(score)

            # Filter based on safety
            if score.is_safe or safety_level == SafetyLevel.RELAXED:
                filtered.append(candidate)
            else:
                # Log filtered content
                self._log_filtered(candidate, score)

        self.filtered_count += len(candidates) - len(filtered)
        self.total_count += len(candidates)

        return filtered, scores

    def _get_thresholds(self, level: SafetyLevel) -> Dict[str, float]:
        """Get safety thresholds for level."""

        # Use custom thresholds if provided
        if self.custom_thresholds:
            return self.custom_thresholds

        # Default thresholds
        if level == SafetyLevel.STRICT:
            return {
                'toxicity': 0.3,
                'hate_speech': 0.2,
                'misinformation': 0.4,
                'nsfw': 0.2,
                'violence': 0.3,
            }
        elif level == SafetyLevel.MODERATE:
            return {
                'toxicity': 0.5,
                'hate_speech': 0.4,
                'misinformation': 0.6,
                'nsfw': 0.4,
                'violence': 0.5,
            }
        else:  # RELAXED
            return {
                'toxicity': 0.7,
                'hate_speech': 0.6,
                'misinformation': 0.8,
                'nsfw': 0.6,
                'violence': 0.7,
            }

    def _log_filtered(self, candidate: Dict, score: SafetyScore):
        """Log filtered content for monitoring."""
        # In production, send to logging/monitoring system
        pass

    def get_stats(self) -> Dict[str, float]:
        """Get filtering statistics."""
        return {
            'total_processed': self.total_count,
            'total_filtered': self.filtered_count,
            'filter_rate': self.filtered_count / max(self.total_count, 1),
        }


# Example usage
if __name__ == "__main__":
    # Create safety filter
    safety_filter = SafetyFilter(default_level=SafetyLevel.MODERATE)

    # Test examples
    test_texts = [
        "This is a normal, safe post about technology.",
        "I hate everyone and everything! Kill them all!",
        "FAKE NEWS! The government is hiding the truth!",
        "Check out this NSFW content...",
    ]

    print("Safety Filter Test Results:")
    print("=" * 60)

    for text in test_texts:
        score = safety_filter.compute_safety_score(text)

        print(f"\nText: {text[:50]}...")
        print(f"  Overall Safety: {score.overall_safety:.3f}")
        print(f"  Toxicity: {score.toxicity:.3f}")
        print(f"  Hate Speech: {score.hate_speech:.3f}")
        print(f"  Misinformation: {score.misinformation:.3f}")
        print(f"  NSFW: {score.nsfw:.3f}")
        print(f"  Is Safe: {score.is_safe}")
        if not score.is_safe:
            print(f"  Reasons: {', '.join(score.reasons)}")

    print("\n" + "=" * 60)
    print(f"Stats: {safety_filter.get_stats()}")

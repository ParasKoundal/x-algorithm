"""Phoenix V4: Cutting-Edge ML Features

This module implements three advanced ML features for the X Recommendation Algorithm:

1. **Multimodal Content Understanding** - CLIP/ViT for images, VideoMAE for videos
2. **Graph Neural Networks** - Social graph encoding with behavioral features only
3. **Advanced Causal Learning** - Pearl's do-calculus and counterfactual fairness

Focus: Prevent discrimination, not enforce demographic outcomes.
- NO protected attributes for scoring
- Individual fairness (similar users treated similarly)
- Content quality, not demographics
"""

__version__ = "4.0.0"

from phoenix_v4.config import PhoenixV4Config

__all__ = ["PhoenixV4Config"]

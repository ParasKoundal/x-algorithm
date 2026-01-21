"""Phoenix V4 Configuration

Configuration for cutting-edge ML features with feature flags for gradual rollout.

Focus: Prevent discrimination, not enforce demographic outcomes.
- Individual fairness (similar users treated similarly)
- NO protected attributes for scoring
- Content quality metrics only
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhoenixV4Config:
    """Configuration for Phoenix V4 features.

    Features can be enabled incrementally using feature flags for safe rollout.
    """

    # ===== Multimodal Content Understanding =====
    use_multimodal: bool = False
    """Enable multimodal content understanding (images + videos)."""

    clip_model_name: str = "ViT-B/32"
    """CLIP model for image encoding. Options: ViT-B/32, ViT-B/16, ViT-L/14."""

    videomae_model_name: str = "videomae-base"
    """VideoMAE model for video encoding."""

    multimodal_fusion_dim: int = 1024
    """Dimension of fused multimodal embeddings."""

    multimodal_num_heads: int = 8
    """Number of attention heads in multimodal fusion."""

    # Multimodal feature flags (enable incrementally)
    enable_multimodal_images: bool = False
    """Enable image encoding (requires use_multimodal=True)."""

    enable_multimodal_videos: bool = False
    """Enable video encoding (requires use_multimodal=True)."""

    # Content quality (NO beauty/demographics)
    compute_content_quality: bool = True
    """Compute objective content quality (resolution, clarity, composition)."""

    # ===== Graph Neural Networks =====
    use_gnn: bool = False
    """Enable Graph Neural Network for social graph encoding."""

    gnn_embedding_dim: int = 256
    """Dimension of social graph embeddings."""

    gnn_num_hops: int = 2
    """Number of hops for neighborhood aggregation."""

    gnn_homophily_debiasing: bool = True
    """Apply homophily debiasing to prevent echo chamber effects."""

    gnn_aggregation_method: str = "mean"
    """Aggregation method: mean, max, sum, attention."""

    # GNN feature flags (enable incrementally)
    enable_gnn_followings: bool = False
    """Encode user's followings (requires use_gnn=True)."""

    enable_gnn_followers: bool = False
    """Encode user's followers (requires use_gnn=True)."""

    enable_gnn_engagement_patterns: bool = False
    """Encode engagement patterns in social graph."""

    # Safe behavioral features (NO demographics)
    gnn_behavioral_features: list = field(default_factory=lambda: [
        "following_count",
        "follower_count",
        "avg_engagement_rate",
        "account_age_days",
        "is_verified",
        "posting_frequency"
    ])
    """Whitelist of safe behavioral features. NO demographics allowed."""

    # ===== Advanced Causal Learning =====
    use_advanced_causal: bool = False
    """Enable advanced causal learning with Pearl's do-calculus."""

    causal_graph_path: str = "configs/causal_graph.json"
    """Path to causal graph definition (DAG structure)."""

    counterfactual_fairness_threshold: float = 0.05
    """Maximum allowed gap in counterfactual fairness test (5%)."""

    causal_debiasing_strength: float = 1.0
    """Strength of causal debiasing (0.0 = no debiasing, 1.0 = full)."""

    use_backdoor_criterion: bool = True
    """Use backdoor criterion to identify confounders."""

    use_frontdoor_criterion: bool = False
    """Use frontdoor criterion for mediator analysis."""

    # ===== Integration & Performance =====
    batch_size: int = 128
    """Batch size for model inference."""

    enable_caching: bool = True
    """Enable caching for V4 features (leverages V3 caching)."""

    cache_ttl_seconds: int = 300
    """TTL for cached embeddings (5 minutes)."""

    enable_precompute: bool = False
    """Precompute embeddings for popular content."""

    precompute_threshold: int = 1000
    """Min engagement count to trigger precomputation."""

    # ===== Fairness & Testing =====
    run_fairness_tests: bool = True
    """Run fairness tests on model outputs."""

    fairness_test_frequency: int = 1000
    """Run fairness tests every N requests."""

    enable_counterfactual_testing: bool = True
    """Enable counterfactual fairness testing."""

    log_fairness_metrics: bool = True
    """Log fairness metrics for monitoring."""

    # Protected attributes (ONLY for testing, NEVER for scoring)
    protected_attributes_for_testing: list = field(default_factory=lambda: [])
    """Protected attributes for fairness testing ONLY. Never used for scoring."""

    # ===== Ablation & Analysis =====
    enable_ablation_mode: bool = False
    """Enable ablation mode to measure feature contributions."""

    ablation_features: list = field(default_factory=lambda: [
        "multimodal",
        "gnn",
        "causal"
    ])
    """Features to ablate for analysis."""

    enable_feature_importance: bool = False
    """Compute feature importance scores."""

    enable_deep_analysis: bool = False
    """Enable deep analysis and visualization."""

    # ===== Logging & Monitoring =====
    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR."""

    enable_metrics: bool = True
    """Enable Prometheus metrics."""

    enable_tracing: bool = False
    """Enable distributed tracing."""

    # ===== Safety =====
    max_embedding_norm: float = 10.0
    """Maximum L2 norm for embeddings (prevent exploding values)."""

    enable_gradient_clipping: bool = True
    """Clip gradients during training."""

    gradient_clip_norm: float = 1.0
    """Gradient clipping threshold."""

    def validate(self) -> None:
        """Validate configuration for consistency."""

        # Multimodal validation
        if self.enable_multimodal_images or self.enable_multimodal_videos:
            if not self.use_multimodal:
                raise ValueError(
                    "Multimodal features enabled but use_multimodal=False. "
                    "Set use_multimodal=True to enable multimodal features."
                )

        # GNN validation
        if (self.enable_gnn_followings or self.enable_gnn_followers or
            self.enable_gnn_engagement_patterns):
            if not self.use_gnn:
                raise ValueError(
                    "GNN features enabled but use_gnn=False. "
                    "Set use_gnn=True to enable GNN features."
                )

        # Behavioral features validation (prevent demographics)
        prohibited_features = {
            "age", "gender", "race", "ethnicity", "location",
            "religion", "sexual_orientation", "disability"
        }
        for feature in self.gnn_behavioral_features:
            if any(prohibited in feature.lower() for prohibited in prohibited_features):
                raise ValueError(
                    f"Prohibited demographic feature detected: {feature}. "
                    f"Only behavioral features allowed."
                )

        # Dimension validation
        if self.multimodal_fusion_dim <= 0:
            raise ValueError(f"Invalid multimodal_fusion_dim: {self.multimodal_fusion_dim}")

        if self.gnn_embedding_dim <= 0:
            raise ValueError(f"Invalid gnn_embedding_dim: {self.gnn_embedding_dim}")

        # Fairness threshold validation
        if not 0.0 <= self.counterfactual_fairness_threshold <= 1.0:
            raise ValueError(
                f"counterfactual_fairness_threshold must be in [0, 1], "
                f"got {self.counterfactual_fairness_threshold}"
            )

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "multimodal": {
                "enabled": self.use_multimodal,
                "images": self.enable_multimodal_images,
                "videos": self.enable_multimodal_videos,
                "clip_model": self.clip_model_name,
                "videomae_model": self.videomae_model_name,
                "fusion_dim": self.multimodal_fusion_dim,
            },
            "gnn": {
                "enabled": self.use_gnn,
                "followings": self.enable_gnn_followings,
                "followers": self.enable_gnn_followers,
                "embedding_dim": self.gnn_embedding_dim,
                "num_hops": self.gnn_num_hops,
                "homophily_debiasing": self.gnn_homophily_debiasing,
            },
            "causal": {
                "enabled": self.use_advanced_causal,
                "graph_path": self.causal_graph_path,
                "fairness_threshold": self.counterfactual_fairness_threshold,
                "debiasing_strength": self.causal_debiasing_strength,
            },
            "fairness": {
                "run_tests": self.run_fairness_tests,
                "counterfactual": self.enable_counterfactual_testing,
                "test_frequency": self.fairness_test_frequency,
            },
            "performance": {
                "batch_size": self.batch_size,
                "caching": self.enable_caching,
                "cache_ttl": self.cache_ttl_seconds,
            }
        }

    @classmethod
    def create_baseline(cls) -> "PhoenixV4Config":
        """Create baseline config with all V4 features disabled."""
        return cls(
            use_multimodal=False,
            use_gnn=False,
            use_advanced_causal=False,
        )

    @classmethod
    def create_full_v4(cls) -> "PhoenixV4Config":
        """Create full V4 config with all features enabled."""
        return cls(
            # Multimodal
            use_multimodal=True,
            enable_multimodal_images=True,
            enable_multimodal_videos=True,

            # GNN
            use_gnn=True,
            enable_gnn_followings=True,
            enable_gnn_followers=True,
            enable_gnn_engagement_patterns=True,

            # Causal
            use_advanced_causal=True,

            # Fairness
            run_fairness_tests=True,
            enable_counterfactual_testing=True,
        )

    @classmethod
    def create_multimodal_only(cls) -> "PhoenixV4Config":
        """Create config with only multimodal features enabled."""
        return cls(
            use_multimodal=True,
            enable_multimodal_images=True,
            enable_multimodal_videos=True,
            use_gnn=False,
            use_advanced_causal=False,
        )

    @classmethod
    def create_gnn_only(cls) -> "PhoenixV4Config":
        """Create config with only GNN features enabled."""
        return cls(
            use_multimodal=False,
            use_gnn=True,
            enable_gnn_followings=True,
            enable_gnn_followers=True,
            use_advanced_causal=False,
        )

    @classmethod
    def create_causal_only(cls) -> "PhoenixV4Config":
        """Create config with only causal features enabled."""
        return cls(
            use_multimodal=False,
            use_gnn=False,
            use_advanced_causal=True,
        )

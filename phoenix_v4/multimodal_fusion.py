"""Multimodal Fusion Module for Phoenix V4

Fuses text, image, and video embeddings using cross-modal attention.

Architecture:
- Cross-modal attention (text attends to image + video)
- Gated fusion for selective information flow
- Projection to unified embedding space

Fairness: All modalities treated equally, no demographic signals.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CrossModalAttention(hk.Module):
    """Cross-modal attention mechanism.

    Allows text embeddings to attend to image and video embeddings for
    richer multimodal representation.
    """

    def __init__(
        self,
        num_heads: int = 8,
        key_size: int = 64,
        w_init_scale: float = 2.0,
        name: Optional[str] = None
    ):
        """Initialize cross-modal attention.

        Args:
            num_heads: Number of attention heads
            key_size: Dimension of keys (and queries)
            w_init_scale: Weight initialization scale
            name: Module name
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.w_init_scale = w_init_scale

    def __call__(
        self,
        query: jnp.ndarray,  # (query_len, d_model)
        key: jnp.ndarray,    # (kv_len, d_model)
        value: jnp.ndarray,  # (kv_len, d_model)
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Apply cross-modal attention.

        Args:
            query: Query embeddings (typically text)
            key: Key embeddings (text + image + video)
            value: Value embeddings (text + image + video)
            mask: Optional attention mask

        Returns:
            attended: Attended embeddings
        """
        batch_size, query_len, d_model = query.shape
        kv_len = key.shape[1]

        # Linear projections
        q_proj = hk.Linear(self.num_heads * self.key_size, name="q_proj")(query)
        k_proj = hk.Linear(self.num_heads * self.key_size, name="k_proj")(key)
        v_proj = hk.Linear(self.num_heads * self.key_size, name="v_proj")(value)

        # Reshape for multi-head attention
        q = q_proj.reshape(batch_size, query_len, self.num_heads, self.key_size)
        k = k_proj.reshape(batch_size, kv_len, self.num_heads, self.key_size)
        v = v_proj.reshape(batch_size, kv_len, self.num_heads, self.key_size)

        # Transpose to (batch_size, num_heads, seq_len, key_size)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
        scores = scores / jnp.sqrt(self.key_size)

        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)

        attention_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention to values
        attended = jnp.matmul(attention_weights, v)

        # Reshape back
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, query_len, self.num_heads * self.key_size)

        # Final projection
        output = hk.Linear(d_model, name="output_proj")(attended)

        return output


class GatedFusion(hk.Module):
    """Gated fusion mechanism for selective information flow.

    Uses learned gates to control how much information flows from each modality.
    """

    def __init__(self, fusion_dim: int, name: Optional[str] = None):
        """Initialize gated fusion.

        Args:
            fusion_dim: Dimension of fused embeddings
            name: Module name
        """
        super().__init__(name=name)
        self.fusion_dim = fusion_dim

    def __call__(
        self,
        text_emb: jnp.ndarray,
        image_emb: Optional[jnp.ndarray],
        video_emb: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        """Apply gated fusion.

        Args:
            text_emb: Text embeddings (batch_size, d_text)
            image_emb: Image embeddings (batch_size, d_image) or None
            video_emb: Video embeddings (batch_size, d_video) or None

        Returns:
            fused: Fused embeddings (batch_size, fusion_dim)
        """
        batch_size = text_emb.shape[0]

        # Project to common dimension
        text_proj = hk.Linear(self.fusion_dim, name="text_proj")(text_emb)

        # Collect available modalities
        modalities = [text_proj]
        modality_names = ["text"]

        if image_emb is not None:
            image_proj = hk.Linear(self.fusion_dim, name="image_proj")(image_emb)
            modalities.append(image_proj)
            modality_names.append("image")

        if video_emb is not None:
            video_proj = hk.Linear(self.fusion_dim, name="video_proj")(video_emb)
            modalities.append(video_proj)
            modality_names.append("video")

        num_modalities = len(modalities)

        # Compute gates
        # Concatenate all modalities
        concat = jnp.concatenate(modalities, axis=-1)

        # Gate network: learns importance of each modality
        gates = hk.Sequential([
            hk.Linear(self.fusion_dim, name="gate_hidden"),
            jax.nn.relu,
            hk.Linear(num_modalities, name="gate_output"),
            jax.nn.softmax
        ])(concat)

        # Apply gates
        fused = jnp.zeros((batch_size, self.fusion_dim))
        for i, modality in enumerate(modalities):
            gate = gates[:, i:i+1]  # (batch_size, 1)
            fused = fused + gate * modality

        return fused


class MultimodalFusion(hk.Module):
    """Multimodal fusion module.

    Combines text, image, and video embeddings using:
    1. Cross-modal attention
    2. Gated fusion
    3. Projection to unified space

    Architecture:
        Text + Image + Video
            |
        Cross-Modal Attention (text attends to image+video)
            |
        Gated Fusion (learned weights)
            |
        Projection to fusion_dim
            |
        LayerNorm + Residual
            |
        Final fused embedding
    """

    def __init__(
        self,
        fusion_dim: int = 1024,
        num_heads: int = 8,
        key_size: int = 64,
        use_residual: bool = True,
        name: Optional[str] = None
    ):
        """Initialize multimodal fusion.

        Args:
            fusion_dim: Dimension of fused embeddings
            num_heads: Number of attention heads
            key_size: Dimension of attention keys
            use_residual: Whether to use residual connections
            name: Module name
        """
        super().__init__(name=name)
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.key_size = key_size
        self.use_residual = use_residual

        logger.info(
            f"MultimodalFusion initialized: "
            f"fusion_dim={fusion_dim}, num_heads={num_heads}"
        )

    def __call__(
        self,
        text_embedding: jnp.ndarray,
        image_embedding: Optional[jnp.ndarray] = None,
        video_embedding: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Fuse multimodal embeddings.

        Args:
            text_embedding: Text embeddings (batch_size, d_text)
            image_embedding: Image embeddings (batch_size, d_image) or None
            video_embedding: Video embeddings (batch_size, d_video) or None

        Returns:
            fused_embedding: (batch_size, fusion_dim)
        """
        batch_size = text_embedding.shape[0]

        # Ensure batch dimension
        if len(text_embedding.shape) == 1:
            text_embedding = text_embedding[None, :]

        # Step 1: Collect available modalities
        modalities = [text_embedding]

        if image_embedding is not None:
            if len(image_embedding.shape) == 1:
                image_embedding = image_embedding[None, :]
            modalities.append(image_embedding)

        if video_embedding is not None:
            if len(video_embedding.shape) == 1:
                video_embedding = video_embedding[None, :]
            modalities.append(video_embedding)

        # If only text, just project and return
        if len(modalities) == 1:
            text_proj = hk.Linear(self.fusion_dim, name="text_only_proj")(text_embedding)
            text_proj = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(text_proj)
            return text_proj

        # Step 2: Stack modalities for attention
        # Project to common dimension first
        text_proj = hk.Linear(self.fusion_dim, name="text_pre_proj")(text_embedding)

        modality_projs = [text_proj]
        if image_embedding is not None:
            img_proj = hk.Linear(self.fusion_dim, name="image_pre_proj")(image_embedding)
            modality_projs.append(img_proj)

        if video_embedding is not None:
            vid_proj = hk.Linear(self.fusion_dim, name="video_pre_proj")(video_embedding)
            modality_projs.append(vid_proj)

        # Stack: (batch_size, num_modalities, fusion_dim)
        stacked = jnp.stack(modality_projs, axis=1)

        # Step 3: Cross-modal attention
        # Query: text, Key/Value: all modalities
        cross_attn = CrossModalAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            name="cross_modal_attn"
        )

        query = text_proj[:, None, :]  # (batch_size, 1, fusion_dim)
        attended = cross_attn(query, stacked, stacked)
        attended = attended.squeeze(1)  # (batch_size, fusion_dim)

        # Step 4: Gated fusion
        gated_fusion = GatedFusion(self.fusion_dim, name="gated_fusion")

        # For gated fusion, we need individual embeddings
        image_emb_for_gate = image_embedding if image_embedding is not None else None
        video_emb_for_gate = video_embedding if video_embedding is not None else None

        fused = gated_fusion(text_embedding, image_emb_for_gate, video_emb_for_gate)

        # Step 5: Combine attention and gating
        # Average the two fusion approaches
        combined = 0.5 * attended + 0.5 * fused

        # Step 6: Residual connection (optional)
        if self.use_residual:
            residual = hk.Linear(self.fusion_dim, name="residual_proj")(text_embedding)
            combined = combined + residual

        # Step 7: LayerNorm and final projection
        combined = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(combined)
        output = hk.Linear(self.fusion_dim, name="final_proj")(combined)
        output = jax.nn.relu(output)

        return output


def create_multimodal_fusion_fn(config):
    """Create multimodal fusion function.

    Args:
        config: PhoenixV4Config

    Returns:
        fusion_fn: Function that fuses embeddings
        init_fn: Function to initialize parameters
    """

    def fusion_fn(text_emb, image_emb=None, video_emb=None):
        fusion = MultimodalFusion(
            fusion_dim=config.multimodal_fusion_dim,
            num_heads=config.multimodal_num_heads,
        )
        return fusion(text_emb, image_emb, video_emb)

    # Transform to pure function
    fusion_fn = hk.transform(fusion_fn)

    return fusion_fn.apply, fusion_fn.init


# Utility functions

def combine_embeddings_simple(
    text_emb: jnp.ndarray,
    image_emb: Optional[jnp.ndarray] = None,
    video_emb: Optional[jnp.ndarray] = None,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
) -> jnp.ndarray:
    """Simple weighted combination of embeddings (no attention).

    Useful for ablation studies and baseline comparisons.

    Args:
        text_emb: Text embeddings (d_text,)
        image_emb: Image embeddings (d_image,) or None
        video_emb: Video embeddings (d_video,) or None
        weights: (w_text, w_image, w_video)

    Returns:
        combined: Combined embedding (same dim as text_emb)
    """
    w_text, w_image, w_video = weights

    # Start with text
    combined = w_text * text_emb

    # Add image if available
    if image_emb is not None:
        # Project to text dimension if needed
        if image_emb.shape != text_emb.shape:
            # Simple projection via truncation or padding
            if len(image_emb) > len(text_emb):
                image_emb = image_emb[:len(text_emb)]
            else:
                padding = jnp.zeros(len(text_emb) - len(image_emb))
                image_emb = jnp.concatenate([image_emb, padding])

        combined = combined + w_image * image_emb

    # Add video if available
    if video_emb is not None:
        # Project to text dimension if needed
        if video_emb.shape != text_emb.shape:
            if len(video_emb) > len(text_emb):
                video_emb = video_emb[:len(text_emb)]
            else:
                padding = jnp.zeros(len(text_emb) - len(video_emb))
                video_emb = jnp.concatenate([video_emb, padding])

        combined = combined + w_video * video_emb

    # Normalize weights
    total_weight = w_text
    if image_emb is not None:
        total_weight += w_image
    if video_emb is not None:
        total_weight += w_video

    combined = combined / total_weight

    return combined


def test_multimodal_fusion():
    """Test multimodal fusion module."""
    import jax.numpy as jnp
    from jax import random

    # Create test data
    batch_size = 4
    d_text = 768
    d_image = 512
    d_video = 512
    fusion_dim = 1024

    key = random.PRNGKey(0)
    key1, key2, key3, key4 = random.split(key, 4)

    text_emb = random.normal(key1, (batch_size, d_text))
    image_emb = random.normal(key2, (batch_size, d_image))
    video_emb = random.normal(key3, (batch_size, d_video))

    # Create fusion function
    def fusion_fn(text, img, vid):
        fusion = MultimodalFusion(fusion_dim=fusion_dim)
        return fusion(text, img, vid)

    fusion_fn = hk.transform(fusion_fn)

    # Initialize
    params = fusion_fn.init(key4, text_emb, image_emb, video_emb)

    # Forward pass
    fused = fusion_fn.apply(params, None, text_emb, image_emb, video_emb)

    print(f"Input shapes:")
    print(f"  Text: {text_emb.shape}")
    print(f"  Image: {image_emb.shape}")
    print(f"  Video: {video_emb.shape}")
    print(f"Output shape: {fused.shape}")
    print(f"Expected: ({batch_size}, {fusion_dim})")

    assert fused.shape == (batch_size, fusion_dim), "Shape mismatch!"
    print("✓ Multimodal fusion test passed!")


if __name__ == "__main__":
    test_multimodal_fusion()

"""Multimodal Content Encoders for Phoenix V4

Implements CLIP/ViT for images and VideoMAE for videos.

CRITICAL FAIRNESS SAFEGUARDS:
- NO face detection or recognition
- NO demographic inference
- Content quality metrics ONLY (resolution, clarity, composition)
- Embeddings capture content semantics, NOT people

Focus: Prevent discrimination through content-only analysis.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class CLIPImageEncoder:
    """CLIP ViT encoder for images.

    Encodes images using pretrained CLIP model. Returns semantic embeddings
    that capture content (objects, scenes, composition) without demographic information.

    Model: ViT-B/32 (default), produces 512-dim embeddings.
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        """Initialize CLIP encoder.

        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14)
        """
        self.model_name = model_name
        self.embedding_dim = 512  # ViT-B/32 output dim
        self.image_size = 224  # CLIP input size

        # Model initialization (placeholder - in production, load pretrained CLIP)
        logger.info(f"Initialized CLIP encoder: {model_name}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for CLIP.

        Args:
            image: RGB image (H, W, 3) in range [0, 255]

        Returns:
            Preprocessed image (224, 224, 3) normalized
        """
        # Resize to 224x224
        # In production: use cv2.resize or PIL.Image.resize
        image = image.astype(np.float32) / 255.0

        # Normalize with CLIP stats
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        image = (image - mean) / std

        return image

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode image to embedding.

        Args:
            image: RGB image (H, W, 3) in range [0, 255]

        Returns:
            image_embedding: (512,) float32 array
        """
        # Preprocess
        processed = self.preprocess(image)

        # CLIP ViT forward pass (placeholder - use pretrained model in production)
        # This is a simplified version showing the structure
        embedding = self._clip_forward(processed)

        # L2 normalize (CLIP convention)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def _clip_forward(self, image: np.ndarray) -> np.ndarray:
        """CLIP ViT forward pass (placeholder).

        In production, this would use the pretrained CLIP model:
        - Vision Transformer (ViT) backbone
        - Patch embedding + positional encoding
        - Multi-head self-attention layers
        - CLS token as final embedding

        Args:
            image: Preprocessed image (224, 224, 3)

        Returns:
            embedding: (512,) float32 array
        """
        # Placeholder: In production, load and run pretrained CLIP
        # For now, return random embeddings for structure
        # key = jax.random.PRNGKey(0)
        # embedding = jax.random.normal(key, (self.embedding_dim,))

        # Simulate semantic embedding based on image statistics
        # This is just for demonstration - real CLIP learns semantics
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)

        return embedding

    def batch_encode(self, images: np.ndarray) -> np.ndarray:
        """Encode batch of images.

        Args:
            images: Batch of images (batch_size, H, W, 3)

        Returns:
            embeddings: (batch_size, 512) float32 array
        """
        batch_size = images.shape[0]
        embeddings = np.zeros((batch_size, self.embedding_dim), dtype=np.float32)

        for i in range(batch_size):
            embeddings[i] = self.encode(images[i])

        return embeddings


class VideoMAEEncoder:
    """VideoMAE encoder for videos.

    Encodes videos using pretrained VideoMAE model. Returns temporal embeddings
    that capture video content (actions, scenes, dynamics) without demographic information.

    Model: VideoMAE-base, produces 512-dim embeddings.
    """

    def __init__(self, model_name: str = "videomae-base"):
        """Initialize VideoMAE encoder.

        Args:
            model_name: VideoMAE model variant
        """
        self.model_name = model_name
        self.embedding_dim = 512
        self.num_frames = 16  # VideoMAE uses 16 frames
        self.frame_size = 224

        logger.info(f"Initialized VideoMAE encoder: {model_name}")

    def sample_frames(
        self,
        video: np.ndarray,
        num_frames: int = 16
    ) -> np.ndarray:
        """Sample frames uniformly from video.

        Args:
            video: Video frames (T, H, W, 3)
            num_frames: Number of frames to sample

        Returns:
            sampled_frames: (num_frames, H, W, 3)
        """
        total_frames = video.shape[0]

        if total_frames <= num_frames:
            # Repeat last frame if video too short
            indices = np.arange(total_frames)
            padding = np.full(num_frames - total_frames, total_frames - 1)
            indices = np.concatenate([indices, padding])
        else:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        return video[indices]

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        """Preprocess frames for VideoMAE.

        Args:
            frames: Video frames (T, H, W, 3) in range [0, 255]

        Returns:
            Preprocessed frames (T, 224, 224, 3) normalized
        """
        # Resize to 224x224
        frames = frames.astype(np.float32) / 255.0

        # Normalize with ImageNet stats (VideoMAE uses these)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames = (frames - mean) / std

        return frames

    def encode(self, video: np.ndarray) -> np.ndarray:
        """Encode video to embedding.

        Args:
            video: Video frames (T, H, W, 3) in range [0, 255]

        Returns:
            video_embedding: (512,) float32 array
        """
        # Sample 16 frames uniformly
        frames = self.sample_frames(video, self.num_frames)

        # Preprocess
        processed = self.preprocess(frames)

        # VideoMAE forward pass (placeholder - use pretrained model in production)
        embedding = self._videomae_forward(processed)

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def _videomae_forward(self, frames: np.ndarray) -> np.ndarray:
        """VideoMAE forward pass (placeholder).

        In production, this would use the pretrained VideoMAE model:
        - 3D Vision Transformer backbone
        - Spatio-temporal attention
        - Pooled embedding from final layer

        Args:
            frames: Preprocessed frames (16, 224, 224, 3)

        Returns:
            embedding: (512,) float32 array
        """
        # Placeholder: In production, load and run pretrained VideoMAE
        # For now, return random embeddings for structure
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)

        return embedding


class ContentQualityAnalyzer:
    """Analyzes objective content quality without demographic inference.

    Measures:
    - Image: Resolution, clarity (sharpness), composition quality
    - Video: Production quality, stability, frame rate consistency

    NEVER measures: Faces, demographics, beauty, people appearance

    Focus: Objective technical quality, not subjective or demographic features.
    """

    def __init__(self):
        """Initialize content quality analyzer."""
        logger.info("Initialized ContentQualityAnalyzer (NO demographic features)")

    def analyze_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze objective image quality.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            quality_metrics: {
                'resolution_score': 0-1 (higher resolution = higher score),
                'sharpness_score': 0-1 (sharper = higher score),
                'composition_score': 0-1 (better rule of thirds, balance = higher),
                'overall_quality': 0-1 (weighted average)
            }
        """
        H, W = image.shape[:2]

        # 1. Resolution score (higher resolution = higher quality)
        resolution_pixels = H * W
        resolution_score = min(resolution_pixels / (1920 * 1080), 1.0)

        # 2. Sharpness score (Laplacian variance)
        gray = self._rgb_to_gray(image)
        sharpness = self._compute_sharpness(gray)
        sharpness_score = min(sharpness / 100.0, 1.0)  # Normalize

        # 3. Composition score (rule of thirds, balance)
        composition_score = self._compute_composition(image)

        # Overall quality (weighted average)
        overall_quality = (
            0.3 * resolution_score +
            0.4 * sharpness_score +
            0.3 * composition_score
        )

        return {
            'resolution_score': float(resolution_score),
            'sharpness_score': float(sharpness_score),
            'composition_score': float(composition_score),
            'overall_quality': float(overall_quality)
        }

    def analyze_video_quality(self, video: np.ndarray) -> Dict[str, float]:
        """Analyze objective video quality.

        Args:
            video: Video frames (T, H, W, 3)

        Returns:
            quality_metrics: {
                'resolution_score': 0-1,
                'stability_score': 0-1 (less motion blur = higher),
                'framerate_consistency': 0-1 (consistent framerate = higher),
                'overall_quality': 0-1
            }
        """
        T, H, W = video.shape[:3]

        # 1. Resolution score
        resolution_pixels = H * W
        resolution_score = min(resolution_pixels / (1920 * 1080), 1.0)

        # 2. Stability score (low inter-frame difference = more stable)
        stability_score = self._compute_stability(video)

        # 3. Frame rate consistency
        # In production, analyze actual timestamps
        framerate_consistency = 0.9  # Placeholder

        # Overall quality
        overall_quality = (
            0.3 * resolution_score +
            0.4 * stability_score +
            0.3 * framerate_consistency
        )

        return {
            'resolution_score': float(resolution_score),
            'stability_score': float(stability_score),
            'framerate_consistency': float(framerate_consistency),
            'overall_quality': float(overall_quality)
        }

    def _rgb_to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to grayscale."""
        if len(image.shape) == 2:
            return image
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    def _compute_sharpness(self, gray_image: np.ndarray) -> float:
        """Compute sharpness using Laplacian variance.

        Higher variance = sharper image.
        """
        # Laplacian kernel
        laplacian = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])

        # Convolve (simplified - in production use scipy.ndimage.convolve)
        # For now, approximate with gradient
        grad_x = np.diff(gray_image, axis=1)
        grad_y = np.diff(gray_image, axis=0)
        sharpness = np.std(grad_x) + np.std(grad_y)

        return float(sharpness)

    def _compute_composition(self, image: np.ndarray) -> float:
        """Compute composition quality (rule of thirds, balance).

        Simplified version - in production, use more sophisticated metrics.
        """
        H, W = image.shape[:2]

        # Check if interesting content is near rule of thirds lines
        # (x = W/3, 2W/3; y = H/3, 2H/3)

        # For now, use simple color variance as proxy for composition
        # Better composition = balanced variance across quadrants
        gray = self._rgb_to_gray(image)

        # Divide into 9 regions (3x3 grid)
        h_step = H // 3
        w_step = W // 3

        variances = []
        for i in range(3):
            for j in range(3):
                region = gray[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                variances.append(np.var(region))

        # Good composition = relatively balanced variances
        variance_balance = 1.0 - (np.std(variances) / (np.mean(variances) + 1e-6))
        composition_score = max(0.0, min(variance_balance, 1.0))

        return float(composition_score)

    def _compute_stability(self, video: np.ndarray) -> float:
        """Compute video stability (low motion blur/shake).

        Lower inter-frame difference = more stable.
        """
        if len(video) < 2:
            return 1.0

        # Compute mean inter-frame difference
        diffs = []
        for t in range(len(video) - 1):
            frame_diff = np.mean(np.abs(video[t+1] - video[t]))
            diffs.append(frame_diff)

        mean_diff = np.mean(diffs)

        # Lower diff = more stable
        # Normalize: typical diff is around 10-50 for stable videos
        stability_score = max(0.0, 1.0 - mean_diff / 50.0)

        return float(stability_score)


class MultimodalEncoders:
    """Main interface for multimodal encoding.

    Combines CLIP (images) and VideoMAE (videos) with content quality analysis.

    FAIRNESS GUARANTEE:
    - NO face detection
    - NO demographic inference
    - Content semantics and technical quality ONLY
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        videomae_model_name: str = "videomae-base",
        compute_quality: bool = True
    ):
        """Initialize multimodal encoders.

        Args:
            clip_model_name: CLIP model variant
            videomae_model_name: VideoMAE model variant
            compute_quality: Whether to compute content quality scores
        """
        self.clip_encoder = CLIPImageEncoder(clip_model_name)
        self.videomae_encoder = VideoMAEEncoder(videomae_model_name)
        self.quality_analyzer = ContentQualityAnalyzer() if compute_quality else None

        logger.info("MultimodalEncoders initialized with fairness safeguards")

    def encode_image(
        self,
        image: Optional[np.ndarray],
        compute_quality: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
        """Encode image with optional quality analysis.

        Args:
            image: RGB image (H, W, 3) or None
            compute_quality: Whether to compute quality metrics

        Returns:
            embedding: (512,) float32 array or None if image is None
            quality: Quality metrics dict or None
        """
        if image is None:
            return None, None

        # Encode image
        embedding = self.clip_encoder.encode(image)

        # Analyze quality (optional)
        quality = None
        if compute_quality and self.quality_analyzer is not None:
            quality = self.quality_analyzer.analyze_image_quality(image)

        return embedding, quality

    def encode_video(
        self,
        video: Optional[np.ndarray],
        compute_quality: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
        """Encode video with optional quality analysis.

        Args:
            video: Video frames (T, H, W, 3) or None
            compute_quality: Whether to compute quality metrics

        Returns:
            embedding: (512,) float32 array or None if video is None
            quality: Quality metrics dict or None
        """
        if video is None:
            return None, None

        # Encode video
        embedding = self.videomae_encoder.encode(video)

        # Analyze quality (optional)
        quality = None
        if compute_quality and self.quality_analyzer is not None:
            quality = self.quality_analyzer.analyze_video_quality(video)

        return embedding, quality

    def encode_post(
        self,
        image: Optional[np.ndarray] = None,
        video: Optional[np.ndarray] = None,
        compute_quality: bool = True
    ) -> Dict[str, any]:
        """Encode post with images and/or videos.

        Args:
            image: Optional RGB image
            video: Optional video frames
            compute_quality: Whether to compute quality metrics

        Returns:
            result: {
                'image_embedding': (512,) or None,
                'video_embedding': (512,) or None,
                'image_quality': dict or None,
                'video_quality': dict or None,
                'has_media': bool
            }
        """
        image_emb, image_quality = self.encode_image(image, compute_quality)
        video_emb, video_quality = self.encode_video(video, compute_quality)

        return {
            'image_embedding': image_emb,
            'video_embedding': video_emb,
            'image_quality': image_quality,
            'video_quality': video_quality,
            'has_media': (image is not None) or (video is not None)
        }

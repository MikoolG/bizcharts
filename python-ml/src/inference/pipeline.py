"""Production inference pipeline for batch processing.

Optimized for processing scraped posts efficiently.
Supports both GPU inference and CPU-optimized ONNX models.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline.

    Attributes:
        setfit_path: Path to SetFit model
        cryptobert_path: Path to CryptoBERT model (optional)
        llava_path: Path to LLaVA model (optional)
        use_onnx: Use ONNX-optimized models for CPU inference
        device: Device for inference ('cpu', 'cuda')
        batch_size: Batch size for processing
        enable_image: Whether to enable image analysis
    """

    setfit_path: str = "models/setfit"
    cryptobert_path: str | None = None
    llava_path: str | None = None
    use_onnx: bool = False
    device: str = "cpu"
    batch_size: int = 32
    enable_image: bool = False


@dataclass
class InferenceResult:
    """Result for a single post inference.

    Attributes:
        thread_id: Thread identifier
        sentiment_score: Score from -1 to +1
        sentiment_confidence: Confidence (0-1)
        sentiment_label: Discrete label
        method: Model(s) used for prediction
        image_score: Optional image sentiment score
        sarcasm_probability: Optional sarcasm probability
    """

    thread_id: int | None
    sentiment_score: float
    sentiment_confidence: float
    sentiment_label: str
    method: str = "setfit"
    image_score: float | None = None
    sarcasm_probability: float | None = None

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "sentiment_label": self.sentiment_label,
            "method": self.method,
            "image_score": self.image_score,
            "sarcasm_probability": self.sarcasm_probability,
        }


class SentimentInferencePipeline:
    """Production inference pipeline optimized for batch processing.

    Supports multiple model configurations:
    - SetFit only (fast, lightweight)
    - SetFit + CryptoBERT ensemble
    - Full multi-modal with LLaVA

    Usage:
        config = InferenceConfig(setfit_path="models/setfit")
        pipeline = SentimentInferencePipeline(config)
        results = pipeline.process_batch(posts)
    """

    def __init__(self, config: InferenceConfig):
        """Initialize inference pipeline.

        Args:
            config: Inference configuration
        """
        self.config = config
        self._text_model = None
        self._cryptobert_model = None
        self._llava_model = None
        self._loaded = False

    def _load_models(self):
        """Lazy load models on first use."""
        if self._loaded:
            return

        logger.info(f"Loading inference models (ONNX={self.config.use_onnx})...")

        # Load SetFit model
        if self.config.use_onnx and Path(f"{self.config.setfit_path}/onnx").exists():
            self._load_onnx_setfit()
        else:
            self._load_pytorch_setfit()

        # Load CryptoBERT if configured
        if self.config.cryptobert_path:
            self._load_cryptobert()

        # Load LLaVA if configured and enabled
        if self.config.llava_path and self.config.enable_image:
            self._load_llava()

        self._loaded = True
        logger.info("Models loaded successfully")

    def _load_pytorch_setfit(self):
        """Load PyTorch SetFit model."""
        from ..models.setfit_model import SetFitSentimentModel

        self._text_model = SetFitSentimentModel(
            model_path=self.config.setfit_path,
            device=self.config.device,
        )

    def _load_onnx_setfit(self):
        """Load ONNX-optimized SetFit model."""
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from sentence_transformers import SentenceTransformer

            onnx_path = f"{self.config.setfit_path}/onnx"
            self._text_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path)
            logger.info(f"Loaded ONNX model from {onnx_path}")
        except ImportError:
            logger.warning("optimum not available, falling back to PyTorch")
            self._load_pytorch_setfit()
        except Exception as e:
            logger.warning(f"ONNX load failed ({e}), falling back to PyTorch")
            self._load_pytorch_setfit()

    def _load_cryptobert(self):
        """Load CryptoBERT model."""
        from ..models.cryptobert_model import CryptoBERTModel

        self._cryptobert_model = CryptoBERTModel(
            model_path=self.config.cryptobert_path,
            device=self.config.device,
        )

    def _load_llava(self):
        """Load LLaVA model for image analysis."""
        from ..models.llava_model import LLaVAMemeModel

        self._llava_model = LLaVAMemeModel(
            model_path=self.config.llava_path,
            load_in_4bit=self.config.device == "cuda",
        )

    def predict(self, text: str, image_path: str | None = None) -> InferenceResult:
        """Predict sentiment for a single post.

        Args:
            text: Post text
            image_path: Optional path to image

        Returns:
            InferenceResult with sentiment prediction
        """
        self._load_models()

        # Get text prediction
        text_result = self._text_model.predict(text)

        score = text_result.score
        confidence = text_result.confidence
        method = "setfit"

        # Ensemble with CryptoBERT if available
        if self._cryptobert_model:
            crypto_result = self._cryptobert_model.predict(text)
            # Weight: 40% SetFit, 60% CryptoBERT
            score = score * 0.4 + crypto_result.score * 0.6
            confidence = confidence * 0.4 + crypto_result.confidence * 0.6
            method = "ensemble"

        # Add image analysis if available
        image_score = None
        sarcasm_prob = None

        if image_path and self._llava_model:
            try:
                img_result = self._llava_model.analyze_meme(image_path, text)
                image_score = img_result.score

                # Fuse text and image (70/30 weight)
                score = score * 0.7 + img_result.score * 0.3

                if img_result.is_sarcastic:
                    sarcasm_prob = img_result.sarcasm_confidence

                method = "multimodal"
            except Exception as e:
                logger.warning(f"Image analysis failed: {e}")

        # Determine label
        if score < -0.2:
            label = "bearish"
        elif score > 0.2:
            label = "bullish"
        else:
            label = "neutral"

        return InferenceResult(
            thread_id=None,
            sentiment_score=score,
            sentiment_confidence=confidence,
            sentiment_label=label,
            method=method,
            image_score=image_score,
            sarcasm_probability=sarcasm_prob,
        )

    def process_batch(
        self,
        posts: list[dict],
        show_progress: bool = True,
    ) -> list[InferenceResult]:
        """Process a batch of posts.

        Args:
            posts: List of post dicts with 'text', optional 'image_path', 'thread_id'
            show_progress: Show progress bar

        Returns:
            List of InferenceResult objects
        """
        self._load_models()

        results = []

        # Process in batches for text-only predictions
        from tqdm import tqdm

        iterator = tqdm(posts, desc="Processing", disable=not show_progress)

        batch_texts = []
        batch_indices = []

        for i, post in enumerate(iterator):
            text = post.get("text", "")
            if text:
                batch_texts.append(text)
                batch_indices.append(i)

            # Process when batch is full
            if len(batch_texts) >= self.config.batch_size:
                batch_results = self._process_text_batch(batch_texts)

                for j, result in enumerate(batch_results):
                    idx = batch_indices[j]
                    result.thread_id = posts[idx].get("thread_id")

                    # Add image analysis if enabled
                    image_path = posts[idx].get("image_path")
                    if image_path and self._llava_model:
                        self._add_image_analysis(result, image_path, batch_texts[j])

                    results.append((idx, result))

                batch_texts = []
                batch_indices = []

        # Process remaining
        if batch_texts:
            batch_results = self._process_text_batch(batch_texts)

            for j, result in enumerate(batch_results):
                idx = batch_indices[j]
                result.thread_id = posts[idx].get("thread_id")

                image_path = posts[idx].get("image_path")
                if image_path and self._llava_model:
                    self._add_image_analysis(result, image_path, batch_texts[j])

                results.append((idx, result))

        # Handle posts with no text
        for i, post in enumerate(posts):
            if not post.get("text"):
                results.append(
                    (
                        i,
                        InferenceResult(
                            thread_id=post.get("thread_id"),
                            sentiment_score=0.0,
                            sentiment_confidence=0.0,
                            sentiment_label="neutral",
                            method="none",
                        ),
                    )
                )

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _process_text_batch(self, texts: list[str]) -> list[InferenceResult]:
        """Process a batch of texts."""
        # Get SetFit predictions
        setfit_results = self._text_model.predict_batch(texts)

        results = []
        for i, text_result in enumerate(setfit_results):
            score = text_result.score
            confidence = text_result.confidence
            method = "setfit"

            # Ensemble with CryptoBERT if available
            if self._cryptobert_model:
                crypto_result = self._cryptobert_model.predict(texts[i])
                score = score * 0.4 + crypto_result.score * 0.6
                confidence = confidence * 0.4 + crypto_result.confidence * 0.6
                method = "ensemble"

            # Determine label
            if score < -0.2:
                label = "bearish"
            elif score > 0.2:
                label = "bullish"
            else:
                label = "neutral"

            results.append(
                InferenceResult(
                    thread_id=None,
                    sentiment_score=score,
                    sentiment_confidence=confidence,
                    sentiment_label=label,
                    method=method,
                )
            )

        return results

    def _add_image_analysis(
        self,
        result: InferenceResult,
        image_path: str,
        text: str,
    ):
        """Add image analysis to existing result."""
        try:
            img_result = self._llava_model.analyze_meme(image_path, text)
            result.image_score = img_result.score

            # Update fused score
            result.sentiment_score = result.sentiment_score * 0.7 + img_result.score * 0.3

            # Update label
            if result.sentiment_score < -0.2:
                result.sentiment_label = "bearish"
            elif result.sentiment_score > 0.2:
                result.sentiment_label = "bullish"
            else:
                result.sentiment_label = "neutral"

            if img_result.is_sarcastic:
                result.sarcasm_probability = img_result.sarcasm_confidence

            result.method = "multimodal"
        except Exception as e:
            logger.warning(f"Image analysis failed for {image_path}: {e}")

    @classmethod
    def from_config_file(cls, path: str | Path) -> "SentimentInferencePipeline":
        """Load pipeline from config file.

        Args:
            path: Path to YAML config file

        Returns:
            Configured pipeline instance
        """
        import yaml

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        config = InferenceConfig(**config_dict)
        return cls(config)

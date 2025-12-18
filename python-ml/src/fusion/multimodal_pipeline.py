"""Complete multi-modal fusion pipeline for sentiment analysis.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                      INPUT PROCESSING                        │
├──────────────────┬──────────────────┬───────────────────────┤
│ Text Pipeline    │ Image Pipeline   │ Context Pipeline      │
│ VADER + SetFit + │ LLaVA           │ Thread history        │
│ CryptoBERT       │                  │                       │
└────────┬─────────┴────────┬─────────┴───────────┬───────────┘
         │                  │                     │
         ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    FUSION + SARCASM DETECTION                │
│  Weighted average with incongruity-based sarcasm inversion  │
│  Output: {sentiment, confidence, sarcasm_probability}       │
└─────────────────────────────────────────────────────────────┘
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .sarcasm_detector import SarcasmDetector

logger = logging.getLogger(__name__)


@dataclass
class FusedSentiment:
    """Final fused sentiment prediction.

    Attributes:
        score: Fused sentiment score from -1 (bearish) to +1 (bullish)
        confidence: Overall confidence in prediction (0-1)
        label: Discrete label ('bearish', 'neutral', 'bullish')
        sarcasm_probability: Probability that content is sarcastic
        components: Individual model contributions for debugging
    """

    score: float
    confidence: float
    label: str
    sarcasm_probability: float
    components: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "label": self.label,
            "sarcasm_probability": self.sarcasm_probability,
            "components": self.components,
        }


@dataclass
class FusionWeights:
    """Weights for multi-modal fusion.

    Empirically tuned starting point:
    - Text: 0.5 (most reliable signal)
    - Image: 0.3 (important for memes)
    - OCR text: 0.1 (text extracted from images)
    - Context: 0.1 (thread history)
    """

    text: float = 0.5
    image: float = 0.3
    ocr_text: float = 0.1
    context: float = 0.1

    def normalize(self, available: set[str]) -> dict[str, float]:
        """Normalize weights to sum to 1 for available components."""
        weights = {}
        total = 0.0

        for component in available:
            weight = getattr(self, component, 0.0)
            weights[component] = weight
            total += weight

        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights


class MultiModalFusionPipeline:
    """Complete multi-modal fusion pipeline.

    Combines text models (VADER, SetFit, CryptoBERT), image analysis
    (LLaVA), and context to produce final sentiment predictions.
    Handles missing modalities gracefully.
    """

    def __init__(
        self,
        text_models: list | None = None,
        image_model=None,
        weights: FusionWeights | None = None,
        sarcasm_threshold: float = 0.7,
    ):
        """Initialize fusion pipeline.

        Args:
            text_models: List of text sentiment models (BaseSentimentModel)
            image_model: Image sentiment model (LLaVAMemeModel)
            weights: Fusion weights for different modalities
            sarcasm_threshold: Threshold for sarcasm detection
        """
        self.text_models = text_models or []
        self.image_model = image_model
        self.weights = weights or FusionWeights()
        self.sarcasm_detector = SarcasmDetector(
            incongruity_threshold=sarcasm_threshold,
            apply_inversion=True,
        )

    def predict(
        self,
        text: str,
        image_path: str | Path | None = None,
        context_texts: list[str] | None = None,
        ocr_text: str | None = None,
    ) -> FusedSentiment:
        """Full multi-modal prediction.

        Args:
            text: Post text content
            image_path: Optional path to image
            context_texts: Optional previous posts in thread
            ocr_text: Optional text extracted from image

        Returns:
            FusedSentiment with combined prediction
        """
        components: dict = {}
        available_components: set[str] = set()

        # 1. Text sentiment (ensemble of models)
        text_score, text_confidence = self._get_text_sentiment(text)
        components["text"] = {
            "score": text_score,
            "confidence": text_confidence,
        }
        available_components.add("text")

        # 2. Image sentiment (if image provided)
        image_score = 0.0
        image_confidence = 0.0
        image_sarcasm = False

        if image_path and self.image_model:
            try:
                result = self.image_model.analyze_meme(image_path, text)
                image_score = result.score
                image_confidence = result.confidence
                image_sarcasm = result.is_sarcastic

                components["image"] = {
                    "score": image_score,
                    "confidence": image_confidence,
                    "is_sarcastic": image_sarcasm,
                    "explanation": result.explanation,
                }
                available_components.add("image")
            except Exception as e:
                logger.warning(f"Image analysis failed: {e}")

        # 3. OCR text sentiment (if provided)
        if ocr_text:
            ocr_score, ocr_confidence = self._get_text_sentiment(ocr_text)
            components["ocr_text"] = {
                "score": ocr_score,
                "confidence": ocr_confidence,
            }
            available_components.add("ocr_text")
        else:
            ocr_score = 0.0
            ocr_confidence = 0.0

        # 4. Context adjustment (if provided)
        context_adjustment = 0.0
        context_confidence = 0.0

        if context_texts:
            context_scores = []
            for ctx in context_texts[-5:]:  # Last 5 posts max
                score, _ = self._get_text_sentiment(ctx)
                context_scores.append(score)

            if context_scores:
                context_adjustment = sum(context_scores) / len(context_scores) * 0.2
                context_confidence = 0.5  # Lower confidence for context signal

                components["context"] = {
                    "adjustment": context_adjustment,
                    "posts_analyzed": len(context_scores),
                    "avg_sentiment": sum(context_scores) / len(context_scores),
                }
                available_components.add("context")

        # 5. Sarcasm detection
        sarcasm_result = None
        if "image" in available_components:
            sarcasm_result = self.sarcasm_detector.detect(
                text_sentiment=text_score,
                image_sentiment=image_score,
                text_confidence=text_confidence,
                image_confidence=image_confidence,
            )

            components["sarcasm"] = {
                "is_sarcastic": sarcasm_result.is_sarcastic,
                "confidence": sarcasm_result.confidence,
                "incongruity": sarcasm_result.incongruity_score,
            }

        # Also consider LLaVA's sarcasm detection
        if image_sarcasm and sarcasm_result:
            # Combine both sarcasm signals
            combined_sarcasm = (sarcasm_result.confidence + 0.5) / 2
            sarcasm_result.confidence = combined_sarcasm
            sarcasm_result.is_sarcastic = True
            components["sarcasm"]["llava_detected"] = True

        # 6. Weighted fusion
        normalized_weights = self.weights.normalize(available_components)

        fused_score = 0.0
        fused_confidence = 0.0
        total_weight = 0.0

        # Add text contribution
        if "text" in normalized_weights:
            effective_text_score = text_score
            # Apply sarcasm inversion if detected
            if sarcasm_result and sarcasm_result.is_sarcastic and sarcasm_result.confidence > 0.6:
                inversion_strength = sarcasm_result.confidence
                effective_text_score = text_score * (1 - 2 * inversion_strength)

            fused_score += effective_text_score * normalized_weights["text"]
            fused_confidence += text_confidence * normalized_weights["text"]
            total_weight += normalized_weights["text"]

        # Add image contribution
        if "image" in normalized_weights:
            fused_score += image_score * normalized_weights["image"]
            fused_confidence += image_confidence * normalized_weights["image"]
            total_weight += normalized_weights["image"]

        # Add OCR contribution
        if "ocr_text" in normalized_weights:
            fused_score += ocr_score * normalized_weights["ocr_text"]
            fused_confidence += ocr_confidence * normalized_weights["ocr_text"]
            total_weight += normalized_weights["ocr_text"]

        # Add context contribution
        if "context" in normalized_weights:
            fused_score += context_adjustment * normalized_weights["context"]
            fused_confidence += context_confidence * normalized_weights["context"]
            total_weight += normalized_weights["context"]

        # Normalize confidence
        if total_weight > 0:
            fused_confidence /= total_weight

        # Clamp score to [-1, 1]
        fused_score = max(-1.0, min(1.0, fused_score))

        # Determine label
        label = self._score_to_label(fused_score)

        # Get sarcasm probability
        sarcasm_prob = 0.0
        if sarcasm_result and sarcasm_result.is_sarcastic:
            sarcasm_prob = sarcasm_result.confidence
        elif image_sarcasm:
            sarcasm_prob = 0.5

        return FusedSentiment(
            score=fused_score,
            confidence=fused_confidence,
            label=label,
            sarcasm_probability=sarcasm_prob,
            components=components,
        )

    def _get_text_sentiment(self, text: str) -> tuple[float, float]:
        """Get ensemble text sentiment.

        Args:
            text: Input text

        Returns:
            Tuple of (score, confidence)
        """
        if not self.text_models:
            return 0.0, 0.0

        scores = []
        confidences = []

        for model in self.text_models:
            try:
                result = model.predict(text)
                scores.append(result.score)
                confidences.append(result.confidence)
            except Exception as e:
                logger.warning(f"Text model failed: {e}")

        if not scores:
            return 0.0, 0.0

        # Average ensemble
        avg_score = sum(scores) / len(scores)
        avg_confidence = sum(confidences) / len(confidences)

        return avg_score, avg_confidence

    @staticmethod
    def _score_to_label(score: float, neutral_threshold: float = 0.2) -> str:
        """Convert continuous score to discrete label."""
        if score < -neutral_threshold:
            return "bearish"
        elif score > neutral_threshold:
            return "bullish"
        return "neutral"

    def predict_batch(
        self,
        posts: list[dict],
    ) -> list[FusedSentiment]:
        """Predict sentiment for multiple posts.

        Args:
            posts: List of post dicts with 'text', optional 'image_path',
                   'context_texts', 'ocr_text'

        Returns:
            List of FusedSentiment predictions
        """
        results = []

        for post in posts:
            result = self.predict(
                text=post.get("text", ""),
                image_path=post.get("image_path"),
                context_texts=post.get("context_texts"),
                ocr_text=post.get("ocr_text"),
            )
            results.append(result)

        return results

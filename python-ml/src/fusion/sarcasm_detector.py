"""Sarcasm detection via text-image incongruity.

Sarcasm in chan culture often manifests as disagreement between text
and image sentiment. For example:
- "This is fine" (positive text) + Pink Wojak (negative image)
- "WAGMI" (positive text) + Chart going down (negative image)

Human accuracy on sarcasm averages only 81.6%, and models without
context drop to 49% F1. This module provides heuristic-based
sarcasm detection through sentiment incongruity.
"""

from dataclasses import dataclass


@dataclass
class SarcasmResult:
    """Result of sarcasm detection.

    Attributes:
        is_sarcastic: Whether sarcasm was detected
        confidence: Confidence in the sarcasm prediction (0-1)
        incongruity_score: Raw text-image sentiment disagreement
        text_sentiment: Text sentiment score (-1 to +1)
        image_sentiment: Image sentiment score (-1 to +1)
        inferred_sentiment: Adjusted sentiment after sarcasm inversion
    """

    is_sarcastic: bool
    confidence: float
    incongruity_score: float
    text_sentiment: float
    image_sentiment: float
    inferred_sentiment: float | None = None


class SarcasmDetector:
    """Detect sarcasm through text-image sentiment disagreement.

    When text expresses positive sentiment but the accompanying image
    shows negative sentiment (or vice versa), this often indicates
    sarcasm or irony. The detector flags these cases and can optionally
    invert the sentiment score.

    Realistic accuracy expectations:
    - Without context: 55-65% F1 on sarcasm
    - With text-image signals: 65-75% F1
    """

    def __init__(
        self,
        incongruity_threshold: float = 0.7,
        confidence_weight: float = 0.5,
        apply_inversion: bool = True,
    ):
        """Initialize sarcasm detector.

        Args:
            incongruity_threshold: Minimum text-image sentiment difference
                                   to flag as sarcasm (0-2 range)
            confidence_weight: How much to weight individual confidences
            apply_inversion: Whether to invert sentiment when sarcastic
        """
        self.incongruity_threshold = incongruity_threshold
        self.confidence_weight = confidence_weight
        self.apply_inversion = apply_inversion

    def detect(
        self,
        text_sentiment: float,
        image_sentiment: float,
        text_confidence: float = 1.0,
        image_confidence: float = 1.0,
    ) -> SarcasmResult:
        """Detect sarcasm through text-image sentiment disagreement.

        Args:
            text_sentiment: Text sentiment score (-1 to +1)
            image_sentiment: Image sentiment score (-1 to +1)
            text_confidence: Confidence in text sentiment (0-1)
            image_confidence: Confidence in image sentiment (0-1)

        Returns:
            SarcasmResult with detection and adjusted sentiment
        """
        # Calculate raw incongruity (0-2 range)
        incongruity = abs(text_sentiment - image_sentiment)

        # Weight by confidence - low confidence = less reliable signal
        min_confidence = min(text_confidence, image_confidence)
        weighted_incongruity = incongruity * (
            self.confidence_weight + (1 - self.confidence_weight) * min_confidence
        )

        # Determine if sarcastic
        is_sarcastic = weighted_incongruity > self.incongruity_threshold

        # Calculate sarcasm confidence
        if is_sarcastic:
            # Confidence increases with incongruity beyond threshold
            excess = weighted_incongruity - self.incongruity_threshold
            max_excess = 2.0 - self.incongruity_threshold  # Maximum possible excess
            confidence = min(1.0, 0.5 + 0.5 * (excess / max_excess))
        else:
            confidence = 0.0

        # Calculate inferred sentiment
        inferred_sentiment = None
        if self.apply_inversion and is_sarcastic and confidence > 0.5:
            # Invert text sentiment, weighted by confidence
            inversion_factor = confidence
            inferred_sentiment = text_sentiment * (1 - 2 * inversion_factor)
        elif not is_sarcastic:
            # No sarcasm - trust text sentiment
            inferred_sentiment = text_sentiment

        return SarcasmResult(
            is_sarcastic=is_sarcastic,
            confidence=confidence,
            incongruity_score=incongruity,
            text_sentiment=text_sentiment,
            image_sentiment=image_sentiment,
            inferred_sentiment=inferred_sentiment,
        )

    def detect_from_signals(
        self,
        text_signals: list[str],
        image_signals: list[str],
    ) -> bool:
        """Quick sarcasm check from detected signals.

        Looks for contradictory signals between text and image.

        Args:
            text_signals: List of detected text signals (e.g., ['WAGMI', 'moon'])
            image_signals: List of detected image signals (e.g., ['pink_wojak', 'crash'])

        Returns:
            True if signals suggest sarcasm
        """
        # Define contradictory pairs
        bullish_text = {"wagmi", "moon", "pump", "bullish", "lfg", "gmi"}
        bearish_text = {"ngmi", "rekt", "dump", "bearish", "its_over", "rugged"}

        bullish_image = {"green_wojak", "chad", "gigachad", "bull", "pump_chart"}
        bearish_image = {"pink_wojak", "crying_wojak", "bobo", "crash_chart", "rug"}

        # Normalize signals
        text_lower = {s.lower().replace(" ", "_") for s in text_signals}
        image_lower = {s.lower().replace(" ", "_") for s in image_signals}

        # Check for contradictions
        text_is_bullish = bool(text_lower & bullish_text)
        text_is_bearish = bool(text_lower & bearish_text)
        image_is_bullish = bool(image_lower & bullish_image)
        image_is_bearish = bool(image_lower & bearish_image)

        # Contradiction = sarcasm likely
        if text_is_bullish and image_is_bearish:
            return True
        if text_is_bearish and image_is_bullish:
            return True

        return False


class ContextAwareSarcasmDetector(SarcasmDetector):
    """Sarcasm detector that considers thread context.

    Thread context is essential for sarcasm detection. The phrase
    "WAGMI" during a market crash versus during a rally carries
    opposite sentiment.
    """

    def __init__(
        self,
        incongruity_threshold: float = 0.7,
        context_weight: float = 0.3,
        **kwargs,
    ):
        """Initialize context-aware detector.

        Args:
            incongruity_threshold: Base threshold for incongruity
            context_weight: How much context affects threshold
            **kwargs: Additional args for parent class
        """
        super().__init__(incongruity_threshold=incongruity_threshold, **kwargs)
        self.context_weight = context_weight

    def detect_with_context(
        self,
        text_sentiment: float,
        image_sentiment: float,
        context_sentiments: list[float],
        text_confidence: float = 1.0,
        image_confidence: float = 1.0,
    ) -> SarcasmResult:
        """Detect sarcasm considering thread context.

        If the thread context has consistently different sentiment from
        the current post, it's more likely to be sarcasm.

        Args:
            text_sentiment: Current text sentiment (-1 to +1)
            image_sentiment: Current image sentiment (-1 to +1)
            context_sentiments: List of sentiments from previous posts
            text_confidence: Confidence in text sentiment
            image_confidence: Confidence in image sentiment

        Returns:
            SarcasmResult with context-adjusted detection
        """
        # Get context sentiment (average of previous posts)
        if context_sentiments:
            context_sentiment = sum(context_sentiments) / len(context_sentiments)
        else:
            context_sentiment = 0.0

        # Adjust threshold based on context disagreement
        context_disagreement = abs(text_sentiment - context_sentiment)
        adjusted_threshold = self.incongruity_threshold * (
            1 - self.context_weight * context_disagreement
        )

        # Run base detection with adjusted threshold
        original_threshold = self.incongruity_threshold
        self.incongruity_threshold = adjusted_threshold

        result = self.detect(
            text_sentiment=text_sentiment,
            image_sentiment=image_sentiment,
            text_confidence=text_confidence,
            image_confidence=image_confidence,
        )

        # Restore original threshold
        self.incongruity_threshold = original_threshold

        return result

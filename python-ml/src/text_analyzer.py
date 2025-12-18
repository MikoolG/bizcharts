"""Text sentiment analysis using VADER, SetFit, and CryptoBERT ensemble."""

import html
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    score: float  # -1 to +1
    confidence: float  # 0 to 1
    method: str  # 'vader', 'ensemble', etc.
    is_greentext: bool
    bullish_signals: list[str]
    bearish_signals: list[str]


class TextAnalyzer:
    """Analyze text sentiment using VADER with custom /biz/ lexicon.

    Supports optional ensemble mode combining:
    - VADER (rule-based, fast)
    - SetFit (few-shot learned)
    - CryptoBERT (crypto domain-specific)

    Ensemble weights: VADER 30%, ML models 70%
    """

    def __init__(
        self,
        lexicon_path: str | Path = "config/lexicon.json",
        use_ensemble: bool = False,
        setfit_path: str | Path | None = None,
        cryptobert_path: str | Path | None = None,
        device: str | None = None,
    ):
        """Initialize text analyzer.

        Args:
            lexicon_path: Path to custom /biz/ lexicon JSON
            use_ensemble: Enable ML model ensemble (requires trained models)
            setfit_path: Path to trained SetFit model
            cryptobert_path: Path to trained CryptoBERT model (optional)
            device: Device for ML inference ('cuda', 'cpu', or auto)
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self.lexicon = self._load_lexicon(lexicon_path)
        self._extend_vader_lexicon()

        # Ensemble components
        self.use_ensemble = use_ensemble
        self._setfit_model = None
        self._cryptobert_model = None

        if use_ensemble:
            self._init_ensemble(setfit_path, cryptobert_path, device)

    def _init_ensemble(
        self,
        setfit_path: str | Path | None,
        cryptobert_path: str | Path | None,
        device: str | None,
    ):
        """Initialize ML models for ensemble."""
        # Load SetFit model
        if setfit_path and Path(setfit_path).exists():
            try:
                from .models.setfit_model import SetFitSentimentModel

                self._setfit_model = SetFitSentimentModel(
                    model_path=setfit_path,
                    device=device,
                )
                logger.info(f"Loaded SetFit model from {setfit_path}")
            except Exception as e:
                logger.warning(f"Failed to load SetFit model: {e}")
        elif setfit_path:
            logger.warning(f"SetFit model not found at {setfit_path}")

        # Load CryptoBERT model (optional)
        if cryptobert_path and Path(cryptobert_path).exists():
            try:
                from .models.cryptobert_model import CryptoBERTModel

                self._cryptobert_model = CryptoBERTModel(
                    model_path=cryptobert_path,
                    device=device,
                )
                logger.info(f"Loaded CryptoBERT model from {cryptobert_path}")
            except Exception as e:
                logger.warning(f"Failed to load CryptoBERT model: {e}")

        # Verify at least one ML model loaded
        if not self._setfit_model and not self._cryptobert_model:
            logger.warning("No ML models loaded, falling back to VADER only")
            self.use_ensemble = False

    def _load_lexicon(self, path: str | Path) -> dict:
        """Load custom lexicon from JSON file."""
        with open(path) as f:
            return json.load(f)

    def _extend_vader_lexicon(self) -> None:
        """Add custom /biz/ terms to VADER's lexicon."""
        for term, weight in self.lexicon.get("bullish", {}).items():
            self.analyzer.lexicon[term.lower()] = weight

        for term, weight in self.lexicon.get("bearish", {}).items():
            self.analyzer.lexicon[term.lower()] = weight

        for term, weight in self.lexicon.get("context_dependent", {}).items():
            self.analyzer.lexicon[term.lower()] = weight

    def preprocess(self, text: str | None) -> tuple[str, bool]:
        """
        Preprocess 4chan post text.

        Returns:
            Tuple of (cleaned_text, is_greentext)
        """
        if not text:
            return "", False

        # Decode HTML entities
        text = html.unescape(text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Detect greentext (lines starting with >)
        lines = text.split("\n")
        greentext_count = sum(1 for line in lines if line.strip().startswith(">"))
        is_greentext = greentext_count > len(lines) * 0.3

        # Remove quote references (>>12345)
        text = re.sub(r">>\\d+", "", text)

        # Normalize whitespace
        text = " ".join(text.split())

        return text, is_greentext

    def analyze(self, text: str | None) -> SentimentResult:
        """
        Analyze sentiment of a post.

        Args:
            text: Raw post text (may include HTML)

        Returns:
            SentimentResult with score, confidence, and detected signals
        """
        cleaned_text, is_greentext = self.preprocess(text)

        if not cleaned_text:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                method="vader",
                is_greentext=False,
                bullish_signals=[],
                bearish_signals=[],
            )

        # Get VADER scores
        scores = self.analyzer.polarity_scores(cleaned_text)

        # Extract detected signals
        bullish_signals = self._find_signals(cleaned_text, "bullish")
        bearish_signals = self._find_signals(cleaned_text, "bearish")

        # Calculate base VADER confidence
        vader_confidence = self._calculate_confidence(scores, bullish_signals, bearish_signals)

        # Apply greentext discount
        greentext_factor = self.lexicon.get("greentext_discount", 0.6)
        if is_greentext:
            vader_confidence *= greentext_factor

        # Use ensemble if enabled
        if self.use_ensemble:
            return self._ensemble_analyze(
                cleaned_text,
                vader_score=scores["compound"],
                vader_confidence=vader_confidence,
                is_greentext=is_greentext,
                bullish_signals=bullish_signals,
                bearish_signals=bearish_signals,
            )

        return SentimentResult(
            score=scores["compound"],
            confidence=vader_confidence,
            method="vader",
            is_greentext=is_greentext,
            bullish_signals=bullish_signals,
            bearish_signals=bearish_signals,
        )

    def _ensemble_analyze(
        self,
        text: str,
        vader_score: float,
        vader_confidence: float,
        is_greentext: bool,
        bullish_signals: list[str],
        bearish_signals: list[str],
    ) -> SentimentResult:
        """Combine VADER with ML model predictions.

        Weights:
        - VADER: 30%
        - SetFit: 35% (or 70% if no CryptoBERT)
        - CryptoBERT: 35% (if available)
        """
        ml_scores = []
        ml_confidences = []

        # Get SetFit prediction
        if self._setfit_model:
            try:
                result = self._setfit_model.predict(text)
                ml_scores.append(result.score)
                ml_confidences.append(result.confidence)
            except Exception as e:
                logger.warning(f"SetFit prediction failed: {e}")

        # Get CryptoBERT prediction
        if self._cryptobert_model:
            try:
                result = self._cryptobert_model.predict(text)
                ml_scores.append(result.score)
                ml_confidences.append(result.confidence)
            except Exception as e:
                logger.warning(f"CryptoBERT prediction failed: {e}")

        # Combine scores
        if ml_scores:
            # Average ML scores
            ml_score = sum(ml_scores) / len(ml_scores)
            ml_confidence = sum(ml_confidences) / len(ml_confidences)

            # Weighted combination: 30% VADER, 70% ML
            final_score = vader_score * 0.3 + ml_score * 0.7
            final_confidence = vader_confidence * 0.3 + ml_confidence * 0.7
            method = "ensemble"
        else:
            # Fall back to VADER only
            final_score = vader_score
            final_confidence = vader_confidence
            method = "vader"

        return SentimentResult(
            score=final_score,
            confidence=final_confidence,
            method=method,
            is_greentext=is_greentext,
            bullish_signals=bullish_signals,
            bearish_signals=bearish_signals,
        )

    def _find_signals(self, text: str, signal_type: str) -> list[str]:
        """Find matching sentiment signals in text."""
        signals = []
        text_lower = text.lower()

        for term in self.lexicon.get(signal_type, {}):
            if term.lower() in text_lower:
                signals.append(term)

        return signals

    def _calculate_confidence(
        self,
        scores: dict,
        bullish_signals: list[str],
        bearish_signals: list[str],
    ) -> float:
        """Calculate confidence in sentiment score."""
        # Base confidence from VADER's certainty
        compound = abs(scores["compound"])
        base_confidence = min(compound * 1.2, 0.9)

        # Boost confidence if explicit signals found
        signal_count = len(bullish_signals) + len(bearish_signals)
        signal_boost = min(signal_count * 0.1, 0.3)

        # Mixed signals reduce confidence
        if bullish_signals and bearish_signals:
            signal_boost *= 0.5

        return min(base_confidence + signal_boost, 1.0)

    def is_ambiguous(self, result: SentimentResult, threshold: float = 0.6) -> bool:
        """Check if result is ambiguous and needs Claude analysis."""
        # Low confidence
        if result.confidence < threshold:
            return True

        # Mixed signals
        if result.bullish_signals and result.bearish_signals:
            return True

        # Near-neutral score with signals
        if abs(result.score) < 0.1 and (result.bullish_signals or result.bearish_signals):
            return True

        return False


def main() -> None:
    """Test the text analyzer."""
    analyzer = TextAnalyzer()

    test_posts = [
        "WAGMI bros, BTC is going to the moon!",
        "It's so over. NGMI. I got rugged.",
        ">be me\n>buy the top\n>it dumps\n>mfw",
        "Should I buy ETH? DYOR NFA",
        "This is financial advice: sell everything",
    ]

    for post in test_posts:
        result = analyzer.analyze(post)
        print(f"\nPost: {post[:50]}...")
        print(f"  Score: {result.score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Greentext: {result.is_greentext}")
        print(f"  Bullish signals: {result.bullish_signals}")
        print(f"  Bearish signals: {result.bearish_signals}")
        print(f"  Needs Claude: {analyzer.is_ambiguous(result)}")


if __name__ == "__main__":
    main()

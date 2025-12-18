"""ADWIN drift detection for retraining triggers.

ADWIN (ADaptive WINdowing) is a change detector that automatically adjusts
its window size based on the observed data. When statistical drift is
detected in prediction accuracy, retraining should be triggered.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from river.drift import ADWIN

logger = logging.getLogger(__name__)


@dataclass
class DriftEvent:
    """Record of a detected drift event.

    Attributes:
        timestamp: Unix timestamp when drift was detected
        error_rate_before: Error rate before drift
        error_rate_after: Error rate after drift (if available)
        window_size: ADWIN window size at detection
    """

    timestamp: int
    error_rate_before: float
    error_rate_after: float | None = None
    window_size: int | None = None


class DriftMonitor:
    """Monitor prediction accuracy for drift using ADWIN.

    ADWIN maintains a variable-length window of recent errors and
    detects when the error rate has changed significantly. This signals
    that the model may need retraining.

    Typical usage:
        monitor = DriftMonitor()
        for prediction, actual in labeled_stream:
            correct = (prediction == actual)
            if monitor.update(correct, timestamp):
                trigger_retraining()
                monitor.reset()
    """

    def __init__(self, delta: float = 0.002):
        """Initialize drift monitor.

        Args:
            delta: Confidence parameter for ADWIN. Lower values = more
                   sensitive to drift. Default 0.002 is good for
                   sentiment accuracy monitoring.
        """
        self.delta = delta
        self.detector = ADWIN(delta=delta)
        self.error_history: list[tuple[int, int]] = []  # (timestamp, error)
        self.drift_events: list[DriftEvent] = []
        self.total_predictions = 0
        self.total_errors = 0

    def update(self, prediction_correct: bool, timestamp: int) -> bool:
        """Update with new prediction result.

        Args:
            prediction_correct: Whether the prediction was correct
            timestamp: Unix timestamp of the prediction

        Returns:
            True if drift was detected
        """
        error = 0 if prediction_correct else 1
        self.error_history.append((timestamp, error))
        self.total_predictions += 1
        self.total_errors += error

        # Update ADWIN
        self.detector.update(error)

        if self.detector.drift_detected:
            event = DriftEvent(
                timestamp=timestamp,
                error_rate_before=self.current_error_rate,
                window_size=self.detector.width,
            )
            self.drift_events.append(event)

            logger.warning(
                f"Drift detected at {timestamp}! "
                f"Error rate: {self.current_error_rate:.2%}, "
                f"Window size: {self.detector.width}"
            )
            return True

        return False

    @property
    def current_error_rate(self) -> float:
        """Current error rate from ADWIN window."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_errors / self.total_predictions

    @property
    def should_retrain(self) -> bool:
        """Check if retraining is needed based on recent drift."""
        return len(self.drift_events) > 0

    def reset(self, keep_history: bool = False):
        """Reset detector after retraining.

        Args:
            keep_history: Whether to keep error history
        """
        self.detector = ADWIN(delta=self.delta)
        self.drift_events = []
        self.total_predictions = 0
        self.total_errors = 0

        if not keep_history:
            self.error_history = []

        logger.info("Drift monitor reset")

    def get_recent_accuracy(self, n: int = 100) -> float:
        """Get accuracy over last n predictions.

        Args:
            n: Number of recent predictions to consider

        Returns:
            Accuracy as a fraction (0-1)
        """
        if not self.error_history:
            return 1.0

        recent = self.error_history[-n:]
        errors = sum(e[1] for e in recent)
        return 1 - (errors / len(recent))

    def get_accuracy_trend(self, window_size: int = 50) -> list[float]:
        """Get rolling accuracy over time.

        Args:
            window_size: Size of rolling window

        Returns:
            List of accuracy values
        """
        if len(self.error_history) < window_size:
            return []

        accuracies = []
        for i in range(window_size, len(self.error_history) + 1):
            window = self.error_history[i - window_size : i]
            errors = sum(e[1] for e in window)
            accuracies.append(1 - errors / window_size)

        return accuracies

    def save(self, path: str | Path) -> None:
        """Save monitor state to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "delta": self.delta,
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "error_history": self.error_history,
            "drift_events": [
                {
                    "timestamp": e.timestamp,
                    "error_rate_before": e.error_rate_before,
                    "error_rate_after": e.error_rate_after,
                    "window_size": e.window_size,
                }
                for e in self.drift_events
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DriftMonitor":
        """Load monitor from JSON file."""
        with open(path) as f:
            data = json.load(f)

        monitor = cls(delta=data["delta"])
        monitor.total_predictions = data["total_predictions"]
        monitor.total_errors = data["total_errors"]
        monitor.error_history = [tuple(e) for e in data["error_history"]]
        monitor.drift_events = [
            DriftEvent(
                timestamp=e["timestamp"],
                error_rate_before=e["error_rate_before"],
                error_rate_after=e.get("error_rate_after"),
                window_size=e.get("window_size"),
            )
            for e in data["drift_events"]
        ]

        return monitor

    def __repr__(self) -> str:
        return (
            f"DriftMonitor(predictions={self.total_predictions}, "
            f"error_rate={self.current_error_rate:.2%}, "
            f"drift_events={len(self.drift_events)})"
        )


class VocabularyMonitor:
    """Monitor for new crypto slang detection.

    Tracks when the tokenizer produces heavily fragmented tokens,
    which indicates novel vocabulary that the model hasn't seen.
    """

    def __init__(
        self,
        tokenizer_name: str = "ElKulako/cryptobert",
        fragmentation_threshold: int = 3,
    ):
        """Initialize vocabulary monitor.

        Args:
            tokenizer_name: Name of tokenizer to use for fragmentation check
            fragmentation_threshold: Max subword pieces before flagging
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.fragmentation_threshold = fragmentation_threshold
        self.unknown_terms: dict[str, int] = {}

    def check_text(self, text: str) -> list[str]:
        """Find heavily fragmented (likely novel) terms.

        Args:
            text: Input text to analyze

        Returns:
            List of novel terms found
        """
        # Simple word tokenization
        words = text.split()
        novel_terms = []

        for word in words:
            # Skip short words
            if len(word) < 3:
                continue

            # Clean word (remove punctuation)
            clean_word = "".join(c for c in word if c.isalnum())
            if not clean_word:
                continue

            # Check tokenizer fragmentation
            tokens = self.tokenizer.tokenize(clean_word)

            if len(tokens) > self.fragmentation_threshold:
                novel_terms.append(clean_word)
                self.unknown_terms[clean_word] = self.unknown_terms.get(clean_word, 0) + 1

        return novel_terms

    def get_trending_unknown(self, min_count: int = 10) -> list[tuple[str, int]]:
        """Get frequently seen unknown terms for lexicon expansion.

        Args:
            min_count: Minimum occurrence count

        Returns:
            List of (term, count) tuples sorted by frequency
        """
        return sorted(
            [(k, v) for k, v in self.unknown_terms.items() if v >= min_count],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_fragmentation_rate(self, texts: list[str]) -> float:
        """Calculate fragmentation rate over a corpus.

        Args:
            texts: List of texts to analyze

        Returns:
            Fraction of words that are heavily fragmented
        """
        total_words = 0
        fragmented_words = 0

        for text in texts:
            novel = self.check_text(text)
            words = [w for w in text.split() if len(w) >= 3]
            total_words += len(words)
            fragmented_words += len(novel)

        if total_words == 0:
            return 0.0

        return fragmented_words / total_words

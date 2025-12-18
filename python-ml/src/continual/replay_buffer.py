"""Experience replay buffer with reservoir sampling.

Implements reservoir sampling to maintain a representative buffer of past examples
for continual learning. This prevents catastrophic forgetting by mixing old
examples with new training data during retraining.
"""

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Example:
    """A labeled training example.

    Attributes:
        text: Input text
        label: Sentiment label ('bearish', 'neutral', 'bullish')
        timestamp: Unix timestamp when example was added
        source: Origin ('human', 'pseudo', 'active')
        confidence: Model confidence if pseudo-labeled
        thread_id: Optional thread ID for traceability
    """

    text: str
    label: str
    timestamp: int
    source: str = "human"
    confidence: float = 1.0
    thread_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(**d)


class ReplayBuffer:
    """Experience replay buffer using reservoir sampling.

    Maintains a fixed-size buffer of past examples that is representative
    of all examples seen. Uses reservoir sampling to ensure uniform
    probability of selection regardless of arrival order.

    The buffer should be mixed with new training data during retraining:
    - 70% new data
    - 30% replay buffer samples

    This single technique provides most forgetting prevention benefit
    with minimal complexity.
    """

    def __init__(self, max_size: int = 2000):
        """Initialize replay buffer.

        Args:
            max_size: Maximum number of examples to store
        """
        self.buffer: list[Example] = []
        self.max_size = max_size
        self.total_seen = 0

    def add(self, examples: list[Example]) -> int:
        """Add examples using reservoir sampling.

        Reservoir sampling maintains a uniform distribution over all examples
        seen, regardless of when they were added. Each example has probability
        max_size/total_seen of being in the buffer.

        Args:
            examples: List of examples to add

        Returns:
            Number of examples actually added/replaced in buffer
        """
        added = 0

        for example in examples:
            self.total_seen += 1

            if len(self.buffer) < self.max_size:
                # Buffer not full, just append
                self.buffer.append(example)
                added += 1
            else:
                # Reservoir sampling: replace with probability max_size/total_seen
                idx = random.randint(0, self.total_seen - 1)
                if idx < self.max_size:
                    self.buffer[idx] = example
                    added += 1

        return added

    def add_one(self, example: Example) -> bool:
        """Add a single example.

        Args:
            example: Example to add

        Returns:
            True if example was added to buffer
        """
        return self.add([example]) > 0

    def sample(self, n: int) -> list[Example]:
        """Sample n random examples from buffer.

        Args:
            n: Number of examples to sample

        Returns:
            List of sampled examples
        """
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def get_training_mix(
        self,
        new_data: list[Example],
        replay_ratio: float = 0.3,
    ) -> list[Example]:
        """Mix new data with replay buffer samples.

        Creates a training batch that combines new examples with historical
        examples from the replay buffer.

        Args:
            new_data: New training examples
            replay_ratio: Fraction of final mix from replay buffer

        Returns:
            Combined list of new and replay examples
        """
        if not self.buffer:
            return new_data

        # Calculate how many replay samples needed
        n_replay = int(len(new_data) * replay_ratio / (1 - replay_ratio))
        replay_samples = self.sample(n_replay)

        # Combine and shuffle
        combined = new_data + replay_samples
        random.shuffle(combined)

        return combined

    def get_label_distribution(self) -> dict[str, int]:
        """Get distribution of labels in buffer.

        Returns:
            Dictionary mapping labels to counts
        """
        distribution = {"bearish": 0, "neutral": 0, "bullish": 0}
        for example in self.buffer:
            if example.label in distribution:
                distribution[example.label] += 1
        return distribution

    def get_source_distribution(self) -> dict[str, int]:
        """Get distribution of sources in buffer.

        Returns:
            Dictionary mapping sources to counts
        """
        distribution: dict[str, int] = {}
        for example in self.buffer:
            distribution[example.source] = distribution.get(example.source, 0) + 1
        return distribution

    def save(self, path: str | Path) -> None:
        """Save buffer to JSON file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "max_size": self.max_size,
            "total_seen": self.total_seen,
            "buffer": [e.to_dict() for e in self.buffer],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved replay buffer with {len(self.buffer)} examples to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        """Load buffer from JSON file.

        Args:
            path: Path to saved file

        Returns:
            Loaded ReplayBuffer instance
        """
        with open(path) as f:
            data = json.load(f)

        buffer = cls(max_size=data["max_size"])
        buffer.total_seen = data["total_seen"]
        buffer.buffer = [Example.from_dict(e) for e in data["buffer"]]

        logger.info(f"Loaded replay buffer with {len(buffer.buffer)} examples")
        return buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self.buffer)}, max={self.max_size}, seen={self.total_seen})"

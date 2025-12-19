#!/usr/bin/env python3
"""
Prepare training data for Qwen2.5-VL fine-tuning.

Converts auto-labels JSON and downloaded images into ShareGPT format
suitable for training.

Usage:
    python -m src.training.prepare_training_data \
        --labels data/auto_labels.json \
        --images data/images \
        --output data/training_data.json

The output format is ShareGPT style:
[
  {
    "messages": [
      {"role": "user", "content": "<image>\nClassify this crypto post..."},
      {"role": "assistant", "content": "bearish"}
    ],
    "images": ["thread_12345.jpg"]
  }
]
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Classification prompt for training
TRAINING_PROMPT = """<image>
Analyze this 4chan /biz/ cryptocurrency post for market sentiment.

Consider the image and this post text: {post_text}

Classify as bearish, neutral, or bullish."""


def download_images(
    db_path: Path,
    output_dir: Path,
    thread_ids: set[int],
    max_workers: int = 5,
) -> dict[int, str]:
    """Download images for specified thread IDs.

    Returns:
        Dict mapping thread_id to local filename
    """
    import concurrent.futures

    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        """
        SELECT thread_id, thumbnail_url
        FROM thread_ops
        WHERE thread_id IN ({})
        """.format(
            ",".join(str(tid) for tid in thread_ids)
        )
    )

    urls = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    downloaded = {}

    def download_one(thread_id: int, url: str) -> tuple[int, str | None]:
        filename = f"thread_{thread_id}.jpg"
        filepath = output_dir / filename

        if filepath.exists():
            return thread_id, filename

        try:
            with httpx.Client() as client:
                response = client.get(url, timeout=10.0)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(response.content)
            return thread_id, filename
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return thread_id, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_one, tid, url) for tid, url in urls.items()
        ]

        for future in concurrent.futures.as_completed(futures):
            thread_id, filename = future.result()
            if filename:
                downloaded[thread_id] = filename

    logger.info(f"Downloaded {len(downloaded)}/{len(urls)} images")
    return downloaded


def convert_to_sharegpt(
    labels: list[dict],
    image_map: dict[int, str],
) -> list[dict]:
    """Convert auto-labels to ShareGPT training format.

    Args:
        labels: List of label dicts from auto_labeler
        image_map: Dict mapping thread_id to local image filename

    Returns:
        List of ShareGPT-formatted training samples
    """
    samples = []

    for label in labels:
        thread_id = label["thread_id"]

        if thread_id not in image_map:
            continue

        post_text = label.get("post_text", "") or "[no text]"
        sentiment = label["label"]

        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": TRAINING_PROMPT.format(post_text=post_text),
                },
                {
                    "role": "assistant",
                    "content": sentiment,
                },
            ],
            "images": [image_map[thread_id]],
            "metadata": {
                "thread_id": thread_id,
                "source": label.get("api_used", "unknown"),
                "confidence": label.get("confidence", 0.0),
            },
        }
        samples.append(sample)

    return samples


def split_dataset(
    samples: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split dataset into train and validation sets.

    Args:
        samples: List of training samples
        val_ratio: Fraction to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples)
    """
    import random

    random.seed(seed)
    samples = samples.copy()
    random.shuffle(samples)

    split_idx = int(len(samples) * (1 - val_ratio))
    return samples[:split_idx], samples[split_idx:]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for Qwen2.5-VL"
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to auto_labels.json from auto_labeler",
    )
    parser.add_argument(
        "--db",
        help="Path to posts.db (for downloading images)",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Directory for images (will download if --db provided)",
    )
    parser.add_argument(
        "--output",
        default="data/training_data.json",
        help="Output path for training data JSON",
    )
    parser.add_argument(
        "--val-output",
        help="Optional: Output path for validation split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence to include in training",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load auto-labels
    labels_path = Path(args.labels)
    with open(labels_path) as f:
        labels = json.load(f)

    logger.info(f"Loaded {len(labels)} labels from {labels_path}")

    # Filter by confidence
    if args.min_confidence > 0:
        labels = [l for l in labels if l.get("confidence", 0) >= args.min_confidence]
        logger.info(f"After confidence filter: {len(labels)} labels")

    # Get thread IDs
    thread_ids = {l["thread_id"] for l in labels}

    # Download images if database provided
    images_dir = Path(args.images)
    if args.db:
        db_path = Path(args.db)
        image_map = download_images(db_path, images_dir, thread_ids)
    else:
        # Assume images already exist with naming convention thread_{id}.jpg
        image_map = {}
        for tid in thread_ids:
            filename = f"thread_{tid}.jpg"
            if (images_dir / filename).exists():
                image_map[tid] = filename
        logger.info(f"Found {len(image_map)} existing images")

    # Convert to training format
    samples = convert_to_sharegpt(labels, image_map)
    logger.info(f"Created {len(samples)} training samples")

    if not samples:
        logger.error("No valid samples created!")
        return 1

    # Split if validation output specified
    if args.val_output:
        train_samples, val_samples = split_dataset(samples, args.val_ratio)
        logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val")

        # Save training set
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(train_samples, f, indent=2)
        logger.info(f"Saved training data: {output_path}")

        # Save validation set
        val_path = Path(args.val_output)
        val_path.parent.mkdir(parents=True, exist_ok=True)
        with open(val_path, "w") as f:
            json.dump(val_samples, f, indent=2)
        logger.info(f"Saved validation data: {val_path}")
    else:
        # Save all as training
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)
        logger.info(f"Saved training data: {output_path}")

    # Print distribution
    dist = {"bearish": 0, "neutral": 0, "bullish": 0}
    for s in samples:
        label = s["messages"][1]["content"]
        dist[label] = dist.get(label, 0) + 1

    print("\n" + "=" * 50)
    print("Training data preparation complete!")
    print(f"Total samples: {len(samples)}")
    print(f"Distribution: {dist}")
    print(f"Output: {args.output}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())

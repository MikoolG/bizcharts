"""Load training data from SQLite database."""

import sqlite3
from pathlib import Path

from datasets import Dataset


def rating_to_label(rating: int) -> str:
    """Convert 1-3 rating scale to sentiment label.

    Mapping:
        1: bearish
        2: neutral
        3: bullish

    Args:
        rating: Integer rating from 1 to 3

    Returns:
        Sentiment label string
    """
    mapping = {1: "bearish", 2: "neutral", 3: "bullish"}
    return mapping.get(rating, "neutral")


def label_to_rating(label: str) -> int:
    """Convert sentiment label back to rating.

    Args:
        label: Sentiment label ('bearish', 'neutral', 'bullish')

    Returns:
        Rating value (1-3)
    """
    mapping = {"bearish": 1, "neutral": 2, "bullish": 3}
    return mapping.get(label, 2)


def load_labeled_data(
    db_path: str | Path,
    min_confidence: float = 0.0,
    exclude_skipped: bool = True,
) -> Dataset:
    """Load labeled posts from training_labels table.

    Args:
        db_path: Path to SQLite database
        min_confidence: Minimum confidence threshold (if stored)
        exclude_skipped: Whether to exclude skipped posts

    Returns:
        HuggingFace Dataset with 'text', 'label', 'rating', 'image_url' columns
    """
    conn = sqlite3.connect(str(db_path))

    # Build query
    query = """
        SELECT
            l.text_snapshot as text,
            l.sentiment_rating as rating,
            l.image_url_snapshot as image_url,
            l.notes
        FROM training_labels l
        WHERE l.sentiment_rating IS NOT NULL
    """

    if exclude_skipped:
        query += " AND (l.skipped = 0 OR l.skipped IS NULL)"

    cursor = conn.execute(query)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        # Return empty dataset with correct schema
        return Dataset.from_dict(
            {"text": [], "label": [], "rating": [], "image_url": [], "notes": []}
        )

    # Process rows
    data = {
        "text": [],
        "label": [],
        "rating": [],
        "image_url": [],
        "notes": [],
    }

    for row in rows:
        text, rating, image_url, notes = row

        # Skip if no text
        if not text or not text.strip():
            continue

        data["text"].append(text.strip())
        data["label"].append(rating_to_label(rating))
        data["rating"].append(rating)
        data["image_url"].append(image_url)
        data["notes"].append(notes or "")

    return Dataset.from_dict(data)


def load_unlabeled_data(
    db_path: str | Path,
    limit: int = 1000,
    with_images_only: bool = False,
) -> Dataset:
    """Load unlabeled posts for active learning.

    Args:
        db_path: Path to SQLite database
        limit: Maximum number of posts to return
        with_images_only: Only return posts that have images

    Returns:
        HuggingFace Dataset with 'thread_id', 'text', 'image_url' columns
    """
    conn = sqlite3.connect(str(db_path))

    query = """
        SELECT
            t.thread_id,
            t.op_text_clean as text,
            t.thumbnail_url as image_url
        FROM thread_ops t
        LEFT JOIN training_labels l ON t.thread_id = l.thread_id
        WHERE l.id IS NULL
          AND t.op_text_clean IS NOT NULL
          AND t.op_text_clean != ''
    """

    if with_images_only:
        query += " AND t.thumbnail_url IS NOT NULL"

    query += f" ORDER BY t.reply_count DESC LIMIT {limit}"

    cursor = conn.execute(query)
    rows = cursor.fetchall()
    conn.close()

    data = {
        "thread_id": [r[0] for r in rows],
        "text": [r[1] for r in rows],
        "image_url": [r[2] for r in rows],
    }

    return Dataset.from_dict(data)


def get_label_distribution(db_path: str | Path) -> dict[str, int]:
    """Get distribution of labels in training data.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dictionary mapping labels to counts
    """
    conn = sqlite3.connect(str(db_path))

    cursor = conn.execute("""
        SELECT sentiment_rating, COUNT(*) as count
        FROM training_labels
        WHERE sentiment_rating IS NOT NULL
          AND (skipped = 0 OR skipped IS NULL)
        GROUP BY sentiment_rating
        ORDER BY sentiment_rating
    """)

    rows = cursor.fetchall()
    conn.close()

    # Aggregate by label
    distribution = {"bearish": 0, "neutral": 0, "bullish": 0}
    for rating, count in rows:
        label = rating_to_label(rating)
        distribution[label] += count

    return distribution


def export_to_csv(db_path: str | Path, output_path: str | Path) -> int:
    """Export labeled data to CSV for external tools.

    Args:
        db_path: Path to SQLite database
        output_path: Path to output CSV file

    Returns:
        Number of rows exported
    """
    import csv

    dataset = load_labeled_data(db_path)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "rating"])
        writer.writeheader()

        for row in dataset:
            writer.writerow(
                {
                    "text": row["text"],
                    "label": row["label"],
                    "rating": row["rating"],
                }
            )

    return len(dataset)

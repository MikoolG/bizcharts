"""Zero-shot pseudo-labeling with CryptoBERT.

Uses CryptoBERT's pre-trained crypto sentiment classification to generate
pseudo-labels for unlabeled data. High-confidence predictions can be used
for training or to bootstrap the active learning process.
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def pseudo_label_posts(
    db_path: str | Path,
    output_table: str = "pseudo_labels",
    batch_size: int = 32,
    min_confidence: float = 0.0,
    limit: int | None = None,
    device: str | None = None,
) -> dict:
    """Run CryptoBERT on unlabeled posts and store pseudo-labels.

    Args:
        db_path: Path to SQLite database
        output_table: Table name to store pseudo-labels
        batch_size: Batch size for inference
        min_confidence: Only store predictions above this confidence
        limit: Maximum number of posts to process (None = all)
        device: Device for inference ('cuda', 'cpu', or None for auto)

    Returns:
        Statistics dict with counts and confidence distribution
    """
    from ..models.cryptobert_model import CryptoBERTModel

    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))

    # Create output table
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {output_table} (
            thread_id INTEGER PRIMARY KEY,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            prob_bearish REAL,
            prob_neutral REAL,
            prob_bullish REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Load unlabeled posts
    query = """
        SELECT t.thread_id, t.op_text_clean as text
        FROM thread_ops t
        LEFT JOIN training_labels l ON t.thread_id = l.thread_id
        WHERE l.id IS NULL
          AND t.op_text_clean IS NOT NULL
          AND t.op_text_clean != ''
        ORDER BY t.reply_count DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)
    rows = cursor.fetchall()
    logger.info(f"Found {len(rows)} unlabeled posts")

    if not rows:
        return {"total": 0, "labeled": 0}

    # Initialize model
    logger.info("Loading CryptoBERT model...")
    model = CryptoBERTModel(use_lora=False, device=device)

    # Process in batches
    stats = {
        "total": len(rows),
        "labeled": 0,
        "by_label": {"bearish": 0, "neutral": 0, "bullish": 0},
        "confidence_bins": {
            "0.5-0.6": 0,
            "0.6-0.7": 0,
            "0.7-0.8": 0,
            "0.8-0.9": 0,
            "0.9-1.0": 0,
        },
    }

    for i in tqdm(range(0, len(rows), batch_size), desc="Pseudo-labeling"):
        batch = rows[i : i + batch_size]
        thread_ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        # Get predictions
        results = model.predict_batch(texts)

        # Store results
        for thread_id, result in zip(thread_ids, results):
            if result.confidence < min_confidence:
                continue

            conn.execute(
                f"""
                INSERT OR REPLACE INTO {output_table}
                (thread_id, label, confidence, prob_bearish, prob_neutral, prob_bullish)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    thread_id,
                    result.label,
                    result.confidence,
                    result.probabilities["bearish"],
                    result.probabilities["neutral"],
                    result.probabilities["bullish"],
                ),
            )

            stats["labeled"] += 1
            stats["by_label"][result.label] += 1

            # Confidence bins
            if result.confidence >= 0.9:
                stats["confidence_bins"]["0.9-1.0"] += 1
            elif result.confidence >= 0.8:
                stats["confidence_bins"]["0.8-0.9"] += 1
            elif result.confidence >= 0.7:
                stats["confidence_bins"]["0.7-0.8"] += 1
            elif result.confidence >= 0.6:
                stats["confidence_bins"]["0.6-0.7"] += 1
            else:
                stats["confidence_bins"]["0.5-0.6"] += 1

        conn.commit()

    conn.close()
    return stats


def get_high_confidence_labels(
    db_path: str | Path,
    min_confidence: float = 0.85,
    table: str = "pseudo_labels",
) -> list[dict]:
    """Get pseudo-labels above confidence threshold.

    Args:
        db_path: Path to SQLite database
        min_confidence: Minimum confidence threshold
        table: Table name with pseudo-labels

    Returns:
        List of dicts with thread_id, text, label, confidence
    """
    conn = sqlite3.connect(str(db_path))

    cursor = conn.execute(
        f"""
        SELECT p.thread_id, t.op_text_clean as text, p.label, p.confidence
        FROM {table} p
        JOIN thread_ops t ON p.thread_id = t.thread_id
        WHERE p.confidence >= ?
        ORDER BY p.confidence DESC
    """,
        (min_confidence,),
    )

    results = []
    for row in cursor.fetchall():
        results.append(
            {
                "thread_id": row[0],
                "text": row[1],
                "label": row[2],
                "confidence": row[3],
            }
        )

    conn.close()
    return results


def get_uncertain_labels(
    db_path: str | Path,
    max_confidence: float = 0.6,
    table: str = "pseudo_labels",
    limit: int = 200,
) -> list[dict]:
    """Get low-confidence pseudo-labels for manual review.

    These are the cases where CryptoBERT is uncertain and human
    labels would be most valuable (active learning candidates).

    Args:
        db_path: Path to SQLite database
        max_confidence: Maximum confidence threshold
        table: Table name with pseudo-labels
        limit: Maximum number to return

    Returns:
        List of dicts with thread_id, text, label, confidence, probabilities
    """
    conn = sqlite3.connect(str(db_path))

    cursor = conn.execute(
        f"""
        SELECT
            p.thread_id,
            t.op_text_clean as text,
            p.label,
            p.confidence,
            p.prob_bearish,
            p.prob_neutral,
            p.prob_bullish,
            t.thumbnail_url as image_url
        FROM {table} p
        JOIN thread_ops t ON p.thread_id = t.thread_id
        WHERE p.confidence < ?
        ORDER BY p.confidence ASC
        LIMIT ?
    """,
        (max_confidence, limit),
    )

    results = []
    for row in cursor.fetchall():
        results.append(
            {
                "thread_id": row[0],
                "text": row[1],
                "predicted_label": row[2],
                "confidence": row[3],
                "probabilities": {
                    "bearish": row[4],
                    "neutral": row[5],
                    "bullish": row[6],
                },
                "image_url": row[7],
            }
        )

    conn.close()
    return results


def export_training_set(
    db_path: str | Path,
    output_path: str | Path,
    min_confidence: float = 0.85,
    include_manual_labels: bool = True,
    table: str = "pseudo_labels",
) -> int:
    """Export combined training set (manual + high-confidence pseudo).

    Args:
        db_path: Path to SQLite database
        output_path: Path to output CSV
        min_confidence: Minimum confidence for pseudo-labels
        include_manual_labels: Whether to include manual training_labels
        table: Table name with pseudo-labels

    Returns:
        Number of rows exported
    """
    import csv
    from .data_loader import rating_to_label

    conn = sqlite3.connect(str(db_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # Get manual labels
    if include_manual_labels:
        cursor = conn.execute("""
            SELECT text_snapshot, sentiment_rating, 'manual' as source
            FROM training_labels
            WHERE sentiment_rating IS NOT NULL
              AND (skipped = 0 OR skipped IS NULL)
        """)
        for row in cursor.fetchall():
            rows.append(
                {
                    "text": row[0],
                    "label": rating_to_label(row[1]),
                    "source": row[2],
                    "confidence": 1.0,
                }
            )

    # Get high-confidence pseudo-labels
    cursor = conn.execute(
        f"""
        SELECT t.op_text_clean, p.label, 'pseudo' as source, p.confidence
        FROM {table} p
        JOIN thread_ops t ON p.thread_id = t.thread_id
        WHERE p.confidence >= ?
    """,
        (min_confidence,),
    )
    for row in cursor.fetchall():
        rows.append(
            {
                "text": row[0],
                "label": row[1],
                "source": row[2],
                "confidence": row[3],
            }
        )

    conn.close()

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} samples to {output_path}")
    return len(rows)


def main():
    """CLI entry point for pseudo-labeling."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate pseudo-labels with CryptoBERT")
    parser.add_argument("--db", required=True, help="Path to posts.db")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Min confidence to store")
    parser.add_argument("--limit", type=int, default=None, help="Max posts to process")
    parser.add_argument("--export", type=str, default=None, help="Export training set to CSV")
    parser.add_argument("--export-confidence", type=float, default=0.85, help="Min confidence for export")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.export:
        # Export mode
        count = export_training_set(
            args.db,
            args.export,
            min_confidence=args.export_confidence,
        )
        print(f"Exported {count} samples to {args.export}")
    else:
        # Labeling mode
        stats = pseudo_label_posts(
            args.db,
            batch_size=args.batch_size,
            min_confidence=args.min_confidence,
            limit=args.limit,
        )

        print(f"\nPseudo-labeling complete!")
        print(f"Total processed: {stats['total']}")
        print(f"Labels stored: {stats['labeled']}")
        print(f"\nLabel distribution:")
        for label, count in stats["by_label"].items():
            print(f"  {label}: {count}")
        print(f"\nConfidence distribution:")
        for bin_name, count in stats["confidence_bins"].items():
            print(f"  {bin_name}: {count}")


if __name__ == "__main__":
    main()

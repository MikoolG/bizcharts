#!/usr/bin/env python3
"""
Analysis utilities for labeled sentiment data.

Usage:
    python -m src.label_analysis --summary
    python -m src.label_analysis --distribution
    python -m src.label_analysis --errors
    python -m src.label_analysis --export training_data.csv
"""

import argparse
import sqlite3
from pathlib import Path


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def show_summary(conn: sqlite3.Connection):
    """Show labeling summary statistics."""
    print("\n" + "=" * 60)
    print("LABELING SUMMARY")
    print("=" * 60)

    # Overall stats
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN skipped THEN 1 ELSE 0 END) as skipped,
            SUM(CASE WHEN NOT skipped AND sentiment_rating IS NOT NULL THEN 1 ELSE 0 END) as rated,
            COUNT(DISTINCT labeler_id) as labelers
        FROM training_labels
    """)
    row = cursor.fetchone()
    print(f"\nTotal labels: {row['total']}")
    print(f"Rated: {row['rated']}")
    print(f"Skipped: {row['skipped']}")
    print(f"Unique labelers: {row['labelers']}")

    # Rating stats
    cursor = conn.execute("""
        SELECT
            AVG(sentiment_rating) as avg,
            MIN(sentiment_rating) as min,
            MAX(sentiment_rating) as max,
            COUNT(*) as count
        FROM training_labels
        WHERE NOT skipped AND sentiment_rating IS NOT NULL
    """)
    row = cursor.fetchone()
    if row['count'] > 0:
        print(f"\nRating Statistics:")
        print(f"  Average: {row['avg']:.2f}")
        print(f"  Range: {row['min']} - {row['max']}")

    # By labeler
    cursor = conn.execute("""
        SELECT
            labeler_id,
            COUNT(*) as count,
            AVG(sentiment_rating) as avg_rating,
            SUM(CASE WHEN skipped THEN 1 ELSE 0 END) as skipped
        FROM training_labels
        GROUP BY labeler_id
    """)
    print(f"\nBy Labeler:")
    for row in cursor:
        avg = f"{row['avg_rating']:.2f}" if row['avg_rating'] else "N/A"
        print(f"  {row['labeler_id']}: {row['count']} labels, avg={avg}, skipped={row['skipped']}")


def show_distribution(conn: sqlite3.Connection):
    """Show rating distribution."""
    print("\n" + "=" * 60)
    print("RATING DISTRIBUTION")
    print("=" * 60)

    cursor = conn.execute("""
        SELECT
            sentiment_rating,
            COUNT(*) as count
        FROM training_labels
        WHERE NOT skipped AND sentiment_rating IS NOT NULL
        GROUP BY sentiment_rating
        ORDER BY sentiment_rating
    """)

    rows = list(cursor)
    if not rows:
        print("\nNo ratings yet.")
        return

    total = sum(r['count'] for r in rows)
    max_count = max(r['count'] for r in rows)

    print(f"\nTotal rated: {total}\n")
    print("Rating  Count  Bar")
    print("-" * 50)

    for row in rows:
        rating = row['sentiment_rating']
        count = row['count']
        pct = count / total * 100
        bar_len = int(count / max_count * 30)
        bar = "#" * bar_len

        # Color indicator
        if rating <= 3:
            sentiment = "BEAR"
        elif rating <= 4:
            sentiment = "bear"
        elif rating <= 6:
            sentiment = "neut"
        elif rating <= 7:
            sentiment = "bull"
        else:
            sentiment = "BULL"

        print(f"  {rating:2d}    {count:4d}  {bar} ({pct:.1f}%) [{sentiment}]")

    # Aggregate sentiment
    bearish = sum(r['count'] for r in rows if r['sentiment_rating'] <= 4)
    neutral = sum(r['count'] for r in rows if 5 <= r['sentiment_rating'] <= 6)
    bullish = sum(r['count'] for r in rows if r['sentiment_rating'] >= 7)

    print(f"\nAggregate:")
    print(f"  Bearish (1-4): {bearish} ({bearish/total*100:.1f}%)")
    print(f"  Neutral (5-6): {neutral} ({neutral/total*100:.1f}%)")
    print(f"  Bullish (7-10): {bullish} ({bullish/total*100:.1f}%)")


def show_errors(conn: sqlite3.Connection):
    """Show where automated scores differ from human labels."""
    print("\n" + "=" * 60)
    print("AUTOMATED vs HUMAN COMPARISON")
    print("=" * 60)

    cursor = conn.execute("""
        SELECT
            t.thread_id,
            t.subject,
            t.sentiment_score as auto_score,
            t.sentiment_confidence as auto_confidence,
            t.sentiment_method as method,
            l.sentiment_rating as human_rating,
            t.op_text_clean
        FROM thread_ops t
        JOIN training_labels l ON t.thread_id = l.thread_id
        WHERE t.sentiment_score IS NOT NULL
          AND l.skipped = FALSE
          AND l.sentiment_rating IS NOT NULL
    """)

    rows = list(cursor)
    if not rows:
        print("\nNo posts with both automated and human scores.")
        print("Run sentiment analysis first, then label some posts.")
        return

    # Calculate errors
    errors = []
    for row in rows:
        auto = row['auto_score']  # -1 to +1
        human = (row['human_rating'] - 5.5) / 4.5  # Convert 1-10 to -1 to +1
        error = abs(auto - human)
        errors.append({
            'thread_id': row['thread_id'],
            'subject': row['subject'],
            'auto_score': auto,
            'auto_confidence': row['auto_confidence'],
            'human_rating': row['human_rating'],
            'human_score': human,
            'error': error,
            'text': (row['op_text_clean'] or "")[:100],
        })

    # Sort by error
    errors.sort(key=lambda x: x['error'], reverse=True)

    # Overall metrics
    avg_error = sum(e['error'] for e in errors) / len(errors)
    correlation = calculate_correlation(
        [e['auto_score'] for e in errors],
        [e['human_score'] for e in errors]
    )

    print(f"\nTotal compared: {len(errors)}")
    print(f"Average error: {avg_error:.3f}")
    print(f"Correlation: {correlation:.3f}")

    # Show worst errors
    print(f"\nTop 10 Largest Discrepancies:")
    print("-" * 80)
    for e in errors[:10]:
        auto_str = f"{e['auto_score']:+.2f}"
        human_str = f"{e['human_rating']}/10"
        print(f"\nThread #{e['thread_id']} (error: {e['error']:.2f})")
        print(f"  Auto: {auto_str} | Human: {human_str}")
        print(f"  Subject: {e['subject'] or '(none)'}")
        print(f"  Text: {e['text']}...")


def calculate_correlation(x: list, y: list) -> float:
    """Calculate Pearson correlation coefficient."""
    n = len(x)
    if n == 0:
        return 0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return 0

    return numerator / (denom_x * denom_y)


def export_training_data(conn: sqlite3.Connection, output_path: str):
    """Export labeled data for model training."""
    import csv

    cursor = conn.execute("""
        SELECT
            l.thread_id,
            l.sentiment_rating,
            l.text_snapshot as text,
            l.image_url_snapshot as image_url,
            l.notes,
            l.labeler_id,
            l.labeled_at,
            t.subject,
            t.reply_count,
            t.source,
            t.has_image
        FROM training_labels l
        JOIN thread_ops t ON l.thread_id = t.thread_id
        WHERE l.skipped = FALSE
          AND l.sentiment_rating IS NOT NULL
        ORDER BY l.labeled_at
    """)

    rows = list(cursor)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'thread_id', 'sentiment_rating', 'sentiment_normalized',
            'text', 'image_url', 'notes', 'labeler_id', 'labeled_at',
            'subject', 'reply_count', 'source', 'has_image'
        ])

        for row in rows:
            # Normalize rating to -1 to +1 scale
            normalized = (row['sentiment_rating'] - 5.5) / 4.5

            writer.writerow([
                row['thread_id'],
                row['sentiment_rating'],
                f"{normalized:.3f}",
                row['text'],
                row['image_url'],
                row['notes'],
                row['labeler_id'],
                row['labeled_at'],
                row['subject'],
                row['reply_count'],
                row['source'],
                row['has_image'],
            ])

    print(f"\nExported {len(rows)} labeled posts to {output_path}")
    print("\nColumns:")
    print("  - sentiment_rating: Original 1-10 scale")
    print("  - sentiment_normalized: Converted to -1 to +1 scale")


def show_recent(conn: sqlite3.Connection, limit: int = 20):
    """Show recently labeled posts."""
    print("\n" + "=" * 60)
    print(f"RECENT LABELS (last {limit})")
    print("=" * 60)

    cursor = conn.execute("""
        SELECT
            l.thread_id,
            l.sentiment_rating,
            l.skipped,
            l.labeled_at,
            l.labeler_id,
            t.subject
        FROM training_labels l
        JOIN thread_ops t ON l.thread_id = t.thread_id
        ORDER BY l.labeled_at DESC
        LIMIT ?
    """, (limit,))

    print(f"\n{'ID':>10} {'Rating':>8} {'Labeler':>10} {'Subject':<40}")
    print("-" * 75)

    for row in cursor:
        rating = "SKIP" if row['skipped'] else f"{row['sentiment_rating']}/10"
        subject = (row['subject'] or "(none)")[:38]
        print(f"{row['thread_id']:>10} {rating:>8} {row['labeler_id']:>10} {subject:<40}")


def main():
    parser = argparse.ArgumentParser(description="Analyze labeled sentiment data")
    parser.add_argument("--db", default="../data/posts.db", help="Database path")
    parser.add_argument("--summary", action="store_true", help="Show summary statistics")
    parser.add_argument("--distribution", action="store_true", help="Show rating distribution")
    parser.add_argument("--errors", action="store_true", help="Compare auto vs human scores")
    parser.add_argument("--recent", type=int, metavar="N", help="Show N recent labels")
    parser.add_argument("--export", metavar="FILE", help="Export training data to CSV")
    parser.add_argument("--all", action="store_true", help="Show all analysis")

    args = parser.parse_args()

    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        script_dir = Path(__file__).parent.parent
        db_path = script_dir / "data" / "posts.db"

    if not db_path.exists():
        print(f"Database not found: {args.db}")
        return

    conn = get_db_connection(str(db_path))

    # Default to summary if no options specified
    if not any([args.summary, args.distribution, args.errors, args.recent, args.export, args.all]):
        args.summary = True

    if args.all:
        args.summary = True
        args.distribution = True
        args.errors = True

    if args.summary:
        show_summary(conn)

    if args.distribution:
        show_distribution(conn)

    if args.errors:
        show_errors(conn)

    if args.recent:
        show_recent(conn, args.recent)

    if args.export:
        export_training_data(conn, args.export)

    conn.close()


if __name__ == "__main__":
    main()

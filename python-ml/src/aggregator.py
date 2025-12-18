"""Sentiment aggregation and Fear/Greed index calculation."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

from .text_analyzer import TextAnalyzer


@dataclass
class MarketSentiment:
    """Market-wide sentiment snapshot."""

    timestamp: datetime
    fear_greed_index: float  # 0-100
    avg_sentiment: float  # -1 to +1
    bullish_pct: float
    bearish_pct: float
    neutral_pct: float
    total_posts: int
    unique_threads: int


@dataclass
class CoinSentiment:
    """Sentiment for a specific coin."""

    symbol: str
    timestamp: datetime
    avg_sentiment: float
    post_count: int
    confidence: float


class SentimentAggregator:
    """Aggregate sentiment scores and calculate indices."""

    def __init__(
        self,
        sqlite_path: str | Path = "data/posts.db",
        duckdb_path: str | Path = "data/analytics.duckdb",
    ):
        self.sqlite_path = Path(sqlite_path)
        self.duckdb_path = Path(duckdb_path)
        self.text_analyzer = TextAnalyzer()

        # Ensure DuckDB directory exists
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_duckdb()

    def _init_duckdb(self) -> None:
        """Initialize DuckDB schema."""
        with duckdb.connect(str(self.duckdb_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    post_id BIGINT PRIMARY KEY,
                    thread_id BIGINT,
                    timestamp TIMESTAMP,
                    overall_score FLOAT,
                    confidence FLOAT,
                    method VARCHAR,
                    is_greentext BOOLEAN,
                    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS coin_sentiment (
                    coin_symbol VARCHAR,
                    bucket_start TIMESTAMP,
                    bucket_size VARCHAR,
                    avg_sentiment FLOAT,
                    weighted_sentiment FLOAT,
                    post_count INTEGER,
                    bullish_count INTEGER,
                    bearish_count INTEGER,
                    neutral_count INTEGER,
                    PRIMARY KEY (coin_symbol, bucket_start, bucket_size)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_sentiment (
                    bucket_start TIMESTAMP,
                    bucket_size VARCHAR,
                    fear_greed_index FLOAT,
                    avg_sentiment FLOAT,
                    bullish_pct FLOAT,
                    bearish_pct FLOAT,
                    neutral_pct FLOAT,
                    total_posts INTEGER,
                    unique_threads INTEGER,
                    PRIMARY KEY (bucket_start, bucket_size)
                )
            """)

    def analyze_new_posts(self, batch_size: int = 1000) -> int:
        """
        Analyze posts that haven't been scored yet.

        Returns:
            Number of posts analyzed
        """
        # Get unanalyzed posts from SQLite
        with sqlite3.connect(self.sqlite_path) as sqlite_conn:
            sqlite_conn.row_factory = sqlite3.Row

            # Find posts not yet in DuckDB
            with duckdb.connect(str(self.duckdb_path)) as duck_conn:
                existing_ids = duck_conn.execute(
                    "SELECT post_id FROM sentiment_scores"
                ).fetchall()
                existing_set = {row[0] for row in existing_ids}

            cursor = sqlite_conn.execute(
                """
                SELECT post_id, thread_id, timestamp, text
                FROM posts
                WHERE text IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (batch_size * 2,),  # Fetch extra to account for already-processed
            )

            posts_to_analyze = []
            for row in cursor:
                if row["post_id"] not in existing_set:
                    posts_to_analyze.append(dict(row))
                    if len(posts_to_analyze) >= batch_size:
                        break

        if not posts_to_analyze:
            return 0

        # Analyze each post
        results = []
        for post in posts_to_analyze:
            result = self.text_analyzer.analyze(post["text"])
            results.append({
                "post_id": post["post_id"],
                "thread_id": post["thread_id"],
                "timestamp": datetime.fromtimestamp(post["timestamp"]),
                "overall_score": result.score,
                "confidence": result.confidence,
                "method": result.method,
                "is_greentext": result.is_greentext,
            })

        # Store in DuckDB
        df = pl.DataFrame(results)
        with duckdb.connect(str(self.duckdb_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sentiment_scores
                SELECT * FROM df
            """)

        return len(results)

    def aggregate_hourly(self) -> None:
        """Aggregate sentiment scores by hour."""
        with duckdb.connect(str(self.duckdb_path)) as conn:
            # Market-wide aggregation
            conn.execute("""
                INSERT OR REPLACE INTO market_sentiment
                SELECT
                    date_trunc('hour', timestamp) as bucket_start,
                    'hour' as bucket_size,
                    -- Fear/Greed calculation (simplified)
                    GREATEST(0, LEAST(100,
                        50 + (AVG(overall_score) * 50)
                    )) as fear_greed_index,
                    AVG(overall_score) as avg_sentiment,
                    100.0 * SUM(CASE WHEN overall_score > 0.05 THEN 1 ELSE 0 END) / COUNT(*) as bullish_pct,
                    100.0 * SUM(CASE WHEN overall_score < -0.05 THEN 1 ELSE 0 END) / COUNT(*) as bearish_pct,
                    100.0 * SUM(CASE WHEN overall_score BETWEEN -0.05 AND 0.05 THEN 1 ELSE 0 END) / COUNT(*) as neutral_pct,
                    COUNT(*) as total_posts,
                    COUNT(DISTINCT thread_id) as unique_threads
                FROM sentiment_scores
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY date_trunc('hour', timestamp)
            """)

    def aggregate_coin_sentiment(self) -> None:
        """Aggregate sentiment by coin."""
        with (
            sqlite3.connect(self.sqlite_path) as sqlite_conn,
            duckdb.connect(str(self.duckdb_path)) as duck_conn,
        ):
            # Get coin mentions from SQLite
            cursor = sqlite_conn.execute("""
                SELECT cm.post_id, cm.coin_symbol
                FROM coin_mentions cm
                JOIN posts p ON cm.post_id = p.post_id
                WHERE p.timestamp >= strftime('%s', 'now', '-7 days')
            """)
            mentions = cursor.fetchall()

            if not mentions:
                return

            mentions_df = pl.DataFrame(
                {"post_id": [m[0] for m in mentions], "coin_symbol": [m[1] for m in mentions]}
            )

            # Join with sentiment scores
            duck_conn.execute("""
                INSERT OR REPLACE INTO coin_sentiment
                SELECT
                    m.coin_symbol,
                    date_trunc('hour', s.timestamp) as bucket_start,
                    'hour' as bucket_size,
                    AVG(s.overall_score) as avg_sentiment,
                    SUM(s.overall_score * s.confidence) / SUM(s.confidence) as weighted_sentiment,
                    COUNT(*) as post_count,
                    SUM(CASE WHEN s.overall_score > 0.05 THEN 1 ELSE 0 END) as bullish_count,
                    SUM(CASE WHEN s.overall_score < -0.05 THEN 1 ELSE 0 END) as bearish_count,
                    SUM(CASE WHEN s.overall_score BETWEEN -0.05 AND 0.05 THEN 1 ELSE 0 END) as neutral_count
                FROM mentions_df m
                JOIN sentiment_scores s ON m.post_id = s.post_id
                GROUP BY m.coin_symbol, date_trunc('hour', s.timestamp)
            """)

    def get_current_sentiment(self) -> MarketSentiment | None:
        """Get the most recent market sentiment."""
        with duckdb.connect(str(self.duckdb_path)) as conn:
            result = conn.execute("""
                SELECT *
                FROM market_sentiment
                WHERE bucket_size = 'hour'
                ORDER BY bucket_start DESC
                LIMIT 1
            """).fetchone()

            if not result:
                return None

            return MarketSentiment(
                timestamp=result[0],
                fear_greed_index=result[2],
                avg_sentiment=result[3],
                bullish_pct=result[4],
                bearish_pct=result[5],
                neutral_pct=result[6],
                total_posts=result[7],
                unique_threads=result[8],
            )

    def get_coin_sentiment(self, symbol: str, hours: int = 24) -> list[CoinSentiment]:
        """Get sentiment history for a specific coin."""
        with duckdb.connect(str(self.duckdb_path)) as conn:
            results = conn.execute("""
                SELECT coin_symbol, bucket_start, avg_sentiment, post_count,
                       weighted_sentiment / avg_sentiment as confidence
                FROM coin_sentiment
                WHERE coin_symbol = ?
                  AND bucket_size = 'hour'
                  AND bucket_start >= NOW() - INTERVAL ? HOUR
                ORDER BY bucket_start DESC
            """, [symbol, hours]).fetchall()

            return [
                CoinSentiment(
                    symbol=r[0],
                    timestamp=r[1],
                    avg_sentiment=r[2],
                    post_count=r[3],
                    confidence=r[4] if r[4] else 0.5,
                )
                for r in results
            ]

    def calculate_fear_greed(
        self,
        avg_sentiment: float,
        bullish_pct: float,
        volume_zscore: float = 0.0,
    ) -> float:
        """
        Calculate Fear/Greed index on 0-100 scale.

        Components:
        - Sentiment score: 40%
        - Bullish/bearish ratio: 35%
        - Volume momentum: 25%
        """
        # Normalize sentiment from [-1, 1] to [0, 100]
        sentiment_component = (avg_sentiment + 1) * 50

        # Bullish percentage is already 0-100
        ratio_component = bullish_pct

        # Volume z-score normalized
        volume_clamped = max(-3, min(3, volume_zscore))
        volume_component = (volume_clamped + 3) / 6 * 100

        index = (
            sentiment_component * 0.40
            + ratio_component * 0.35
            + volume_component * 0.25
        )

        return round(max(0, min(100, index)), 1)


def main() -> None:
    """Run sentiment aggregation."""
    aggregator = SentimentAggregator()

    print("Analyzing new posts...")
    count = aggregator.analyze_new_posts()
    print(f"Analyzed {count} posts")

    print("Aggregating hourly sentiment...")
    aggregator.aggregate_hourly()

    print("Aggregating coin sentiment...")
    aggregator.aggregate_coin_sentiment()

    sentiment = aggregator.get_current_sentiment()
    if sentiment:
        print(f"\nCurrent Market Sentiment:")
        print(f"  Fear/Greed Index: {sentiment.fear_greed_index:.1f}")
        print(f"  Average Sentiment: {sentiment.avg_sentiment:.3f}")
        print(f"  Bullish: {sentiment.bullish_pct:.1f}%")
        print(f"  Bearish: {sentiment.bearish_pct:.1f}%")
        print(f"  Neutral: {sentiment.neutral_pct:.1f}%")
        print(f"  Total Posts: {sentiment.total_posts}")
    else:
        print("\nNo sentiment data available yet")


if __name__ == "__main__":
    main()

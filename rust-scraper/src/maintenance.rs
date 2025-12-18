//! Database maintenance and storage optimization
//!
//! Provides retention policies, cleanup routines, and storage optimization
//! to keep the database size manageable over time.

use anyhow::Result;
use rusqlite::{Connection, params};
use tracing::info;

/// Retention configuration for database cleanup
#[derive(Debug, Clone)]
pub struct RetentionConfig {
    /// Days to keep full thread data (with raw HTML)
    pub full_data_days: i64,
    /// Days to keep thread metadata (cleaned text only)
    pub metadata_days: i64,
    /// Days to keep catalog snapshots at full resolution
    pub snapshot_full_days: i64,
    /// Days to keep hourly aggregated snapshots
    pub snapshot_hourly_days: i64,
    /// Whether to keep analyzed threads longer
    pub preserve_analyzed: bool,
    /// Minimum reply count to preserve indefinitely
    pub preserve_min_replies: i32,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            full_data_days: 30,        // Keep raw HTML for 30 days
            metadata_days: 365,        // Keep metadata for 1 year
            snapshot_full_days: 7,     // Full resolution snapshots for 7 days
            snapshot_hourly_days: 90,  // Hourly aggregates for 90 days
            preserve_analyzed: true,   // Keep threads with sentiment scores
            preserve_min_replies: 50,  // Keep popular threads forever
        }
    }
}

/// Database maintenance operations
pub struct Maintenance<'a> {
    conn: &'a Connection,
    config: RetentionConfig,
}

impl<'a> Maintenance<'a> {
    pub fn new(conn: &'a Connection, config: RetentionConfig) -> Self {
        Self { conn, config }
    }

    /// Run all maintenance tasks
    pub fn run_all(&self) -> Result<MaintenanceStats> {
        let mut stats = MaintenanceStats::default();

        // 1. Clean up raw HTML from old threads (keep clean text)
        stats.html_stripped = self.strip_old_html()?;

        // 2. Aggregate old catalog snapshots
        stats.snapshots_aggregated = self.aggregate_snapshots()?;

        // 3. Remove very old threads (beyond metadata retention)
        stats.threads_deleted = self.delete_old_threads()?;

        // 4. Clean orphaned coin mentions
        stats.mentions_cleaned = self.clean_orphan_mentions()?;

        // 5. Drop legacy tables if empty
        stats.legacy_cleaned = self.clean_legacy_tables()?;

        // 6. Optimize database
        self.vacuum()?;

        Ok(stats)
    }

    /// Strip raw HTML from old threads, keeping only cleaned text
    /// This saves ~40% storage for threads past full_data_days
    fn strip_old_html(&self) -> Result<i64> {
        let cutoff = chrono::Utc::now().timestamp() - (self.config.full_data_days * 86400);

        let affected = self.conn.execute(
            r#"
            UPDATE thread_ops
            SET op_text = NULL
            WHERE created_at < ?1
              AND op_text IS NOT NULL
              AND op_text_clean IS NOT NULL
              AND (reply_count < ?2 OR ?3 = 0)
              AND (?4 = 0 OR sentiment_score IS NULL)
            "#,
            params![
                cutoff,
                self.config.preserve_min_replies,
                self.config.preserve_min_replies,  // 0 means don't filter
                self.config.preserve_analyzed as i32,
            ],
        )?;

        if affected > 0 {
            info!("Stripped raw HTML from {} old threads", affected);
        }

        Ok(affected as i64)
    }

    /// Aggregate catalog snapshots to reduce storage
    /// - Keep full resolution for snapshot_full_days
    /// - Aggregate to hourly for snapshot_hourly_days
    /// - Aggregate to daily beyond that
    fn aggregate_snapshots(&self) -> Result<i64> {
        let now = chrono::Utc::now().timestamp();
        let hourly_cutoff = now - (self.config.snapshot_full_days * 86400);
        let daily_cutoff = now - (self.config.snapshot_hourly_days * 86400);

        // Create aggregated snapshots table if not exists
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS catalog_snapshots_hourly (
                hour_timestamp INTEGER PRIMARY KEY,
                avg_total_threads REAL,
                avg_threads_with_images REAL,
                avg_reply_count REAL,
                sample_count INTEGER,
                top_coins TEXT
            );

            CREATE TABLE IF NOT EXISTS catalog_snapshots_daily (
                day_timestamp INTEGER PRIMARY KEY,
                avg_total_threads REAL,
                avg_threads_with_images REAL,
                avg_reply_count REAL,
                sample_count INTEGER,
                top_coins TEXT
            );
            "#
        )?;

        // Aggregate old snapshots to hourly
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO catalog_snapshots_hourly
            SELECT
                (snapshot_at / 3600) * 3600 as hour_timestamp,
                AVG(total_threads) as avg_total_threads,
                AVG(threads_with_images) as avg_threads_with_images,
                AVG(avg_reply_count) as avg_reply_count,
                COUNT(*) as sample_count,
                -- Keep the top_coins from the most recent snapshot in the hour
                (SELECT top_coins FROM catalog_snapshots cs2
                 WHERE (cs2.snapshot_at / 3600) = (catalog_snapshots.snapshot_at / 3600)
                 ORDER BY cs2.snapshot_at DESC LIMIT 1) as top_coins
            FROM catalog_snapshots
            WHERE snapshot_at < ?1 AND snapshot_at >= ?2
            GROUP BY (snapshot_at / 3600)
            "#,
            params![hourly_cutoff, daily_cutoff],
        )?;

        // Aggregate very old snapshots to daily
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO catalog_snapshots_daily
            SELECT
                (snapshot_at / 86400) * 86400 as day_timestamp,
                AVG(total_threads) as avg_total_threads,
                AVG(threads_with_images) as avg_threads_with_images,
                AVG(avg_reply_count) as avg_reply_count,
                COUNT(*) as sample_count,
                (SELECT top_coins FROM catalog_snapshots cs2
                 WHERE (cs2.snapshot_at / 86400) = (catalog_snapshots.snapshot_at / 86400)
                 ORDER BY cs2.snapshot_at DESC LIMIT 1) as top_coins
            FROM catalog_snapshots
            WHERE snapshot_at < ?1
            GROUP BY (snapshot_at / 86400)
            "#,
            params![daily_cutoff],
        )?;

        // Delete aggregated raw snapshots
        let deleted = self.conn.execute(
            "DELETE FROM catalog_snapshots WHERE snapshot_at < ?1",
            params![hourly_cutoff],
        )?;

        if deleted > 0 {
            info!("Aggregated and removed {} old catalog snapshots", deleted);
        }

        Ok(deleted as i64)
    }

    /// Delete threads older than metadata_days (except preserved ones)
    fn delete_old_threads(&self) -> Result<i64> {
        let cutoff = chrono::Utc::now().timestamp() - (self.config.metadata_days * 86400);

        let deleted = self.conn.execute(
            r#"
            DELETE FROM thread_ops
            WHERE created_at < ?1
              AND reply_count < ?2
              AND (?3 = 0 OR sentiment_score IS NULL)
              AND thread_id NOT IN (SELECT thread_id FROM training_labels)
            "#,
            params![
                cutoff,
                self.config.preserve_min_replies,
                self.config.preserve_analyzed as i32,
            ],
        )?;

        if deleted > 0 {
            info!("Deleted {} very old threads", deleted);
        }

        Ok(deleted as i64)
    }

    /// Clean up coin mentions for deleted threads
    fn clean_orphan_mentions(&self) -> Result<i64> {
        let deleted = self.conn.execute(
            r#"
            DELETE FROM coin_mentions
            WHERE thread_id NOT IN (SELECT thread_id FROM thread_ops)
            "#,
            params![],
        )?;

        if deleted > 0 {
            info!("Cleaned {} orphan coin mentions", deleted);
        }

        Ok(deleted as i64)
    }

    /// Drop legacy tables if they're empty
    fn clean_legacy_tables(&self) -> Result<bool> {
        let posts_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM posts",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        let threads_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM threads",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        if posts_count == 0 && threads_count == 0 {
            self.conn.execute_batch(
                r#"
                DROP TABLE IF EXISTS posts;
                DROP TABLE IF EXISTS threads;
                "#
            )?;
            info!("Dropped empty legacy tables (posts, threads)");
            return Ok(true);
        }

        Ok(false)
    }

    /// Run VACUUM to reclaim space
    fn vacuum(&self) -> Result<()> {
        info!("Running VACUUM to reclaim space...");
        self.conn.execute_batch("VACUUM;")?;
        Ok(())
    }

    /// Get current storage statistics
    pub fn get_stats(&self) -> Result<StorageStats> {
        let thread_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM thread_ops",
            [],
            |row| row.get(0),
        )?;

        let threads_with_html: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM thread_ops WHERE op_text IS NOT NULL",
            [],
            |row| row.get(0),
        )?;

        let mention_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM coin_mentions",
            [],
            |row| row.get(0),
        )?;

        let snapshot_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM catalog_snapshots",
            [],
            |row| row.get(0),
        )?;

        let label_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM training_labels",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        // Estimate text storage
        let text_bytes: i64 = self.conn.query_row(
            r#"
            SELECT COALESCE(SUM(
                COALESCE(LENGTH(op_text), 0) +
                COALESCE(LENGTH(op_text_clean), 0) +
                COALESCE(LENGTH(subject), 0)
            ), 0) FROM thread_ops
            "#,
            [],
            |row| row.get(0),
        )?;

        // Get page count (SQLite internal)
        let page_count: i64 = self.conn.query_row(
            "PRAGMA page_count",
            [],
            |row| row.get(0),
        )?;

        let page_size: i64 = self.conn.query_row(
            "PRAGMA page_size",
            [],
            |row| row.get(0),
        )?;

        Ok(StorageStats {
            thread_count,
            threads_with_html,
            mention_count,
            snapshot_count,
            label_count,
            text_bytes,
            db_size_bytes: page_count * page_size,
        })
    }
}

#[derive(Debug, Default)]
pub struct MaintenanceStats {
    pub html_stripped: i64,
    pub snapshots_aggregated: i64,
    pub threads_deleted: i64,
    pub mentions_cleaned: i64,
    pub legacy_cleaned: bool,
}

#[derive(Debug)]
pub struct StorageStats {
    pub thread_count: i64,
    pub threads_with_html: i64,
    pub mention_count: i64,
    pub snapshot_count: i64,
    pub label_count: i64,
    pub text_bytes: i64,
    pub db_size_bytes: i64,
}

impl StorageStats {
    pub fn print_report(&self) {
        println!("\n{}", "=".repeat(50));
        println!("STORAGE STATISTICS");
        println!("{}", "=".repeat(50));
        println!("\nRecords:");
        println!("  Threads:          {:>10}", self.thread_count);
        println!("  With raw HTML:    {:>10} ({:.1}%)",
            self.threads_with_html,
            self.threads_with_html as f64 / self.thread_count.max(1) as f64 * 100.0
        );
        println!("  Coin mentions:    {:>10}", self.mention_count);
        println!("  Catalog snapshots:{:>10}", self.snapshot_count);
        println!("  Training labels:  {:>10}", self.label_count);
        println!("\nStorage:");
        println!("  Text content:     {:>10}", format_bytes(self.text_bytes));
        println!("  Database size:    {:>10}", format_bytes(self.db_size_bytes));
        println!("\nEstimated annual growth at current rate:");
        let daily_threads = self.thread_count as f64 / 30.0; // Rough estimate
        let annual_mb = daily_threads * 365.0 * 2.5 / 1024.0 / 1024.0;
        println!("  ~{:.0} MB/year (threads only)", annual_mb);
    }
}

fn format_bytes(bytes: i64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / 1024.0 / 1024.0)
    } else {
        format!("{:.2} GB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_retention_config() {
        let config = RetentionConfig::default();
        assert_eq!(config.full_data_days, 30);
        assert_eq!(config.metadata_days, 365);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1500), "1.5 KB");
        assert_eq!(format_bytes(1500000), "1.4 MB");
    }
}

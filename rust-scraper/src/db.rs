//! Database operations for SQLite storage - OP-focused design

use anyhow::Result;
use rusqlite::{Connection, params};
use std::path::Path;

/// Database handle for SQLite operations
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Create new database connection
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Create parent directories if needed
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;

        // Enable WAL mode for better concurrency
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;

        Ok(Self { conn })
    }

    /// Get a reference to the underlying connection
    /// Used for maintenance operations
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Run database migrations
    pub fn run_migrations(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            -- Thread OPs table (primary focus - one row per thread)
            CREATE TABLE IF NOT EXISTS thread_ops (
                thread_id INTEGER PRIMARY KEY,

                -- Thread metadata
                subject TEXT,                    -- Thread title/subject
                board TEXT DEFAULT 'biz',

                -- OP content
                op_text TEXT,                    -- Full OP comment (HTML)
                op_text_clean TEXT,              -- Preprocessed text (stripped HTML)
                op_name TEXT,                    -- Poster name
                op_tripcode TEXT,                -- Tripcode if present

                -- Image data
                has_image BOOLEAN DEFAULT FALSE,
                image_url TEXT,                  -- Full image URL
                thumbnail_url TEXT,              -- Thumbnail URL
                local_thumbnail_path TEXT,       -- Local path to downloaded thumbnail
                image_md5 TEXT,                  -- For deduplication
                image_filename TEXT,             -- Original filename (may contain coin hints)
                image_width INTEGER,
                image_height INTEGER,
                image_size INTEGER,              -- File size in bytes

                -- Timestamps
                created_at INTEGER NOT NULL,     -- Thread creation time (Unix)
                last_modified INTEGER,           -- Last activity time
                first_seen_at INTEGER,           -- When we first scraped it
                last_seen_at INTEGER,            -- When we last saw it in catalog
                archived_at INTEGER,             -- When thread was archived (if from archive)

                -- Popularity metrics (for weighting)
                reply_count INTEGER DEFAULT 0,   -- Total replies
                image_count INTEGER DEFAULT 0,   -- Total images in thread
                unique_ips INTEGER,              -- Unique posters (if available)
                page_position INTEGER,           -- Catalog page (1-10, lower = more active)
                bump_limit_reached BOOLEAN DEFAULT FALSE,
                image_limit_reached BOOLEAN DEFAULT FALSE,
                is_sticky BOOLEAN DEFAULT FALSE,
                is_closed BOOLEAN DEFAULT FALSE,

                -- Analysis state
                sentiment_score REAL,            -- -1 to +1
                sentiment_confidence REAL,       -- 0 to 1
                sentiment_method TEXT,           -- 'vader', 'claude', etc.
                image_sentiment_score REAL,      -- From CLIP/YOLO
                image_analyzed BOOLEAN DEFAULT FALSE,
                analyzed_at INTEGER,             -- When sentiment was computed
                needs_reanalysis BOOLEAN DEFAULT FALSE,

                -- Source tracking
                source TEXT DEFAULT 'live',      -- 'live', 'warosu', 'archived.moe'
                source_url TEXT                  -- Original URL if from archive
            );

            CREATE INDEX IF NOT EXISTS idx_ops_created ON thread_ops(created_at);
            CREATE INDEX IF NOT EXISTS idx_ops_last_modified ON thread_ops(last_modified);
            CREATE INDEX IF NOT EXISTS idx_ops_reply_count ON thread_ops(reply_count);
            CREATE INDEX IF NOT EXISTS idx_ops_analyzed ON thread_ops(analyzed_at);
            CREATE INDEX IF NOT EXISTS idx_ops_source ON thread_ops(source);

            -- Coin mentions (linked to thread OPs)
            CREATE TABLE IF NOT EXISTS coin_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER REFERENCES thread_ops(thread_id),
                coin_symbol TEXT NOT NULL,
                coin_name TEXT,
                confidence REAL DEFAULT 1.0,
                mention_source TEXT DEFAULT 'text',  -- 'text', 'subject', 'filename'
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_mentions_symbol ON coin_mentions(coin_symbol);
            CREATE INDEX IF NOT EXISTS idx_mentions_thread ON coin_mentions(thread_id);

            -- Scraper state and statistics
            CREATE TABLE IF NOT EXISTS scraper_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            -- Catalog snapshots for tracking board activity over time
            CREATE TABLE IF NOT EXISTS catalog_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_at INTEGER NOT NULL,        -- Unix timestamp
                total_threads INTEGER,
                threads_with_images INTEGER,
                avg_reply_count REAL,
                top_coins TEXT,                      -- JSON array of most mentioned coins
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_time ON catalog_snapshots(snapshot_at);

            -- Human-labeled training data for sentiment model training/validation
            CREATE TABLE IF NOT EXISTS training_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER REFERENCES thread_ops(thread_id),

                -- Human rating (1-10 scale: 1=extremely bearish, 10=extremely bullish)
                sentiment_rating INTEGER CHECK(sentiment_rating >= 1 AND sentiment_rating <= 10),

                -- Skip flag for irrelevant posts
                skipped BOOLEAN DEFAULT FALSE,

                -- Optional notes from labeler
                notes TEXT,

                -- Labeling session metadata
                labeler_id TEXT DEFAULT 'default',  -- For multi-user labeling
                labeled_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                -- Snapshot of data at labeling time (for reproducibility)
                text_snapshot TEXT,                  -- OP text at time of labeling
                image_url_snapshot TEXT,             -- Image URL at time of labeling

                UNIQUE(thread_id, labeler_id)
            );

            CREATE INDEX IF NOT EXISTS idx_labels_thread ON training_labels(thread_id);
            CREATE INDEX IF NOT EXISTS idx_labels_rating ON training_labels(sentiment_rating);
            CREATE INDEX IF NOT EXISTS idx_labels_labeled ON training_labels(labeled_at);

            -- Migration: Add local_thumbnail_path if not exists (for existing databases)
            -- SQLite doesn't have IF NOT EXISTS for ALTER TABLE, so we ignore errors

            -- Legacy posts table (kept for backward compatibility, may be removed)
            CREATE TABLE IF NOT EXISTS posts (
                post_id INTEGER PRIMARY KEY,
                thread_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                name TEXT,
                text TEXT,
                has_image BOOLEAN DEFAULT FALSE,
                image_url TEXT,
                image_md5 TEXT,
                thumbnail_url TEXT,
                replies_to TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_posts_thread ON posts(thread_id);
            CREATE INDEX IF NOT EXISTS idx_posts_timestamp ON posts(timestamp);

            -- Legacy threads table (kept for backward compatibility)
            CREATE TABLE IF NOT EXISTS threads (
                thread_id INTEGER PRIMARY KEY,
                subject TEXT,
                op_post_id INTEGER,
                board TEXT DEFAULT 'biz',
                created_at INTEGER,
                last_updated INTEGER,
                reply_count INTEGER,
                image_count INTEGER
            );
            "#,
        )?;

        // Migration: Add local_thumbnail_path column if it doesn't exist
        // SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we try and ignore errors
        let _ = self.conn.execute(
            "ALTER TABLE thread_ops ADD COLUMN local_thumbnail_path TEXT",
            [],
        );

        Ok(())
    }

    /// Insert or update a thread OP from catalog
    pub fn upsert_thread_op(&self, op: &ThreadOp) -> Result<bool> {
        // Check if this is a new thread or update
        let existing: Option<i64> = self.conn.query_row(
            "SELECT reply_count FROM thread_ops WHERE thread_id = ?1",
            params![op.thread_id],
            |row| row.get(0),
        ).ok();

        let is_new = existing.is_none();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.conn.execute(
            r#"
            INSERT INTO thread_ops (
                thread_id, subject, board, op_text, op_text_clean, op_name, op_tripcode,
                has_image, image_url, thumbnail_url, local_thumbnail_path, image_md5, image_filename,
                image_width, image_height, image_size,
                created_at, last_modified, first_seen_at, last_seen_at,
                reply_count, image_count, unique_ips, page_position,
                bump_limit_reached, image_limit_reached, is_sticky, is_closed,
                source
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7,
                ?8, ?9, ?10, ?11, ?12, ?13,
                ?14, ?15, ?16,
                ?17, ?18, COALESCE((SELECT first_seen_at FROM thread_ops WHERE thread_id = ?1), ?19), ?20,
                ?21, ?22, ?23, ?24,
                ?25, ?26, ?27, ?28,
                ?29
            )
            ON CONFLICT(thread_id) DO UPDATE SET
                last_modified = excluded.last_modified,
                last_seen_at = excluded.last_seen_at,
                reply_count = excluded.reply_count,
                image_count = excluded.image_count,
                unique_ips = COALESCE(excluded.unique_ips, unique_ips),
                page_position = excluded.page_position,
                bump_limit_reached = excluded.bump_limit_reached,
                image_limit_reached = excluded.image_limit_reached,
                is_sticky = excluded.is_sticky,
                is_closed = excluded.is_closed,
                -- Update local thumbnail path if newly downloaded
                local_thumbnail_path = COALESCE(excluded.local_thumbnail_path, local_thumbnail_path),
                -- Mark for reanalysis if reply count changed significantly
                needs_reanalysis = CASE
                    WHEN excluded.reply_count > reply_count + 10 THEN 1
                    ELSE needs_reanalysis
                END
            "#,
            params![
                op.thread_id, op.subject, op.board, op.op_text, op.op_text_clean, op.op_name, op.op_tripcode,
                op.has_image, op.image_url, op.thumbnail_url, op.local_thumbnail_path, op.image_md5, op.image_filename,
                op.image_width, op.image_height, op.image_size,
                op.created_at, op.last_modified, now, now,
                op.reply_count, op.image_count, op.unique_ips, op.page_position,
                op.bump_limit_reached, op.image_limit_reached, op.is_sticky, op.is_closed,
                op.source,
            ],
        )?;

        Ok(is_new)
    }

    /// Insert coin mentions for a thread OP
    pub fn insert_op_mentions(&self, thread_id: i64, mentions: &[CoinMention]) -> Result<()> {
        // Clear existing mentions for this thread (in case of reanalysis)
        self.conn.execute(
            "DELETE FROM coin_mentions WHERE thread_id = ?1",
            params![thread_id],
        )?;

        let mut stmt = self.conn.prepare(
            "INSERT INTO coin_mentions (thread_id, coin_symbol, coin_name, confidence, mention_source)
             VALUES (?1, ?2, ?3, ?4, ?5)"
        )?;

        for mention in mentions {
            stmt.execute(params![
                thread_id,
                mention.symbol,
                mention.name,
                mention.confidence,
                mention.source,
            ])?;
        }

        Ok(())
    }

    /// Get threads that need sentiment analysis
    pub fn get_unanalyzed_threads(&self, limit: i32) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT thread_id FROM thread_ops
             WHERE analyzed_at IS NULL OR needs_reanalysis = 1
             ORDER BY reply_count DESC, created_at DESC
             LIMIT ?1"
        )?;

        let thread_ids: Vec<i64> = stmt
            .query_map(params![limit], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(thread_ids)
    }

    /// Update sentiment scores for a thread
    pub fn update_sentiment(&self, thread_id: i64, score: f64, confidence: f64, method: &str) -> Result<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.conn.execute(
            "UPDATE thread_ops SET
             sentiment_score = ?1, sentiment_confidence = ?2, sentiment_method = ?3,
             analyzed_at = ?4, needs_reanalysis = 0
             WHERE thread_id = ?5",
            params![score, confidence, method, now, thread_id],
        )?;

        Ok(())
    }

    /// Update the local thumbnail path for a thread
    pub fn update_thumbnail_path(&self, thread_id: i64, path: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE thread_ops SET local_thumbnail_path = ?1 WHERE thread_id = ?2",
            params![path, thread_id],
        )?;
        Ok(())
    }

    // === Import Coverage Analysis ===

    /// Get coverage statistics by source
    pub fn get_coverage_by_source(&self) -> Result<Vec<SourceCoverage>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT
                source,
                COUNT(*) as thread_count,
                MIN(created_at) as oldest_timestamp,
                MAX(created_at) as newest_timestamp,
                COUNT(CASE WHEN has_image THEN 1 END) as with_images,
                COUNT(CASE WHEN local_thumbnail_path IS NOT NULL THEN 1 END) as with_local_images
            FROM thread_ops
            GROUP BY source
            ORDER BY source
            "#
        )?;

        let results = stmt.query_map([], |row| {
            Ok(SourceCoverage {
                source: row.get(0)?,
                thread_count: row.get(1)?,
                oldest_timestamp: row.get(2)?,
                newest_timestamp: row.get(3)?,
                with_images: row.get(4)?,
                with_local_images: row.get(5)?,
            })
        })?;

        results.collect::<std::result::Result<Vec<_>, _>>().map_err(|e| e.into())
    }

    /// Get daily thread counts for a source (for gap detection)
    pub fn get_daily_coverage(&self, source: &str, limit: i32) -> Result<Vec<DailyCoverage>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT
                date(created_at, 'unixepoch') as date,
                COUNT(*) as thread_count
            FROM thread_ops
            WHERE source = ?1
            GROUP BY date
            ORDER BY date DESC
            LIMIT ?2
            "#
        )?;

        let results = stmt.query_map(params![source, limit], |row| {
            Ok(DailyCoverage {
                date: row.get(0)?,
                thread_count: row.get(1)?,
            })
        })?;

        results.collect::<std::result::Result<Vec<_>, _>>().map_err(|e| e.into())
    }

    /// Get the newest thread timestamp for a source (for continuation)
    pub fn get_newest_timestamp(&self, source: &str) -> Result<Option<i64>> {
        let result: Option<i64> = self.conn.query_row(
            "SELECT MAX(created_at) FROM thread_ops WHERE source = ?1",
            params![source],
            |row| row.get(0),
        ).ok().flatten();
        Ok(result)
    }

    /// Get the oldest thread timestamp for a source (for backfilling)
    pub fn get_oldest_timestamp(&self, source: &str) -> Result<Option<i64>> {
        let result: Option<i64> = self.conn.query_row(
            "SELECT MIN(created_at) FROM thread_ops WHERE source = ?1",
            params![source],
            |row| row.get(0),
        ).ok().flatten();
        Ok(result)
    }

    /// Check if a thread already exists
    pub fn thread_exists(&self, thread_id: i64) -> Result<bool> {
        let count: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM thread_ops WHERE thread_id = ?1",
            params![thread_id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Save a catalog snapshot for historical tracking
    pub fn save_catalog_snapshot(&self, total: i32, with_images: i32, avg_replies: f64, top_coins: &str) -> Result<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.conn.execute(
            "INSERT INTO catalog_snapshots (snapshot_at, total_threads, threads_with_images, avg_reply_count, top_coins)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![now, total, with_images, avg_replies, top_coins],
        )?;

        Ok(())
    }

    /// Get thread OP for analysis
    pub fn get_thread_op(&self, thread_id: i64) -> Result<Option<ThreadOp>> {
        let result = self.conn.query_row(
            "SELECT thread_id, subject, board, op_text, op_text_clean, op_name, op_tripcode,
                    has_image, image_url, thumbnail_url, local_thumbnail_path, image_md5, image_filename,
                    image_width, image_height, image_size,
                    created_at, last_modified, reply_count, image_count, unique_ips, page_position,
                    bump_limit_reached, image_limit_reached, is_sticky, is_closed, source
             FROM thread_ops WHERE thread_id = ?1",
            params![thread_id],
            |row| {
                Ok(ThreadOp {
                    thread_id: row.get(0)?,
                    subject: row.get(1)?,
                    board: row.get(2)?,
                    op_text: row.get(3)?,
                    op_text_clean: row.get(4)?,
                    op_name: row.get(5)?,
                    op_tripcode: row.get(6)?,
                    has_image: row.get(7)?,
                    image_url: row.get(8)?,
                    thumbnail_url: row.get(9)?,
                    local_thumbnail_path: row.get(10)?,
                    image_md5: row.get(11)?,
                    image_filename: row.get(12)?,
                    image_width: row.get(13)?,
                    image_height: row.get(14)?,
                    image_size: row.get(15)?,
                    created_at: row.get(16)?,
                    last_modified: row.get(17)?,
                    reply_count: row.get(18)?,
                    image_count: row.get(19)?,
                    unique_ips: row.get(20)?,
                    page_position: row.get(21)?,
                    bump_limit_reached: row.get(22)?,
                    image_limit_reached: row.get(23)?,
                    is_sticky: row.get(24)?,
                    is_closed: row.get(25)?,
                    source: row.get(26)?,
                })
            },
        );

        match result {
            Ok(op) => Ok(Some(op)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // === Legacy compatibility methods ===

    /// Insert or update a post (legacy - for backward compatibility)
    pub fn upsert_post(&self, post: &Post) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO posts
            (post_id, thread_id, timestamp, name, text, has_image, image_url, image_md5, thumbnail_url, replies_to)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            params![
                post.post_id,
                post.thread_id,
                post.timestamp,
                post.name,
                post.text,
                post.has_image,
                post.image_url,
                post.image_md5,
                post.thumbnail_url,
                post.replies_to,
            ],
        )?;
        Ok(())
    }

    /// Insert coin mentions for a post (legacy)
    pub fn insert_mentions(&self, post_id: i64, mentions: &[CoinMention]) -> Result<()> {
        let mut stmt = self.conn.prepare(
            "INSERT INTO coin_mentions (thread_id, coin_symbol, coin_name, confidence) VALUES (?1, ?2, ?3, ?4)"
        )?;

        for mention in mentions {
            stmt.execute(params![
                post_id,
                mention.symbol,
                mention.name,
                mention.confidence,
            ])?;
        }

        Ok(())
    }

    /// Get last updated timestamp for a thread (legacy)
    pub fn get_thread_last_updated(&self, thread_id: i64) -> Result<Option<i64>> {
        let result: Option<i64> = self.conn.query_row(
            "SELECT last_modified FROM thread_ops WHERE thread_id = ?1",
            params![thread_id],
            |row| row.get(0),
        ).ok();

        Ok(result)
    }

    /// Update thread metadata (legacy)
    pub fn upsert_thread(&self, thread: &Thread) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO threads
            (thread_id, subject, op_post_id, board, created_at, last_updated, reply_count, image_count)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#,
            params![
                thread.thread_id,
                thread.subject,
                thread.op_post_id,
                thread.board,
                thread.created_at,
                thread.last_updated,
                thread.reply_count,
                thread.image_count,
            ],
        )?;
        Ok(())
    }

    /// Get or set scraper state
    pub fn get_state(&self, key: &str) -> Result<Option<String>> {
        let result: Option<String> = self.conn.query_row(
            "SELECT value FROM scraper_state WHERE key = ?1",
            params![key],
            |row| row.get(0),
        ).ok();

        Ok(result)
    }

    pub fn set_state(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO scraper_state (key, value, updated_at) VALUES (?1, ?2, CURRENT_TIMESTAMP)",
            params![key, value],
        )?;
        Ok(())
    }
}

/// Thread OP data structure (primary focus)
#[derive(Debug, Clone)]
pub struct ThreadOp {
    pub thread_id: i64,
    pub subject: Option<String>,
    pub board: String,
    pub op_text: Option<String>,
    pub op_text_clean: Option<String>,
    pub op_name: Option<String>,
    pub op_tripcode: Option<String>,
    pub has_image: bool,
    pub image_url: Option<String>,
    pub thumbnail_url: Option<String>,
    pub local_thumbnail_path: Option<String>,
    pub image_md5: Option<String>,
    pub image_filename: Option<String>,
    pub image_width: Option<i32>,
    pub image_height: Option<i32>,
    pub image_size: Option<i32>,
    pub created_at: i64,
    pub last_modified: Option<i64>,
    pub reply_count: i32,
    pub image_count: i32,
    pub unique_ips: Option<i32>,
    pub page_position: Option<i32>,
    pub bump_limit_reached: bool,
    pub image_limit_reached: bool,
    pub is_sticky: bool,
    pub is_closed: bool,
    pub source: String,
}

impl Default for ThreadOp {
    fn default() -> Self {
        Self {
            thread_id: 0,
            subject: None,
            board: "biz".to_string(),
            op_text: None,
            op_text_clean: None,
            op_name: None,
            op_tripcode: None,
            has_image: false,
            image_url: None,
            thumbnail_url: None,
            local_thumbnail_path: None,
            image_md5: None,
            image_filename: None,
            image_width: None,
            image_height: None,
            image_size: None,
            created_at: 0,
            last_modified: None,
            reply_count: 0,
            image_count: 0,
            unique_ips: None,
            page_position: None,
            bump_limit_reached: false,
            image_limit_reached: false,
            is_sticky: false,
            is_closed: false,
            source: "live".to_string(),
        }
    }
}

// === Legacy data structures (kept for compatibility) ===

/// Post data structure (legacy)
#[derive(Debug, Clone)]
pub struct Post {
    pub post_id: i64,
    pub thread_id: i64,
    pub timestamp: i64,
    pub name: Option<String>,
    pub text: Option<String>,
    pub has_image: bool,
    pub image_url: Option<String>,
    pub image_md5: Option<String>,
    pub thumbnail_url: Option<String>,
    pub replies_to: Option<String>,
}

/// Thread data structure (legacy)
#[derive(Debug, Clone)]
pub struct Thread {
    pub thread_id: i64,
    pub subject: Option<String>,
    pub op_post_id: i64,
    pub board: String,
    pub created_at: i64,
    pub last_updated: i64,
    pub reply_count: i32,
    pub image_count: i32,
}

/// Coin mention data structure
#[derive(Debug, Clone)]
pub struct CoinMention {
    pub symbol: String,
    pub name: Option<String>,
    pub confidence: f64,
    pub source: String,  // 'text', 'subject', 'filename'
}

impl CoinMention {
    pub fn new(symbol: String, name: Option<String>, confidence: f64) -> Self {
        Self {
            symbol,
            name,
            confidence,
            source: "text".to_string(),
        }
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = source.to_string();
        self
    }
}

// === Coverage Analysis Structures ===

/// Coverage statistics for a data source
#[derive(Debug, Clone)]
pub struct SourceCoverage {
    pub source: String,
    pub thread_count: i64,
    pub oldest_timestamp: Option<i64>,
    pub newest_timestamp: Option<i64>,
    pub with_images: i64,
    pub with_local_images: i64,
}

/// Daily thread count for gap analysis
#[derive(Debug, Clone)]
pub struct DailyCoverage {
    pub date: String,
    pub thread_count: i64,
}

//! Database operations for SQLite storage

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

    /// Run database migrations
    pub fn run_migrations(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            -- Posts table
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

            -- Threads table
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

            -- Coin mentions
            CREATE TABLE IF NOT EXISTS coin_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER REFERENCES posts(post_id),
                coin_symbol TEXT NOT NULL,
                coin_name TEXT,
                confidence REAL DEFAULT 1.0
            );

            CREATE INDEX IF NOT EXISTS idx_mentions_symbol ON coin_mentions(coin_symbol);
            CREATE INDEX IF NOT EXISTS idx_mentions_post ON coin_mentions(post_id);

            -- Scraper state
            CREATE TABLE IF NOT EXISTS scraper_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            "#,
        )?;

        Ok(())
    }

    /// Insert or update a post
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

    /// Insert coin mentions for a post
    pub fn insert_mentions(&self, post_id: i64, mentions: &[CoinMention]) -> Result<()> {
        let mut stmt = self.conn.prepare(
            "INSERT INTO coin_mentions (post_id, coin_symbol, coin_name, confidence) VALUES (?1, ?2, ?3, ?4)"
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

    /// Get last updated timestamp for a thread
    pub fn get_thread_last_updated(&self, thread_id: i64) -> Result<Option<i64>> {
        let result: Option<i64> = self.conn.query_row(
            "SELECT last_updated FROM threads WHERE thread_id = ?1",
            params![thread_id],
            |row| row.get(0),
        ).ok();

        Ok(result)
    }

    /// Update thread metadata
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

/// Post data structure
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

/// Thread data structure
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
}

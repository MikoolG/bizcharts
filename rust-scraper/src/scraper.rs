//! 4chan Catalog-Only Scraper - OP-focused data collection
//!
//! This scraper efficiently collects thread OP data directly from the catalog,
//! avoiding the need to fetch individual threads. The catalog provides full OP
//! text, images, and popularity metrics in a single request.

use anyhow::Result;
use governor::{Quota, RateLimiter as GovRateLimiter};
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::db::{Database, ThreadOp};
use crate::extractor::extract_coins;

const API_BASE: &str = "https://a.4cdn.org";
const IMAGE_BASE: &str = "https://i.4cdn.org";

/// Main scraper struct - catalog-only approach
pub struct BizScraper {
    client: Client,
    rate_limiter: GovRateLimiter<governor::state::NotKeyed, governor::state::InMemoryState, governor::clock::DefaultClock>,
    board: String,
    poll_interval: Duration,
    download_thumbnails: bool,
}

impl BizScraper {
    /// Create new scraper instance
    pub fn new(config: &Config) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .gzip(true)
            .user_agent("BizCharts/1.0 (Sentiment Analysis Research)")
            .build()?;

        // Rate limiter: 1 request per second (conservative)
        let quota = Quota::per_second(NonZeroU32::new(config.scraper.rate_limit_per_second).unwrap());
        let rate_limiter = GovRateLimiter::direct(quota);

        Ok(Self {
            client,
            rate_limiter,
            board: config.scraper.board.clone(),
            poll_interval: Duration::from_secs(config.scraper.poll_interval_seconds),
            download_thumbnails: config.scraper.download_thumbnails,
        })
    }

    /// Run the scraper forever
    pub async fn run_forever(&self, db: &Database) -> Result<()> {
        info!("Starting catalog-only scraper for /{}/", self.board);
        info!("Poll interval: {}s", self.poll_interval.as_secs());

        loop {
            match self.scrape_catalog(db).await {
                Ok(stats) => {
                    info!(
                        "Catalog scraped: {} threads ({} new, {} updated), top coins: {:?}",
                        stats.total, stats.new_threads, stats.updated_threads, stats.top_coins
                    );
                }
                Err(e) => {
                    warn!("Scrape cycle error: {}", e);
                }
            }

            tokio::time::sleep(self.poll_interval).await;
        }
    }

    /// Scrape the catalog and extract all OP data
    pub async fn scrape_catalog(&self, db: &Database) -> Result<ScrapeStats> {
        // Rate limit before fetching
        self.rate_limiter.until_ready().await;

        let url = format!("{}/{}/catalog.json", API_BASE, self.board);
        debug!("Fetching catalog: {}", url);

        let response = self.client.get(&url).send().await?;

        if response.status() == reqwest::StatusCode::NOT_MODIFIED {
            debug!("Catalog unchanged (304)");
            return Ok(ScrapeStats::default());
        }

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Catalog fetch failed: {}", response.status()));
        }

        let catalog: Vec<CatalogPage> = response.json().await?;

        let mut stats = ScrapeStats::default();
        let mut coin_counts: HashMap<String, i32> = HashMap::new();

        for page in &catalog {
            for thread in &page.threads {
                let thread_op = self.catalog_thread_to_op(thread, page.page);

                // Extract coins from OP text, subject, and filename
                let mut mentions = Vec::new();

                if let Some(ref text) = thread_op.op_text {
                    for mention in extract_coins(text) {
                        *coin_counts.entry(mention.symbol.clone()).or_insert(0) += 1;
                        mentions.push(mention.with_source("text"));
                    }
                }

                if let Some(ref subject) = thread_op.subject {
                    for mention in extract_coins(subject) {
                        if !mentions.iter().any(|m| m.symbol == mention.symbol) {
                            *coin_counts.entry(mention.symbol.clone()).or_insert(0) += 1;
                            mentions.push(mention.with_source("subject"));
                        }
                    }
                }

                if let Some(ref filename) = thread_op.image_filename {
                    for mention in extract_coins(filename) {
                        if !mentions.iter().any(|m| m.symbol == mention.symbol) {
                            *coin_counts.entry(mention.symbol.clone()).or_insert(0) += 1;
                            mentions.push(mention.with_source("filename"));
                        }
                    }
                }

                // Store thread OP
                let is_new = db.upsert_thread_op(&thread_op)?;

                if is_new {
                    stats.new_threads += 1;
                } else {
                    stats.updated_threads += 1;
                }

                // Store coin mentions (only for new threads or if mentions exist)
                if !mentions.is_empty() {
                    db.insert_op_mentions(thread_op.thread_id, &mentions)?;
                }

                stats.total += 1;
            }
        }

        // Get top coins for snapshot
        let mut top_coins: Vec<_> = coin_counts.into_iter().collect();
        top_coins.sort_by(|a, b| b.1.cmp(&a.1));
        stats.top_coins = top_coins.into_iter().take(5).map(|(s, _)| s).collect();

        // Calculate and save catalog snapshot
        let threads_with_images = catalog.iter()
            .flat_map(|p| &p.threads)
            .filter(|t| t.tim.is_some())
            .count() as i32;

        let avg_replies = if stats.total > 0 {
            catalog.iter()
                .flat_map(|p| &p.threads)
                .map(|t| t.replies.unwrap_or(0) as f64)
                .sum::<f64>() / stats.total as f64
        } else {
            0.0
        };

        let top_coins_json = serde_json::to_string(&stats.top_coins).unwrap_or_default();
        db.save_catalog_snapshot(stats.total, threads_with_images, avg_replies, &top_coins_json)?;

        Ok(stats)
    }

    /// Convert catalog thread data to ThreadOp struct
    fn catalog_thread_to_op(&self, thread: &CatalogThread, page: i32) -> ThreadOp {
        let image_url = thread.tim.map(|tim| {
            format!(
                "{}/{}/{}{}",
                IMAGE_BASE,
                self.board,
                tim,
                thread.ext.as_deref().unwrap_or(".jpg")
            )
        });

        let thumbnail_url = thread.tim.map(|tim| {
            format!("{}/{}/{}s.jpg", IMAGE_BASE, self.board, tim)
        });

        ThreadOp {
            thread_id: thread.no,
            subject: thread.sub.clone(),
            board: self.board.clone(),
            op_text: thread.com.clone(),
            op_text_clean: thread.com.as_ref().map(|t| strip_html(t)),
            op_name: thread.name.clone(),
            op_tripcode: thread.trip.clone(),
            has_image: thread.tim.is_some(),
            image_url,
            thumbnail_url,
            image_md5: thread.md5.clone(),
            image_filename: thread.filename.clone(),
            image_width: thread.w,
            image_height: thread.h,
            image_size: thread.fsize,
            created_at: thread.time,
            last_modified: Some(thread.last_modified),
            reply_count: thread.replies.unwrap_or(0),
            image_count: thread.images.unwrap_or(0),
            unique_ips: thread.unique_ips,
            page_position: Some(page),
            bump_limit_reached: thread.bumplimit.unwrap_or(0) == 1,
            image_limit_reached: thread.imagelimit.unwrap_or(0) == 1,
            is_sticky: thread.sticky.unwrap_or(0) == 1,
            is_closed: thread.closed.unwrap_or(0) == 1,
            source: "live".to_string(),
        }
    }
}

/// Statistics from a scrape cycle
#[derive(Debug, Default)]
pub struct ScrapeStats {
    pub total: i32,
    pub new_threads: i32,
    pub updated_threads: i32,
    pub top_coins: Vec<String>,
}

/// Strip HTML tags from text (basic implementation)
fn strip_html(text: &str) -> String {
    // First, handle <br> tags as newlines
    let text = text
        .replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n");

    // Remove HTML tags (before decoding entities to avoid &gt; being treated as >)
    let mut result = String::with_capacity(text.len());
    let mut in_tag = false;

    for c in text.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => result.push(c),
            _ => {}
        }
    }

    // Then decode HTML entities
    let result = result
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#039;", "'");

    // Normalize whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

// === API Response Types ===

#[derive(Debug, Deserialize)]
struct CatalogPage {
    page: i32,
    threads: Vec<CatalogThread>,
}

#[derive(Debug, Deserialize)]
struct CatalogThread {
    // Required fields
    no: i64,
    time: i64,

    // Thread metadata
    #[serde(default)]
    last_modified: i64,
    #[serde(default)]
    replies: Option<i32>,
    #[serde(default)]
    images: Option<i32>,

    // OP content
    #[serde(default)]
    sub: Option<String>,        // Subject
    #[serde(default)]
    com: Option<String>,        // Comment (full OP text)
    #[serde(default)]
    name: Option<String>,       // Poster name
    #[serde(default)]
    trip: Option<String>,       // Tripcode

    // Image data
    #[serde(default)]
    tim: Option<i64>,           // Image timestamp (for URL)
    #[serde(default)]
    ext: Option<String>,        // Image extension
    #[serde(default)]
    filename: Option<String>,   // Original filename
    #[serde(default)]
    md5: Option<String>,        // Image MD5
    #[serde(default)]
    w: Option<i32>,             // Image width
    #[serde(default)]
    h: Option<i32>,             // Image height
    #[serde(default)]
    fsize: Option<i32>,         // File size

    // Thread status
    #[serde(default)]
    sticky: Option<i32>,
    #[serde(default)]
    closed: Option<i32>,
    #[serde(default)]
    bumplimit: Option<i32>,
    #[serde(default)]
    imagelimit: Option<i32>,
    #[serde(default)]
    unique_ips: Option<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html() {
        assert_eq!(
            strip_html("<span class=\"quote\">&gt;greentext</span>"),
            ">greentext"
        );
        assert_eq!(
            strip_html("line1<br>line2<br/>line3"),
            "line1 line2 line3"
        );
        assert_eq!(
            strip_html("normal text"),
            "normal text"
        );
    }
}

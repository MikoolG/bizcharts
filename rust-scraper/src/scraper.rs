//! 4chan API scraper implementation

use anyhow::Result;
use governor::{Quota, RateLimiter as GovRateLimiter};
use reqwest::Client;
use serde::Deserialize;
use std::num::NonZeroU32;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use crate::config::Config;
use crate::db::{Database, Post, Thread, CoinMention};
use crate::extractor::extract_coins;

const API_BASE: &str = "https://a.4cdn.org";
const IMAGE_BASE: &str = "https://i.4cdn.org";

/// Main scraper struct
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
            .build()?;

        // Rate limiter: 1 request per second
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
        loop {
            match self.scrape_cycle(db).await {
                Ok(count) => info!("Scrape cycle complete, processed {} posts", count),
                Err(e) => error!("Scrape cycle error: {}", e),
            }

            tokio::time::sleep(self.poll_interval).await;
        }
    }

    /// Single scrape cycle
    async fn scrape_cycle(&self, db: &Database) -> Result<usize> {
        let mut total_posts = 0;

        // Fetch catalog
        let catalog = self.fetch_catalog().await?;

        for page in catalog {
            for thread_info in page.threads {
                // Check if thread needs updating
                let last_updated = db.get_thread_last_updated(thread_info.no)?;
                let needs_update = last_updated
                    .map(|lu| thread_info.last_modified > lu)
                    .unwrap_or(true);

                if !needs_update {
                    debug!("Thread {} unchanged, skipping", thread_info.no);
                    continue;
                }

                // Rate limit before fetching
                self.rate_limiter.until_ready().await;

                // Fetch full thread
                match self.fetch_thread(thread_info.no).await {
                    Ok(thread) => {
                        let post_count = self.process_thread(db, &thread).await?;
                        total_posts += post_count;
                    }
                    Err(e) => {
                        warn!("Failed to fetch thread {}: {}", thread_info.no, e);
                    }
                }
            }
        }

        Ok(total_posts)
    }

    /// Fetch board catalog
    async fn fetch_catalog(&self) -> Result<Vec<CatalogPage>> {
        let url = format!("{}/{}/catalog.json", API_BASE, self.board);

        self.rate_limiter.until_ready().await;

        let response = self.client.get(&url).send().await?;

        if response.status() == reqwest::StatusCode::NOT_MODIFIED {
            return Ok(vec![]);
        }

        let catalog: Vec<CatalogPage> = response.json().await?;
        Ok(catalog)
    }

    /// Fetch a single thread
    async fn fetch_thread(&self, thread_id: i64) -> Result<ThreadResponse> {
        let url = format!("{}/{}/thread/{}.json", API_BASE, self.board, thread_id);

        let response = self.client.get(&url).send().await?;
        let thread: ThreadResponse = response.json().await?;

        Ok(thread)
    }

    /// Process a thread and store posts
    async fn process_thread(&self, db: &Database, thread: &ThreadResponse) -> Result<usize> {
        let mut count = 0;

        for api_post in &thread.posts {
            // Convert to our Post struct
            let post = Post {
                post_id: api_post.no,
                thread_id: thread.posts.first().map(|p| p.no).unwrap_or(0),
                timestamp: api_post.time,
                name: api_post.name.clone(),
                text: api_post.com.clone(),
                has_image: api_post.tim.is_some(),
                image_url: api_post.tim.map(|tim| {
                    format!("{}/{}/{}{}", IMAGE_BASE, self.board, tim, api_post.ext.as_deref().unwrap_or(".jpg"))
                }),
                image_md5: api_post.md5.clone(),
                thumbnail_url: api_post.tim.map(|tim| {
                    format!("{}/{}/{}s.jpg", IMAGE_BASE, self.board, tim)
                }),
                replies_to: None, // TODO: Extract from comment
            };

            // Store post
            db.upsert_post(&post)?;

            // Extract and store coin mentions
            if let Some(ref text) = post.text {
                let mentions = extract_coins(text);
                if !mentions.is_empty() {
                    db.insert_mentions(post.post_id, &mentions)?;
                }
            }

            count += 1;
        }

        // Update thread metadata
        if let Some(op) = thread.posts.first() {
            let thread_data = Thread {
                thread_id: op.no,
                subject: op.sub.clone(),
                op_post_id: op.no,
                board: self.board.clone(),
                created_at: op.time,
                last_updated: thread.posts.last().map(|p| p.time).unwrap_or(op.time),
                reply_count: thread.posts.len() as i32 - 1,
                image_count: thread.posts.iter().filter(|p| p.tim.is_some()).count() as i32,
            };
            db.upsert_thread(&thread_data)?;
        }

        Ok(count)
    }
}

// API response types

#[derive(Debug, Deserialize)]
struct CatalogPage {
    page: i32,
    threads: Vec<CatalogThread>,
}

#[derive(Debug, Deserialize)]
struct CatalogThread {
    no: i64,
    #[serde(default)]
    last_modified: i64,
    replies: Option<i32>,
    images: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct ThreadResponse {
    posts: Vec<ApiPost>,
}

#[derive(Debug, Deserialize)]
struct ApiPost {
    no: i64,
    time: i64,
    name: Option<String>,
    com: Option<String>,
    sub: Option<String>,
    tim: Option<i64>,
    ext: Option<String>,
    md5: Option<String>,
}

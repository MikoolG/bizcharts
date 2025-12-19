//! Warosu Archive Scraper - Historical /biz/ thread collection
//!
//! Warosu.org is a 4chan archive using Fuuka software. Since it has no JSON API,
//! this module parses HTML pages to extract historical thread data.
//!
//! Key features:
//! - Search by coin/subject with date ranges
//! - Browse archived threads by page
//! - Extract full OP data including images
//! - Respectful rate limiting (archive is a free service)

use anyhow::{anyhow, Result};
use chrono::NaiveDateTime;
use governor::{Quota, RateLimiter as GovRateLimiter};
use reqwest::Client;
#[allow(unused_imports)]
use scraper::{Html, Selector};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

use crate::db::{Database, ThreadOp};
use crate::extractor::extract_coins;

const WAROSU_BASE: &str = "https://warosu.org";
#[allow(dead_code)]
const WAROSU_IMAGE_BASE: &str = "https://i.warosu.org/data/biz";

/// Warosu archive scraper
pub struct WarosuScraper {
    client: Client,
    rate_limiter: GovRateLimiter<governor::state::NotKeyed, governor::state::InMemoryState, governor::clock::DefaultClock>,
    board: String,
    images_dir: PathBuf,
}

/// Parsed thread data from Warosu
#[derive(Debug, Clone)]
pub struct WarosuThread {
    pub thread_id: i64,
    pub subject: Option<String>,
    pub op_text: Option<String>,
    pub op_name: Option<String>,
    pub op_tripcode: Option<String>,
    pub timestamp: i64,
    pub image_url: Option<String>,
    pub thumbnail_url: Option<String>,
    pub image_filename: Option<String>,
    pub reply_count: Option<i32>,
}

/// Search parameters for Warosu queries
#[derive(Debug, Default, Clone)]
pub struct SearchParams {
    /// Search in subject line
    pub subject: Option<String>,
    /// Search in post text
    pub text: Option<String>,
    /// Start date (YYYY-MM-DD)
    pub date_start: Option<String>,
    /// End date (YYYY-MM-DD)
    pub date_end: Option<String>,
    /// Only fetch OPs (not replies)
    pub ops_only: bool,
    /// Result offset for pagination
    pub offset: u32,
}

impl WarosuScraper {
    /// Create new Warosu scraper instance
    pub fn new(board: &str, images_dir: &str) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .gzip(true)
            .user_agent("BizCharts/1.0 (Historical Research)")
            .build()?;

        // Rate limiter: 1 request per 2 seconds (be respectful to the archive)
        let quota = Quota::per_second(NonZeroU32::new(1).unwrap())
            .allow_burst(NonZeroU32::new(1).unwrap());
        let rate_limiter = GovRateLimiter::direct(quota);

        Ok(Self {
            client,
            rate_limiter,
            board: board.to_string(),
            images_dir: PathBuf::from(images_dir),
        })
    }

    /// Download thumbnail from Warosu and return local path
    async fn download_thumbnail(&self, thread_id: i64, thumbnail_url: &str) -> Result<PathBuf> {
        // Rate limit image downloads
        self.rate_limiter.until_ready().await;

        debug!("Downloading Warosu thumbnail for thread {}", thread_id);

        let response = self.client.get(thumbnail_url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!("Thumbnail download failed: {}", response.status()));
        }

        let bytes = response.bytes().await?;

        // Save to local file: data/images/{thread_id}.jpg
        let local_path = self.images_dir.join(format!("{}.jpg", thread_id));
        let mut file = fs::File::create(&local_path).await?;
        file.write_all(&bytes).await?;

        debug!("Saved thumbnail to {:?}", local_path);
        Ok(local_path)
    }

    /// Search for threads matching criteria
    pub async fn search(&self, params: &SearchParams) -> Result<Vec<WarosuThread>> {
        self.rate_limiter.until_ready().await;

        let mut url = format!("{}/{}/?task=search", WAROSU_BASE, self.board);

        if let Some(ref subject) = params.subject {
            url.push_str(&format!("&search_subject={}", urlencoding::encode(subject)));
        }
        if let Some(ref text) = params.text {
            url.push_str(&format!("&search_text={}", urlencoding::encode(text)));
        }
        if let Some(ref date_start) = params.date_start {
            url.push_str(&format!("&search_datefrom={}", date_start));
        }
        if let Some(ref date_end) = params.date_end {
            url.push_str(&format!("&search_dateto={}", date_end));
        }
        if params.ops_only {
            url.push_str("&search_op=op");
        }
        if params.offset > 0 {
            url.push_str(&format!("&offset={}", params.offset));
        }

        debug!("Warosu search: {}", url);

        let response = self.client.get(&url).send().await?;
        if !response.status().is_success() {
            return Err(anyhow!("Warosu search failed: {}", response.status()));
        }

        let html = response.text().await?;
        self.parse_search_results(&html)
    }

    /// Browse archived threads by page number
    pub async fn browse_page(&self, page: u32) -> Result<Vec<WarosuThread>> {
        self.rate_limiter.until_ready().await;

        let url = format!("{}/{}/?task=page&page={}", WAROSU_BASE, self.board, page);
        debug!("Warosu browse page: {}", url);

        let response = self.client.get(&url).send().await?;
        if !response.status().is_success() {
            return Err(anyhow!("Warosu page fetch failed: {}", response.status()));
        }

        let html = response.text().await?;
        self.parse_index_page(&html)
    }

    /// Fetch a specific thread's full details
    #[allow(dead_code)]
    pub async fn fetch_thread(&self, thread_id: i64) -> Result<Option<WarosuThread>> {
        self.rate_limiter.until_ready().await;

        let url = format!("{}/{}/thread/{}", WAROSU_BASE, self.board, thread_id);
        debug!("Warosu fetch thread: {}", url);

        let response = self.client.get(&url).send().await?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }

        if !response.status().is_success() {
            return Err(anyhow!("Warosu thread fetch failed: {}", response.status()));
        }

        let html = response.text().await?;
        self.parse_thread_page(&html, thread_id)
    }

    /// Import historical threads to database
    pub async fn import_historical(
        &self,
        db: &Database,
        params: &SearchParams,
        max_threads: usize,
    ) -> Result<ImportStats> {
        let mut stats = ImportStats::default();
        let mut offset = params.offset;

        // Ensure images directory exists
        fs::create_dir_all(&self.images_dir).await?;

        info!("Starting Warosu historical import (max: {} threads)", max_threads);

        loop {
            let mut search_params = params.clone();
            search_params.offset = offset;
            search_params.ops_only = true; // We only want OPs

            let threads = self.search(&search_params).await?;

            if threads.is_empty() {
                debug!("No more threads found at offset {}", offset);
                break;
            }

            for thread in &threads {
                if stats.total >= max_threads {
                    break;
                }

                // Convert to ThreadOp and store
                let thread_op = self.warosu_to_thread_op(thread);

                // Extract coin mentions
                let mut mentions = Vec::new();
                if let Some(ref text) = thread_op.op_text {
                    mentions.extend(extract_coins(text).into_iter().map(|m| m.with_source("text")));
                }
                if let Some(ref subject) = thread_op.subject {
                    for mention in extract_coins(subject) {
                        if !mentions.iter().any(|m| m.symbol == mention.symbol) {
                            mentions.push(mention.with_source("subject"));
                        }
                    }
                }

                let is_new = db.upsert_thread_op(&thread_op)?;

                if is_new {
                    stats.new_threads += 1;
                    if !mentions.is_empty() {
                        db.insert_op_mentions(thread_op.thread_id, &mentions)?;
                        stats.mentions_added += mentions.len();
                    }

                    // Download thumbnail for new threads with images
                    if thread_op.has_image {
                        if let Some(ref thumb_url) = thread_op.thumbnail_url {
                            match self.download_thumbnail(thread_op.thread_id, thumb_url).await {
                                Ok(local_path) => {
                                    let path_str = local_path.to_string_lossy().to_string();
                                    if let Err(e) = db.update_thumbnail_path(thread_op.thread_id, &path_str) {
                                        warn!("Failed to update thumbnail path: {}", e);
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to download thumbnail for {}: {}", thread_op.thread_id, e);
                                }
                            }
                        }
                    }
                } else {
                    stats.skipped += 1;
                }

                stats.total += 1;
            }

            if stats.total >= max_threads {
                break;
            }

            offset += 24; // Warosu pagination increment

            // Brief pause between pages
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        info!(
            "Warosu import complete: {} total, {} new, {} skipped, {} mentions",
            stats.total, stats.new_threads, stats.skipped, stats.mentions_added
        );

        Ok(stats)
    }

    /// Import threads by browsing pages (general activity, no search filter)
    pub async fn import_by_pages(
        &self,
        db: &Database,
        num_pages: u32,
    ) -> Result<ImportStats> {
        let mut stats = ImportStats::default();

        // Ensure images directory exists
        fs::create_dir_all(&self.images_dir).await?;

        info!("Starting Warosu page browse (pages: {})", num_pages);

        for page in 1..=num_pages {
            info!("Fetching page {}/{}", page, num_pages);

            let threads = match self.browse_page(page).await {
                Ok(t) => t,
                Err(e) => {
                    warn!("Failed to fetch page {}: {}", page, e);
                    continue;
                }
            };

            if threads.is_empty() {
                debug!("No threads found on page {}", page);
                continue;
            }

            for thread in &threads {
                // Convert to ThreadOp and store
                let thread_op = self.warosu_to_thread_op(thread);

                // Extract coin mentions
                let mut mentions = Vec::new();
                if let Some(ref text) = thread_op.op_text {
                    mentions.extend(extract_coins(text).into_iter().map(|m| m.with_source("text")));
                }
                if let Some(ref subject) = thread_op.subject {
                    for mention in extract_coins(subject) {
                        if !mentions.iter().any(|m| m.symbol == mention.symbol) {
                            mentions.push(mention.with_source("subject"));
                        }
                    }
                }

                let is_new = db.upsert_thread_op(&thread_op)?;

                if is_new {
                    stats.new_threads += 1;
                    if !mentions.is_empty() {
                        db.insert_op_mentions(thread_op.thread_id, &mentions)?;
                        stats.mentions_added += mentions.len();
                    }

                    // Download thumbnail for new threads with images
                    if thread_op.has_image {
                        if let Some(ref thumb_url) = thread_op.thumbnail_url {
                            match self.download_thumbnail(thread_op.thread_id, thumb_url).await {
                                Ok(local_path) => {
                                    let path_str = local_path.to_string_lossy().to_string();
                                    if let Err(e) = db.update_thumbnail_path(thread_op.thread_id, &path_str) {
                                        warn!("Failed to update thumbnail path: {}", e);
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to download thumbnail for {}: {}", thread_op.thread_id, e);
                                }
                            }
                        }
                    }
                } else {
                    stats.skipped += 1;
                }

                stats.total += 1;
            }

            // Brief pause between pages
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        info!(
            "Warosu page browse complete: {} total, {} new, {} skipped, {} mentions",
            stats.total, stats.new_threads, stats.skipped, stats.mentions_added
        );

        Ok(stats)
    }

    /// Parse search results HTML
    fn parse_search_results(&self, html: &str) -> Result<Vec<WarosuThread>> {
        let mut threads = Vec::new();

        // Search results use table-based layout with each post in a <table> block
        // Structure: <table><tr><td class=doubledash>>><td class="comment reply">...</table>
        let table_regex = regex::Regex::new(
            r#"<table><tr><td class=doubledash>>><td class="comment reply">([\s\S]*?)</table>"#
        ).unwrap();

        let ts_regex = regex::Regex::new(r#"title=(\d{13})"#).unwrap();
        let thread_id_regex = regex::Regex::new(r#"No\.(\d+)</a>\s*\[<a href=/biz/thread/\d+>View</a>\]"#).unwrap();
        let subject_regex = regex::Regex::new(r#"<span class=filetitle>([^<]+)</span>"#).unwrap();
        let image_regex = regex::Regex::new(r#"<a href=(https://i\.warosu\.org/data/biz/img/[^\s>]+)>"#).unwrap();
        let thumb_regex = regex::Regex::new(r#"src=(https://i\.warosu\.org/data/biz/thumb/[^\s>]+)"#).unwrap();
        let text_regex = regex::Regex::new(r#"<blockquote>([\s\S]*?)</blockquote>"#).unwrap();

        for table_cap in table_regex.captures_iter(html) {
            let section = &table_cap[1];

            // Extract thread ID - must have [View] link to be an OP
            let thread_id = match thread_id_regex.captures(section) {
                Some(cap) => cap[1].parse::<i64>().unwrap_or(0),
                None => continue,
            };

            if thread_id == 0 {
                continue;
            }

            let mut thread = WarosuThread {
                thread_id,
                subject: None,
                op_text: None,
                op_name: Some("Anonymous".to_string()),
                op_tripcode: None,
                timestamp: 0,
                image_url: None,
                thumbnail_url: None,
                image_filename: None,
                reply_count: None,
            };

            // Extract timestamp
            if let Some(ts_cap) = ts_regex.captures(section) {
                thread.timestamp = ts_cap[1].parse::<i64>().unwrap_or(0) / 1000;
            }

            // Extract subject
            if let Some(subj_cap) = subject_regex.captures(section) {
                thread.subject = Some(html_escape::decode_html_entities(&subj_cap[1]).to_string());
            }

            // Extract image URL
            if let Some(img_cap) = image_regex.captures(section) {
                thread.image_url = Some(img_cap[1].to_string());
            }

            // Extract thumbnail URL
            if let Some(thumb_cap) = thumb_regex.captures(section) {
                thread.thumbnail_url = Some(thumb_cap[1].to_string());
            }

            // Extract text from blockquote
            if let Some(text_cap) = text_regex.captures(section) {
                thread.op_text = Some(strip_html_warosu(&text_cap[1]));
            }

            threads.push(thread);
        }

        // Deduplicate by thread_id
        threads.sort_by_key(|t| t.thread_id);
        threads.dedup_by_key(|t| t.thread_id);

        debug!("Parsed {} threads from search results", threads.len());
        Ok(threads)
    }

    /// Parse index/browse page HTML
    fn parse_index_page(&self, html: &str) -> Result<Vec<WarosuThread>> {
        let mut threads = Vec::new();

        // Thread OPs are in <div class=comment id=p{thread_id}> containers
        // followed by <br class=newthr> separators
        // Key structure for OPs:
        //   <div class=comment id=p61508089>
        //     ...
        //     <span class=posttime title=1766048430000>...</span>
        //     ...
        //     No.61508089</a> [<a href=/biz/thread/61508089>Reply</a>]
        //     <blockquote>...</blockquote>
        //   </div>
        //
        // Replies are in <td class="comment reply" id=p...> (note "reply" class)

        // Split by thread OP containers (div with just "comment" class, not "comment reply")
        let op_container_regex = regex::Regex::new(
            r#"<div class=comment id=p(\d+)>([\s\S]*?)(?:<br class=newthr>|<table>)"#
        ).unwrap();

        // Note: Warosu uses unquoted class attributes like class=filetitle
        let subject_regex = regex::Regex::new(
            r#"<span[^>]*class=(?:"[^"]*)?(?:subject|filetitle)[^>]*>([^<]+)</span>"#
        ).unwrap();

        // Note: Warosu HTML doesn't quote attribute values, so we match without quotes
        let image_regex = regex::Regex::new(
            r#"href=(https?://i\.warosu\.org/data/biz/img/[^\s>]+)"#
        ).unwrap();

        let thumb_regex = regex::Regex::new(
            r#"src=(https?://i\.warosu\.org/data/biz/thumb/[^\s>]+)"#
        ).unwrap();

        let replies_regex = regex::Regex::new(
            r#"(\d+)\s+repl(?:y|ies)\s+omitted"#
        ).unwrap();

        let ts_regex = regex::Regex::new(r"title=(\d{13})").unwrap();
        let text_regex = regex::Regex::new(r"<blockquote[^>]*>([\s\S]*?)</blockquote>").unwrap();

        for cap in op_container_regex.captures_iter(html) {
            if let Ok(thread_id) = cap[1].parse::<i64>() {
                let section = &cap[2];

                // Verify this is really an OP by checking for [Reply] link
                if !section.contains(&format!("href=/biz/thread/{}>Reply", thread_id)) {
                    continue;
                }

                let mut thread = WarosuThread {
                    thread_id,
                    subject: None,
                    op_text: None,
                    op_name: Some("Anonymous".to_string()),
                    op_tripcode: None,
                    timestamp: 0,
                    image_url: None,
                    thumbnail_url: None,
                    image_filename: None,
                    reply_count: None,
                };

                // Extract timestamp from title attribute (Unix milliseconds)
                if let Some(ts_cap) = ts_regex.captures(section) {
                    thread.timestamp = ts_cap[1].parse::<i64>().unwrap_or(0) / 1000;
                }

                // Extract subject
                if let Some(subj_cap) = subject_regex.captures(section) {
                    thread.subject = Some(html_escape::decode_html_entities(&subj_cap[1]).to_string());
                }

                // Extract image URLs
                if let Some(img_cap) = image_regex.captures(section) {
                    thread.image_url = Some(img_cap[1].to_string());
                }
                if let Some(thumb_cap) = thumb_regex.captures(section) {
                    thread.thumbnail_url = Some(thumb_cap[1].to_string());
                }

                // Extract reply count from "X replies omitted"
                if let Some(reply_cap) = replies_regex.captures(section) {
                    thread.reply_count = reply_cap[1].parse().ok();
                }

                // Extract OP text from blockquote
                if let Some(text_cap) = text_regex.captures(section) {
                    thread.op_text = Some(strip_html_warosu(&text_cap[1]));
                }

                threads.push(thread);
            }
        }

        debug!("Parsed {} threads from index page", threads.len());
        Ok(threads)
    }

    /// Parse a single thread page
    #[allow(dead_code)]
    fn parse_thread_page(&self, html: &str, thread_id: i64) -> Result<Option<WarosuThread>> {
        let mut thread = WarosuThread {
            thread_id,
            subject: None,
            op_text: None,
            op_name: Some("Anonymous".to_string()),
            op_tripcode: None,
            timestamp: 0,
            image_url: None,
            thumbnail_url: None,
            image_filename: None,
            reply_count: None,
        };

        // Extract timestamp from title attribute (Unix milliseconds)
        let ts_regex = regex::Regex::new(r"title=(\d{13})").unwrap();
        if let Some(cap) = ts_regex.captures(html) {
            thread.timestamp = cap[1].parse::<i64>().unwrap_or(0) / 1000;
        }

        // Extract subject (Warosu uses unquoted class attributes)
        let subject_regex = regex::Regex::new(r#"<span[^>]*class=(?:"[^"]*)?filetitle[^>]*>([^<]+)</span>"#).unwrap();
        if let Some(cap) = subject_regex.captures(html) {
            thread.subject = Some(html_escape::decode_html_entities(&cap[1]).to_string());
        }

        // Extract poster name (Warosu uses unquoted class attributes)
        let name_regex = regex::Regex::new(r#"<span[^>]*class=(?:"[^"]*)?postername[^>]*>([^<]+)</span>"#).unwrap();
        if let Some(cap) = name_regex.captures(html) {
            let name = html_escape::decode_html_entities(&cap[1]).to_string();
            if name != "Anonymous" {
                thread.op_name = Some(name);
            }
        }

        // Extract tripcode
        let trip_regex = regex::Regex::new(r#"<span[^>]*class="[^"]*postertrip[^"]*"[^>]*>([^<]+)</span>"#).unwrap();
        if let Some(cap) = trip_regex.captures(html) {
            thread.op_tripcode = Some(cap[1].to_string());
        }

        // Extract full image URL
        let img_regex = regex::Regex::new(r#"href="(https?://i\.warosu\.org/data/biz/img/[^"]+)""#).unwrap();
        if let Some(cap) = img_regex.captures(html) {
            thread.image_url = Some(cap[1].to_string());
        }

        // Extract thumbnail URL
        let thumb_regex = regex::Regex::new(r#"src="(https?://i\.warosu\.org/data/biz/thumb/[^"]+)""#).unwrap();
        if let Some(cap) = thumb_regex.captures(html) {
            thread.thumbnail_url = Some(cap[1].to_string());
        }

        // Extract filename
        let filename_regex = regex::Regex::new(r#"title="([^"]+\.\w{3,4})""#).unwrap();
        if let Some(cap) = filename_regex.captures(html) {
            thread.image_filename = Some(cap[1].to_string());
        }

        // Extract OP text - look for first blockquote (OP post)
        let text_regex = regex::Regex::new(r"(?:<blockquote[^>]*>)([\s\S]*?)(?:</blockquote>)").unwrap();
        if let Some(cap) = text_regex.captures(html) {
            thread.op_text = Some(strip_html_warosu(&cap[1]));
        }

        // Count replies (number of post divs/articles after OP)
        let reply_count = html.matches("No.").count().saturating_sub(1) as i32;
        thread.reply_count = Some(reply_count);

        Ok(Some(thread))
    }

    /// Parse Warosu timestamp format: "Fri, Jan 18, 2026 03:54:27"
    fn parse_warosu_timestamp(&self, ts: &str) -> Option<i64> {
        // Warosu format: "Day, Mon DD, YYYY HH:MM:SS"
        // Example: "Fri, Jan 18, 2026 03:54:27"

        let ts = ts.trim();

        // Split and parse manually since chrono has issues with this format
        let parts: Vec<&str> = ts.split(|c| c == ',' || c == ' ').filter(|s| !s.is_empty()).collect();

        if parts.len() < 5 {
            warn!("Failed to parse Warosu timestamp (wrong parts): {}", ts);
            return None;
        }

        // parts: ["Fri", "Jan", "18", "2026", "03:54:27"]
        let month_str = parts[1];
        let day: u32 = parts[2].parse().ok()?;
        let year: i32 = parts[3].parse().ok()?;
        let time_parts: Vec<&str> = parts[4].split(':').collect();

        if time_parts.len() != 3 {
            warn!("Failed to parse Warosu timestamp (bad time): {}", ts);
            return None;
        }

        let hour: u32 = time_parts[0].parse().ok()?;
        let minute: u32 = time_parts[1].parse().ok()?;
        let second: u32 = time_parts[2].parse().ok()?;

        let month = match month_str.to_lowercase().as_str() {
            "jan" => 1, "feb" => 2, "mar" => 3, "apr" => 4,
            "may" => 5, "jun" => 6, "jul" => 7, "aug" => 8,
            "sep" => 9, "oct" => 10, "nov" => 11, "dec" => 12,
            _ => {
                warn!("Failed to parse Warosu timestamp (bad month): {}", ts);
                return None;
            }
        };

        let date = chrono::NaiveDate::from_ymd_opt(year, month, day)?;
        let time = chrono::NaiveTime::from_hms_opt(hour, minute, second)?;
        let dt = NaiveDateTime::new(date, time);

        Some(dt.and_utc().timestamp())
    }

    /// Convert WarosuThread to ThreadOp for database storage
    fn warosu_to_thread_op(&self, thread: &WarosuThread) -> ThreadOp {
        ThreadOp {
            thread_id: thread.thread_id,
            subject: thread.subject.clone(),
            board: self.board.clone(),
            op_text: thread.op_text.clone(),
            op_text_clean: thread.op_text.as_ref().map(|t| strip_html_warosu(t)),
            op_name: thread.op_name.clone(),
            op_tripcode: thread.op_tripcode.clone(),
            has_image: thread.image_url.is_some(),
            image_url: thread.image_url.clone(),
            thumbnail_url: thread.thumbnail_url.clone(),
            local_thumbnail_path: None, // Set after download
            image_md5: None, // Warosu doesn't expose this
            image_filename: thread.image_filename.clone(),
            image_width: None,
            image_height: None,
            image_size: None,
            created_at: thread.timestamp,
            last_modified: None,
            reply_count: thread.reply_count.unwrap_or(0),
            image_count: if thread.image_url.is_some() { 1 } else { 0 },
            unique_ips: None, // Warosu doesn't expose this
            page_position: None,
            bump_limit_reached: false,
            image_limit_reached: false,
            is_sticky: false,
            is_closed: true, // Archived threads are closed
            source: "warosu".to_string(),
        }
    }
}

/// Statistics from historical import
#[derive(Debug, Default)]
pub struct ImportStats {
    pub total: usize,
    pub new_threads: usize,
    pub skipped: usize,
    pub mentions_added: usize,
}

/// Strip HTML tags from Warosu content
fn strip_html_warosu(text: &str) -> String {
    // Handle <br> as newlines
    let text = text
        .replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n")
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#039;", "'")
        .replace("&nbsp;", " ");

    // Remove remaining HTML tags
    let tag_regex = regex::Regex::new(r"<[^>]+>").unwrap();
    let cleaned = tag_regex.replace_all(&text, "");

    // Normalize whitespace but preserve intentional line breaks
    cleaned
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

// Re-export for use by urlencoding
mod urlencoding {
    pub fn encode(s: &str) -> String {
        url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_warosu() {
        assert_eq!(
            strip_html_warosu("<span class=\"quote\">&gt;greentext</span>"),
            ">greentext"
        );
        assert_eq!(
            strip_html_warosu("line1<br>line2"),
            "line1\nline2"
        );
        assert_eq!(
            strip_html_warosu("<a href=\"test\">link text</a>"),
            "link text"
        );
    }

    #[test]
    fn test_timestamp_parsing() {
        let scraper = WarosuScraper::new("biz").unwrap();

        // Test valid timestamp
        let ts = scraper.parse_warosu_timestamp("Fri, Jan 18, 2026 03:54:27");
        assert!(ts.is_some());

        // The timestamp should be reasonable (after 2020)
        assert!(ts.unwrap() > 1577836800); // 2020-01-01
    }
}

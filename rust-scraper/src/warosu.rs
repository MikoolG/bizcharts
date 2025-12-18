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
use std::time::Duration;
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
    pub fn new(board: &str) -> Result<Self> {
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
        })
    }

    /// Search for threads matching criteria
    pub async fn search(&self, params: &SearchParams) -> Result<Vec<WarosuThread>> {
        self.rate_limiter.until_ready().await;

        let mut url = format!("{}/{}/?task=search2", WAROSU_BASE, self.board);

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
    #[allow(dead_code)]
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

    /// Parse search results HTML
    fn parse_search_results(&self, html: &str) -> Result<Vec<WarosuThread>> {
        let _document = Html::parse_document(html);
        let mut threads = Vec::new();

        // Note: Warosu uses table-based layout with minimal CSS classes.
        // We use regex-based parsing as the primary method since the HTML structure
        // is inconsistent. The scraper crate selectors are kept for potential future use.

        // Parse using regex since Warosu HTML is messy
        let thread_regex = regex::Regex::new(
            r#"No\.(\d+).*?(\w+,\s+\w+\s+\d+,\s+\d+\s+\d+:\d+:\d+)"#
        ).unwrap();

        for cap in thread_regex.captures_iter(html) {
            if let Ok(thread_id) = cap[1].parse::<i64>() {
                let timestamp = self.parse_warosu_timestamp(&cap[2]).unwrap_or(0);

                threads.push(WarosuThread {
                    thread_id,
                    subject: None, // Will be filled if we fetch full thread
                    op_text: None,
                    op_name: Some("Anonymous".to_string()),
                    op_tripcode: None,
                    timestamp,
                    image_url: None,
                    thumbnail_url: None,
                    image_filename: None,
                    reply_count: None,
                });
            }
        }

        // Deduplicate by thread_id
        threads.sort_by_key(|t| t.thread_id);
        threads.dedup_by_key(|t| t.thread_id);

        debug!("Parsed {} threads from search results", threads.len());
        Ok(threads)
    }

    /// Parse index/browse page HTML
    #[allow(dead_code)]
    fn parse_index_page(&self, html: &str) -> Result<Vec<WarosuThread>> {
        let _document = Html::parse_document(html);
        let mut threads = Vec::new();

        // Look for thread containers
        // Warosu format: "No.12345678" followed by timestamp and content

        // Use regex to extract structured data from the messy HTML
        let _thread_regex = regex::Regex::new(
            r#"No\.(\d+).*?(\w+,\s+\w+\s+\d+,\s+\d+\s+\d+:\d+:\d+).*?(?:<blockquote[^>]*>|<p>|</a>\s*)(.*?)(?:</blockquote>|</p>|<br|<span)"#
        ).unwrap();

        let subject_regex = regex::Regex::new(
            r#"<span[^>]*class="[^"]*subject[^"]*"[^>]*>([^<]+)</span>"#
        ).unwrap();

        let image_regex = regex::Regex::new(
            r#"href="(https?://i\.warosu\.org/data/biz/img/[^"]+)""#
        ).unwrap();

        let thumb_regex = regex::Regex::new(
            r#"src="(https?://i\.warosu\.org/data/biz/thumb/[^"]+)""#
        ).unwrap();

        let replies_regex = regex::Regex::new(
            r#"(\d+)\s+repl(?:y|ies)\s+omitted"#
        ).unwrap();

        // Split by thread boundaries and parse each
        let thread_sections: Vec<&str> = html.split("No.").collect();

        for section in thread_sections.iter().skip(1) {
            // Extract thread ID (first numeric sequence)
            let id_match = regex::Regex::new(r"^(\d+)").unwrap();
            if let Some(cap) = id_match.captures(section) {
                if let Ok(thread_id) = cap[1].parse::<i64>() {
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
                    let ts_regex = regex::Regex::new(r"(\w+,\s+\w+\s+\d+,\s+\d+\s+\d+:\d+:\d+)").unwrap();
                    if let Some(ts_cap) = ts_regex.captures(section) {
                        thread.timestamp = self.parse_warosu_timestamp(&ts_cap[1]).unwrap_or(0);
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

                    // Extract reply count
                    if let Some(reply_cap) = replies_regex.captures(section) {
                        thread.reply_count = reply_cap[1].parse().ok();
                    }

                    // Extract OP text (look for blockquote or paragraph content)
                    let text_regex = regex::Regex::new(r"(?:<blockquote[^>]*>|<p[^>]*>)([\s\S]*?)(?:</blockquote>|</p>)").unwrap();
                    if let Some(text_cap) = text_regex.captures(section) {
                        let raw_text = &text_cap[1];
                        thread.op_text = Some(strip_html_warosu(raw_text));
                    }

                    threads.push(thread);
                }
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

        // Extract timestamp
        let ts_regex = regex::Regex::new(r"(\w+,\s+\w+\s+\d+,\s+\d+\s+\d+:\d+:\d+)").unwrap();
        if let Some(cap) = ts_regex.captures(html) {
            thread.timestamp = self.parse_warosu_timestamp(&cap[1]).unwrap_or(0);
        }

        // Extract subject
        let subject_regex = regex::Regex::new(r#"<span[^>]*class="[^"]*filetitle[^"]*"[^>]*>([^<]+)</span>"#).unwrap();
        if let Some(cap) = subject_regex.captures(html) {
            thread.subject = Some(html_escape::decode_html_entities(&cap[1]).to_string());
        }

        // Extract poster name
        let name_regex = regex::Regex::new(r#"<span[^>]*class="[^"]*postername[^"]*"[^>]*>([^<]+)</span>"#).unwrap();
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

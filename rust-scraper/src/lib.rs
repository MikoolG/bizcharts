//! BizCharts Scraper Library
//!
//! Core functionality for scraping 4chan /biz/ board.

pub mod config;
pub mod db;
pub mod extractor;
pub mod scraper;

pub use config::Config;
pub use db::Database;
pub use scraper::BizScraper;

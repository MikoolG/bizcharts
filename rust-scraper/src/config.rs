//! Configuration loading and management

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub scraper: ScraperConfig,
    pub database: DatabaseConfig,
    pub sentiment: SentimentConfig,
}

#[derive(Debug, Deserialize)]
pub struct ScraperConfig {
    pub poll_interval_seconds: u64,
    pub rate_limit_per_second: u32,
    pub thread_update_interval: u64,
    pub board: String,
    pub download_thumbnails: bool,
    pub download_full_images: bool,
}

#[derive(Debug, Deserialize)]
pub struct DatabaseConfig {
    pub sqlite_path: String,
    pub duckdb_path: String,
    pub images_dir: String,
}

#[derive(Debug, Deserialize)]
pub struct SentimentConfig {
    pub confidence_threshold: f64,
    pub vader_confidence_threshold: f64,
    pub use_claude_for_ambiguous: bool,
    pub bullish_threshold: f64,
    pub bearish_threshold: f64,
}

impl Config {
    /// Load configuration from TOML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        // Test will be implemented when config file is finalized
    }
}

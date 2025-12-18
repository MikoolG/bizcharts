//! BizCharts Scraper
//!
//! Scrapes the 4chan /biz/ board for sentiment analysis.
//! Stores posts in SQLite for downstream processing by Python ML services.

use anyhow::Result;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

mod config;
mod db;
mod extractor;
mod scraper;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting BizCharts scraper...");

    // Load configuration
    let config = config::Config::load("../config/settings.toml")?;
    info!("Loaded configuration");

    // Initialize database
    let db = db::Database::new(&config.database.sqlite_path)?;
    db.run_migrations()?;
    info!("Database initialized");

    // Initialize scraper
    let scraper = scraper::BizScraper::new(&config)?;

    // Run main scraping loop
    info!("Starting scrape loop with {}s interval", config.scraper.poll_interval_seconds);
    scraper.run_forever(&db).await?;

    Ok(())
}

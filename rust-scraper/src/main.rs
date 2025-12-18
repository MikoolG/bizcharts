//! BizCharts Scraper
//!
//! Scrapes the 4chan /biz/ board for sentiment analysis.
//! Supports both live catalog scraping and historical archive imports.
//!
//! Usage:
//!   biz-scraper              # Run live catalog scraper
//!   biz-scraper --warosu     # Import historical data from Warosu
//!   biz-scraper --warosu --search "bitcoin" --max 1000

use anyhow::Result;
use std::env;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

mod config;
mod db;
mod extractor;
mod scraper;
mod warosu;

/// Command-line arguments
struct Args {
    /// Run Warosu historical import instead of live scraper
    warosu: bool,
    /// Search term for Warosu (subject search)
    search: Option<String>,
    /// Date range start (YYYY-MM-DD)
    date_from: Option<String>,
    /// Date range end (YYYY-MM-DD)
    date_to: Option<String>,
    /// Maximum threads to import
    max_threads: usize,
    /// Show help
    help: bool,
}

impl Args {
    fn parse() -> Self {
        let args: Vec<String> = env::args().collect();
        let mut result = Args {
            warosu: false,
            search: None,
            date_from: None,
            date_to: None,
            max_threads: 5000,
            help: false,
        };

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--warosu" | "-w" => result.warosu = true,
                "--search" | "-s" => {
                    i += 1;
                    if i < args.len() {
                        result.search = Some(args[i].clone());
                    }
                }
                "--from" => {
                    i += 1;
                    if i < args.len() {
                        result.date_from = Some(args[i].clone());
                    }
                }
                "--to" => {
                    i += 1;
                    if i < args.len() {
                        result.date_to = Some(args[i].clone());
                    }
                }
                "--max" | "-m" => {
                    i += 1;
                    if i < args.len() {
                        result.max_threads = args[i].parse().unwrap_or(5000);
                    }
                }
                "--help" | "-h" => result.help = true,
                _ => {}
            }
            i += 1;
        }

        result
    }

    fn print_help() {
        println!("BizCharts Scraper - 4chan /biz/ sentiment data collection\n");
        println!("USAGE:");
        println!("  biz-scraper [OPTIONS]\n");
        println!("OPTIONS:");
        println!("  --warosu, -w      Run Warosu historical import instead of live scraper");
        println!("  --search, -s      Search term for Warosu (searches subject line)");
        println!("  --from DATE       Start date for Warosu search (YYYY-MM-DD)");
        println!("  --to DATE         End date for Warosu search (YYYY-MM-DD)");
        println!("  --max, -m NUM     Maximum threads to import (default: 5000)");
        println!("  --help, -h        Show this help message\n");
        println!("EXAMPLES:");
        println!("  biz-scraper                                    # Live catalog scraping");
        println!("  biz-scraper --warosu                           # Import recent archives");
        println!("  biz-scraper --warosu --search bitcoin --max 1000");
        println!("  biz-scraper --warosu --from 2024-01-01 --to 2024-12-31");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.help {
        Args::print_help();
        return Ok(());
    }

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

    if args.warosu {
        // Run Warosu historical import
        run_warosu_import(&db, &config, &args).await
    } else {
        // Run live catalog scraper
        run_live_scraper(&db, &config).await
    }
}

/// Run the live 4chan catalog scraper
async fn run_live_scraper(db: &db::Database, config: &config::Config) -> Result<()> {
    let scraper = scraper::BizScraper::new(config)?;

    info!(
        "Starting live scrape loop with {}s interval",
        config.scraper.poll_interval_seconds
    );
    scraper.run_forever(db).await?;

    Ok(())
}

/// Run Warosu historical import
async fn run_warosu_import(db: &db::Database, config: &config::Config, args: &Args) -> Result<()> {
    info!("Starting Warosu historical import...");

    let scraper = warosu::WarosuScraper::new(&config.scraper.board)?;

    let params = warosu::SearchParams {
        subject: args.search.clone(),
        text: None,
        date_start: args.date_from.clone(),
        date_end: args.date_to.clone(),
        ops_only: true,
        offset: 0,
    };

    info!("Search params: {:?}", params);
    info!("Max threads: {}", args.max_threads);

    let stats = scraper.import_historical(db, &params, args.max_threads).await?;

    info!("Import complete!");
    info!("  Total processed: {}", stats.total);
    info!("  New threads: {}", stats.new_threads);
    info!("  Skipped (existing): {}", stats.skipped);
    info!("  Coin mentions added: {}", stats.mentions_added);

    Ok(())
}

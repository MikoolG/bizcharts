//! BizCharts Scraper
//!
//! Scrapes the 4chan /biz/ board for sentiment analysis.
//! Supports both live catalog scraping and historical archive imports.
//!
//! Usage:
//!   biz-scraper              # Run live catalog scraper
//!   biz-scraper --warosu     # Import historical data from Warosu
//!   biz-scraper --warosu --search "bitcoin" --max 1000
//!   biz-scraper --maintain   # Run database maintenance
//!   biz-scraper --stats      # Show storage statistics

use anyhow::Result;
use std::env;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

mod config;
mod db;
mod extractor;
mod maintenance;
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
    /// Number of pages to browse (general activity, no search filter)
    pages: Option<u32>,
    /// Continue from last known date (for incremental imports)
    continue_import: bool,
    /// Backfill: import older data before our oldest record
    backfill: bool,
    /// Run database maintenance
    maintain: bool,
    /// Show storage statistics
    stats: bool,
    /// Show data coverage report
    coverage: bool,
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
            pages: None,
            continue_import: false,
            backfill: false,
            maintain: false,
            stats: false,
            coverage: false,
            help: false,
        };

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--warosu" | "-w" => result.warosu = true,
                "--maintain" | "--maintenance" => result.maintain = true,
                "--stats" | "--storage" => result.stats = true,
                "--coverage" => result.coverage = true,
                "--continue" | "-c" => result.continue_import = true,
                "--backfill" => result.backfill = true,
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
                "--pages" | "-p" => {
                    i += 1;
                    if i < args.len() {
                        result.pages = args[i].parse().ok();
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
        println!("MODES:");
        println!("  (default)         Run live catalog scraper");
        println!("  --warosu, -w      Run Warosu historical import");
        println!("  --maintain        Run database maintenance (cleanup old data)");
        println!("  --stats           Show storage statistics");
        println!("  --coverage        Show data coverage report (what dates we have)\n");
        println!("WAROSU OPTIONS:");
        println!("  --pages, -p NUM   Browse N pages of general activity (no search filter)");
        println!("  --search, -s      Search term for Warosu (searches subject line)");
        println!("  --from DATE       Start date for Warosu search (YYYY-MM-DD)");
        println!("  --to DATE         End date for Warosu search (YYYY-MM-DD)");
        println!("  --max, -m NUM     Maximum threads for search import (default: 5000)");
        println!("  --continue, -c    Continue from last known date (import newer data)");
        println!("  --backfill        Import older data before our oldest record\n");
        println!("OTHER:");
        println!("  --help, -h        Show this help message\n");
        println!("EXAMPLES:");
        println!("  biz-scraper                                    # Live catalog scraping");
        println!("  biz-scraper --coverage                         # See what data we have");
        println!("  biz-scraper --warosu --pages 10                # 10 pages of general /biz/");
        println!("  biz-scraper --warosu --continue                # Import from last date to now");
        println!("  biz-scraper --warosu --backfill                # Import older historical data");
        println!("  biz-scraper --warosu --from 2024-01-01 --to 2024-06-30");
        println!("  biz-scraper --maintain                         # Cleanup old data");
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

    if args.stats {
        // Show storage statistics
        run_storage_stats(&db)
    } else if args.coverage {
        // Show data coverage report
        run_coverage_report(&db)
    } else if args.maintain {
        // Run database maintenance
        run_maintenance(&db)
    } else if args.warosu {
        // Run Warosu historical import
        run_warosu_import(&db, &config, &args).await
    } else {
        // Run live catalog scraper
        run_live_scraper(&db, &config).await
    }
}

/// Show storage statistics
fn run_storage_stats(db: &db::Database) -> Result<()> {
    info!("Gathering storage statistics...");

    let maint = maintenance::Maintenance::new(
        db.connection(),
        maintenance::RetentionConfig::default(),
    );

    let stats = maint.get_stats()?;
    stats.print_report();

    Ok(())
}

/// Show data coverage report
fn run_coverage_report(db: &db::Database) -> Result<()> {
    use chrono::{TimeZone, Utc};

    println!("\n{}", "=".repeat(60));
    println!("DATA COVERAGE REPORT");
    println!("{}", "=".repeat(60));

    // Get coverage by source
    let coverage = db.get_coverage_by_source()?;

    if coverage.is_empty() {
        println!("\nNo data collected yet. Run the scraper first.");
        return Ok(());
    }

    for src in &coverage {
        println!("\n{} Source: {}", "─".repeat(20), src.source.to_uppercase());

        let oldest = src.oldest_timestamp
            .map(|t| Utc.timestamp_opt(t, 0).single())
            .flatten()
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "N/A".to_string());

        let newest = src.newest_timestamp
            .map(|t| Utc.timestamp_opt(t, 0).single())
            .flatten()
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "N/A".to_string());

        println!("  Threads:        {:>10}", src.thread_count);
        println!("  With images:    {:>10} ({:.1}%)",
            src.with_images,
            src.with_images as f64 / src.thread_count.max(1) as f64 * 100.0
        );
        println!("  Local images:   {:>10} ({:.1}%)",
            src.with_local_images,
            src.with_local_images as f64 / src.with_images.max(1) as f64 * 100.0
        );
        println!("  Date range:     {} to {}", oldest, newest);

        // Show recent daily coverage
        let daily = db.get_daily_coverage(&src.source, 14)?;
        if !daily.is_empty() {
            println!("\n  Recent daily thread counts:");
            for day in daily.iter().take(7) {
                println!("    {}: {:>5} threads", day.date, day.thread_count);
            }
            if daily.len() > 7 {
                println!("    ... ({} more days)", daily.len() - 7);
            }
        }
    }

    // Show continuation hints
    println!("\n{}", "─".repeat(60));
    println!("IMPORT SUGGESTIONS:");

    for src in &coverage {
        if src.source == "warosu" {
            if let Some(newest_ts) = src.newest_timestamp {
                let newest = Utc.timestamp_opt(newest_ts, 0).single()
                    .map(|dt| dt.format("%Y-%m-%d").to_string())
                    .unwrap_or_default();
                println!("  To get newer data:   --warosu --continue");
                println!("    (will import from {} onwards)", newest);
            }
            if let Some(oldest_ts) = src.oldest_timestamp {
                let oldest = Utc.timestamp_opt(oldest_ts, 0).single()
                    .map(|dt| dt.format("%Y-%m-%d").to_string())
                    .unwrap_or_default();
                println!("  To get older data:   --warosu --backfill");
                println!("    (will import before {})", oldest);
            }
        }
    }

    println!();
    Ok(())
}

/// Run database maintenance
fn run_maintenance(db: &db::Database) -> Result<()> {
    info!("Running database maintenance...");

    let config = maintenance::RetentionConfig::default();
    info!("Retention config: {:?}", config);

    let maint = maintenance::Maintenance::new(db.connection(), config);

    let stats = maint.run_all()?;

    println!("\nMaintenance complete:");
    println!("  HTML stripped:         {} threads", stats.html_stripped);
    println!("  Snapshots aggregated:  {} rows", stats.snapshots_aggregated);
    println!("  Threads deleted:       {} rows", stats.threads_deleted);
    println!("  Mentions cleaned:      {} rows", stats.mentions_cleaned);
    println!("  Legacy tables dropped: {}", stats.legacy_cleaned);

    Ok(())
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
    use chrono::{Duration, TimeZone, Utc};

    info!("Starting Warosu historical import...");

    let scraper = warosu::WarosuScraper::new(&config.scraper.board, &config.database.images_dir)?;

    // If --pages is specified, use page browsing (general activity)
    // Otherwise, use search-based import
    let stats = if let Some(num_pages) = args.pages {
        info!("Mode: Page browsing ({} pages of general /biz/ activity)", num_pages);
        scraper.import_by_pages(db, num_pages).await?
    } else {
        // Determine date range based on flags
        let mut date_from = args.date_from.clone();
        let mut date_to = args.date_to.clone();

        if args.continue_import {
            // Continue from last known date - import newer data
            if let Some(newest_ts) = db.get_newest_timestamp("warosu")? {
                let newest_date = Utc.timestamp_opt(newest_ts, 0).single()
                    .ok_or_else(|| anyhow::anyhow!("Invalid timestamp"))?;
                // Start from the day after the newest record
                let start_date = newest_date + Duration::days(1);
                date_from = Some(start_date.format("%Y-%m-%d").to_string());
                info!("Continue mode: importing from {} onwards", date_from.as_ref().unwrap());
            } else {
                info!("Continue mode: no existing warosu data, will import all available");
            }
        }

        if args.backfill {
            // Backfill - import older data before our oldest record
            if let Some(oldest_ts) = db.get_oldest_timestamp("warosu")? {
                let oldest_date = Utc.timestamp_opt(oldest_ts, 0).single()
                    .ok_or_else(|| anyhow::anyhow!("Invalid timestamp"))?;
                // End at the day before the oldest record
                let end_date = oldest_date - Duration::days(1);
                date_to = Some(end_date.format("%Y-%m-%d").to_string());
                info!("Backfill mode: importing data up to {}", date_to.as_ref().unwrap());
            } else {
                info!("Backfill mode: no existing warosu data, will import all available");
            }
        }

        let params = warosu::SearchParams {
            subject: args.search.clone(),
            text: None,
            date_start: date_from,
            date_end: date_to,
            ops_only: true,
            offset: 0,
        };

        info!("Mode: Search-based import");
        info!("Search params: {:?}", params);
        info!("Max threads: {}", args.max_threads);

        scraper.import_historical(db, &params, args.max_threads).await?
    };

    info!("Import complete!");
    info!("  Total processed: {}", stats.total);
    info!("  New threads: {}", stats.new_threads);
    info!("  Skipped (existing): {}", stats.skipped);
    info!("  Coin mentions added: {}", stats.mentions_added);

    Ok(())
}

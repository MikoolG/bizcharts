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
    /// Run database maintenance
    maintain: bool,
    /// Show storage statistics
    stats: bool,
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
            maintain: false,
            stats: false,
            help: false,
        };

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--warosu" | "-w" => result.warosu = true,
                "--maintain" | "--maintenance" => result.maintain = true,
                "--stats" | "--storage" => result.stats = true,
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
        println!("MODES:");
        println!("  (default)         Run live catalog scraper");
        println!("  --warosu, -w      Run Warosu historical import");
        println!("  --maintain        Run database maintenance (cleanup old data)");
        println!("  --stats           Show storage statistics\n");
        println!("WAROSU OPTIONS:");
        println!("  --search, -s      Search term for Warosu (searches subject line)");
        println!("  --from DATE       Start date for Warosu search (YYYY-MM-DD)");
        println!("  --to DATE         End date for Warosu search (YYYY-MM-DD)");
        println!("  --max, -m NUM     Maximum threads to import (default: 5000)\n");
        println!("OTHER:");
        println!("  --help, -h        Show this help message\n");
        println!("EXAMPLES:");
        println!("  biz-scraper                                    # Live catalog scraping");
        println!("  biz-scraper --warosu                           # Import recent archives");
        println!("  biz-scraper --warosu --search bitcoin --max 1000");
        println!("  biz-scraper --maintain                         # Cleanup old data");
        println!("  biz-scraper --stats                            # Show storage usage");
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

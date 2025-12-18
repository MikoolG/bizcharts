# BizCharts

4chan /biz/ sentiment analysis platform producing Fear/Greed indices and per-coin sentiment tracking.

## Current Status (Dec 2024)

| Component | Status | Notes |
|-----------|--------|-------|
| Live Scraper | ✅ Tested | Pulls ~150 threads from 4chan catalog, downloads thumbnails |
| Warosu Importer | ✅ Tested | Fixed timestamp parsing, captures OPs only (not replies) |
| Labeling GUI | ✅ Tested | 200 posts labeled (180 live, 20 warosu), avg sentiment ~2.4 (bearish) |
| Text Sentiment | 🔄 Pending | VADER + lexicon implementation |
| Image Sentiment | 🔄 Pending | CLIP/color analysis |
| Dashboard | 🔄 Pending | Streamlit UI |

## Architecture

**OP-focused design**: Scrapes catalog only (not individual threads). One catalog request = ~150 thread OPs with full text, images, and engagement metrics.

```
Data Collection (Rust)              Analysis (Python)
┌─────────────────────┐            ┌─────────────────────┐
│ Live Scraper        │            │ Text: VADER+lexicon │
│ - catalog.json/60s  │───────────▶│ Image: Color/CLIP   │
│ - 1 req/sec limit   │  SQLite    │ Fusion: 60/40 split │
├─────────────────────┤  (shared)  ├─────────────────────┤
│ Warosu Importer     │            │ Manual Labeling UI  │
│ - Historical data   │───────────▶│ Analysis/Export     │
├─────────────────────┤            ├─────────────────────┤
│ Maintenance         │            │ Price Provider      │
│ - Retention/cleanup │            │ - Multi-source API  │
└─────────────────────┘            └─────────────────────┘
```

## Project Structure

```
bizcharts/
├── rust-scraper/
│   └── src/
│       ├── main.rs          # CLI entry (--warosu, --maintain, --stats)
│       ├── scraper.rs       # Live catalog scraper
│       ├── warosu.rs        # Historical archive importer
│       ├── db.rs            # SQLite schema + operations
│       ├── extractor.rs     # Coin ticker extraction
│       └── maintenance.rs   # Retention policies, cleanup
├── python-ml/
│   └── src/
│       ├── text_analyzer.py   # VADER + custom lexicon
│       ├── image_analyzer.py  # CLIP + color analysis
│       ├── aggregator.py      # Sentiment fusion
│       ├── price_provider.py  # CoinGecko/Binance/CMC rotation
│       ├── labeler.py         # Manual labeling GUI (Tkinter)
│       ├── label_analysis.py  # Training data analysis/export
│       └── dashboard.py       # Streamlit UI
├── config/
│   └── settings.toml        # All configuration
├── docs/
│   ├── sentiment-strategy.md  # Analysis approach
│   └── labeling-guide.md      # Manual labeling process
└── data/                      # Databases (gitignored)
    └── posts.db
```

## CLI Usage

```bash
# Live catalog scraping
cargo run -- --live --once    # Single snapshot of current catalog (~150 threads)
cargo run -- --live           # Continuous polling (every 60s)

# Import historical data from Warosu archive
cargo run -- --warosu --search bitcoin --max 1000
cargo run -- --warosu --from 2024-01-01 --to 2024-12-31
cargo run -- --warosu --pages 10          # Browse N pages of general activity

# Incremental imports (avoids duplicates)
cargo run -- --coverage                   # Show data coverage report
cargo run -- --warosu --continue          # Import from last date forward
cargo run -- --warosu --backfill          # Import older data before earliest

# Database maintenance
cargo run -- --stats      # Show storage statistics
cargo run -- --maintain   # Run cleanup (strips old HTML, aggregates snapshots)

# Manual labeling (Python)
cd python-ml
python3 -m src.labeler                              # Start labeling UI
python3 -m src.labeler --source live                # Label only live catalog data
python3 -m src.labeler --source warosu              # Label only Warosu data
python3 -m src.labeler --from 2024-01-01            # Label from date onward
python3 -m src.labeler --stats                      # Show labeling statistics
python3 -m src.labeler --export labels.csv          # Export to CSV
```

## Database Schema

Primary table: `thread_ops` (one row per thread OP)

```sql
thread_ops (
    thread_id INTEGER PRIMARY KEY,
    subject TEXT,
    op_text TEXT,                    -- Raw HTML
    op_text_clean TEXT,              -- Stripped for analysis
    reply_count INTEGER,             -- Engagement weight
    image_url TEXT,
    sentiment_score REAL,            -- -1 to +1
    sentiment_confidence REAL,       -- 0 to 1
    source TEXT                      -- 'live' or 'warosu'
)

coin_mentions (thread_id, coin_symbol, confidence, mention_source)
training_labels (thread_id, sentiment_rating 1-5, labeler_id)  -- 1=bearish, 3=neutral, 5=bullish
catalog_snapshots (snapshot_at, total_threads, avg_reply_count, top_coins)
```

## Sentiment Strategy

See [docs/sentiment-strategy.md](docs/sentiment-strategy.md) for full details.

**Key concepts:**
- **Multi-signal fusion**: Text (60%) + Image (40%) sentiment
- **Confidence weighting**: Uncertain posts get less weight in aggregations
- **Reply-based importance**: Popular threads (high reply count) influence aggregate more
- **Color heuristics**: Red-dominant images = bearish, green = bullish (before ML)

**Lexicon examples** (VADER additions):
- WAGMI: +3.5, NGMI: -3.5, LFG: +3.0
- "we're so back": +2.5, "it's over": -3.5
- moon/mooning: +3.0, rekt: -3.5

## Storage & Retention

Default retention (configurable in `maintenance.rs`):
- **Full HTML**: 30 days (then stripped, keep clean text)
- **Thread metadata**: 1 year
- **Catalog snapshots**: Full 7d → Hourly 90d → Daily thereafter
- **Popular threads (50+ replies)**: Preserved indefinitely
- **Training labels**: Never deleted

Run `biz-scraper --maintain` periodically to apply retention policies.

## Price Data

Multi-source rotation to stay within free tier limits:
1. **CoinGecko** - 10k calls/month free
2. **Binance** - Unlimited public API
3. **CoinMarketCap** - 10k calls/month free (needs API key)

Configure in `settings.toml` under `[price_data]`.

## Development

```bash
# Build Rust scraper
cd rust-scraper && cargo build --release

# Setup Python
cd python-ml
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run tests
cargo test                    # Rust
pytest                        # Python
```

## API Reference

**4chan API** (1 req/sec max):
```
GET https://a.4cdn.org/biz/catalog.json  # All thread OPs
GET https://i.4cdn.org/biz/{tim}{ext}    # Full image
GET https://i.4cdn.org/biz/{tim}s.jpg    # Thumbnail
```

**Warosu Archive** (1 req/2sec recommended):
```
https://warosu.org/biz/?task=search&search_subject=bitcoin
```

## Key Files

| File | Purpose |
|------|---------|
| [rust-scraper/src/db.rs](rust-scraper/src/db.rs) | Database schema and all SQL |
| [docs/sentiment-strategy.md](docs/sentiment-strategy.md) | Full sentiment analysis approach |
| [docs/labeling-guide.md](docs/labeling-guide.md) | Manual labeling instructions |
| [config/settings.toml](config/settings.toml) | All configuration options |

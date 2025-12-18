# BizCharts

A comprehensive 4chan /biz/ board market sentiment analysis platform that produces Fear/Greed indices, tracks individual coin sentiment, and correlates social sentiment with price movements.

## Project Vision

BizCharts analyzes the 4chan /biz/ (Business & Finance) board to extract market sentiment signals. The board is a raw, unfiltered source of retail crypto sentiment with unique cultural markers (Wojak memes, specific slang) that traditional sentiment tools miss entirely.

**Core Features:**
- Overall market Fear/Greed index (0-100 scale)
- Per-coin sentiment tracking for every mentioned cryptocurrency
- Historical sentiment trends and time-series analysis
- Price correlation with CoinGecko data
- Meme image analysis (Wojak/Pepe variants carry strong sentiment signals)
- Irony/sarcasm detection for /biz/-specific language patterns

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         BizCharts Architecture                          │
├────────────────────────────────────────────────────────────────────────┤
│  Rust Scraper (dot4ch/reqwest)                                         │
│  - Polls /biz/catalog.json every 60 seconds                            │
│  - Respects 1 req/sec rate limit with token bucket                     │
│  - Uses If-Modified-Since for efficient caching                        │
│      │                                                                  │
│      ▼                                                                  │
│  SQLite (Operational Store)                                            │
│  - Raw posts, threads, timestamps                                      │
│  - Coin mention extraction                                             │
│  - Image metadata and MD5 hashes                                       │
│      │                                                                  │
│      ▼                                                                  │
│  Python ML Service                                                      │
│  ├── Text Pipeline                                                      │
│  │   - HTML decode, greentext strip, ticker normalize                  │
│  │   - VADER + custom /biz/ lexicon (fast baseline)                    │
│  │   - Claude Batch API (ambiguous cases, 95% cost savings)            │
│  │                                                                      │
│  └── Image Pipeline                                                     │
│      - CLIP zero-shot bullish/bearish screening                        │
│      - YOLOv8 + EfficientNet for Wojak/Pepe classification            │
│      - PaddleOCR for meme text extraction                              │
│      - 40% visual + 60% textual fusion                                 │
│      │                                                                  │
│      ▼                                                                  │
│  DuckDB (Analytics Store)                                              │
│  - Sentiment aggregations (minute/hour/day/week)                       │
│  - Per-coin sentiment time series                                      │
│  - Market-wide fear/greed calculations                                 │
│      │                                                                  │
│      ▼                                                                  │
│  Streamlit Dashboard (Multi-Page App)                                  │
│  ├── Main: Fear/Greed gauge + trends                                   │
│  ├── Coins: Per-coin explorer with drill-down                          │
│  ├── History: Time-series analysis + price correlation                 │
│  ├── Posts: Raw post browser with images                               │
│  └── Data: SQL query interface + exports                               │
└────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Scraper | Rust + dot4ch | Compile-time safety, single binary, built-in rate limiting |
| ML/Analysis | Python 3.11+ | Rich ecosystem (transformers, CLIP, VADER, Claude SDK) |
| Operational DB | SQLite | Simple, excellent bindings in both Rust and Python |
| Analytics DB | DuckDB | 10-100x faster than SQLite for aggregations |
| Data Processing | Polars | 5-100x faster than Pandas, native Rust/Python |
| Text Sentiment | VADER + custom lexicon | Fast baseline with /biz/-specific vocabulary |
| Image Sentiment | CLIP + YOLOv8 + EfficientNet | Zero-shot + fine-tuned meme detection |
| LLM | Claude Batch API | 95% cost reduction with prompt caching |
| Dashboard | Streamlit + Plotly | Interactive exploration, multiple views, Python-native |

## Project Structure

```
bizcharts/
├── claude.md                      # This file - project documentation
├── rust-scraper/                  # Rust scraping engine
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs               # Entry point, scheduler
│       ├── lib.rs                # Library exports
│       ├── scraper.rs            # 4chan API client
│       ├── db.rs                 # SQLite operations
│       └── extractor.rs          # Coin ticker extraction
├── python-ml/                     # Python ML services
│   ├── pyproject.toml
│   └── src/
│       ├── __init__.py
│       ├── text_analyzer.py      # VADER + lexicon + Claude
│       ├── image_analyzer.py     # CLIP + YOLOv8 + OCR
│       ├── aggregator.py         # Sentiment fusion + DuckDB
│       └── dashboard.py          # Streamlit multi-page dashboard
├── config/
│   ├── settings.toml             # Application configuration
│   └── lexicon.json              # Custom sentiment vocabulary
├── data/                          # Databases (gitignored)
│   ├── posts.db                  # SQLite operational
│   └── analytics.duckdb          # DuckDB analytical
├── docs/
│   └── plans/
│       └── implementation-roadmap.md
├── dashboards/                    # Grafana JSON exports
└── ClaudeTrader-reference/        # Reference implementation
```

## Quick Start

### Prerequisites
- Rust 1.70+ (for scraper)
- Python 3.11+ (for ML services and dashboard)

### Setup

```bash
# Enter project directory
cd bizcharts

# Build Rust scraper
cd rust-scraper
cargo build --release

# Setup Python environment
cd ../python-ml
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -e .

# Configure (settings.toml is ready to use, add API keys as needed)
# Edit config/settings.toml with your Anthropic API key if using Claude
```

### Running

```bash
# Terminal 1: Start the scraper (runs continuously)
cd rust-scraper
./target/release/biz-scraper

# Terminal 2: Run sentiment analysis (can be scheduled via cron)
cd python-ml
source .venv/bin/activate
python -m src.aggregator

# Terminal 3: Launch dashboard
cd python-ml
source .venv/bin/activate
streamlit run src/dashboard.py
# Opens browser at http://localhost:8501
```

### First-Time Data Collection
The scraper needs to run for a while to collect data before meaningful analysis:
- **1 hour**: Enough for basic testing
- **24 hours**: Good baseline for hourly trends
- **1 week**: Sufficient for meaningful Fear/Greed index

## Configuration

See `config/settings.toml` for all options. Key settings:

```toml
[scraper]
poll_interval_seconds = 60
rate_limit_per_second = 1

[sentiment]
confidence_threshold = 0.6
use_claude_for_ambiguous = true

[claude]
model = "claude-3-5-haiku-20241022"
use_batch_api = true
enable_prompt_caching = true
```

## Custom Sentiment Lexicon

The `/biz/ board has unique vocabulary. See `config/lexicon.json`:

**Bullish signals:**
- WAGMI ("We're All Gonna Make It") → +3.5
- LFG ("Let's Fucking Go") → +3.0
- Moon/Mooning → +3.0
- Diamond hands → +2.5
- "We're so back" → +2.5

**Bearish signals:**
- NGMI ("Not Gonna Make It") → -3.5
- "It's over" → -3.5
- Rekt → -3.5
- Rugged → -4.0
- Paper hands → -2.5

**Image sentiment (Wojak variants):**
- Pink Wojak (crying, red/pink skin) → -0.8
- Green Wojak → +0.7
- Doomer (black hoodie) → -0.6
- Bloomer (smiling) → +0.6
- Gigachad → +0.5

## API Reference

### 4chan API Endpoints Used

```
GET https://a.4cdn.org/biz/catalog.json     # All threads with OPs
GET https://a.4cdn.org/biz/thread/{no}.json # Full thread
GET https://i.4cdn.org/biz/{tim}{ext}       # Full image
GET https://i.4cdn.org/biz/{tim}s.jpg       # Thumbnail
```

**Rate limits:** Max 1 request/second. Thread updates minimum 10 seconds.
**Headers:** Always use `If-Modified-Since` for efficient polling.

### Database Schemas

**SQLite (posts.db):**
```sql
CREATE TABLE posts (
    post_id INTEGER PRIMARY KEY,
    thread_id INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    text TEXT,
    has_image BOOLEAN,
    image_url TEXT,
    image_md5 TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE coin_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER REFERENCES posts(post_id),
    coin_symbol TEXT NOT NULL,
    coin_name TEXT
);

CREATE TABLE threads (
    thread_id INTEGER PRIMARY KEY,
    subject TEXT,
    op_post_id INTEGER,
    created_at INTEGER,
    last_updated INTEGER
);
```

**DuckDB (analytics.duckdb):**
```sql
CREATE TABLE sentiment_scores (
    post_id BIGINT PRIMARY KEY,
    overall_score FLOAT,
    confidence FLOAT,
    method VARCHAR  -- 'vader', 'claude', 'image', 'fused'
);

CREATE TABLE coin_sentiment (
    coin_symbol VARCHAR,
    timestamp TIMESTAMP,
    sentiment_score FLOAT,
    post_count INTEGER,
    confidence FLOAT
);

CREATE TABLE market_sentiment (
    timestamp TIMESTAMP PRIMARY KEY,
    fear_greed_index FLOAT,  -- 0-100 scale
    bullish_pct FLOAT,
    bearish_pct FLOAT,
    neutral_pct FLOAT,
    total_posts INTEGER
);
```

## Development Guidelines

### Code Style
- **Rust:** Follow `rustfmt` defaults, use `clippy` for linting
- **Python:** Follow `ruff` formatting, type hints required

### Testing
```bash
# Rust tests
cd rust-scraper && cargo test

# Python tests
cd python-ml && pytest
```

### Adding New Sentiment Terms
1. Edit `config/lexicon.json`
2. Add term with weight (-5.0 to +5.0 scale)
3. Restart the Python analyzer service

### Debugging
- Scraper logs: `rust-scraper/logs/`
- Check SQLite: `sqlite3 data/posts.db ".schema"`
- Check DuckDB: `duckdb data/analytics.duckdb "SELECT * FROM market_sentiment ORDER BY timestamp DESC LIMIT 10"`

## Cost Management

Claude API costs with optimization:

| Method | 10k posts/day |
|--------|---------------|
| Regular API | ~$10/day |
| Batch API (50% off) | ~$5/day |
| Batch + Caching (95% off) | ~$0.50-2/day |

**Best practices:**
1. Use VADER for clear-cut sentiment (free, fast)
2. Reserve Claude for ambiguous posts only (~20% of total)
3. Always use Batch API for non-real-time analysis
4. Enable prompt caching with static system prompts

## Dashboard Views

The Streamlit dashboard provides multiple pages for different analysis needs:

### 1. Main Dashboard (Fear/Greed Overview)
- **Fear/Greed Gauge**: Current index value (0-100) with color coding
- **Trend Sparkline**: 24h/7d/30d sentiment trajectory
- **Composition Breakdown**: Bullish/bearish/neutral percentages
- **Volume Indicator**: Post activity relative to baseline
- **Top Movers**: Coins with biggest sentiment shifts
- **Data Source**: `market_sentiment` table

### 2. Coin Explorer
- **Search/Filter**: Find any mentioned cryptocurrency
- **Sentiment History**: Per-coin time-series chart
- **Mention Volume**: How often the coin is discussed
- **Leaderboards**: Most bullish/bearish coins
- **Drill-Down**: Click to see actual posts mentioning the coin
- **Price Overlay**: CoinGecko price data for correlation
- **Data Source**: `coin_sentiment`, `coin_mentions` tables

### 3. Historical Analysis
- **Multi-Timeframe Charts**: Minute/hour/day/week aggregations
- **Sentiment vs Price**: Correlation analysis with lag options
- **Regime Detection**: Identify sentiment shift points
- **Comparison View**: Compare multiple coins side-by-side
- **Export**: Download data for external analysis
- **Data Source**: `market_sentiment`, `coin_sentiment` with time filters

### 4. Post Browser
- **Raw Posts**: View actual /biz/ posts with their scores
- **Filters**: By coin, sentiment range, time, has_image
- **Image Display**: See meme images inline
- **Score Breakdown**: Why was this post scored this way?
- **Validation Mode**: Mark posts as correctly/incorrectly scored
- **Data Source**: `posts`, `sentiment_scores` tables joined

### 5. Data Explorer
- **SQL Query Interface**: Run ad-hoc queries against DuckDB
- **Pre-built Queries**: Common analysis templates
- **Export Options**: CSV, JSON, Parquet
- **Schema Browser**: See all tables and columns
- **Data Source**: Direct DuckDB access

### 6. System Health
- **Scraper Status**: Last run, posts collected, errors
- **Pipeline Status**: Analysis queue, processing rate
- **API Usage**: Claude API calls, costs
- **Database Stats**: Table sizes, index health
- **Data Source**: `scraper_state`, logs

## Known Gaps & Future Enhancements

Areas identified for improvement (from project-considerations.md):

### Not Yet Implemented
1. **Spam Filtering**: The considerations doc notes "90%+ of raw crypto Twitter is spam". We need filters for:
   - Bot detection (repetitive posts, suspicious patterns)
   - Shill detection (coordinated pump campaigns)
   - Off-topic filtering (non-crypto discussions)

2. **Fine-tuned RoBERTa**: For nuanced sentiment beyond VADER's capabilities, especially sarcasm detection (~91.7% accuracy reported in research)

3. **ONNX Runtime**: Export models to ONNX for 3-5x inference speedup and 60-80% memory reduction

4. **Alternative.me-Style Components**: Their index uses:
   - Market volatility (25%)
   - Market momentum (25%)
   - Social media volume (15%)
   - Surveys (15%)
   - BTC dominance (10%)
   - Google Trends (10%)

   We currently only have text/image sentiment. Adding volatility and momentum from price data would improve accuracy.

5. **FinBERT**: Financial domain BERT model, potentially better than generic VADER for market sentiment

### Future Features
- Real-time alerts when sentiment shifts dramatically
- API endpoint for external consumption
- Mobile-responsive dashboard
- Historical backfill from archives
- Multi-board support (/biz/, crypto Twitter, Reddit)

## References

- [4chan API Documentation](https://github.com/4chan/4chan-API)
- [Alternative.me Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [Claude API Documentation](https://docs.anthropic.com/)
- [dot4ch Rust crate](https://crates.io/crates/dot4ch)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [DuckDB Documentation](https://duckdb.org/docs/)

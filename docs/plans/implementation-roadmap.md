# BizCharts Implementation Roadmap

## Phase Overview

| Phase | Focus | Key Deliverables |
|-------|-------|------------------|
| 1 | Data Collection | Rust scraper, SQLite schema, rate limiting |
| 2 | Text Sentiment | VADER + lexicon, DuckDB aggregations |
| 3 | Image Sentiment | CLIP, YOLOv8, OCR pipeline |
| 4 | Visualization | Grafana dashboards, price correlation |
| 5 | Advanced | Alerts, API, ML refinements |

---

## Phase 1: Data Collection Foundation

### Goals
- Build reliable Rust scraper for 4chan /biz/ board
- Store raw posts in SQLite with proper schema
- Extract coin/ticker mentions from post text
- Handle rate limiting and caching properly

### Tasks

#### 1.1 Rust Project Setup
```bash
cd rust-scraper
cargo init
```

**Cargo.toml dependencies:**
- `dot4ch` - 4chan API client with built-in rate limiting
- `reqwest` - HTTP client for image downloads
- `rusqlite` - SQLite bindings
- `tokio` - Async runtime
- `serde`, `serde_json` - Serialization
- `chrono` - Timestamp handling
- `tracing` - Logging

#### 1.2 SQLite Schema Implementation

```sql
-- posts table
CREATE TABLE posts (
    post_id INTEGER PRIMARY KEY,
    thread_id INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    name TEXT,
    text TEXT,
    has_image BOOLEAN DEFAULT FALSE,
    image_url TEXT,
    image_md5 TEXT,
    thumbnail_url TEXT,
    replies_to TEXT,  -- JSON array of post IDs
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_posts_thread ON posts(thread_id);
CREATE INDEX idx_posts_timestamp ON posts(timestamp);

-- threads table
CREATE TABLE threads (
    thread_id INTEGER PRIMARY KEY,
    subject TEXT,
    op_post_id INTEGER,
    board TEXT DEFAULT 'biz',
    created_at INTEGER,
    last_updated INTEGER,
    reply_count INTEGER,
    image_count INTEGER
);

-- coin mentions extraction
CREATE TABLE coin_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER REFERENCES posts(post_id),
    coin_symbol TEXT NOT NULL,
    coin_name TEXT,
    confidence REAL DEFAULT 1.0
);

CREATE INDEX idx_mentions_symbol ON coin_mentions(coin_symbol);
CREATE INDEX idx_mentions_post ON coin_mentions(post_id);

-- scraper state
CREATE TABLE scraper_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 1.3 Core Scraper Logic

```rust
// Pseudocode structure
async fn main() {
    // Initialize database
    let db = Database::open("data/posts.db")?;
    db.run_migrations()?;

    // Load coin symbols for extraction
    let coin_symbols = load_coingecko_symbols().await?;

    // Main polling loop
    loop {
        // Fetch catalog with If-Modified-Since
        let catalog = fetch_catalog_if_modified().await?;

        if let Some(threads) = catalog {
            for thread in threads {
                // Check if thread updated since last scrape
                if thread.last_modified > db.get_thread_last_updated(thread.no)? {
                    let full_thread = fetch_thread(thread.no).await?;

                    for post in full_thread.posts {
                        // Store post
                        db.upsert_post(&post)?;

                        // Extract coin mentions
                        let mentions = extract_coins(&post.text, &coin_symbols);
                        db.insert_mentions(post.no, &mentions)?;

                        // Queue image download if present
                        if post.has_image() {
                            download_thumbnail(&post).await?;
                        }
                    }
                }
            }
        }

        // Sleep respecting rate limits
        tokio::time::sleep(Duration::from_secs(60)).await;
    }
}
```

#### 1.4 Ticker/Coin Extraction

```rust
fn extract_coins(text: &str, known_symbols: &HashSet<String>) -> Vec<CoinMention> {
    let mut mentions = Vec::new();

    // Pattern 1: $SYMBOL format
    let dollar_pattern = Regex::new(r"\$([A-Z]{2,10})\b").unwrap();

    // Pattern 2: Known symbols as standalone words
    for cap in dollar_pattern.captures_iter(text) {
        let symbol = &cap[1];
        if known_symbols.contains(symbol) {
            mentions.push(CoinMention {
                symbol: symbol.to_string(),
                confidence: 0.95,
            });
        }
    }

    // Pattern 3: Full names (Bitcoin, Ethereum, etc.)
    let name_patterns = [
        ("bitcoin", "BTC"),
        ("ethereum", "ETH"),
        ("solana", "SOL"),
        // ... more mappings
    ];

    let text_lower = text.to_lowercase();
    for (name, symbol) in name_patterns {
        if text_lower.contains(name) {
            mentions.push(CoinMention {
                symbol: symbol.to_string(),
                confidence: 0.9,
            });
        }
    }

    mentions
}
```

#### 1.5 Rate Limiting Implementation

```rust
struct RateLimiter {
    tokens: AtomicU32,
    last_refill: AtomicU64,
    max_tokens: u32,
    refill_rate: Duration,
}

impl RateLimiter {
    async fn acquire(&self) {
        loop {
            self.refill();
            if self.tokens.fetch_sub(1, Ordering::SeqCst) > 0 {
                return;
            }
            self.tokens.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(self.refill_rate).await;
        }
    }
}
```

### Acceptance Criteria
- [ ] Scraper polls catalog every 60 seconds
- [ ] If-Modified-Since headers reduce bandwidth
- [ ] All posts stored in SQLite with correct schema
- [ ] Coin mentions extracted with 90%+ accuracy on common tickers
- [ ] Thumbnails downloaded and stored
- [ ] No rate limit violations (monitor for 24 hours)

---

## Phase 2: Text Sentiment Baseline

### Goals
- Build Python service to analyze post sentiment
- Implement VADER with custom /biz/ lexicon
- Store sentiment scores in DuckDB
- Create aggregation views for time-series analysis

### Tasks

#### 2.1 Python Project Setup

```bash
cd python-ml
python -m venv .venv
source .venv/bin/activate
```

**pyproject.toml dependencies:**
- `polars` - Fast dataframes
- `duckdb` - Analytics database
- `vaderSentiment` - Baseline sentiment
- `anthropic` - Claude SDK
- `sqlite-utils` - SQLite reading

#### 2.2 Text Preprocessing Pipeline

```python
import re
import html

def preprocess_post(text: str) -> str:
    """Clean 4chan post text for sentiment analysis."""
    if not text:
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags (4chan uses <br>, <a>, etc.)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Handle greentext (lines starting with >)
    # Flag for reduced confidence, often ironic
    greentext_lines = []
    regular_lines = []
    for line in text.split('\n'):
        if line.strip().startswith('>'):
            greentext_lines.append(line)
        else:
            regular_lines.append(line)

    # Normalize tickers: $BTC -> BTC
    text = re.sub(r'\$([A-Z]{2,10})', r'\1', text)

    # Expand common abbreviations
    expansions = {
        'wagmi': 'we are going to make it',
        'ngmi': 'not going to make it',
        'gmi': 'going to make it',
        'lfg': 'lets go',
        'dyor': 'do your own research',
        'nfa': 'not financial advice',
    }
    for abbr, expansion in expansions.items():
        text = re.sub(rf'\b{abbr}\b', expansion, text, flags=re.IGNORECASE)

    return text, len(greentext_lines) > 0
```

#### 2.3 Custom Lexicon Integration

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

def create_biz_analyzer() -> SentimentIntensityAnalyzer:
    """Create VADER analyzer with /biz/ lexicon additions."""
    analyzer = SentimentIntensityAnalyzer()

    # Load custom lexicon
    with open('config/lexicon.json') as f:
        lexicon = json.load(f)

    # Add to VADER's lexicon
    for term, weight in lexicon['bullish'].items():
        analyzer.lexicon[term.lower()] = weight

    for term, weight in lexicon['bearish'].items():
        analyzer.lexicon[term.lower()] = weight

    return analyzer

def analyze_sentiment(text: str, is_greentext: bool, analyzer) -> dict:
    """Analyze sentiment with greentext confidence adjustment."""
    scores = analyzer.polarity_scores(text)

    confidence = 0.8
    if is_greentext:
        confidence *= 0.6  # Reduce confidence for ironic content

    return {
        'compound': scores['compound'],
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu'],
        'confidence': confidence,
        'method': 'vader'
    }
```

#### 2.4 DuckDB Schema and Aggregations

```sql
-- Sentiment scores per post
CREATE TABLE sentiment_scores (
    post_id BIGINT PRIMARY KEY,
    thread_id BIGINT,
    timestamp TIMESTAMP,
    overall_score FLOAT,      -- -1 to +1
    confidence FLOAT,         -- 0 to 1
    method VARCHAR,           -- vader, claude, fused
    is_greentext BOOLEAN,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Per-coin sentiment aggregations
CREATE TABLE coin_sentiment (
    coin_symbol VARCHAR,
    bucket_start TIMESTAMP,
    bucket_size VARCHAR,      -- 'minute', 'hour', 'day', 'week'
    avg_sentiment FLOAT,
    weighted_sentiment FLOAT, -- confidence-weighted
    post_count INTEGER,
    bullish_count INTEGER,
    bearish_count INTEGER,
    neutral_count INTEGER,
    PRIMARY KEY (coin_symbol, bucket_start, bucket_size)
);

-- Market-wide sentiment
CREATE TABLE market_sentiment (
    bucket_start TIMESTAMP,
    bucket_size VARCHAR,
    fear_greed_index FLOAT,   -- 0-100 scale
    avg_sentiment FLOAT,      -- -1 to +1
    bullish_pct FLOAT,
    bearish_pct FLOAT,
    neutral_pct FLOAT,
    total_posts INTEGER,
    unique_threads INTEGER,
    PRIMARY KEY (bucket_start, bucket_size)
);

-- Aggregation views
CREATE VIEW hourly_sentiment AS
SELECT
    date_trunc('hour', timestamp) as hour,
    AVG(overall_score) as avg_sentiment,
    SUM(CASE WHEN overall_score > 0.05 THEN 1 ELSE 0 END) as bullish,
    SUM(CASE WHEN overall_score < -0.05 THEN 1 ELSE 0 END) as bearish,
    COUNT(*) as total
FROM sentiment_scores
GROUP BY 1;
```

#### 2.5 Fear/Greed Index Calculation

```python
def calculate_fear_greed_index(
    avg_sentiment: float,
    bullish_pct: float,
    post_volume_zscore: float,
    price_correlation: float | None = None
) -> float:
    """
    Calculate Fear/Greed index on 0-100 scale.

    Components (inspired by Alternative.me):
    - Sentiment score: 40%
    - Bullish/bearish ratio: 30%
    - Volume momentum: 20%
    - Price correlation: 10% (optional)
    """
    # Normalize sentiment from [-1, 1] to [0, 100]
    sentiment_component = (avg_sentiment + 1) * 50

    # Bullish percentage already 0-100
    ratio_component = bullish_pct

    # Volume z-score normalized: high volume in bull = greed, in bear = fear
    # Clamp to [-3, 3] then normalize
    volume_clamped = max(-3, min(3, post_volume_zscore))
    volume_component = (volume_clamped + 3) / 6 * 100

    # Weighted combination
    if price_correlation is not None:
        price_component = (price_correlation + 1) * 50
        index = (
            sentiment_component * 0.35 +
            ratio_component * 0.25 +
            volume_component * 0.20 +
            price_component * 0.20
        )
    else:
        index = (
            sentiment_component * 0.40 +
            ratio_component * 0.35 +
            volume_component * 0.25
        )

    return round(max(0, min(100, index)), 1)
```

### Acceptance Criteria
- [ ] All posts processed with VADER + custom lexicon
- [ ] Greentext posts flagged with reduced confidence
- [ ] DuckDB populated with sentiment scores
- [ ] Hourly/daily aggregation views working
- [ ] Fear/Greed index calculated correctly (validate against gut check)

---

## Phase 3: Image Sentiment Pipeline

### Goals
- Classify meme images for sentiment
- Detect Wojak/Pepe variants
- Extract text from memes via OCR
- Fuse image and text sentiment

### Tasks

#### 3.1 CLIP Zero-Shot Classification

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

class MemeClassifier:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.sentiment_labels = [
            "a meme showing financial success and gains",
            "a meme showing financial loss and despair",
            "a neutral meme about cryptocurrency",
            "a celebration meme",
            "a sad or crying meme",
        ]

    def classify(self, image: Image.Image) -> dict:
        inputs = self.processor(
            text=self.sentiment_labels,
            images=image,
            return_tensors="pt",
            padding=True
        )

        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

        # Map to sentiment score
        bullish_prob = probs[0].item() + probs[3].item()
        bearish_prob = probs[1].item() + probs[4].item()

        sentiment = bullish_prob - bearish_prob
        confidence = max(bullish_prob, bearish_prob)

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'method': 'clip'
        }
```

#### 3.2 Wojak/Pepe Detection with YOLOv8

```python
from ultralytics import YOLO

class WojakDetector:
    # Sentiment mapping for detected variants
    VARIANT_SENTIMENT = {
        'pink_wojak': -0.8,
        'green_wojak': 0.7,
        'doomer': -0.6,
        'bloomer': 0.6,
        'gigachad': 0.5,
        'crying_pepe': -0.7,
        'smug_pepe': 0.3,
        'sad_pepe': -0.5,
    }

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, image_path: str) -> list[dict]:
        results = self.model(image_path)
        detections = []

        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                if class_name in self.VARIANT_SENTIMENT:
                    detections.append({
                        'variant': class_name,
                        'confidence': float(box.conf),
                        'sentiment': self.VARIANT_SENTIMENT[class_name],
                        'bbox': box.xyxy[0].tolist()
                    })

        return detections
```

#### 3.3 PaddleOCR Integration

```python
from paddleocr import PaddleOCR

class MemeOCR:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def extract_text(self, image_path: str) -> str:
        result = self.ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            return ""

        # Concatenate all detected text
        texts = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.5:
                texts.append(text)

        return ' '.join(texts)
```

#### 3.4 Sentiment Fusion

```python
def fuse_sentiment(
    text_sentiment: dict | None,
    image_sentiment: dict | None,
    text_weight: float = 0.6,
    image_weight: float = 0.4
) -> dict:
    """Fuse text and image sentiment with weighted combination."""

    if text_sentiment is None and image_sentiment is None:
        return {'sentiment': 0, 'confidence': 0, 'method': 'none'}

    if text_sentiment is None:
        return {**image_sentiment, 'method': 'image_only'}

    if image_sentiment is None:
        return {**text_sentiment, 'method': 'text_only'}

    # Weighted fusion
    text_score = text_sentiment['sentiment'] * text_sentiment['confidence']
    image_score = image_sentiment['sentiment'] * image_sentiment['confidence']

    total_weight = (
        text_weight * text_sentiment['confidence'] +
        image_weight * image_sentiment['confidence']
    )

    if total_weight == 0:
        return {'sentiment': 0, 'confidence': 0, 'method': 'fused'}

    fused_score = (
        text_weight * text_score + image_weight * image_score
    ) / total_weight

    fused_confidence = (
        text_weight * text_sentiment['confidence'] +
        image_weight * image_sentiment['confidence']
    )

    return {
        'sentiment': fused_score,
        'confidence': fused_confidence,
        'method': 'fused',
        'text_contribution': text_weight * text_sentiment['confidence'] / total_weight,
        'image_contribution': image_weight * image_sentiment['confidence'] / total_weight
    }
```

### Acceptance Criteria
- [ ] CLIP classifies memes with reasonable accuracy
- [ ] YOLOv8 model trained on ~1000 labeled Wojak/Pepe images
- [ ] OCR extracts meme text with 70%+ accuracy
- [ ] Fusion produces sensible combined scores
- [ ] Pipeline processes 100 images/minute

---

## Phase 4: Claude Integration & Visualization

### Goals
- Integrate Claude Batch API for ambiguous posts
- Implement prompt caching for cost optimization
- Build Streamlit multi-page dashboard
- Add CoinGecko price correlation

### Tasks

#### 4.1 Claude Batch API Integration

```python
import anthropic
from anthropic.types import MessageBatch

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are analyzing 4chan /biz/ posts for market sentiment.

Context about /biz/ culture:
- WAGMI/NGMI may be used ironically, especially during market downturns
- Greentext (lines starting with >) often indicates sarcasm or storytelling
- Pink Wojak = financial loss, Green Wojak = gains
- "It's so over" can be ironic copium

Rate sentiment from -1.0 (very bearish) to +1.0 (very bullish).
Also provide a confidence score from 0.0 to 1.0.

Respond in JSON format:
{"sentiment": <float>, "confidence": <float>, "reasoning": "<brief explanation>"}
"""

async def analyze_batch(posts: list[dict]) -> list[dict]:
    """Analyze posts using Claude Batch API with prompt caching."""

    requests = []
    for i, post in enumerate(posts):
        requests.append({
            "custom_id": f"post-{post['post_id']}",
            "params": {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 200,
                "system": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [
                    {"role": "user", "content": f"Analyze this post:\n\n{post['text']}"}
                ]
            }
        })

    # Create batch
    batch = client.messages.batches.create(requests=requests)

    # Poll for completion
    while batch.processing_status == "in_progress":
        await asyncio.sleep(60)
        batch = client.messages.batches.retrieve(batch.id)

    # Collect results
    results = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            response = json.loads(result.result.message.content[0].text)
            results.append({
                'post_id': int(result.custom_id.split('-')[1]),
                **response
            })

    return results
```

#### 4.2 Streamlit Dashboard

The dashboard is implemented as a multi-page Streamlit app in `python-ml/src/dashboard.py`.

**Dashboard Pages:**

1. **Main Dashboard** - Fear/Greed gauge using Plotly gauge chart
   - Real-time index value with color coding
   - Bullish/bearish/neutral composition
   - Post volume and thread counts

2. **Coin Explorer** - Search and analyze individual coins
   - Time-series sentiment charts
   - Drill-down to actual posts
   - Price correlation overlay

3. **Historical Analysis** - Multi-timeframe charts
   - Fear/Greed history
   - Volume trends
   - Regime detection

4. **Post Browser** - View raw posts
   - Filter by coin, sentiment, time
   - See images inline
   - Validate sentiment scores

5. **Data Explorer** - SQL query interface
   - Pre-built query templates
   - CSV/JSON export
   - Direct DuckDB access

6. **System Health** - Monitoring
   - Scraper status
   - Database sizes
   - Processing metrics

**Running the dashboard:**
```bash
cd python-ml
streamlit run src/dashboard.py
# Opens at http://localhost:8501
```

#### 4.3 CoinGecko Price Integration

```python
import httpx

class CoinGeckoClient:
    BASE_URL = "https://api.coingecko.com/api/v3"

    async def get_price_history(
        self,
        coin_id: str,
        days: int = 30
    ) -> list[tuple[int, float]]:
        """Get historical price data."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/coins/{coin_id}/market_chart",
                params={"vs_currency": "usd", "days": days}
            )
            data = response.json()
            return data["prices"]  # [[timestamp_ms, price], ...]

    async def get_coin_list(self) -> dict[str, str]:
        """Get mapping of symbols to CoinGecko IDs."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/coins/list")
            coins = response.json()
            return {c["symbol"].upper(): c["id"] for c in coins}

def calculate_sentiment_price_correlation(
    sentiment_series: list[tuple[datetime, float]],
    price_series: list[tuple[datetime, float]],
    lag_hours: int = 0
) -> float:
    """Calculate Pearson correlation between sentiment and price."""
    import numpy as np

    # Align time series
    # ... alignment logic ...

    return np.corrcoef(sentiment_values, price_values)[0, 1]
```

### Acceptance Criteria
- [ ] Claude Batch API processing 1000+ posts/batch
- [ ] Prompt caching reduces costs by 90%+
- [ ] Grafana dashboard shows Fear/Greed gauge
- [ ] Historical sentiment charts working
- [ ] Per-coin sentiment heatmap functional
- [ ] Price correlation calculated and displayed

---

## Phase 5: Advanced Features (Future)

### Real-Time Alerts
```python
async def check_sentiment_shift():
    """Alert on dramatic sentiment changes."""
    current = await get_current_sentiment()
    previous = await get_sentiment_hours_ago(1)

    shift = abs(current.fear_greed - previous.fear_greed)

    if shift > 15:  # 15+ point swing
        await send_alert(
            f"Sentiment shift detected: {previous.fear_greed} → {current.fear_greed}"
        )
```

### API Endpoint
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/v1/sentiment")
async def get_sentiment():
    """Get current market sentiment."""
    return await get_latest_market_sentiment()

@app.get("/api/v1/coins/{symbol}/sentiment")
async def get_coin_sentiment(symbol: str):
    """Get sentiment for specific coin."""
    return await get_coin_sentiment_history(symbol)
```

### ML Refinements
- Train custom sarcasm detection model on /biz/ data
- A/B test different sentiment fusion weights
- Implement feedback loop from price predictions

---

## Timeline Summary

| Phase | Duration | Milestone |
|-------|----------|-----------|
| 1 | Foundational | Scraper running, data flowing |
| 2 | Core | Text sentiment baseline working |
| 3 | Enhanced | Image pipeline integrated |
| 4 | Complete | Full dashboard operational |
| 5 | Ongoing | Continuous improvements |

## Success Metrics

- **Data Quality:** 95%+ posts captured, <1% duplicates
- **Sentiment Accuracy:** Manual validation on 100 posts shows 80%+ agreement
- **Performance:** <5 minute lag from post to sentiment score
- **Cost:** <$2/day Claude API costs with caching
- **Availability:** Dashboard accessible 99%+ of time

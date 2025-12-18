# Building BizCharts: Technical architecture for 4chan /biz/ market analysis

A robust /biz/ sentiment analysis tool should use a **hybrid Rust-Python architecture**: Rust's reqwest for reliable scraping with strict rate limiting, Python for ML pipelines leveraging Claude's Batch API with prompt caching for **up to 95% cost reduction**, and DuckDB for high-performance time-series analytics. The 4chan JSON API provides all necessary data—thread text, timestamps, image URLs, and metadata—at 1 request/second with `If-Modified-Since` headers for efficient polling. For meme sentiment, CLIP-based zero-shot screening combined with fine-tuned EfficientNet classifiers can detect Wojak/Pepe variants with known sentiment associations (pink Wojak = loss, green Wojak = gains), while custom lexicons handle /biz/-specific slang like WAGMI, NGMI, and "we're so back."

## The 4chan API enables efficient real-time board monitoring

The official 4chan read-only JSON API at `a.4cdn.org` provides everything needed for sentiment analysis without scraping complexity. The catalog endpoint (`/biz/catalog.json`) returns all threads with OPs and preview replies in a single request—ideal for initial snapshots. Individual threads at `/biz/thread/[no].json` include full post text (in the `com` field as HTML), timestamps, and image metadata including MD5 hashes for deduplication.

**Rate limiting is non-negotiable**: the API enforces a strict maximum of 1 request per second, with thread updates requiring minimum 10-second intervals. The critical optimization is using `If-Modified-Since` headers—the server returns 304 Not Modified when nothing has changed, dramatically reducing bandwidth and preventing IP blocks. Violating these limits results in bans, so any production system must implement token bucket rate limiting.

For Rust, the **dot4ch** crate enforces these limits automatically and provides typed access to all endpoints. Python developers should use **BASC-py4chan**, which handles caching and implements proper If-Modified-Since logic. Both approaches are vastly superior to web scraping since the API provides structured JSON with exact text, eliminating OCR inaccuracies entirely. Images live on `i.4cdn.org`—thumbnails at `[tim]s.jpg` (small, fast) and full images at `[tim][ext]` for detailed meme analysis.

## Meme image analysis combines detection, classification, and OCR

Analyzing /biz/ meme sentiment requires a multi-stage pipeline because images carry significant signal. Research on meme classification shows that CLIP-based models excel at zero-shot screening—you can classify memes as bullish/bearish without training data by comparing image embeddings against text prompts like "a wojak meme showing financial loss" versus "a celebration meme about crypto gains."

For character detection, **YOLOv8 fine-tuned on Wojak/Pepe variants** identifies specific meme characters and their bounding boxes. The Roboflow Universe contains a "Meme Detection 2" dataset with 294 images across 7 classes including Pepe, Wojak, and Doge—a starting point for custom training. Once detected, a secondary **EfficientNet-B0 classifier** categorizes variants:

| Wojak Variant | Visual Indicators | Sentiment Signal |
|--------------|-------------------|------------------|
| Pink Wojak | Pink/red skin tone | Strong bearish (-0.8) |
| Green Wojak | Green skin tone | Strong bullish (+0.7) |
| Doomer | Black hoodie, cigarette | Pessimism (-0.6) |
| Bloomer | Smiling, bright colors | Optimism (+0.6) |
| Gigachad | Ultra-masculine | Success/confidence (+0.5) |

Text overlays in memes require OCR extraction. **PaddleOCR** achieves the highest accuracy (~80% average) on varied fonts and works well with meme text. The extracted text then feeds into the text sentiment pipeline, with final scores combining visual (40% weight) and textual (60% weight) signals via late fusion.

For complex or ambiguous memes, Claude's vision API can interpret context that rule-based systems miss—understanding irony, cultural references, and combined image-text meaning. At approximately **$0.001-0.003 per image**, this is cost-effective for a subset of unclear cases rather than every image.

## Text sentiment requires domain-specific vocabulary and sarcasm awareness

Standard sentiment tools fail catastrophically on /biz/ text because they don't understand crypto-native vocabulary. A custom lexicon mapping slang to sentiment weights is essential:

| Bullish Terms | Bearish Terms | Neutral/Context-Dependent |
|--------------|---------------|---------------------------|
| WAGMI (+3.5) | NGMI (-3.5) | DYOR (0) |
| LFG (+3.0) | "It's over" (-3.5) | NFA (0) |
| Moon (+3.0) | Rekt (-3.5) | Cope (-2.0, often ironic) |
| Diamond hands (+2.5) | Rugged (-4.0) | Hopium (-2.0) |
| "We're so back" (+2.5) | Paper hands (-2.5) | Whale (context-dependent) |

The critical challenge is **sarcasm and irony**. During bear markets, "WAGMI" is frequently used ironically while portfolios collapse. Research shows greentext format (lines starting with `>`) indicates narrative/ironic storytelling 70%+ of the time. A preprocessing pipeline must flag greentext posts for reduced sentiment confidence or sarcasm-aware processing.

For implementation, **VADER with custom lexicon additions** provides fast baseline sentiment (excellent for high-volume processing), while **fine-tuned RoBERTa** handles nuanced cases. Academic research demonstrates that BERT-based models achieve ~91.7% accuracy on Reddit sarcasm detection when including context from parent posts. Claude API excels at interpreting complex ironic usage through prompts that explicitly acknowledge /biz/ cultural norms:

```
Analyze this 4chan /biz/ post's sentiment. Consider that:
- WAGMI/NGMI may be used ironically, especially in bear markets
- Greentext (lines starting with >) often indicates sarcasm
- Rate sentiment from -1 (very bearish) to +1 (very bullish) with confidence score
```

## Rust for reliability, Python for ML creates the optimal hybrid stack

The language choice affects every layer of the system. For **scraping and data collection**, Rust with reqwest provides superior reliability—compile-time guarantees prevent runtime crashes, async Tokio handles thousands of concurrent connections efficiently, and the resulting binary deploys without runtime dependencies. Python's aiohttp is 10-20% slower and the GIL limits CPU-bound processing, though it's adequate for prototyping.

For **ML and sentiment analysis**, Python remains dominant. The transformers ecosystem has thousands of pre-trained models, including FinBERT for financial text and crypto-specific fine-tuned variants. Training in Rust is impractical—tch-rs requires the full 2GB libtorch runtime and offers minimal ecosystem support. The winning pattern: **train and experiment in Python, deploy inference via ONNX Runtime** for 3-5x speedup and 60-80% memory reduction.

**Polars** deserves special mention as the data processing layer—it's available in both Rust and Python with identical APIs, runs **5-100x faster than Pandas** through multi-threaded parallel processing, and uses 2-4x less memory. For time-series sentiment aggregation across millions of posts, this performance difference is substantial.

For **databases**, the research strongly favors a dual-store approach:

- **SQLite** for operational storage (raw scraped posts, fast lookups)
- **DuckDB** for analytical queries (sentiment aggregations, historical analysis)

DuckDB is 10-100x faster than SQLite for analytical queries while remaining a single-file embedded database requiring zero setup. TimescaleDB or QuestDB are overkill for a local tool unless you need sub-second streaming ingestion at massive scale.

## Claude API integration with batch processing and caching minimizes costs

Claude's Batch API offers **50% cost reduction** on all models by processing up to 100,000 requests per batch with a 24-hour completion window (most finish within an hour). For historical sentiment analysis—processing days or weeks of /biz/ posts—this is the clear choice over real-time API calls.

**Prompt caching** stacks with batch discounts for up to 95% savings. The technique works by including a static system prompt (sentiment guidelines, vocabulary definitions, output format specifications) with `cache_control: {"type": "ephemeral"}`, then varying only the post content. Cached reads cost 90% less than uncached, and the cache persists for 5 minutes by default.

Cost projection for processing **10,000 posts daily** with Claude Haiku 3.5:
- Regular API: ~$10/day  
- Batch API (50% off): ~$5/day  
- Batch + Caching (95% off): **~$0.50-2/day**

For real-time alerts on breaking sentiment shifts, use the standard API with Haiku for speed and cost efficiency. Reserve Sonnet or Opus for complex interpretations where nuance matters—like determining whether a highly-upvoted post is genuine bullishness or coordinated pump rhetoric.

## Existing tools validate the approach and reveal implementation patterns

The Alternative.me Fear & Greed Index demonstrates that multi-signal aggregation works for crypto sentiment. Their methodology combines volatility (25%), market momentum (25%), social media (15%), surveys (15%), Bitcoin dominance (10%), and Google Trends (10%)—weighted factors that smooth noise and capture genuine market psychology. A /biz/-specific index could adapt this framework: text sentiment, meme sentiment, post volume, thread engagement, and keyword frequency as weighted components.

Several GitHub projects have tackled /biz/ analysis directly. **timurscode/4chan-Market-Analysis-Tool** provides a Python GUI using TextBlob and NLTK for sentiment, integrating CoinGecko for price correlation. **distrill/4parse** explicitly targets /biz/ for sentiment analysis. The academic paper "Kek, Cucks, and God Emperor Trump" found that ~84% of 4chan posts are neutral or negative—suggesting calibration is needed to avoid pessimistic bias in raw aggregations.

LunarCrush and Santiment demonstrate the value of **spam filtering**—The TIE reports that 90%+ of raw crypto Twitter is spam or manipulation. Any robust system must filter noise aggressively before sentiment scoring.

## Recommended architecture balances performance with development velocity

```
┌────────────────────────────────────────────────────────────────────────┐
│                         BizCharts Architecture                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐         ┌──────────────────┐                      │
│  │  Rust Scraper   │────────▶│  SQLite (raw)    │                      │
│  │  (dot4ch/reqwest)│         │  - posts, images │                      │
│  │  1 req/sec      │         │  - timestamps    │                      │
│  └─────────────────┘         └────────┬─────────┘                      │
│                                       │                                 │
│         ┌─────────────────────────────┴─────────────────────┐          │
│         ▼                                                    ▼          │
│  ┌─────────────────┐                              ┌─────────────────┐  │
│  │  Python ML Svc  │                              │  Image Pipeline │  │
│  │  - Text preproc │                              │  - CLIP screen  │  │
│  │  - VADER+lexicon│                              │  - YOLOv8 detect│  │
│  │  - Claude batch │                              │  - PaddleOCR    │  │
│  └────────┬────────┘                              └────────┬────────┘  │
│           │                                                 │           │
│           └──────────────────┬──────────────────────────────┘           │
│                              ▼                                          │
│                    ┌─────────────────┐                                  │
│                    │  DuckDB (OLAP)  │                                  │
│                    │  - aggregations │                                  │
│                    │  - time-series  │                                  │
│                    └────────┬────────┘                                  │
│                             │                                           │
│                             ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Grafana + Plotly Dashboard                                       │  │
│  │  - Fear/Greed gauge  - Sentiment timeline  - Crypto correlation  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

For a developer preferring Rust, write the scraping engine, rate limiter, and data storage layer in Rust using dot4ch, reqwest, and rusqlite. This core will be rock-solid and deployable as a single binary. The ML components—sentiment analysis, meme classification, Claude API integration—should remain in Python where the ecosystem is vastly richer. Communication between components can use simple SQLite as a shared data store (both languages have excellent SQLite bindings) or REST endpoints via Actix-web (Rust) calling Python FastAPI services.

## Concrete implementation roadmap for a working prototype

**Phase 1 (Week 1-2): Data collection foundation**
- Implement Rust scraper with dot4ch, polling `/biz/catalog.json` every 60 seconds
- Store raw posts in SQLite with schema: `post_id, thread_id, timestamp, text, has_image, image_url, image_md5`
- Implement If-Modified-Since caching and token bucket rate limiter
- Download thumbnails for posts with images

**Phase 2 (Week 3-4): Text sentiment baseline**
- Python service reading from SQLite
- Preprocess pipeline: HTML decode, strip greentext, normalize tickers, expand slang
- VADER with custom /biz/ lexicon for fast scoring
- Store sentiment scores in DuckDB with minute/hour/day aggregation views

**Phase 3 (Week 5-6): Image sentiment pipeline**
- CLIP zero-shot classification for bullish/bearish/neutral screening
- Fine-tune EfficientNet on manually labeled Wojak/Pepe dataset (~1000 images)
- PaddleOCR for text extraction from memes
- Fuse image + text sentiment scores (40/60 weighting)

**Phase 4 (Week 7-8): Claude integration and visualization**
- Batch API integration for nuanced sentiment on ambiguous posts
- Prompt caching for cost optimization
- Grafana dashboard with fear/greed gauge, time-series charts
- Correlation analysis with BTC/ETH price data via CoinGecko API

## Conclusion and key technical decisions

Building a /biz/ sentiment analysis tool is technically feasible with current tools and APIs. The 4chan API provides structured access to all necessary data without scraping fragility. The hybrid Rust-Python architecture maximizes reliability (Rust scraping) and ML capability (Python analysis) while Polars and DuckDB deliver analytical performance typically reserved for dedicated data warehouses.

The meme analysis problem is harder than text sentiment—Wojak/Pepe detection requires custom training data and the cultural context of 4chan memes evolves rapidly. Start with text-only sentiment using the custom lexicon approach, then layer in image analysis once the foundation is solid. Claude's vision API serves as a powerful fallback for complex cases where rule-based systems fail.

For cost management, the combination of Claude Batch API and prompt caching transforms what could be $300/month in API costs into under $20/month for typical usage patterns. This makes sophisticated LLM-powered sentiment analysis economically viable for a personal tool.

The existing ecosystem—from Alternative.me's methodology to LunarCrush's filtering approaches to academic 4chan research—provides tested patterns to adopt. The technical risk is manageable; the challenge is building sufficient labeled training data and iteratively improving the custom lexicon as /biz/ slang evolves.
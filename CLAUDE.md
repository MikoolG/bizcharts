# BizCharts

4chan /biz/ sentiment analysis platform producing Fear/Greed indices and per-coin sentiment tracking.

## Current Status (Dec 2025)

| Component | Status | Notes |
|-----------|--------|-------|
| Live Scraper | ✅ Complete | Pulls ~150 threads from 4chan catalog, downloads thumbnails |
| Warosu Importer | ✅ Complete | Date-range search, captures OPs with subjects/images/text |
| Labeling GUI | ✅ Complete | 1,322 posts labeled (1-3 scale), supports active learning mode |
| Text Sentiment | ✅ Complete | VADER + lexicon, optional ML ensemble (SetFit/CryptoBERT) |
| ML Training | ✅ Complete | SetFit, CryptoBERT, LLaVA wrappers + RunPod scripts |
| Active Learning | ✅ Complete | Hybrid uncertainty-diversity sampling |
| Image Sentiment | 🔄 Pending | LLaVA wrapper ready, integration pending |
| Dashboard | 🔄 Pending | Streamlit UI |

## Architecture

**OP-focused design**: Scrapes catalog only (not individual threads). One catalog request = ~150 thread OPs with full text, images, and engagement metrics.

```
Data Collection (Rust)              Analysis (Python)
┌─────────────────────┐            ┌─────────────────────┐
│ Live Scraper        │            │ Text: VADER+lexicon │
│ - catalog.json/60s  │───────────▶│ + SetFit/CryptoBERT │
│ - 1 req/sec limit   │  SQLite    │ Image: LLaVA fusion │
├─────────────────────┤  (shared)  ├─────────────────────┤
│ Warosu Importer     │            │ Labeling UI         │
│ - Historical data   │───────────▶│ + Active Learning   │
├─────────────────────┤            ├─────────────────────┤
│ Maintenance         │            │ ML Training         │
│ - Retention/cleanup │            │ - RunPod scripts    │
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
│       ├── text_analyzer.py   # VADER + custom lexicon + ensemble
│       ├── labeler.py         # Manual labeling GUI + active learning
│       ├── models/            # ML model wrappers
│       │   ├── setfit_model.py    # SetFit few-shot learning
│       │   ├── cryptobert_model.py # CryptoBERT with LoRA
│       │   ├── llava_model.py     # LLaVA 7B for memes
│       │   └── qwen_model.py      # Qwen2.5-VL-7B (recommended for multimodal)
│       ├── training/          # Training pipelines
│       │   ├── data_loader.py     # Load from SQLite
│       │   ├── setfit_trainer.py  # SetFit training
│       │   ├── auto_labeler.py    # GPT-4o-mini/Gemini API labeling
│       │   ├── prepare_training_data.py  # Convert labels to ShareGPT format
│       │   └── runpod/            # Cloud GPU scripts
│       │       ├── train_qwen.py  # Qwen2.5-VL QLoRA training
│       │       └── train_setfit.py
│       ├── active_learning/   # Uncertainty-diversity sampling
│       ├── continual/         # Replay buffer, drift detection
│       ├── fusion/            # Multi-modal sarcasm detection
│       └── inference/         # Batch pipeline, ONNX export
├── config/
│   └── settings.toml        # All configuration (incl. [ml.*] sections)
├── docs/
│   ├── self-training.md       # ML architecture & research
│   ├── labeling-guide.md      # Manual labeling + active learning
│   └── plans/                 # Implementation plans
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

# Active learning (prioritize uncertain posts)
python3 -m src.labeler --active-learning --model models/setfit

# ML Training
python3 -m src.training.setfit_trainer --db ../data/posts.db --output models/setfit
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
training_labels (thread_id, sentiment_rating 1-3, labeler_id)  -- 1=bearish, 2=neutral, 3=bullish
catalog_snapshots (snapshot_at, total_threads, avg_reply_count, top_coins)
```

## Current Labeled Dataset

**1,322 manually labeled posts** for training/evaluation:

| Period | Posts | Market Condition |
|--------|-------|------------------|
| Dec 15-19, 2025 | 822 | Bearish |
| Oct 7-10, 2025 | 500 | Bullish |

**Label distribution:**
- Bearish: 453 (34.3%)
- Neutral: 631 (47.7%)
- Bullish: 238 (18.0%)

Backups stored in `rust-scraper/backups/` (not in git).

## Sentiment Strategy

See [docs/sentiment-strategy.md](docs/sentiment-strategy.md) for full details.
See [docs/self-training.md](docs/self-training.md) for ML architecture.

**Key concepts:**
- **Multi-signal fusion**: Text (50%) + Image (30%) + OCR (10%) + Context (10%)
- **Ensemble mode**: VADER (30%) + ML models (70%) when enabled
- **Confidence weighting**: Uncertain posts get less weight in aggregations
- **Reply-based importance**: Popular threads (high reply count) influence aggregate more
- **Sarcasm detection**: Text-image incongruity signals ironic content

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
| [docs/training-flow.md](docs/training-flow.md) | **Complete training walkthrough (start here)** |
| [docs/automated-training.md](docs/automated-training.md) | Research decisions and pipeline rationale |
| [docs/labeling-guide.md](docs/labeling-guide.md) | Manual labeling + active learning |
| [docs/plans/implementation-roadmap.md](docs/plans/implementation-roadmap.md) | Project phases and progress |
| [config/settings.toml](config/settings.toml) | All configuration (incl. `[ml.*]` sections) |

## Multimodal Sentiment Training

**Core principle**: Sentiment is ALWAYS determined from image + text together, never text alone.

### Training Pipeline Overview

```
Phase 1: Data Collection (Local)
┌─────────────────────────────────────────────────────────────┐
│ 1. Scrape 10,000+ posts with images (Rust scraper)          │
│ 2. Manual labeling: 500-1000 posts as TEST SET (ground truth)│
│    - This is your held-out evaluation set                   │
│    - Used to measure final model accuracy                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
Phase 2: Auto-Labeling (API, ~$0-2.50)
┌─────────────────────────────────────────────────────────────┐
│ Use vision API to label 10k+ posts for TRAINING data        │
│ Options:                                                    │
│   - Gemini 2.0 Flash: FREE (rate limited, ~7 days for 10k) │
│   - GPT-4o-mini: ~$1.70-2.50 (fast, ~1-2 hours)            │
│ Input: image + text → Output: bearish/neutral/bullish       │
│ Store labels locally as JSON for training                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
Phase 3: Fine-tuning (RunPod, ~$2-5)
┌─────────────────────────────────────────────────────────────┐
│ Fine-tune Qwen2.5-VL-7B with QLoRA on auto-labeled data     │
│ - LoRA config: r=16, alpha=32, target q/v projections       │
│ - 1-3 epochs, ~3-4 hours on RTX 4090                        │
│ - Evaluate on manual test set (Phase 1 labels)              │
│ Output: LoRA adapter (~100-500MB) you own                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
Phase 4: Production Inference (RunPod Serverless)
┌─────────────────────────────────────────────────────────────┐
│ Deploy trained model for live sentiment analysis            │
│ - RunPod Serverless: ~$0.002-0.01 per image                │
│ - Process 200 catalog OPs: ~$0.40-2.00 per batch           │
│ - Monthly cost (hourly checks): ~$3-10                      │
└─────────────────────────────────────────────────────────────┘
```

### Why This Pipeline?

**Why API for labeling instead of Qwen2-VL on RunPod?**
- GPT-4o-mini/Gemini: 80-83% label quality, ~$0-2.50 for 10k
- Qwen2-VL on RunPod: 55-65% label quality, ~$2-14 for 10k
- API is both **cheaper AND higher quality**

**Why Qwen2.5-VL-7B for training?**
- Best OCR for meme text overlays
- Strong fine-tuning ecosystem (TRL, LLaMA-Factory)
- Documented 48% → 66%+ accuracy improvement with LoRA
- Fits on RTX 4090 (24GB) with QLoRA

### Model Choice: Qwen2.5-VL-7B

| Model | VRAM (QLoRA) | Why |
|-------|--------------|-----|
| **Qwen2.5-VL-7B** | ~18-24GB | Best OCR, strong benchmarks, excellent fine-tuning support |
| LLaVA-1.5-7B | ~12-15GB | Alternative, LLaVAC paper shows 79-83% on sentiment |

### Auto-Labeling Options

| API | Cost (10k images) | Quality | Speed |
|-----|-------------------|---------|-------|
| **Gemini 2.0 Flash** (free tier) | $0 | Good (70-78%) | ~7 days (rate limited) |
| **GPT-4o-mini** | ~$1.70-2.50 | Very good (80-83%) | ~1-2 hours |

Use Gemini free tier if not in a hurry. Use GPT-4o-mini for faster turnaround.

### Expected Accuracy

| Stage | Accuracy |
|-------|----------|
| Text-only baseline (CryptoBERT) | 40% |
| Zero-shot Qwen2.5-VL | 55-65% |
| Fine-tuned Qwen2.5-VL | 70-80% |
| With self-training iteration | 75-85% |

### Cost Summary

| Phase | Cost | One-time? |
|-------|------|-----------|
| Manual labeling (500-1000) | Your time | Yes |
| Auto-labeling (10k+) | $0-2.50 | Yes |
| QLoRA training | $2-5 | Yes (retrain as needed) |
| **Total one-time** | **~$5-10** | |
| Production inference | $3-10/month | Ongoing |

### Dataset Format (ShareGPT style)

```json
[
  {
    "messages": [
      {"role": "user", "content": "<image>\nClassify this 4chan /biz/ crypto post as bearish, neutral, or bullish. Consider both the image and text together."},
      {"role": "assistant", "content": "bearish"}
    ],
    "images": ["/data/images/thread_12345.jpg"]
  }
]
```

### Training Resources

- [Qwen2.5-VL Fine-tuning Guide](https://datature.io/blog/how-to-fine-tune-qwen2-5-vl)
- [TRL VLM Fine-tuning Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)
- [Qwen-VL-Finetune Repository](https://github.com/2U1/Qwen2-VL-Finetune)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - supports Qwen2.5-VL

### Training Workflow Commands

**Step 1: Auto-label posts using vision API (run locally or on any machine)**
```bash
cd python-ml

# Using GPT-4o-mini (fast, ~$2 for 10k)
OPENAI_API_KEY=sk-xxx python -m src.training.auto_labeler \
    --db ../data/posts.db \
    --api openai \
    --output ../data/auto_labels.json \
    --rate-limit 0.5

# Using Gemini 2.0 Flash (free tier, slower)
GEMINI_API_KEY=xxx python -m src.training.auto_labeler \
    --db ../data/posts.db \
    --api gemini \
    --output ../data/auto_labels.json \
    --rate-limit 2.0
```

**Step 2: Prepare training data (run locally)**
```bash
cd python-ml

# Download images and convert labels to ShareGPT format
python -m src.training.prepare_training_data \
    --labels ../data/auto_labels.json \
    --db ../data/posts.db \
    --images ../data/images \
    --output ../data/training_data.json \
    --val-output ../data/val_data.json
```

**Step 3: Push to RunPod**
```bash
# Push training data and images
scp -P $PORT -i ~/.ssh/runpod_bizcharts \
    data/training_data.json data/val_data.json \
    root@$IP:/workspace/data/

scp -P $PORT -i ~/.ssh/runpod_bizcharts -r \
    data/images/ \
    root@$IP:/workspace/data/

# Push training script
scp -P $PORT -i ~/.ssh/runpod_bizcharts \
    python-ml/src/training/runpod/train_qwen.py \
    root@$IP:/workspace/scripts/
```

**Step 4: Train on RunPod (GPU required)**
```bash
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "
  source /workspace/venv/bin/activate
  pip install qwen-vl-utils trl peft bitsandbytes accelerate
  python /workspace/scripts/train_qwen.py \
    --dataset /workspace/data/training_data.json \
    --images /workspace/data/images \
    --output /workspace/models/qwen-sentiment \
    --epochs 3
"
```

**Step 5: Pull trained model**
```bash
scp -P $PORT -i ~/.ssh/runpod_bizcharts -r \
    "root@$IP:/workspace/models/qwen-sentiment" \
    python-ml/models/
```

---

## RunPod Training Environment

### Core Philosophy: Local = Data/Config, RunPod = Execution

| Responsibility | Local | RunPod |
|----------------|-------|--------|
| **Database & Labels** | ✅ Store here | Push for training |
| **Training Config** | ✅ Edit here | Push from local |
| **Code Changes** | ✅ Edit here | Push from local |
| **Model Training** | — | ✅ Runs here |
| **Trained Models** | Pull after training | ✅ Output here |

**Why:** RunPod pods can be stopped/terminated. Local persists. Edit locally, execute remotely.

### RunPod Storage Configuration

**Recommended pod setup:**
- **Container Disk:** 20 GB (OS, temp files)
- **Volume Disk:** 25 GB for text models, 35 GB if using LLaVA (persistent `/workspace`)
- **GPU:** RTX 4090 ($0.69/hr) for fast training, RTX 3090 ($0.44/hr) for budget

**Storage breakdown:**
- Python venv (PyTorch, transformers): ~10 GB
- Base models (SetFit + CryptoBERT): ~1 GB
- LLaVA 7B 4-bit (optional): ~4 GB
- Trained outputs + database: ~1 GB

With Volume Disk configured:
- **STOP** → Data persists, GPU released, ~$5/month storage cost
- **TERMINATE** → All data erased

**Note:** IP and PORT change with each new pod. Check RunPod dashboard for current values.

### SSH Key Setup

Create a dedicated SSH key for RunPod:

```bash
# Generate key (one-time)
ssh-keygen -t ed25519 -f ~/.ssh/runpod_bizcharts -C "bizcharts-runpod"

# Add public key to RunPod dashboard: Settings > SSH Public Keys
cat ~/.ssh/runpod_bizcharts.pub
```

**Always use this key for connections:**
```bash
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP
scp -P $PORT -i ~/.ssh/runpod_bizcharts file root@$IP:/path/
```

### Connection Setup

```bash
# Set connection vars (from RunPod dashboard)
export IP="xxx.xxx.xxx.xxx"
export PORT="xxxxx"

# Test connection
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "nvidia-smi"
```

### CRITICAL: Package Persistence

**System packages do NOT persist across pod restarts.** Only `/workspace` survives.

**Solution:** Always use venvs inside `/workspace`:

```bash
# Create persistent venv (one-time setup)
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "
  python3 -m venv /workspace/venv
  source /workspace/venv/bin/activate
  pip install --upgrade pip
  pip install torch transformers setfit datasets accelerate peft bitsandbytes
"
```

**All training commands MUST activate the venv:**
```bash
source /workspace/venv/bin/activate && python train_setfit.py ...
```

### RunPod Directory Structure

```
/workspace/                      # PERSISTENT (survives STOP)
├── venv/                        # Python virtual environment
├── data/
│   └── posts.db                 # Training database (pushed from local)
├── models/                      # Trained model outputs
│   ├── setfit/
│   ├── cryptobert/
│   └── llava/
├── scripts/                     # Training scripts (pushed from local)
└── logs/                        # Training logs
```

### File Transfer Commands

**Push database for training:**
```bash
scp -P $PORT -i ~/.ssh/runpod_bizcharts \
  rust-scraper/data/posts.db \
  root@$IP:/workspace/data/
```

**Push training scripts:**
```bash
scp -P $PORT -i ~/.ssh/runpod_bizcharts \
  python-ml/src/training/runpod/*.py \
  root@$IP:/workspace/scripts/
```

**Pull trained models:**
```bash
scp -P $PORT -i ~/.ssh/runpod_bizcharts \
  "root@$IP:/workspace/models/setfit/*" \
  python-ml/models/setfit/
```

**Bulk transfer with tar (faster for many files):**
```bash
# Pack on RunPod
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP \
  "cd /workspace/models && tar -cvf /tmp/models.tar setfit/"

# Download
scp -P $PORT -i ~/.ssh/runpod_bizcharts \
  root@$IP:/tmp/models.tar ./

# Extract locally
tar -xvf models.tar -C python-ml/models/
```

### Training Workflow

**1. Push data and scripts:**
```bash
# Database
scp -P $PORT -i ~/.ssh/runpod_bizcharts rust-scraper/data/posts.db root@$IP:/workspace/data/

# Scripts
scp -P $PORT -i ~/.ssh/runpod_bizcharts python-ml/src/training/runpod/*.py root@$IP:/workspace/scripts/
```

**2. Run training:**
```bash
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "
  source /workspace/venv/bin/activate
  cd /workspace/scripts
  python train_setfit.py --db /workspace/data/posts.db --output /workspace/models/setfit
"
```

**3. Monitor progress:**
```bash
# Watch logs
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "tail -f /workspace/logs/training.log"

# Check GPU usage
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "nvidia-smi"
```

**4. Pull trained model:**
```bash
scp -P $PORT -i ~/.ssh/runpod_bizcharts \
  "root@$IP:/workspace/models/setfit/*" \
  python-ml/models/setfit/
```

### Background Operations

**Don't wait idle during long training runs.** Use background execution:

```bash
# Start training in background
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "
  source /workspace/venv/bin/activate
  nohup python /workspace/scripts/train_setfit.py \
    --db /workspace/data/posts.db \
    --output /workspace/models/setfit \
    &> /workspace/logs/training.log &
"

# Check progress periodically
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "tail -20 /workspace/logs/training.log"

# Check if still running
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "ps aux | grep train"
```

### Session End Checklist

**Before stopping/terminating a RunPod pod:**

1. **Pull trained models:**
```bash
scp -P $PORT -i ~/.ssh/runpod_bizcharts "root@$IP:/workspace/models/setfit/*" python-ml/models/setfit/
```

2. **Pull any logs you want to keep:**
```bash
scp -P $PORT -i ~/.ssh/runpod_bizcharts "root@$IP:/workspace/logs/*.log" ./training-logs/
```

3. **Verify transfer:**
```bash
ls -la python-ml/models/setfit/
```

**IMPORTANT:** With Volume Disk, STOP preserves `/workspace`. Without it, both STOP and TERMINATE erase everything.

### GitHub Releases for Trained Models

Trained model files can be large (100MB+). Use GitHub Releases to store them.

**Prerequisites:**
```bash
sudo apt install gh -y && gh auth login
```

**Create a release with trained model:**
```bash
# Tag the version
git tag setfit-v1 -m "SetFit model trained on 200 labels"
git push origin setfit-v1

# Create release with model files
gh release create setfit-v1 \
  python-ml/models/setfit/model.safetensors \
  python-ml/models/setfit/config.json \
  --title "SetFit Sentiment Model v1" \
  --notes "Trained on 200 labeled posts. Accuracy: 78%. F1: 0.75"
```

**Download model on new RunPod pod:**
```bash
gh auth login
gh release download setfit-v1 --repo YOUR_USERNAME/bizcharts \
  --pattern "*.safetensors" --dir /workspace/models/setfit/
```

### HuggingFace Model Downloads

Download base models directly on RunPod (faster than uploading):

```bash
# SetFit base model
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "
  source /workspace/venv/bin/activate
  python -c \"
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
model.save('/workspace/models/base/paraphrase-mpnet')
\"
"

# CryptoBERT
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "
  source /workspace/venv/bin/activate
  python -c \"
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('ElKulako/cryptobert')
tokenizer = AutoTokenizer.from_pretrained('ElKulako/cryptobert')
model.save_pretrained('/workspace/models/base/cryptobert')
tokenizer.save_pretrained('/workspace/models/base/cryptobert')
\"
"
```

### Cost Reference

| Task | GPU | Time | Cost |
|------|-----|------|------|
| SetFit (200 labels) | RTX 4090 | 5 min | ~$0.06 |
| CryptoBERT LoRA | RTX 4090 | 30 min | ~$0.35 |
| LLaVA QLoRA | RTX 4090 | 2-4 hr | ~$1.50-3.00 |
| Weekly retrain | RTX 3090 | 15 min | ~$0.10 |

**Monthly estimate:** ~$5-10 for regular retraining

# Training Flow: From Raw Data to Trained Model

Complete walkthrough for training a multimodal sentiment model for /biz/ crypto posts.

**End result:** A LoRA adapter for Qwen2.5-VL-7B that classifies crypto memes as bearish/neutral/bullish.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: Scrape Data        Step 2: Manual Label       Step 3: API Label│
│  ┌─────────────────┐        ┌─────────────────┐        ┌───────────────┐│
│  │ Rust scraper    │        │ 500-1000 posts  │        │ 10k+ posts    ││
│  │ 10k+ posts      │───────▶│ (TEST SET)      │───────▶│ (TRAIN SET)   ││
│  │ with images     │        │ Your labels     │        │ API labels    ││
│  └─────────────────┘        └─────────────────┘        └───────────────┘│
│                                                               │         │
│                                                               ▼         │
│                              Step 5: Pull Model    Step 4: Train on GPU │
│                              ┌─────────────────┐   ┌───────────────────┐│
│                              │ LoRA adapter    │◀──│ Qwen2.5-VL QLoRA  ││
│                              │ (~100-500MB)    │   │ RunPod RTX 4090   ││
│                              └─────────────────┘   └───────────────────┘│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- **Local machine**: Any computer (no GPU needed for steps 1-3)
- **RunPod account**: For GPU training (step 4)
- **API key**: OpenAI or Google (for auto-labeling)
- **Time estimate**:
  - Steps 1-3: 1-2 days (mostly waiting for scraping/labeling)
  - Step 4: 3-4 hours on GPU
  - Step 5: 10 minutes

---

## Step 1: Scrape Data

**Goal:** Collect 10,000+ posts with images from /biz/.

### 1.1 Live Catalog Scraping

```bash
cd rust-scraper

# Single snapshot (~150 threads)
cargo run --release -- --live --once

# Continuous scraping (run for a few days to build up data)
cargo run --release -- --live
```

### 1.2 Historical Data from Warosu

```bash
# Search for crypto-related threads
cargo run --release -- --warosu --search bitcoin --max 2000
cargo run --release -- --warosu --search ethereum --max 2000
cargo run --release -- --warosu --search crypto --max 2000

# Browse general activity
cargo run --release -- --warosu --pages 50

# Check coverage
cargo run --release -- --coverage
```

### 1.3 Verify Data

```bash
# Check how many posts we have
cargo run --release -- --stats
```

**Target:** 10,000+ posts with `thumbnail_url` populated.

---

## Step 2: Manual Labeling (Test Set)

**Goal:** Create 500-1000 human-labeled posts as ground truth for evaluation.

### 2.1 Start Labeling GUI

```bash
cd python-ml
source .venv/bin/activate

# Start labeling
python -m src.labeler --source all

# Or focus on high-engagement posts
python -m src.labeler --min-replies 10
```

### 2.2 Labeling Guidelines

For each post, look at **BOTH image AND text together**:

| Rating | Label | When to use |
|--------|-------|-------------|
| 1 | Bearish | Doom memes, pink wojaks, "it's over", price crashes, despair |
| 2 | Neutral/Irrelevant | News, questions, memes without clear sentiment, off-topic |
| 3 | Bullish | Green candles, happy pepes, "we're gonna make it", moon, euphoria |

**Important considerations:**
- Sarcasm is common - "this is fine" with fire = bearish
- Wojak variants: Pink = bearish, Green = bullish, Bog = manipulation
- Check text overlays on images
- When unsure, lean toward neutral (2)
- Use Skip (0/S) for broken images or unreadable posts

### 2.3 Check Progress

```bash
python -m src.labeler --stats
```

**Target:** 500-1000 labeled posts with balanced distribution.

**Tip:** Sample from different market periods (bullish AND bearish) to balance sentiment classes. If your initial dataset is skewed (e.g., 2:1 bearish:bullish during a crash), use Warosu date-range search to add posts from opposite market conditions:

```bash
# Add posts from a bullish period (Oct 2025 example)
cargo run --release -- --warosu --from 2025-10-07 --to 2025-10-10 --max 500
```

---

## Step 3: Auto-Labeling (Training Set)

**Goal:** Use vision API to label 10,000+ posts for training data.

### 3.1 Choose Your API

| API | Cost (10k) | Quality | Speed | Best for |
|-----|------------|---------|-------|----------|
| **Gemini 2.0 Flash** | $0 | 70-78% | ~7 days | Budget, not time-sensitive |
| **GPT-4o-mini** | ~$2 | 80-83% | ~2 hours | Best quality/speed balance |

### 3.2 Set Up API Key

**For OpenAI (GPT-4o-mini):**
```bash
# Get key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-proj-..."
```

**For Google (Gemini):**
```bash
# Get key from https://aistudio.google.com/apikey
export GEMINI_API_KEY="..."
```

### 3.3 Run Auto-Labeling

```bash
cd python-ml
source .venv/bin/activate

# Using GPT-4o-mini (recommended)
python -m src.training.auto_labeler \
    --db ../data/posts.db \
    --api openai \
    --output ../data/auto_labels.json \
    --rate-limit 0.5

# Using Gemini (free but slower)
python -m src.training.auto_labeler \
    --db ../data/posts.db \
    --api gemini \
    --output ../data/auto_labels.json \
    --rate-limit 2.0
```

**Options:**
- `--limit 1000` - Label only first N posts (for testing)
- `--resume` - Continue from where you left off
- `--batch-size 50` - Save checkpoint every N posts

### 3.4 Prepare Training Data

```bash
# Download images and convert to ShareGPT format
python -m src.training.prepare_training_data \
    --labels ../data/auto_labels.json \
    --db ../data/posts.db \
    --images ../data/images \
    --output ../data/training_data.json \
    --val-output ../data/val_data.json \
    --min-confidence 0.7
```

**Output files:**
- `training_data.json` - 90% of samples for training
- `val_data.json` - 10% for validation during training
- `images/` - Downloaded post images

---

## Step 4: Train on RunPod

**Goal:** Fine-tune Qwen2.5-VL-7B with QLoRA on your labeled data.

### 4.1 Create RunPod Pod

1. Go to [RunPod](https://runpod.io)
2. Create new pod:
   - **GPU:** RTX 4090 ($0.34-0.69/hr)
   - **Template:** PyTorch 2.x
   - **Container Disk:** 20 GB
   - **Volume Disk:** 50 GB (for model + data)

3. Get connection details from dashboard:
```bash
export IP="xxx.xxx.xxx.xxx"
export PORT="xxxxx"
```

### 4.2 Push Data to RunPod

```bash
# Create directories
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP "mkdir -p /workspace/{data,scripts,models}"

# Push training data
scp -P $PORT -i ~/.ssh/runpod_bizcharts \
    data/training_data.json data/val_data.json \
    root@$IP:/workspace/data/

# Push images (this may take a while)
scp -P $PORT -i ~/.ssh/runpod_bizcharts -r \
    data/images/ \
    root@$IP:/workspace/data/

# Push training script
scp -P $PORT -i ~/.ssh/runpod_bizcharts \
    python-ml/src/training/runpod/train_qwen.py \
    root@$IP:/workspace/scripts/
```

### 4.3 Set Up Environment on RunPod

```bash
ssh -p $PORT -i ~/.ssh/runpod_bizcharts root@$IP

# On RunPod:
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes accelerate trl
pip install qwen-vl-utils datasets pillow
```

### 4.4 Run Training

```bash
# On RunPod:
source /workspace/venv/bin/activate

python /workspace/scripts/train_qwen.py \
    --dataset /workspace/data/training_data.json \
    --images /workspace/data/images \
    --output /workspace/models/qwen-sentiment \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation 8

# Monitor GPU usage in another terminal:
watch -n 1 nvidia-smi
```

**Expected duration:** 3-4 hours for 10k samples, 3 epochs.

### 4.5 Verify Training Completed

```bash
# On RunPod:
ls -la /workspace/models/qwen-sentiment/

# Should see:
# - adapter_config.json
# - adapter_model.safetensors
# - training_args.json
```

---

## Step 5: Pull Trained Model

**Goal:** Download the LoRA adapter to your local machine.

### 5.1 Download Model

```bash
# From local machine:
scp -P $PORT -i ~/.ssh/runpod_bizcharts -r \
    "root@$IP:/workspace/models/qwen-sentiment" \
    python-ml/models/

# Verify
ls -la python-ml/models/qwen-sentiment/
```

### 5.2 Stop RunPod Pod

Don't forget to stop the pod to avoid charges!

1. Go to RunPod dashboard
2. Click "Stop" on your pod
3. Or terminate if you don't need the data anymore

---

## Step 6: Test the Model (Optional)

Test inference on RunPod before pulling:

```bash
# On RunPod:
source /workspace/venv/bin/activate

python -c "
from src.models.qwen_model import Qwen25VLModel

model = Qwen25VLModel(adapter_path='/workspace/models/qwen-sentiment')

# Test on a sample image
result = model.analyze('/workspace/data/images/thread_12345.jpg', 'WAGMI')
print(f'Sentiment: {result.sentiment}')
print(f'Confidence: {result.confidence}')
print(f'Explanation: {result.explanation}')
"
```

---

## Evaluation

After training, evaluate on your manual test set:

```bash
cd python-ml
source .venv/bin/activate

# Export manual labels as test set
python -m src.labeler --export ../data/test_labels.csv

# Run evaluation (requires GPU or RunPod)
python -c "
from src.models.qwen_model import Qwen25VLModel
import pandas as pd

model = Qwen25VLModel(adapter_path='models/qwen-sentiment')
test_df = pd.read_csv('../data/test_labels.csv')

correct = 0
for _, row in test_df.iterrows():
    result = model.analyze(row['image_path'], row['text'])
    if result.sentiment == row['label']:
        correct += 1

print(f'Accuracy: {correct}/{len(test_df)} = {correct/len(test_df):.1%}')
"
```

**Expected accuracy:** 70-80% after training.

---

## Cost Summary

| Step | Cost | Time |
|------|------|------|
| Scraping | $0 | 1-2 days |
| Manual labeling | $0 (your time) | 2-4 hours |
| Auto-labeling (GPT-4o-mini) | ~$2 | 2 hours |
| RunPod training | ~$2-5 | 3-4 hours |
| **Total** | **~$5-10** | **2-3 days** |

---

## Troubleshooting

### Auto-labeling fails with rate limit
```bash
# Add longer delay between requests
python -m src.training.auto_labeler ... --rate-limit 5.0

# Or use --resume to continue
python -m src.training.auto_labeler ... --resume
```

### Out of VRAM on RunPod
```bash
# Reduce batch size
python train_qwen.py ... --batch-size 1 --gradient-accumulation 16
```

### Training loss not decreasing
- Check data format in training_data.json
- Ensure images exist at specified paths
- Try lower learning rate: `--learning-rate 1e-5`

### Model outputs gibberish
- Training may not have converged - try more epochs
- Check that training data has balanced classes
- Verify labels are correct format: "bearish", "neutral", "bullish"

---

## Next Steps

After training:

1. **Deploy for inference** - Set up RunPod Serverless endpoint
2. **Integrate with dashboard** - Connect to Streamlit UI
3. **Continuous improvement** - Collect more data, retrain periodically
4. **Self-training** - Use high-confidence predictions to expand training set

See [CLAUDE.md](../CLAUDE.md) for production deployment details.

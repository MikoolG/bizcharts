# BizCharts Sentiment Labeling Guide

## Overview

The labeling tool allows you to manually rate thread OP sentiment to create training/validation data. This data supplements (not replaces) the automated analysis systems like VADER, color analysis, and CLIP.

## Purpose

Human labels serve multiple purposes:

1. **Validation**: Check if automated systems are accurate
2. **Training**: Provide ground truth for future model training if needed
3. **Pattern Discovery**: Find cases where heuristics fail
4. **Calibration**: Understand the baseline sentiment distribution

## Installation

```bash
cd python-ml

# Install dependencies
pip install Pillow  # For image display

# Or add to requirements.txt:
# Pillow>=10.0.0
```

## Usage

### Basic Labeling Session

```bash
# Start labeling (loads unlabeled posts ordered by reply count)
python -m src.labeler

# Use a named session (for multiple labelers or sessions)
python -m src.labeler --session my_session

# Limit posts loaded
python -m src.labeler --limit 500

# Review all posts (including already labeled)
python -m src.labeler --review
```

### Filtering by Source and Date

```bash
# Label only live catalog data
python -m src.labeler --source live

# Label only Warosu historical data
python -m src.labeler --source warosu

# Label data from a specific date range
python -m src.labeler --from 2024-01-01 --to 2024-06-30

# Combine filters: Warosu data from Q1 2024
python -m src.labeler --source warosu --from 2024-01-01 --to 2024-03-31
```

### Active Learning Mode

Once you have trained an initial model (200+ labels), use active learning to prioritize the most valuable posts to label:

```bash
# Use trained SetFit model for priority ordering
python -m src.labeler --active-learning --model models/setfit

# Customize candidate pool size
python -m src.labeler --active-learning --pool-size 2000 --limit 100

# Combine with source filtering
python -m src.labeler --active-learning --source live
```

Active learning uses **hybrid uncertainty-diversity sampling**:
- **Uncertainty**: Selects posts where the model is least confident
- **Diversity**: Ensures coverage across different topics via clustering

This approach provides 500 well-selected labels equivalent to ~2,000 random labels.

### Export & Statistics

```bash
# View labeling statistics
python -m src.labeler --stats

# Export labels to CSV
python -m src.labeler --export labels.csv

# Export labels for specific source
python -m src.labeler --export warosu_labels.csv --source warosu
```

### Controls

| Key | Action |
|-----|--------|
| `1` | Bearish (negative sentiment, panic, despair) |
| `2` | Neutral/Irrelevant (no clear sentiment, questions, off-topic) |
| `3` | Bullish (positive sentiment, euphoria, WAGMI) |
| `0` or `S` | Skip (broken image, unreadable) |
| `Left Arrow` | Go to previous post |
| `Right Arrow` | Go to next post |
| `N` | Add a note to current post |
| `Q` / `Esc` | Quit and save |

## Rating Scale

Use a 1-3 scale for sentiment:

| Rating | Meaning | Examples |
|--------|---------|----------|
| 1 | Bearish | "It's over", despair, pink wojak, predictions of crash, panic |
| 2 | Neutral/Irrelevant | Questions, news without opinion, off-topic, mixed signals |
| 3 | Bullish | "WAGMI", "to the moon", green wojak, euphoria, optimism |

### Skip Criteria

Use Skip (0/S) for posts that:
- Have broken or missing images
- Are unreadable or spam
- Have corrupt data

## Data Storage

Labels are stored in the SQLite database (`data/posts.db`) in the `training_labels` table:

```sql
CREATE TABLE training_labels (
    id INTEGER PRIMARY KEY,
    thread_id INTEGER,              -- Links to thread_ops
    sentiment_rating INTEGER,       -- 1-3 scale (1=bearish, 2=neutral, 3=bullish)
    skipped BOOLEAN,                -- True if marked for skip
    notes TEXT,                     -- Optional labeler notes
    labeler_id TEXT,                -- Session/labeler identifier
    labeled_at DATETIME,            -- When labeled
    text_snapshot TEXT,             -- OP text at labeling time
    image_url_snapshot TEXT         -- Image URL at labeling time
);
```

Labels are source-agnostic - they link to `thread_ops` by `thread_id` regardless of whether the data came from live scraping or Warosu. You can filter by source/date when querying for analysis.

## Recommended Workflow

### Initial Data Collection

1. **Run the Rust scraper** to collect live data:
   ```bash
   cd rust-scraper
   cargo run
   # Let it run for a few hours to collect threads
   ```

2. **Import historical data** from Warosu:
   ```bash
   cargo run -- --warosu --max 1000
   # Or search for specific topics:
   cargo run -- --warosu --search bitcoin --max 500
   ```

### Labeling Session

1. **Start with high-engagement posts** (default sorting by reply count):
   ```bash
   python -m src.labeler --limit 200
   ```

2. **Label in batches** - 50-100 posts per session is reasonable

3. **Be consistent** - If unsure between two ratings, pick the same direction each time

4. **Use notes** for edge cases you want to review later

### Quality Guidelines

- **Rate based on overall sentiment**, not just specific words
- **Consider both text and image** when rating
- **Irony/sarcasm**: If clearly ironic, rate the intended sentiment (e.g., ironic "WAGMI" during crash = bearish)
- **Mixed signals**: If genuinely mixed, rate as neutral (2)
- **Greentext stories**: Often ironic - read the full context

## Using Labeled Data

### Validation

Compare automated scores against human labels:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/posts.db")

# Get posts with both human labels and automated scores
df = pd.read_sql("""
    SELECT
        t.thread_id,
        t.sentiment_score as auto_score,
        l.sentiment_rating as human_rating,
        t.source
    FROM thread_ops t
    JOIN training_labels l ON t.thread_id = l.thread_id
    WHERE t.sentiment_score IS NOT NULL
      AND l.skipped = FALSE
""", conn)

# Convert human rating (1-3) to -1 to +1 scale
df['human_score'] = (df['human_rating'] - 2) / 1.0

# Calculate correlation
correlation = df['auto_score'].corr(df['human_score'])
print(f"Correlation: {correlation:.3f}")

# Can also filter by source for source-specific analysis
warosu_df = df[df['source'] == 'warosu']
```

### Training ML Models

The self-training system uses your labels to train sentiment models:

```bash
# Train SetFit model (few-shot learning, ~5 min)
cd python-ml
python -m src.training.setfit_trainer --db ../data/posts.db --output models/setfit

# Or use RunPod for GPU training
# Upload data to /workspace/data/posts.db, then:
python train_setfit.py --db /workspace/data/posts.db --output /workspace/models/setfit
```

The training pipeline automatically:
1. Loads labels from `training_labels` table
2. Maps 1-3 ratings directly to labels (1=bearish, 2=neutral, 3=bullish)
3. Splits data 80/20 for train/test
4. Reports accuracy and F1 scores

See [self-training.md](self-training.md) for the full ML architecture.

### Manual Data Export

If building a custom model:

```python
# Export for training
df = pd.read_sql("""
    SELECT
        l.text_snapshot as text,
        l.image_url_snapshot as image_url,
        l.sentiment_rating as label
    FROM training_labels l
    WHERE l.skipped = FALSE
      AND l.sentiment_rating IS NOT NULL
""", conn)

# Save for model training
df.to_csv("training_data.csv", index=False)
```

### Pattern Analysis

Find where automated systems fail:

```python
# High-confidence errors (note: human rating 1-3 converted to -1 to +1 scale)
errors = pd.read_sql("""
    SELECT
        t.thread_id,
        t.op_text_clean,
        t.sentiment_score as auto_score,
        t.sentiment_confidence as auto_confidence,
        l.sentiment_rating as human_rating,
        ABS(t.sentiment_score - (l.sentiment_rating - 2) / 1.0) as error
    FROM thread_ops t
    JOIN training_labels l ON t.thread_id = l.thread_id
    WHERE t.sentiment_confidence > 0.7
      AND l.skipped = FALSE
    ORDER BY error DESC
    LIMIT 50
""", conn)
```

## Best Practices

1. **Label diverse content** - Don't just label obvious bullish/bearish posts
2. **Include edge cases** - Posts with images, greentext, sarcasm
3. **Be honest about uncertainty** - Use neutral ratings when genuinely unsure
4. **Review periodically** - Use `--review` to check your consistency
5. **Export backups** - Run `--export` regularly to backup your work

## Metrics

Track these metrics as you label:

- **Distribution**: Should roughly match expected /biz/ sentiment (~60% bearish per research)
- **Skip rate**: High skip rate might mean data quality issues
- **Average rating**: Track if it drifts over sessions
- **Time per label**: Aim for 5-15 seconds per post

## Integration with Sentiment Pipeline

Human labels integrate with the broader system as:

```
                     ┌─────────────────────┐
                     │   Human Labels      │
                     │   (training_labels) │
                     └──────────┬──────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
           ┌────────────────┐    ┌────────────────┐
           │   Validation   │    │    Training    │
           │   (compare to  │    │   (if models   │
           │    heuristics) │    │    needed)     │
           └────────────────┘    └────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    ┌────────────────────────┐
                    │  Refined Heuristics /  │
                    │  Trained Models        │
                    └────────────────────────┘
```

Human labels are NOT the primary source - they validate and improve the automated systems.

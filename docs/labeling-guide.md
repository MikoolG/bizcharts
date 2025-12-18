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

### Export & Statistics

```bash
# View labeling statistics
python -m src.labeler --stats

# Export labels to CSV
python -m src.labeler --export labels.csv
```

### Controls

| Key | Action |
|-----|--------|
| `1-9` | Rate sentiment (1=extremely bearish, 5=neutral, 9=very bullish) |
| `0` | Rate 10 (extremely bullish) |
| `S` | Skip (post is irrelevant to sentiment) |
| `Left Arrow` | Go to previous post |
| `Right Arrow` | Go to next post |
| `N` | Add a note to current post |
| `Q` / `Esc` | Quit and save |

## Rating Scale

Use a 1-10 scale for sentiment:

| Rating | Meaning | Examples |
|--------|---------|----------|
| 1 | Extremely bearish | "It's over", complete despair, pink wojak |
| 2 | Very bearish | Strong negativity, predictions of crash |
| 3 | Bearish | Negative outlook, concern about holdings |
| 4 | Slightly bearish | Mild pessimism, uncertainty |
| 5 | Neutral | Questions, news without opinion, analysis |
| 6 | Slightly bullish | Mild optimism, cautious hope |
| 7 | Bullish | Positive outlook, confident in holdings |
| 8 | Very bullish | Strong optimism, predicting gains |
| 9 | Extremely bullish | "WAGMI", "to the moon", green wojak |
| 10 | Maximum bullish | Euphoria, celebration of massive gains |

### Skip Criteria

Skip posts that:
- Are not about crypto/finance sentiment (off-topic)
- Are purely informational with no opinion
- Are spam or incomprehensible
- Are meta-discussions about the board itself

## Data Storage

Labels are stored in the SQLite database (`data/posts.db`) in the `training_labels` table:

```sql
CREATE TABLE training_labels (
    id INTEGER PRIMARY KEY,
    thread_id INTEGER,              -- Links to thread_ops
    sentiment_rating INTEGER,       -- 1-10 scale
    skipped BOOLEAN,                -- True if marked irrelevant
    notes TEXT,                     -- Optional labeler notes
    labeler_id TEXT,                -- Session/labeler identifier
    labeled_at DATETIME,            -- When labeled
    text_snapshot TEXT,             -- OP text at labeling time
    image_url_snapshot TEXT         -- Image URL at labeling time
);
```

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
- **Mixed signals**: If genuinely mixed, rate toward neutral (5-6)
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
        l.sentiment_rating as human_rating
    FROM thread_ops t
    JOIN training_labels l ON t.thread_id = l.thread_id
    WHERE t.sentiment_score IS NOT NULL
      AND l.skipped = FALSE
""", conn)

# Convert human rating to -1 to +1 scale
df['human_score'] = (df['human_rating'] - 5.5) / 4.5

# Calculate correlation
correlation = df['auto_score'].corr(df['human_score'])
print(f"Correlation: {correlation:.3f}")
```

### Training

If building a supervised model:

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
# High-confidence errors
errors = pd.read_sql("""
    SELECT
        t.thread_id,
        t.op_text_clean,
        t.sentiment_score as auto_score,
        t.sentiment_confidence as auto_confidence,
        l.sentiment_rating as human_rating,
        ABS(t.sentiment_score - (l.sentiment_rating - 5.5) / 4.5) as error
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

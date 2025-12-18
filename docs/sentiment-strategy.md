# BizCharts Sentiment Analysis Strategy

## Overview

This document outlines the sentiment analysis approach for BizCharts. The goal is to accurately measure /biz/ board sentiment toward cryptocurrencies using a **multi-signal, confidence-weighted system** that improves over time through data collection and pattern discovery.

## Core Principles

1. **Free-first**: Use only free services or custom implementations. No paid APIs except Claude subscription.
2. **Confidence-aware**: Uncertain classifications get lower weight in aggregations.
3. **Multi-signal fusion**: Combine text + image signals for final sentiment.
4. **Data-driven refinement**: Use historical data (Warosu) to discover patterns before building complex models.
5. **Reply-weighted importance**: Popular threads (high reply count) have more influence on aggregate sentiment.

---

## Signal Sources

### 1. Text Sentiment

**Primary signal from OP text and subject line.**

| Method | Cost | Accuracy | Notes |
|--------|------|----------|-------|
| VADER + Custom Lexicon | FREE | ~70% | Fast baseline, /biz/ slang dictionary |
| Rule-based patterns | FREE | Variable | Greentext detection, sarcasm flags |
| Claude API | SUBSCRIPTION | ~90%+ | For ambiguous cases only |

**Custom lexicon additions for /biz/:**
```
WAGMI: +3.5 (but flag if market is down - likely ironic)
NGMI: -3.5
LFG: +3.0
"we're so back": +2.5
"it's over": -3.5
rekt: -3.5
moon/mooning: +3.0
dump/dumping: -2.5
```

**Greentext handling:**
- Lines starting with `>` often indicate sarcasm/storytelling
- Reduce confidence by 40% for posts with significant greentext
- Flag for manual review or Claude analysis if uncertain

---

### 2. Image Sentiment

**Visual signals from OP images.**

#### Approach: Heuristic-First, Model-Later

Rather than immediately training ML models, start with **interpretable heuristics** that can be validated and refined:

##### Color Analysis (No Training Required)

```python
def analyze_dominant_colors(image):
    """
    Simple color-based sentiment heuristics.
    Red-dominant = bearish, Green-dominant = bullish
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Count pixels in color ranges
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_ratio = np.count_nonzero(red_mask) / total_pixels
    green_ratio = np.count_nonzero(green_mask) / total_pixels

    # Pink Wojak detection: high red + skin tone
    # Green candles/charts: high green

    return {
        'red_ratio': red_ratio,
        'green_ratio': green_ratio,
        'color_sentiment': green_ratio - red_ratio,  # -1 to +1
        'confidence': abs(green_ratio - red_ratio)   # Higher if clear signal
    }
```

##### Visual Pattern Heuristics

| Pattern | Detection Method | Sentiment Signal |
|---------|-----------------|------------------|
| Red/pink dominant | Color histogram | Bearish (-0.6) |
| Green dominant | Color histogram | Bullish (+0.6) |
| Chart with red candles | Edge detection + red regions | Bearish |
| Chart with green candles | Edge detection + green regions | Bullish |
| Crying/distressed face | Future: facial expression | Bearish |
| Celebration imagery | Future: pose detection | Bullish |

##### CLIP Zero-Shot (Baseline)

If color heuristics are inconclusive, use CLIP:

```python
prompts = [
    "a meme showing financial loss, despair, or bearish sentiment",
    "a meme showing financial gains, celebration, or bullish sentiment",
    "a neutral image or chart without clear sentiment"
]
```

**CLIP limitations:**
- Doesn't learn from our specific data
- May miss /biz/-specific meme variants
- Good baseline but not optimal for edge cases

##### Future: Fine-Tuned Classifier

If heuristics prove insufficient, consider training on accumulated labeled data:

1. Collect 1000+ images with sentiment labels (from manual review)
2. Train EfficientNet-B0 or ViT-tiny classifier
3. Categories: strongly_bullish, bullish, neutral, bearish, strongly_bearish

**Training data collection strategy:**
- Use color heuristics to pre-sort images
- Manual verification of uncertain cases
- Accumulate labeled examples over time
- Only train when dataset is large enough (~1000+ images)

---

### 3. Combined Sentiment Score

**Fusion of text and image signals into final sentiment.**

```python
def calculate_combined_sentiment(text_sentiment, image_sentiment):
    """
    Combine text and image sentiment with confidence weighting.
    """
    # Base weights (can be tuned based on data)
    TEXT_WEIGHT = 0.6
    IMAGE_WEIGHT = 0.4

    # If no image, text is 100%
    if image_sentiment is None:
        return text_sentiment

    # Weighted combination
    text_contribution = text_sentiment.score * text_sentiment.confidence * TEXT_WEIGHT
    image_contribution = image_sentiment.score * image_sentiment.confidence * IMAGE_WEIGHT

    total_weight = (text_sentiment.confidence * TEXT_WEIGHT +
                    image_sentiment.confidence * IMAGE_WEIGHT)

    combined_score = (text_contribution + image_contribution) / total_weight

    # Combined confidence: geometric mean of individual confidences
    combined_confidence = sqrt(text_sentiment.confidence *
                               (image_sentiment.confidence if image_sentiment else 1.0))

    return SentimentResult(
        score=combined_score,          # -1 (bearish) to +1 (bullish)
        confidence=combined_confidence  # 0 to 1
    )
```

---

## Weighting System

### Reply-Based Importance Weight

Popular threads (high engagement) should influence aggregate sentiment more than ignored threads.

```python
def calculate_thread_weight(thread):
    """
    Weight based on thread engagement.
    """
    # Log scale to prevent single viral threads from dominating
    reply_weight = log(1 + thread.reply_count) / log(1 + MAX_EXPECTED_REPLIES)

    # Boost for threads with many unique IPs (if available)
    if thread.unique_ips:
        ip_factor = min(1.0, thread.unique_ips / 50)  # Cap at 50 unique posters
    else:
        ip_factor = 0.5  # Default if unknown

    # Page position factor: front page threads are more visible
    if thread.page_position:
        position_factor = 1.0 - (thread.page_position - 1) * 0.05  # 5% decay per page
    else:
        position_factor = 0.5

    return reply_weight * ip_factor * position_factor
```

### Confidence-Based Weight Modifier

Uncertain classifications should contribute less to aggregate sentiment.

```python
def calculate_final_weight(thread, sentiment_result):
    """
    Final weight = engagement weight * confidence modifier

    High confidence (0.8+): Full weight
    Medium confidence (0.5-0.8): Reduced weight
    Low confidence (<0.5): Minimal weight
    """
    engagement_weight = calculate_thread_weight(thread)

    # Confidence modifier: square confidence to penalize uncertainty more
    confidence_modifier = sentiment_result.confidence ** 2

    return engagement_weight * confidence_modifier
```

### Aggregate Sentiment Calculation

```python
def calculate_aggregate_sentiment(threads_with_sentiment):
    """
    Weighted average of all thread sentiments.
    """
    total_weighted_sentiment = 0
    total_weight = 0

    for thread, sentiment in threads_with_sentiment:
        weight = calculate_final_weight(thread, sentiment)
        total_weighted_sentiment += sentiment.score * weight
        total_weight += weight

    if total_weight == 0:
        return 0  # Neutral if no confident signals

    return total_weighted_sentiment / total_weight
```

---

## Learning Strategy

### Phase 1: Data Collection & Pattern Discovery

**Goal:** Understand what patterns exist before building complex models.

1. **Run scraper** for 1-2 weeks, collecting all OP data + images
2. **Apply heuristics** (color analysis, VADER) to all posts
3. **Manual review** of edge cases to understand failure modes
4. **Catalog patterns** that heuristics miss

**Key questions to answer:**
- What % of images are memes vs charts vs other?
- How well does color analysis correlate with manual sentiment labels?
- What text patterns indicate sarcasm/irony?
- Are there specific image patterns that always indicate sentiment?

### Phase 2: Heuristic Refinement

**Goal:** Tune heuristics based on Phase 1 findings.

1. Adjust color thresholds based on observed distributions
2. Expand custom lexicon with discovered /biz/ slang
3. Add new pattern detectors for common image types
4. Tune confidence thresholds and weights

### Phase 3: Optional Model Training

**Only if heuristics prove insufficient:**

1. Label dataset using heuristic-assisted process:
   - High-confidence heuristic results → auto-label
   - Low-confidence results → manual review
2. Train lightweight classifier (EfficientNet-B0)
3. Compare model accuracy vs heuristics
4. If model significantly better → deploy
5. If marginal improvement → stick with heuristics

**Training infrastructure (free):**
- Google Colab: Free GPU for training
- Local CPU training: Slower but works for small models
- HuggingFace: Free model hosting

### Phase 4: Continuous Improvement

**Ongoing refinement using production data:**

1. Log predictions with confidence scores
2. Periodically review low-confidence predictions
3. Add confirmed patterns to heuristics
4. Retrain model quarterly if using ML approach

---

## Implementation Priority

### Must Have (Phase 1)
- [x] OP text extraction from catalog
- [x] Coin mention extraction
- [x] Reply count / engagement metrics
- [ ] VADER + custom /biz/ lexicon
- [ ] Basic color analysis for images
- [ ] Confidence-weighted aggregation

### Should Have (Phase 2)
- [ ] CLIP zero-shot baseline
- [ ] Greentext/sarcasm detection
- [ ] Image download pipeline
- [ ] Manual review interface for labeling

### Nice to Have (Phase 3+)
- [ ] Fine-tuned image classifier
- [ ] Claude Vision for ambiguous cases
- [ ] Real-time sentiment streaming
- [ ] Historical backtesting framework

---

## Cost Analysis

| Component | Implementation | Cost |
|-----------|---------------|------|
| Text sentiment (VADER) | Python nltk | FREE |
| Custom lexicon | JSON file | FREE |
| Color analysis | OpenCV | FREE |
| CLIP inference | HuggingFace transformers | FREE |
| Image storage | Local filesystem | FREE |
| Model training | Google Colab / Local | FREE |
| Claude (ambiguous only) | Subscription | ~$0.10/day (estimated 50 calls) |

**Total monthly cost:** ~$3-5 (Claude calls for edge cases only)

---

## Success Metrics

### Accuracy Metrics
- **Text sentiment accuracy**: Manual review of 100 random posts
- **Image sentiment accuracy**: Manual review of 100 random images
- **Combined accuracy**: End-to-end sentiment correctness

### System Metrics
- **Coverage**: % of posts with sentiment score
- **Confidence distribution**: Should be bell-curved, not all high/low
- **Processing latency**: Target <100ms per thread

### Business Metrics
- **Correlation with price**: Does sentiment lead or lag price movements?
- **Signal quality**: Do extreme sentiment readings predict volatility?

---

## Open Questions

1. **How to handle image-only posts?** (No text to analyze)
2. **Should we weight coin-specific sentiment differently?** (BTC vs altcoins)
3. **How to detect coordinated shilling/FUD campaigns?**
4. **What's the optimal lookback window for aggregate sentiment?**
5. **Should sentiment decay over time?** (Older posts less relevant)

---

## Appendix: Color Ranges for Analysis

```python
# HSV color ranges for sentiment heuristics
# These should be tuned based on observed /biz/ image data

RED_RANGES = [
    ((0, 100, 100), (10, 255, 255)),    # Red
    ((160, 100, 100), (180, 255, 255)), # Red (wrapping)
    ((0, 50, 150), (20, 150, 255)),     # Pink (Wojak skin)
]

GREEN_RANGES = [
    ((35, 100, 100), (85, 255, 255)),   # Green spectrum
    ((35, 50, 100), (85, 150, 255)),    # Pale green
]

# Confidence thresholds
COLOR_CONFIDENCE_THRESHOLD = 0.15  # Min color ratio difference
CLIP_CONFIDENCE_THRESHOLD = 0.7    # Min CLIP similarity for classification
```

# Research Brief: Fully Automated Sentiment Training for Crypto Social Media

## Goal
**Train a high-accuracy multimodal sentiment model** that analyzes image + text together.

Requirements:
- **Multimodal input**: Every prediction uses BOTH image AND text - never text alone
- **High accuracy**: Must achieve significantly better than the 40% text-only baseline
- **Automated training**: Minimize manual labeling through clever training strategies
- **Domain-specific**: Must handle crypto slang, memes, sarcasm specific to 4chan /biz/

This is NOT about zero-shot inference - we ARE going to train. The question is: what's the best way to train a multimodal model for our domain?

---

## Project Context

### What We're Building
A sentiment analysis system for 4chan /biz/ board posts about cryptocurrency. Posts include:
- Short-form text (similar to tweets)
- Meme images (often with text overlays)
- Crypto-specific slang: "WAGMI", "NGMI", "rekt", "moon", "ape in", etc.
- Heavy sarcasm and irony

### Output Requirements
- **3-class sentiment**: bearish, neutral, bullish
- Must handle both text AND images (images are mandatory for accurate sentiment)
- Low latency for real-time dashboard updates

### Hardware
- Local development: ThinkPad P14s (no GPU, CPU only)
- Training & testing: RunPod cloud GPUs (RTX 4090/3090)
- Production inference: Will require GPU server or RunPod

---

## Current Tech Stack

### Implemented Infrastructure
```
python-ml/src/
├── models/
│   ├── cryptobert_model.py    # ElKulako/cryptobert wrapper (3.2M crypto posts pretrained)
│   ├── setfit_model.py        # Few-shot classifier
│   └── llava_model.py         # LLaVA 7B for image understanding
├── training/
│   ├── data_loader.py         # SQLite data loading
│   ├── setfit_trainer.py      # SetFit training pipeline
│   └── pseudo_labeler.py      # CryptoBERT zero-shot labeling (just created)
├── fusion/
│   ├── multimodal_pipeline.py # Text + image fusion
│   └── sarcasm_detector.py    # Text-image incongruity detection
└── active_learning/
    └── acquisition.py         # Hybrid uncertainty-diversity sampling
```

### Dependencies Available
- PyTorch, Transformers, PEFT (LoRA)
- sentence-transformers, setfit
- datasets (HuggingFace)
- scikit-learn, river (drift detection)
- MLflow for experiment tracking

### Data Available
- **107 manually labeled posts** (bearish=58, neutral=32, bullish=17)
- **202 total posts** in database (need to scrape more for larger dataset)
- **2 unlabeled posts** remaining (most have been labeled)
- Images stored as URLs (can be fetched)
- Can easily scrape more from 4chan /biz/ (Rust scraper ready)

---

## Multimodal Models to Consider (for Training)

We need a vision-language model that can be fine-tuned on our domain:

| Model | Size | VRAM | Trainable? | Notes |
|-------|------|------|------------|-------|
| LLaVA 1.5/1.6 | 7B | 15GB (4-bit) | Yes (QLoRA) | Good baseline, well-documented fine-tuning |
| LLaVA-NeXT | 7-34B | 15-40GB | Yes | Improved version, better detail |
| Qwen-VL | 7B | 14GB | Yes | Strong on Chinese + English |
| InternVL | 2-26B | 8-40GB | Yes | Recent, good performance |
| Idefics2 | 8B | 16GB | Yes | Open, HuggingFace native |
| PaliGemma | 3B | 8GB | Yes | Smaller, faster training |

**Key question for research:** Which model is best for fine-tuning on crypto meme sentiment with limited data?

---

## Research Questions

### Primary Question
**How do we train a multimodal model (image + text → sentiment) to achieve high accuracy on crypto memes with minimal manual labeling?**

### Specific Areas to Explore

1. **Training Data Generation**
   - Can we use Claude/GPT-4V to auto-label thousands of posts (image + text)?
   - What's the cost/accuracy tradeoff of LLM-generated labels?
   - How many training examples do we need for good accuracy?

2. **Fine-tuning Approaches for VLMs**
   - QLoRA vs full fine-tuning for LLaVA/similar models?
   - Best practices for sentiment classification fine-tuning?
   - How to format training data (instruction tuning vs classification head)?

3. **Knowledge Distillation**
   - Train on GPT-4V/Claude labels, run inference with smaller LLaVA?
   - What accuracy loss do we expect from distillation?

4. **Semi-supervised / Self-training**
   - Train on small labeled set, pseudo-label more, iterate?
   - Confidence thresholds and noise handling for multimodal?

5. **Domain Adaptation**
   - How to adapt general VLMs to crypto meme domain?
   - Contrastive learning on unlabeled image-text pairs?
   - Importance of crypto vocabulary (WAGMI, NGMI, etc.)

6. **Sarcasm and Irony**
   - This is critical for /biz/ - text often contradicts image
   - How to train models to detect and invert sarcastic sentiment?
   - Text-image incongruity as a signal

---

## Constraints

- **No manual labeling** (or minimal - the 107 we have is fine to use)
- **Must handle images** - posts are always image + text together, never text alone
- **Reasonable cost** - ideally under $50 total for training
- **No local GPU** - inference must be RunPod or API-based (development machine is CPU-only)
- **Sentiment = image + text** - this is fundamental, not optional

---

## Desired Output

Please research and recommend:

1. **Best automated training pipeline** - step by step
2. **Model choices** - which models to use for text and images
3. **Expected accuracy** - realistic expectations
4. **Implementation complexity** - rough effort estimate
5. **Alternative approaches** - ranked by automation level

Focus on practical, implementable solutions rather than theoretical approaches. We have the infrastructure ready to go.

---

## Baseline Results (Text-Only - Expected to Fail)

These baselines used **text only** without images. Since human labeling was done with image + text together, text-only models were expected to perform poorly.

**CryptoBERT Zero-Shot Performance on 107 Human-Labeled Posts (TEXT ONLY):**

| Metric | Value |
|--------|-------|
| Overall Accuracy | 39.3% |
| Bearish Recall | 22.4% (13/58) |
| Neutral Recall | 53.1% (17/32) |
| Bullish Recall | 70.6% (12/17) |
| Avg Confidence | 0.66 |

**Confusion Matrix:**
```
             Predicted
             bearish  neutral  bullish
True bearish      13       31       14
True neutral       5       17       10
True bullish       0        5       12
```

**Key Observations:**
1. CryptoBERT over-predicts "neutral" - 53/107 predictions
2. Terrible bearish recall - misses 77% of bearish posts
3. Good bullish detection but that's the smallest class
4. Model was likely trained on Twitter, not 4chan's sarcastic style
5. Average confidence of 0.66 means many uncertain predictions

**VADER Comparison:**
| Model | Overall | Bearish | Neutral | Bullish |
|-------|---------|---------|---------|---------|
| CryptoBERT | 39.3% | 22.4% | 53.1% | 70.6% |
| VADER | 39.3% | 41.4% | 31.2% | 47.1% |
| Ensemble (0.3/0.7) | 41.1% | - | - | - |
| Agree/Confidence | 45.8% | - | - | - |

**Note:** VADER and CryptoBERT have complementary strengths (VADER better at bearish, CryptoBERT better at bullish), but even the best ensemble only reaches 45.8%.

**Implication:** Text-only zero-shot approaches are NOT sufficient. This is expected - /biz/ posts often have ambiguous or sarcastic text where the image provides the real sentiment signal. The 107 human labels were made by looking at image + text together.

**The real question:** How well can a multimodal model (LLaVA, GPT-4V, Claude) perform when given both image + text?

---

## Finalized Training Pipeline

Text-only approaches failed (39-46% accuracy) because they can't see the images. After research, here is the finalized approach:

### Phase 1: Data Collection (Local)
1. Scrape 10,000+ posts with images using Rust scraper
2. Manual labeling: 500-1000 posts as **TEST SET** (ground truth for evaluation)
3. Store in SQLite `training_labels` table

### Phase 2: Auto-Labeling (API, ~$0-2.50)
Use vision API to generate training labels for 10k+ posts:

| API | Cost (10k) | Quality | Speed |
|-----|------------|---------|-------|
| **Gemini 2.0 Flash** (free tier) | $0 | 70-78% | ~7 days (rate limited) |
| **GPT-4o-mini** | ~$1.70-2.50 | 80-83% | ~1-2 hours |

**Why API instead of Qwen2-VL on RunPod?**
- API is both **cheaper AND higher quality** for labeling
- Qwen2-VL zero-shot: 55-65% quality, $2-14 compute cost
- GPT-4o-mini: 80-83% quality, ~$2 API cost

### Phase 3: Fine-tuning (RunPod, ~$2-5)
- **Model**: Qwen2.5-VL-7B with QLoRA
- **Why Qwen2.5-VL?**:
  - Best OCR for meme text overlays
  - Strong fine-tuning ecosystem (TRL, LLaMA-Factory)
  - Documented 48% → 66%+ accuracy improvement with LoRA
- **Hardware**: RTX 4090 (~$0.34-0.69/hr), ~3-4 hours
- **Output**: LoRA adapter (~100-500MB) that you own

### Phase 4: Production Inference (RunPod Serverless)
- Deploy trained model for live sentiment analysis
- Cost: ~$0.002-0.01 per image
- Monthly (hourly catalog checks): ~$3-10

---

## Cost Summary

| Phase | Cost | One-time? |
|-------|------|-----------|
| Manual labeling (500-1000) | Your time | Yes |
| Auto-labeling (10k+) | $0-2.50 | Yes |
| QLoRA training | $2-5 | Yes |
| **Total one-time** | **~$5-10** | |
| Production inference | $3-10/month | Ongoing |

---

## Expected Accuracy

| Stage | Accuracy |
|-------|----------|
| Text-only baseline (CryptoBERT) | 40% |
| Zero-shot Qwen2.5-VL | 55-65% |
| Fine-tuned Qwen2.5-VL | 70-80% |
| With self-training iteration | 75-85% |

---

## Research Decisions Made

1. **Which VLM to fine-tune?** → **Qwen2.5-VL-7B** (best OCR, strong ecosystem)
2. **Best fine-tuning approach?** → **QLoRA** (r=16, alpha=32, target q/v projections)
3. **How to generate training data?** → **Vision API labels** (Gemini free or GPT-4o-mini)
4. **Expected accuracy?** → 70-80% after fine-tuning, 75-85% with self-training
5. **Inference?** → RunPod Serverless (~$3-10/month)

---

## Training Resources

- [Qwen2.5-VL Fine-tuning Guide](https://datature.io/blog/how-to-fine-tune-qwen2-5-vl)
- [TRL VLM Fine-tuning Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)
- [Qwen-VL-Finetune Repository](https://github.com/2U1/Qwen2-VL-Finetune)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [LLaVAC Paper](https://arxiv.org/html/2502.02938v1) - LLaVA sentiment fine-tuning (79-83% accuracy)

# BizCharts Self-Training Implementation Plan

## Summary
Implement a self-improving sentiment analysis system using:
- **Text**: VADER + SetFit + CryptoBERT ensemble
- **Images**: LLaVA 7B for end-to-end meme understanding
- **Active Learning**: Hybrid uncertainty-diversity sampling
- **Continual Learning**: Experience replay + ADWIN drift detection

## User Preferences
- Text: Ensemble (VADER + ML models)
- Images: LLaVA 7B
- Local GPU: 24GB+ VRAM (full flexibility)
- Priority: Full pipeline first

---

## Phase 1: Core Infrastructure (Week 1)

### New Directory Structure
```
python-ml/src/
├── models/                 # Model wrappers
│   ├── __init__.py
│   ├── base.py            # Abstract base classes
│   ├── setfit_model.py    # SetFit wrapper
│   ├── cryptobert_model.py
│   ├── llava_model.py
│   └── ensemble.py
├── training/              # Training pipelines
│   ├── data_loader.py     # Load from SQLite
│   ├── setfit_trainer.py
│   └── runpod/            # Cloud training scripts
├── active_learning/       # AL components
│   ├── acquisition.py     # Hybrid sampling
│   └── labeler_integration.py
├── continual/             # Continual learning
│   ├── replay_buffer.py
│   └── drift_detector.py
├── fusion/                # Multi-modal fusion
│   ├── sarcasm_detector.py
│   └── multimodal_pipeline.py
└── inference/             # Production inference
    ├── pipeline.py
    └── onnx_exporter.py
```

### Dependencies to Add (pyproject.toml)
```toml
# ML Training
"setfit>=1.0.0",
"sentence-transformers>=2.2.0",
"datasets>=2.14.0",
"peft>=0.6.0",
"bitsandbytes>=0.41.0",
"accelerate>=0.24.0",

# Inference optimization
"optimum[onnxruntime]>=1.13.0",

# Active learning & drift
"river>=0.21.0",
"scikit-learn>=1.3.0",

# Experiment tracking
"mlflow>=2.8.0",
```

---

## Phase 2: SetFit Bootstrap

### Files to Create

**python-ml/src/models/base.py**
- `PredictionResult` dataclass: label, score (-1 to +1), confidence, probabilities
- `BaseSentimentModel` abstract class: predict(), predict_batch(), get_embeddings()

**python-ml/src/models/setfit_model.py**
- `SetFitSentimentModel` wrapping `sentence-transformers/paraphrase-mpnet-base-v2`
- Maps 3 classes (bearish/neutral/bullish) to continuous score

**python-ml/src/training/data_loader.py**
- Load from `training_labels` table
- Map 1-3 rating directly to labels: 1=bearish, 2=neutral, 3=bullish
- Return HuggingFace Dataset

**python-ml/src/training/setfit_trainer.py**
- Training config: batch_size=16, epochs=4, iterations=20
- Train/test split (80/20)
- Save to `models/setfit/`

### Files to Modify

**python-ml/src/text_analyzer.py:77-121**
- Add `use_ensemble` parameter to `__init__`
- Load SetFit model when enabled
- Combine VADER (30%) + SetFit (70%) for ensemble output
- Use existing `is_ambiguous()` to route uncertain posts

---

## Phase 3: CryptoBERT + Active Learning

### Files to Create

**python-ml/src/models/cryptobert_model.py**
- Load `ElKulako/cryptobert` with LoRA adapters (r=16, alpha=32)
- VRAM: 4-6GB

**python-ml/src/active_learning/acquisition.py**
```python
def hybrid_acquisition(model, texts, embeddings, n_select=50):
    # 1. Compute entropy uncertainty
    # 2. K-means clustering for diversity
    # 3. Select highest uncertainty per cluster
```

**python-ml/src/active_learning/labeler_integration.py**
- `get_suggested_posts()`: Query unlabeled, run acquisition, return ordered thread_ids

### Files to Modify

**python-ml/src/labeler.py**
- Add `--active-learning` flag
- Load model, run acquisition, reorder posts by AL priority

---

## Phase 4: Continual Learning

### Files to Create

**python-ml/src/continual/replay_buffer.py**
- Reservoir sampling buffer (max_size=2000)
- `get_training_mix()`: 70% new + 30% replay

**python-ml/src/continual/drift_detector.py**
- ADWIN detector (delta=0.002)
- Track error rate, trigger retraining on drift

**python-ml/src/training/pseudo_labeling.py**
- FixMatch with τ=0.95 threshold
- Generate pseudo-labels from high-confidence predictions

---

## Phase 5: LLaVA Multi-Modal

### Files to Create

**python-ml/src/models/llava_model.py**
- Load `llava-hf/llava-1.5-7b-hf` with 4-bit quantization
- Prompt template for meme sentiment + sarcasm detection
- VRAM: 14-15GB (fits 24GB card)

**python-ml/src/fusion/sarcasm_detector.py**
- Detect via text-image sentiment incongruity
- Threshold: |text_sentiment - image_sentiment| > 0.7

**python-ml/src/fusion/multimodal_pipeline.py**
- Fusion weights: text=0.5, image=0.3, context=0.1, ocr=0.1
- Sarcasm inversion when detected with high confidence
- Output: score, confidence, label, sarcasm_probability

### Files to Modify

**python-ml/src/image_analyzer.py**
- Add LLaVA path alongside existing CLIP/YOLO plans
- Route based on config: use_llava vs use_clip_stack

---

## Phase 6: Production Inference

### Files to Create

**python-ml/src/inference/pipeline.py**
- Full inference pipeline combining all models
- Batch processing for scraper integration

**python-ml/src/inference/onnx_exporter.py**
- Export SetFit/CryptoBERT to ONNX
- INT8 quantization for CPU inference

---

## RunPod Training Scripts

**python-ml/src/training/runpod/train_setfit.py**
- CLI script with MLflow tracking
- Cost: ~$0.06 (5 min on RTX 4090)

**python-ml/src/training/runpod/train_llava.py**
- QLoRA fine-tuning on meme dataset
- Cost: ~$1.50-3.00 (2-4 hours)

---

## Compute Costs (RunPod)

| Task | GPU | Time | Cost |
|------|-----|------|------|
| SetFit (200 examples) | RTX 4090 | 5 min | $0.06 |
| CryptoBERT LoRA | RTX 4090 | 30 min | $0.35 |
| LLaVA QLoRA | RTX 4090 | 2-4 hr | $1.50-3.00 |
| Weekly retrain | RTX 3090 | 15 min | $0.06 |

**Monthly ongoing**: ~$5-10

---

## Implementation Order

1. **Base infrastructure**: models/, training/, pyproject.toml deps
2. **SetFit training**: data_loader, trainer, model wrapper
3. **Ensemble integration**: Modify text_analyzer.py for ensemble mode
4. **Active learning**: acquisition, labeler integration
5. **Continual learning**: replay buffer, drift detection
6. **LLaVA integration**: model wrapper, fusion pipeline
7. **Production**: ONNX export, batch inference

---

## Critical Files Summary

| File | Action | Purpose |
|------|--------|---------|
| python-ml/pyproject.toml | Modify | Add ML dependencies |
| python-ml/src/text_analyzer.py | Modify | Add ensemble support |
| python-ml/src/labeler.py | Modify | Add active learning mode |
| python-ml/src/image_analyzer.py | Modify | Add LLaVA path |
| config/settings.toml | Modify | Add model paths, training configs |
| python-ml/src/models/* | Create | Model wrappers |
| python-ml/src/training/* | Create | Training pipelines |
| python-ml/src/fusion/* | Create | Multi-modal fusion |

---

## Expected Outcomes

- **Phase 2**: 75-85% accuracy on clear sentiment (SetFit)
- **Phase 3**: 85-90% with CryptoBERT + active learning
- **Phase 5**: 65-75% on sarcastic content with LLaVA fusion
- **Full system**: Self-improving with minimal human intervention

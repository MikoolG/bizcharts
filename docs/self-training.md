# Building a self-improving sentiment analysis system for crypto imageboards

> **Implementation Status (Dec 2024):** The core infrastructure described in this document has been implemented. See [Implementation Details](#implementation-status) at the bottom for what's available.

The most effective approach for your 4chan /biz/ sentiment system combines **SetFit few-shot learning** for immediate results from 200 examples, **CryptoBERT** for domain-specific understanding, and **LLaVA 7B** for multi-modal meme analysis—all achievable with RunPod compute under $10/month. The critical insight: sarcasm detection in chan culture realistically tops out at **65-80% accuracy** even with optimal methods, making context integration and multi-modal fusion essential rather than optional.

Your constraint of no paid APIs is actually an advantage—open-source models like CryptoBERT now match or exceed commercial APIs for crypto sentiment. The self-improving component relies on **active learning** (uncertainty sampling to identify valuable labels) combined with **experience replay** for continual learning without catastrophic forgetting. Start with SetFit, which achieves 91% accuracy with just 8 examples per class and trains in 30 seconds.

## Recommended architecture and model stack

The system requires three parallel processing pipelines that fuse at inference time: text sentiment, image sentiment, and sarcasm detection. Each component has specific open-source models optimized for your constraints.

**Primary text model: CryptoBERT** (`ElKulako/cryptobert`) is pre-trained on 3.2 million crypto social media posts with native Bearish/Neutral/Bullish labels. This eliminates the cold-start problem for crypto-specific vocabulary like "WAGMI," "NGMI," "rekt," and "moon." For few-shot bootstrapping, **SetFit** with `sentence-transformers/paraphrase-mpnet-base-v2` achieves remarkable accuracy—matching RoBERTa-Large fine-tuned on 3,000 examples using just 64 examples.

**Multi-modal model: LLaVA 1.5 7B** (`llava-hf/llava-1.5-7b-hf`) handles meme understanding effectively. Research demonstrates LLaVA v1.6 successfully identifies Pepe and Wojak memes with contextual descriptions. For resource-constrained inference, **moondream2** (`vikhyatk/moondream2`) at ~2B parameters provides a lightweight alternative requiring only 4-5GB VRAM.

| Component | Model | HuggingFace Path | VRAM |
|-----------|-------|------------------|------|
| Text sentiment | CryptoBERT | `ElKulako/cryptobert` | 4-6GB |
| Few-shot learning | SetFit MPNet | `sentence-transformers/paraphrase-mpnet-base-v2` | 2-4GB |
| Multi-modal VLM | LLaVA 1.5 7B | `llava-hf/llava-1.5-7b-hf` | 14-15GB |
| Image embeddings | OpenCLIP | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | 2GB |
| OCR | TrOCR | `microsoft/trocr-base-printed` | 1GB |
| Zero-shot fallback | BART-MNLI | `facebook/bart-large-mnli` | 3GB |

## Phase 1: Bootstrap from 200 examples in one week

Your 200 labeled examples are sufficient for a production-grade initial model using SetFit's contrastive learning approach. SetFit generates text pairs from your labels, trains a sentence transformer to create sentiment-aware embeddings, then fits a simple classification head.

```python
from setfit import SetFitModel, Trainer, TrainingArguments

model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2",
    labels=["bearish", "neutral", "bullish"]
)

args = TrainingArguments(
    batch_size=16,
    num_epochs=4,
    num_iterations=20,  # Contrastive pairs per sample
)

trainer = Trainer(model=model, args=args, train_dataset=your_200_examples)
trainer.train()  # Completes in ~30 seconds on GPU
```

Expected accuracy: **75-85%** on clear-cut sentiment. Training cost on RunPod RTX 4090: approximately **$0.06** (5 minutes at $0.69/hour). This provides immediate value while you build the more sophisticated components.

For posts with images, run **parallel inference**: SetFit on text, CLIP on images with zero-shot prompts like "a sad feeling meme" vs "a happy optimistic meme," then average predictions weighted 0.6 text / 0.4 image. This simple fusion works surprisingly well before implementing cross-attention.

## Self-improving through active learning and pseudo-labeling

The autonomous improvement loop combines **uncertainty sampling** to identify which posts to manually label, **pseudo-labeling** to expand training data automatically, and **drift detection** to trigger retraining when vocabulary evolves.

**Active learning implementation** uses hybrid uncertainty-diversity sampling. Pure uncertainty sampling (selecting lowest-confidence predictions) risks sampling redundant similar examples. Diversity sampling via clustering ensures coverage of the embedding space.

```python
from modAL.uncertainty import entropy_sampling
from sklearn.cluster import KMeans

def hybrid_acquisition(model, X_unlabeled, embeddings, n_select=50):
    # Uncertainty scores
    probs = model.predict_proba(X_unlabeled)
    uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    # Diversity via clustering
    kmeans = KMeans(n_clusters=n_select)
    clusters = kmeans.fit_predict(embeddings)
    
    # Select highest uncertainty from each cluster
    selected = []
    for cluster_id in range(n_select):
        cluster_mask = clusters == cluster_id
        cluster_uncertainties = uncertainty[cluster_mask]
        best_in_cluster = np.argmax(cluster_uncertainties)
        selected.append(np.where(cluster_mask)[0][best_in_cluster])
    
    return selected
```

**Diminishing returns thresholds**: Active learning provides 500 well-selected labels equivalent to 2,000+ random labels. Beyond **1,000-1,500 labels**, prioritize model architecture improvements over additional labeling. With your current 200 labels plus active learning, expect to reach 85-90% accuracy at around 500 total labeled examples.

**Pseudo-labeling with FixMatch** expands training data using high-confidence predictions. Apply a strict threshold (τ=0.95) initially, then lower to 0.90 as model improves. Strong augmentation on pseudo-labeled examples—back-translation, synonym replacement, random deletion—forces learning robust features rather than spurious patterns.

## Handling sarcasm requires context and multi-modal signals

Sarcasm detection is the hardest component. Human accuracy on sarcasm averages only **81.6%**, and models without context drop to 49% F1. Your system must incorporate thread context (3-5 previous posts) and text-image disagreement signals.

**Thread context is essential**. The phrase "WAGMI" during a market crash versus during a rally carries opposite sentiment. Concatenate [CLS] embeddings from the current post with previous posts, or use a summarization model (BART-Large) to compress thread context.

```python
def get_contextual_embedding(current_post, previous_posts, tokenizer, model):
    # Format: [CLS] prev_3 [SEP] prev_2 [SEP] prev_1 [SEP] current [SEP]
    context = " [SEP] ".join(previous_posts[-3:] + [current_post])
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
    return model(**inputs).last_hidden_state[:, 0, :]  # [CLS] token
```

**Multi-modal sarcasm detection** exploits text-image disagreement. When a pink Wojak (panic/despair) accompanies "This is fine, everything is fine," the contradiction signals sarcasm. Train a binary classifier on CLIP embeddings with input: `[text_embedding; image_embedding; |text_sentiment - image_sentiment|]`.

**Wojak/Pepe expression mapping** requires a custom classifier since no pre-trained models handle these:

| Meme Expression | Sentiment Mapping |
|-----------------|-------------------|
| Pink/Red Wojak | Panic, bearish |
| Crying Wojak | Despair, bearish |
| Smug Pepe | Confident, often ironic |
| Gigachad | Bullish or ironic cope |
| Bobo (bear meme) | Explicitly bearish |

Start with a small labeled dataset (~500 meme images) and fine-tune CLIP with LoRA. This classifier's output becomes a feature for the fusion model.

**Realistic accuracy expectations**: Without context, expect 55-65% F1 on sarcasm. With thread context and multi-modal signals, 65-75% is achievable. Design your system to output **both literal and inferred sentiment** with confidence scores, allowing downstream analysis to handle ambiguity appropriately.

## Continual learning prevents catastrophic forgetting

Chan vocabulary evolves rapidly—new slang emerges weekly. Your system needs **experience replay** to maintain performance on old patterns while learning new ones, plus **drift detection** to trigger retraining.

**Experience replay** stores 1,000-2,000 representative examples in a buffer using reservoir sampling. During retraining, mix 70% new data with 30% replay buffer. This single technique provides most forgetting prevention benefit with minimal complexity.

```python
class ReplayBuffer:
    def __init__(self, max_size=2000):
        self.buffer = []
        self.max_size = max_size
        self.total_seen = 0
    
    def add(self, examples):
        for example in examples:
            self.total_seen += 1
            if len(self.buffer) < self.max_size:
                self.buffer.append(example)
            else:
                # Reservoir sampling maintains uniform distribution
                idx = random.randint(0, self.total_seen - 1)
                if idx < self.max_size:
                    self.buffer[idx] = example
```

**Drift detection with River's ADWIN** monitors prediction accuracy over time. When statistical drift is detected, trigger retraining.

```python
from river.drift import ADWIN

drift_detector = ADWIN(delta=0.002)

for prediction, actual in labeled_stream:
    error = 0 if prediction == actual else 1
    drift_detector.update(error)
    
    if drift_detector.drift_detected:
        trigger_retraining()
        drift_detector.reset()
```

**Vocabulary monitoring** catches new slang before it impacts accuracy. Track subword fragmentation—when BPE tokenizes a term into many pieces (e.g., "NGMI" → ["N", "G", "M", "I"]), it's likely novel. Alert when >3% of tokens are heavily fragmented.

**Retraining frequency**: Weekly incremental updates (LoRA fine-tuning on new data) with monthly full evaluation against a held-out "golden test set" spanning different time periods. Quarterly architecture review for major changes.

## Multi-modal fusion architecture for production

The complete architecture processes text and images through separate encoders, extracts sentiment signals from each, detects incongruity, then fuses for final classification.

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT PROCESSING                        │
├──────────────────┬──────────────────┬───────────────────────┤
│ Text Pipeline    │ Image Pipeline   │ OCR Pipeline          │
│ BERTweet/        │ CLIP ViT-L/14    │ TrOCR + EasyOCR      │
│ CryptoBERT       │ + Wojak classifier│                      │
└────────┬─────────┴────────┬─────────┴───────────┬───────────┘
         │                  │                     │
         ▼                  ▼                     ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│ Text Sentiment  │ │ Image Sentiment │ │ Text-in-Image       │
│ Score [-1, +1]  │ │ Score [-1, +1]  │ │ Sentiment           │
└────────┬────────┘ └────────┬────────┘ └──────────┬──────────┘
         │                   │                      │
         └───────────────────┼──────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    INCONGRUITY DETECTOR                      │
│  |text_sentiment - image_sentiment| > threshold → sarcasm   │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    FUSION + CLASSIFICATION                   │
│  Cross-attention or weighted average based on confidence    │
│  Output: {sentiment, confidence, sarcasm_probability}       │
└─────────────────────────────────────────────────────────────┘
```

**Handling posts without images**: Use a mask-based approach where image embeddings are zeroed when absent. The model learns to rely solely on text features in these cases. Alternatively, maintain separate text-only and multi-modal classification heads with automatic routing.

**Fusion weights** (empirically tuned starting point): Text sentiment 0.5, image sentiment 0.3, OCR text 0.1, context 0.1. Adjust based on error analysis—if image-heavy posts underperform, increase image weight.

## Compute requirements and cost estimates for RunPod

All training fits comfortably on an **RTX 4090** ($0.69/hour Community Cloud) or **RTX 3090** ($0.22/hour). The A100 is unnecessary unless training LLaVA from scratch.

| Task | GPU | VRAM | Time | Cost |
|------|-----|------|------|------|
| SetFit initial (200 examples) | RTX 4090 | 4GB | 5 min | $0.06 |
| CryptoBERT LoRA fine-tune | RTX 4090 | 10GB | 30 min | $0.35 |
| LLaVA QLoRA fine-tune (1K images) | RTX 4090 | 12GB | 2-4 hours | $1.50-3.00 |
| Weekly incremental retrain | RTX 3090 | 8GB | 15 min | $0.06 |
| Monthly full retrain | RTX 4090 | 12GB | 1 hour | $0.69 |

**Monthly ongoing cost**: Approximately **$2-8** for weekly retraining cycles.

**Inference runs on CPU** using ONNX Runtime optimization. Export your models to ONNX with INT8 quantization for 4-10x speedup. DistilBERT achieves ~25ms per prediction on CPU, processing **2,500+ posts per minute**—more than sufficient for batch processing ~150 threads per snapshot.

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Export and quantize for CPU inference
model = ORTModelForSequenceClassification.from_pretrained(
    "ElKulako/cryptobert",
    export=True,
    provider="CPUExecutionProvider"
)

quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
quantizer.quantize(save_dir="./quantized_model", quantization_config=qconfig)
```

## Complete phased implementation timeline

**Week 1 (Phase 0-1)**: Deploy zero-shot baseline with `facebook/bart-large-mnli` for immediate results. Train SetFit on your 200 examples. Validate against 50 manually reviewed posts. Expected accuracy: 70-80%.

**Weeks 2-3 (Phase 2)**: Fine-tune CryptoBERT with LoRA. Implement active learning loop to identify high-value labeling targets. Build Wojak/Pepe image classifier on ~500 labeled meme images. Integrate CLIP for multi-modal inference.

**Month 2 (Phase 3)**: Implement continual learning with experience replay buffer. Deploy drift detection monitoring. Set up automated weekly retraining pipeline with MLflow tracking. Target: 85-90% accuracy on clear sentiment, 65-75% on sarcastic content.

**Month 3+ (Phase 4)**: LLaVA QLoRA fine-tuning for sophisticated meme understanding. Cross-attention fusion model. A/B testing infrastructure for model updates. Full autonomous operation with human review only for edge cases.

**Essential libraries**:
```
transformers>=4.35.0
setfit>=1.0.0
peft>=0.6.0
sentence-transformers>=2.2.0
optimum[onnxruntime]>=1.13.0
river>=0.21.0  # drift detection
avalanche-lib>=0.4.0  # continual learning
mlflow>=2.8.0
```

## Key technical decisions and tradeoffs

**Why CryptoBERT over general models**: Pre-training on 3.2M crypto posts means the tokenizer and embeddings already understand domain vocabulary. Fine-tuning from FinBERT or general BERT requires learning this from scratch, wasting your limited labels on vocabulary acquisition rather than sentiment patterns.

**Why SetFit over traditional fine-tuning**: With 200 examples, traditional fine-tuning risks overfitting. SetFit's contrastive approach learns a similarity function that generalizes better from few examples. It also trains in seconds rather than hours.

**Why experience replay over EWC**: Elastic Weight Consolidation (EWC) requires computing Fisher information matrices and struggles with long task sequences in NLP. Experience replay is simpler, more effective, and computationally cheaper—just maintain a buffer and sample from it during training.

**Why not larger LLMs for everything**: A 7B parameter model like Mistral could theoretically handle all tasks but requires expensive GPU inference and is overkill for sentiment classification. The specialized encoder stack (CryptoBERT + CLIP + LLaVA for edge cases) is more efficient and interpretable.

The system architecture prioritizes **debuggability**—when sentiment predictions are wrong, you can trace whether the error came from text understanding, image interpretation, sarcasm detection, or fusion. This modularity accelerates iteration cycles during development and enables targeted improvements in production.

---

## Implementation Status

The following components have been implemented in `python-ml/src/`:

### Model Wrappers (`models/`)

| File | Description | Status |
|------|-------------|--------|
| `base.py` | `PredictionResult` dataclass, `BaseSentimentModel` abstract class | Ready |
| `setfit_model.py` | SetFit wrapper with predict/batch/embeddings | Ready |
| `cryptobert_model.py` | CryptoBERT with LoRA support | Ready |
| `llava_model.py` | LLaVA 7B with 4-bit quantization | Ready |

### Training Infrastructure (`training/`)

| File | Description | Status |
|------|-------------|--------|
| `data_loader.py` | Load labeled data from SQLite, map ratings to labels | Ready |
| `setfit_trainer.py` | Full SetFit training pipeline with MLflow | Ready |
| `runpod/setup.sh` | Environment setup for RunPod | Ready |
| `runpod/train_setfit.py` | CLI for SetFit training (~$0.06) | Ready |
| `runpod/train_cryptobert.py` | CLI for CryptoBERT LoRA (~$0.35) | Ready |
| `runpod/train_llava.py` | CLI for LLaVA QLoRA (~$1.50-3.00) | Ready |

### Active Learning (`active_learning/`)

| File | Description | Status |
|------|-------------|--------|
| `acquisition.py` | Hybrid uncertainty-diversity sampling | Ready |
| `labeler_integration.py` | Integration with labeling GUI | Ready |

### Continual Learning (`continual/`)

| File | Description | Status |
|------|-------------|--------|
| `replay_buffer.py` | Reservoir sampling (2000 examples) | Ready |
| `drift_detector.py` | ADWIN drift detection + vocabulary monitoring | Ready |

### Multi-Modal Fusion (`fusion/`)

| File | Description | Status |
|------|-------------|--------|
| `sarcasm_detector.py` | Text-image incongruity detection | Ready |
| `multimodal_pipeline.py` | Full fusion pipeline with weights | Ready |

### Production Inference (`inference/`)

| File | Description | Status |
|------|-------------|--------|
| `pipeline.py` | Batch inference pipeline | Ready |
| `onnx_exporter.py` | ONNX export with INT8 quantization | Ready |

### Modified Files

| File | Changes |
|------|---------|
| `text_analyzer.py` | Added ensemble mode (VADER 30% + ML 70%) |
| `labeler.py` | Added `--active-learning` flag for priority ordering |
| `config/settings.toml` | Added `[ml.*]` configuration sections |
| `pyproject.toml` | Added ML dependencies |

### Quick Start

```bash
# Install dependencies
cd python-ml
pip install -e .

# Train SetFit model (requires 200+ labeled posts)
python -m src.training.setfit_trainer --db ../data/posts.db --output models/setfit

# Use active learning for labeling
python -m src.labeler --active-learning --model models/setfit

# Enable ensemble mode in config/settings.toml
# [ml.ensemble]
# enabled = true
```

### Configuration

All ML settings are in `config/settings.toml` under `[ml.*]` sections:

```toml
[ml.ensemble]
enabled = false          # Enable after training models
vader_weight = 0.3
ml_weight = 0.7

[ml.fusion]
text_weight = 0.5
image_weight = 0.3
sarcasm_threshold = 0.7

[ml.active_learning]
uncertainty_weight = 0.6
diversity_clusters = 50

[ml.continual]
replay_buffer_size = 2000
drift_delta = 0.002
```

See [self-training-implementation.md](plans/self-training-implementation.md) for the full implementation plan.
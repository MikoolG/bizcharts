#!/bin/bash
# RunPod setup script for BizCharts training
# Run this script after starting a RunPod instance with an RTX 4090

set -e

echo "=== BizCharts Training Environment Setup ==="

# Update pip
pip install --upgrade pip

# Install PyTorch (should be pre-installed on RunPod)
echo "Checking PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Install core dependencies
echo "Installing dependencies..."
pip install transformers>=4.40.0
pip install setfit>=1.0.0
pip install sentence-transformers>=2.2.0
pip install datasets>=2.14.0
pip install peft>=0.6.0
pip install bitsandbytes>=0.41.0
pip install accelerate>=0.24.0
pip install scikit-learn>=1.3.0
pip install mlflow>=2.8.0
pip install tqdm

# Optional: Install for LLaVA training
# pip install llava  # If fine-tuning LLaVA

echo ""
echo "=== Downloading Base Models ==="

# Pre-download models to avoid timeout during training
python -c "
from transformers import AutoModel, AutoTokenizer
from setfit import SetFitModel

print('Downloading sentence-transformers/paraphrase-mpnet-base-v2...')
SetFitModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')

print('Downloading ElKulako/cryptobert...')
AutoModel.from_pretrained('ElKulako/cryptobert')
AutoTokenizer.from_pretrained('ElKulako/cryptobert')

print('Models downloaded successfully!')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To train SetFit:"
echo "  python train_setfit.py --db /workspace/data/posts.db --output /workspace/models/setfit"
echo ""
echo "To train CryptoBERT:"
echo "  python train_cryptobert.py --db /workspace/data/posts.db --output /workspace/models/cryptobert"
echo ""

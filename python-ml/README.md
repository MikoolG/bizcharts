# BizCharts ML

Python ML components for sentiment analysis of 4chan /biz/ board data.

## Components

- **Text Analysis**: VADER + SetFit + CryptoBERT ensemble
- **Image Analysis**: LLaVA for meme understanding
- **Active Learning**: Hybrid uncertainty-diversity sampling
- **Continual Learning**: Experience replay with drift detection

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Usage

See the [labeling guide](../docs/labeling-guide.md) and [self-training docs](../docs/self-training.md).

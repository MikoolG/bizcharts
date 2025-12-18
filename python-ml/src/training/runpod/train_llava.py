#!/usr/bin/env python3
"""
RunPod LLaVA Training Script

Fine-tune LLaVA with QLoRA for meme sentiment analysis.

Usage:
    python train_llava.py --images /workspace/data/images --output /workspace/models/llava

Cost estimate: ~$1.50-3.00 (2-4 hours on RTX 4090 at $0.69/hour)

NOTE: LLaVA fine-tuning requires:
- At least 500 labeled meme images
- ~15GB VRAM (with 4-bit quantization)
- Significant compute time

For most use cases, the pre-trained LLaVA works well with
the custom prompt template. Only fine-tune if accuracy is insufficient.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaVA for meme sentiment analysis"
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing labeled meme images",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="CSV file with image_path,sentiment,is_sarcastic columns",
    )
    parser.add_argument(
        "--output",
        default="/workspace/models/llava",
        help="Output directory",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Check for required packages
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            BitsAndBytesConfig,
            LlavaNextForConditionalGeneration,
            LlavaNextProcessor,
        )
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install with: pip install transformers peft bitsandbytes accelerate")
        sys.exit(1)

    # Verify GPU
    if not torch.cuda.is_available():
        logger.error("CUDA required for LLaVA training!")
        sys.exit(1)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.1f}GB")

    if vram_gb < 12:
        logger.warning("Less than 12GB VRAM - training may fail")

    # Load labels
    import csv

    labels_path = Path(args.labels)
    if not labels_path.exists():
        logger.error(f"Labels file not found: {args.labels}")
        sys.exit(1)

    training_data = []
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = Path(args.images) / row["image_path"]
            if image_path.exists():
                training_data.append({
                    "image": str(image_path),
                    "sentiment": row["sentiment"],
                    "is_sarcastic": row.get("is_sarcastic", "no").lower() == "yes",
                })

    logger.info(f"Found {len(training_data)} valid training examples")

    if len(training_data) < 100:
        logger.warning("Recommend at least 500 examples for LLaVA fine-tuning")

    # Load model with 4-bit quantization
    logger.info("Loading LLaVA 1.5 7B with 4-bit quantization...")

    model_id = "llava-hf/llava-1.5-7b-hf"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Apply LoRA
    logger.info("Applying LoRA adapters...")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create training prompt
    TRAIN_PROMPT = """Analyze this 4chan /biz/ meme for sentiment.
Response format:
SENTIMENT: {sentiment}
SARCASM: {sarcasm}"""

    # Training loop (simplified - use Trainer for production)
    logger.info("Starting training...")
    logger.info("NOTE: Full LLaVA training requires custom dataset class")
    logger.info("This script provides the structure - adapt for your needs")

    # For actual training, you would:
    # 1. Create a custom Dataset class for image-text pairs
    # 2. Use HuggingFace Trainer with custom data collator
    # 3. Handle multi-modal batching carefully

    # Save model structure
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter config
    with open(output_path / "lora_config.json", "w") as f:
        import json
        json.dump({
            "r": args.lora_r,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "base_model": model_id,
        }, f, indent=2)

    print("\n" + "=" * 50)
    print("LLaVA training script initialized.")
    print("For full training, implement custom Dataset class.")
    print("See HuggingFace examples for multi-modal fine-tuning.")
    print("=" * 50)


if __name__ == "__main__":
    main()

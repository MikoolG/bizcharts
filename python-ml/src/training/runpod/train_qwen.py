#!/usr/bin/env python3
"""
RunPod Qwen2.5-VL Training Script

Fine-tune Qwen2.5-VL-7B with QLoRA for crypto meme sentiment analysis.

Usage (on RunPod):
    python train_qwen.py \
        --dataset /workspace/data/training_data.json \
        --images /workspace/data/images \
        --output /workspace/models/qwen-sentiment \
        --epochs 3

Hardware requirements:
    - RTX 4090 (24GB VRAM) - recommended
    - RTX 3090 (24GB VRAM) - works with smaller batch size

Cost estimate: ~$2-5 (3-4 hours on RTX 4090 at $0.34-0.69/hour)

Training data format (ShareGPT style):
[
  {
    "messages": [
      {"role": "user", "content": "<image>\\nClassify this crypto post as bearish, neutral, or bullish."},
      {"role": "assistant", "content": "bearish"}
    ],
    "images": ["thread_12345.jpg"]
  }
]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def check_dependencies():
    """Verify all required packages are installed."""
    missing = []

    try:
        import torch

        if not torch.cuda.is_available():
            logger.error("CUDA not available! This script requires GPU.")
            return False
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram:.1f}GB")
        if vram < 20:
            logger.warning("Less than 20GB VRAM - may need to reduce batch size")
    except ImportError:
        missing.append("torch")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import peft
    except ImportError:
        missing.append("peft")

    try:
        import bitsandbytes
    except ImportError:
        missing.append("bitsandbytes")

    try:
        import trl
    except ImportError:
        missing.append("trl")

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        missing.append("qwen-vl-utils")

    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.info("Install with:")
        logger.info("  pip install torch transformers peft bitsandbytes trl accelerate")
        logger.info("  pip install qwen-vl-utils")
        return False

    return True


def load_dataset(dataset_path: Path, images_dir: Path) -> list[dict]:
    """Load training dataset in ShareGPT format.

    Expected format:
    [
      {
        "messages": [
          {"role": "user", "content": "<image>\nClassify..."},
          {"role": "assistant", "content": "bearish"}
        ],
        "images": ["filename.jpg"]
      }
    ]
    """
    with open(dataset_path) as f:
        data = json.load(f)

    # Validate and resolve image paths
    valid_samples = []
    for sample in data:
        if "images" not in sample or not sample["images"]:
            logger.warning(f"Sample missing images, skipping")
            continue

        # Resolve image path
        image_name = sample["images"][0]
        image_path = images_dir / image_name
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        sample["images"] = [str(image_path)]
        valid_samples.append(sample)

    logger.info(f"Loaded {len(valid_samples)}/{len(data)} valid samples")
    return valid_samples


def prepare_dataset_for_training(samples: list[dict]):
    """Convert samples to HuggingFace dataset format for TRL."""
    from datasets import Dataset

    # TRL SFTTrainer expects specific format
    formatted = []
    for sample in samples:
        formatted.append(
            {
                "messages": sample["messages"],
                "images": sample["images"],
            }
        )

    return Dataset.from_list(formatted)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL for crypto sentiment"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to training data JSON (ShareGPT format)",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing training images",
    )
    parser.add_argument(
        "--output",
        default="/workspace/models/qwen-sentiment",
        help="Output directory for trained model",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--resume",
        help="Resume from checkpoint directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Check dependencies
    if not check_dependencies():
        return 1

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    from trl import SFTConfig, SFTTrainer

    # Load dataset
    dataset_path = Path(args.dataset)
    images_dir = Path(args.images)

    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return 1

    samples = load_dataset(dataset_path, images_dir)
    if len(samples) < 100:
        logger.warning(
            f"Only {len(samples)} samples - recommend 1000+ for good results"
        )

    dataset = prepare_dataset_for_training(samples)

    # Model configuration
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    logger.info(f"Loading model: {model_id}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    # Load model with quantization
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training configuration
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=args.max_length,
        dataset_text_field="",  # Not used with messages format
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="none",  # Disable wandb/mlflow
    )

    # Custom collator for vision-language model
    def collate_fn(examples):
        """Custom collator for Qwen2.5-VL with images."""
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        texts = []
        images_list = []

        for example in examples:
            messages = example["messages"]
            image_path = example["images"][0]

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Format messages for Qwen
            formatted_messages = []
            for msg in messages:
                content = msg["content"]
                if msg["role"] == "user" and "<image>" in content:
                    # Replace <image> placeholder with actual image reference
                    formatted_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {
                                    "type": "text",
                                    "text": content.replace("<image>", "").strip(),
                                },
                            ],
                        }
                    )
                else:
                    formatted_messages.append(
                        {
                            "role": msg["role"],
                            "content": msg["content"],
                        }
                    )

            # Apply chat template
            text = processor.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            # Process vision info
            image_inputs, video_inputs = process_vision_info(formatted_messages)
            images_list.append(image_inputs)

        # Tokenize
        inputs = processor(
            text=texts,
            images=images_list[0] if images_list else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )

        # Add labels (same as input_ids for causal LM)
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        processing_class=processor,
    )

    # Train
    logger.info("Starting training...")
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # Save LoRA config
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(
            {
                "base_model": model_id,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "samples": len(samples),
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print(f"Samples trained on: {len(samples)}")
    print(f"Epochs: {args.epochs}")
    print("=" * 50)
    print("\nTo use this model for inference:")
    print(f"  Load base model: {model_id}")
    print(f"  Load LoRA adapter from: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
RunPod CryptoBERT Training Script

Fine-tune CryptoBERT with LoRA for crypto sentiment classification.

Usage:
    python train_cryptobert.py --db /workspace/data/posts.db --output /workspace/models/cryptobert

Cost estimate: ~$0.35 (30 minutes on RTX 4090 at $0.69/hour)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datetime import datetime

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CryptoBERT for /biz/ sentiment analysis"
    )
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument(
        "--output",
        default="/workspace/models/cryptobert",
        help="Output directory",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Verify GPU
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be slow.")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    from src.training.data_loader import load_labeled_data

    logger.info(f"Loading data from {args.db}")
    dataset = load_labeled_data(args.db)

    if len(dataset) == 0:
        logger.error("No training data found!")
        sys.exit(1)

    logger.info(f"Loaded {len(dataset)} examples")

    # Split data
    split = dataset.train_test_split(test_size=args.test_size, seed=42, stratify_by_column="label")
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Load tokenizer and model
    model_name = "ElKulako/cryptobert"
    logger.info(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "bearish", 1: "neutral", 2: "bullish"},
        label2id={"bearish": 0, "neutral": 1, "bullish": 2},
    )

    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize data
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    # Map labels to integers
    label_map = {"bearish": 0, "neutral": 1, "bullish": 2}

    def add_labels(examples):
        examples["label"] = [label_map[l] for l in examples["label"]]
        return examples

    train_dataset = train_dataset.map(add_labels, batched=True)
    eval_dataset = eval_dataset.map(add_labels, batched=True)
    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="mlflow" if args.mlflow else "none",
    )

    # Compute metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="macro"),
        }

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    start_time = datetime.now()

    trainer.train()

    elapsed = datetime.now() - start_time
    logger.info(f"Training completed in {elapsed}")

    # Evaluate
    metrics = trainer.evaluate()
    logger.info(f"Final metrics: {metrics}")

    # Save model
    logger.info(f"Saving model to {args.output}")

    # Merge LoRA weights and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("\n" + "=" * 50)
    print("SUCCESS! CryptoBERT fine-tuned and saved.")
    print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"F1: {metrics['eval_f1']:.4f}")
    print(f"Download from: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()

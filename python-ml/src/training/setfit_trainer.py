"""SetFit training pipeline."""

import logging
from pathlib import Path

from datasets import Dataset

from .data_loader import load_labeled_data

logger = logging.getLogger(__name__)


def train_setfit(
    db_path: str | Path | None = None,
    dataset: Dataset | None = None,
    output_dir: str | Path = "models/setfit",
    batch_size: int = 16,
    num_epochs: int = 4,
    num_iterations: int = 20,
    test_size: float = 0.2,
    seed: int = 42,
    use_mlflow: bool = False,
):
    """Train SetFit model on labeled data.

    SetFit uses contrastive learning to achieve high accuracy with
    minimal training examples. It generates text pairs from labels,
    trains a sentence transformer to create sentiment-aware embeddings,
    then fits a classification head.

    Args:
        db_path: Path to SQLite database with training_labels
        dataset: Pre-loaded HuggingFace Dataset (alternative to db_path)
        output_dir: Directory to save trained model
        batch_size: Training batch size
        num_epochs: Number of training epochs
        num_iterations: Number of contrastive pairs per sample
        test_size: Fraction of data for evaluation
        seed: Random seed for reproducibility
        use_mlflow: Whether to log metrics to MLflow

    Returns:
        Trained SetFitModel and evaluation metrics
    """
    from setfit import SetFitModel, Trainer, TrainingArguments

    # Load data
    if dataset is None:
        if db_path is None:
            raise ValueError("Either db_path or dataset must be provided")
        dataset = load_labeled_data(db_path)

    if len(dataset) == 0:
        raise ValueError("No training data found")

    logger.info(f"Loaded {len(dataset)} training examples")

    # Check label distribution
    from collections import Counter

    label_counts = Counter(dataset["label"])
    logger.info(f"Label distribution: {dict(label_counts)}")

    # Split data
    split = dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
    train_dataset = split["train"]
    eval_dataset = split["test"]

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Initialize model
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2",
        labels=["bearish", "neutral", "bullish"],
    )

    # Training arguments
    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        seed=seed,
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # MLflow tracking
    if use_mlflow:
        import mlflow

        mlflow.set_experiment("bizcharts-setfit")
        with mlflow.start_run(run_name="setfit-training"):
            mlflow.log_params(
                {
                    "model": "paraphrase-mpnet-base-v2",
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "num_iterations": num_iterations,
                    "train_size": len(train_dataset),
                    "eval_size": len(eval_dataset),
                }
            )

            # Train
            trainer.train()

            # Evaluate
            metrics = trainer.evaluate()
            mlflow.log_metrics(metrics)

            # Save model
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_path))
            mlflow.log_artifacts(str(output_path))

            return model, metrics
    else:
        # Train without MLflow
        trainer.train()

        # Evaluate
        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")

        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path))
        logger.info(f"Model saved to {output_path}")

        return model, metrics


def evaluate_setfit(
    model_path: str | Path,
    db_path: str | Path | None = None,
    dataset: Dataset | None = None,
) -> dict:
    """Evaluate a trained SetFit model.

    Args:
        model_path: Path to trained model
        db_path: Path to database for evaluation data
        dataset: Pre-loaded evaluation dataset

    Returns:
        Dictionary of evaluation metrics
    """
    from setfit import SetFitModel
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    # Load model
    model = SetFitModel.from_pretrained(str(model_path))

    # Load data
    if dataset is None:
        if db_path is None:
            raise ValueError("Either db_path or dataset must be provided")
        dataset = load_labeled_data(db_path)

    # Get predictions
    predictions = model.predict(dataset["text"])
    labels = dataset["label"]

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")

    report = classification_report(labels, predictions, output_dict=True)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
    }


def main():
    """CLI entry point for SetFit training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train SetFit sentiment model")
    parser.add_argument("--db", required=True, help="Path to posts.db")
    parser.add_argument("--output", default="models/setfit", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model, metrics = train_setfit(
        db_path=args.db,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_iterations=args.iterations,
        test_size=args.test_size,
        use_mlflow=args.mlflow,
    )

    print(f"\nTraining complete!")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")


if __name__ == "__main__":
    main()

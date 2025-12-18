#!/usr/bin/env python3
"""
RunPod SetFit Training Script

Train a SetFit model on labeled 4chan /biz/ sentiment data.

Usage:
    python train_setfit.py --db /workspace/data/posts.db --output /workspace/models/setfit

Cost estimate: ~$0.06 (5 minutes on RTX 4090 at $0.69/hour)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Train SetFit model for /biz/ sentiment analysis"
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to SQLite database with training_labels",
    )
    parser.add_argument(
        "--output",
        default="/workspace/models/setfit",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of contrastive pairs per sample",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for evaluation",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking",
    )
    parser.add_argument(
        "--mlflow-uri",
        default="file:///workspace/mlruns",
        help="MLflow tracking URI",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Verify database exists
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {args.db}")
        logger.info("Upload your posts.db to /workspace/data/")
        sys.exit(1)

    # Import training modules
    try:
        from src.training.setfit_trainer import train_setfit, evaluate_setfit
        from src.training.data_loader import load_labeled_data, get_label_distribution
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure project is properly set up")
        sys.exit(1)

    # Check data
    logger.info(f"Loading data from {args.db}")
    distribution = get_label_distribution(args.db)
    logger.info(f"Label distribution: {distribution}")

    total_labels = sum(distribution.values())
    if total_labels < 50:
        logger.warning(f"Only {total_labels} labels found. Recommend at least 200 for good results.")

    # Setup MLflow
    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("bizcharts-setfit")

    # Train
    logger.info("Starting SetFit training...")
    start_time = datetime.now()

    model, metrics = train_setfit(
        db_path=args.db,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_iterations=args.iterations,
        test_size=args.test_size,
        use_mlflow=args.mlflow,
    )

    elapsed = datetime.now() - start_time
    logger.info(f"Training completed in {elapsed}")

    # Log results
    logger.info("=" * 50)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 50)
    logger.info(f"Model saved to: {args.output}")
    logger.info(f"Training time: {elapsed}")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    # Quick evaluation
    logger.info("\nRunning evaluation...")
    eval_metrics = evaluate_setfit(args.output, args.db)
    logger.info(f"Final accuracy: {eval_metrics['accuracy']:.4f}")
    logger.info(f"F1 (macro): {eval_metrics['f1_macro']:.4f}")

    print("\n" + "=" * 50)
    print("SUCCESS! Model trained and saved.")
    print(f"Download from: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()

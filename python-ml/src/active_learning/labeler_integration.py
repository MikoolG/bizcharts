"""Integration with labeler.py for active learning suggestions."""

import logging
import sqlite3
from pathlib import Path

from .acquisition import batch_acquisition, hybrid_acquisition

logger = logging.getLogger(__name__)


class ActiveLearningLabeler:
    """Extends labeler with active learning suggestions.

    Uses hybrid uncertainty-diversity sampling to prioritize which posts
    should be labeled next for maximum model improvement.
    """

    def __init__(
        self,
        db_path: str | Path,
        model=None,
        model_path: str | Path | None = None,
    ):
        """Initialize active learning labeler.

        Args:
            db_path: Path to SQLite database
            model: Pre-loaded model with predict_proba() and get_embeddings()
            model_path: Path to load model from (alternative to passing model)
        """
        self.db_path = str(db_path)
        self.model = model

        if self.model is None and model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str | Path):
        """Load model from path."""
        from ..models.setfit_model import SetFitSentimentModel

        self.model = SetFitSentimentModel(model_path=model_path)
        logger.info(f"Loaded model from {model_path}")

    def get_unlabeled_posts(
        self,
        limit: int = 1000,
        with_images: bool = False,
        min_replies: int = 0,
    ) -> list[tuple[int, str, str | None]]:
        """Get unlabeled posts from database.

        Args:
            limit: Maximum number of posts to return
            with_images: Only return posts with images
            min_replies: Minimum reply count filter

        Returns:
            List of (thread_id, text, image_url) tuples
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                t.thread_id,
                t.op_text_clean as text,
                t.thumbnail_url as image_url
            FROM thread_ops t
            LEFT JOIN training_labels l ON t.thread_id = l.thread_id
            WHERE l.id IS NULL
              AND t.op_text_clean IS NOT NULL
              AND t.op_text_clean != ''
              AND t.reply_count >= ?
        """

        if with_images:
            query += " AND t.thumbnail_url IS NOT NULL"

        query += f" ORDER BY t.reply_count DESC LIMIT {limit}"

        cursor = conn.execute(query, (min_replies,))
        posts = cursor.fetchall()
        conn.close()

        return posts

    def get_suggested_posts(
        self,
        limit: int = 100,
        pool_size: int = 1000,
        with_images: bool = False,
    ) -> list[int]:
        """Get thread_ids of posts most valuable for labeling.

        Uses hybrid acquisition (uncertainty + diversity) to select posts
        that will provide maximum information gain when labeled.

        Args:
            limit: Number of posts to suggest
            pool_size: Size of candidate pool to select from
            with_images: Only consider posts with images

        Returns:
            List of thread_ids ordered by labeling priority
        """
        if self.model is None:
            raise ValueError("Model required for active learning. Provide model or model_path.")

        # Get unlabeled posts
        posts = self.get_unlabeled_posts(limit=pool_size, with_images=with_images)

        if not posts:
            logger.warning("No unlabeled posts found")
            return []

        thread_ids = [p[0] for p in posts]
        texts = [p[1] for p in posts]

        logger.info(f"Running acquisition on {len(texts)} candidates")

        # Get model predictions and embeddings
        probs = self.model.predict_proba(texts)
        embeddings = self.model.get_embeddings(texts)

        # Run hybrid acquisition
        selected_indices = hybrid_acquisition(probs, embeddings, n_select=limit)

        # Return thread_ids in priority order
        return [thread_ids[i] for i in selected_indices]

    def get_posts_by_priority(
        self,
        limit: int = 100,
        pool_size: int = 1000,
    ) -> list[dict]:
        """Get full post data ordered by labeling priority.

        Args:
            limit: Number of posts to return
            pool_size: Size of candidate pool

        Returns:
            List of post dicts with thread_id, text, image_url, uncertainty
        """
        if self.model is None:
            raise ValueError("Model required for active learning")

        posts = self.get_unlabeled_posts(limit=pool_size)

        if not posts:
            return []

        thread_ids = [p[0] for p in posts]
        texts = [p[1] for p in posts]
        image_urls = [p[2] for p in posts]

        # Get predictions
        probs = self.model.predict_proba(texts)
        embeddings = self.model.get_embeddings(texts)

        # Compute uncertainty (entropy)
        uncertainty = -sum(
            probs[:, i] * (probs[:, i] + 1e-10) for i in range(probs.shape[1])
        )

        # Run acquisition
        selected_indices = hybrid_acquisition(probs, embeddings, n_select=limit)

        # Build result with uncertainty scores
        results = []
        for idx in selected_indices:
            pred = self.model.predict(texts[idx])
            results.append(
                {
                    "thread_id": thread_ids[idx],
                    "text": texts[idx],
                    "image_url": image_urls[idx],
                    "uncertainty": float(uncertainty[idx]),
                    "predicted_label": pred.label,
                    "predicted_confidence": pred.confidence,
                }
            )

        return results

    def update_after_label(self, thread_id: int, label: str):
        """Called after a post is labeled to update internal state.

        Can be used to trigger incremental model updates or buffer management.

        Args:
            thread_id: ID of the labeled post
            label: Label that was assigned
        """
        # Placeholder for future incremental learning
        logger.debug(f"Labeled thread {thread_id} as {label}")


def main():
    """CLI for testing active learning suggestions."""
    import argparse

    parser = argparse.ArgumentParser(description="Get active learning suggestions")
    parser.add_argument("--db", required=True, help="Path to posts.db")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--limit", type=int, default=20, help="Number of suggestions")
    parser.add_argument("--pool", type=int, default=500, help="Candidate pool size")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    al = ActiveLearningLabeler(args.db, model_path=args.model)
    posts = al.get_posts_by_priority(limit=args.limit, pool_size=args.pool)

    print(f"\nTop {len(posts)} posts to label:\n")
    for i, post in enumerate(posts, 1):
        print(f"{i}. Thread {post['thread_id']}")
        print(f"   Uncertainty: {post['uncertainty']:.4f}")
        print(f"   Predicted: {post['predicted_label']} ({post['predicted_confidence']:.2%})")
        print(f"   Text: {post['text'][:100]}...")
        print()


if __name__ == "__main__":
    main()

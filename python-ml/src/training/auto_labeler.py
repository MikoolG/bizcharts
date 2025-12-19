#!/usr/bin/env python3
"""
Auto-labeling script using vision APIs (GPT-4o-mini or Gemini 2.0 Flash).

Generates training labels for multimodal sentiment analysis by sending
image + text to a vision API and collecting sentiment predictions.

Usage:
    # Using GPT-4o-mini (fast, ~$2 for 10k images)
    python -m src.training.auto_labeler --db data/posts.db --api openai --output data/auto_labels.json

    # Using Gemini 2.0 Flash (free tier, rate limited)
    python -m src.training.auto_labeler --db data/posts.db --api gemini --output data/auto_labels.json

    # Resume from checkpoint
    python -m src.training.auto_labeler --db data/posts.db --api openai --output data/auto_labels.json --resume

Cost estimates (10k images):
    - GPT-4o-mini: ~$1.70-2.50
    - Gemini 2.0 Flash (free tier): $0 (rate limited to ~1500/day)
"""

import argparse
import base64
import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

# API endpoints
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Sentiment classification prompt
CLASSIFICATION_PROMPT = """Analyze this 4chan /biz/ cryptocurrency post for market sentiment.

Look at BOTH the image AND the text together. Consider:
- Meme imagery (Wojak variants, Pepe, charts)
- Text overlays on the image
- The post text below the image
- Crypto slang: WAGMI (bullish), NGMI (bearish), rekt, moon, etc.
- Sarcasm and irony (common on /biz/)

Post text: {post_text}

Classify as exactly one of: bearish, neutral, bullish

Respond with ONLY the single word: bearish, neutral, or bullish"""


@dataclass
class LabelResult:
    """Result from auto-labeling a single post."""

    thread_id: int
    label: str  # bearish, neutral, bullish
    confidence: float  # 0-1 based on model's certainty
    api_used: str  # openai or gemini
    post_text: str
    image_url: str
    timestamp: str


def encode_image_from_url(image_url: str, client: httpx.Client) -> str | None:
    """Download image and encode to base64."""
    try:
        response = client.get(image_url, timeout=10.0)
        response.raise_for_status()
        return base64.standard_b64encode(response.content).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to download image {image_url}: {e}")
        return None


def encode_image_from_path(image_path: Path) -> str | None:
    """Read local image and encode to base64."""
    try:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to read image {image_path}: {e}")
        return None


def call_openai(
    image_base64: str,
    post_text: str,
    api_key: str,
    client: httpx.Client,
) -> tuple[str, float]:
    """Call GPT-4o-mini vision API.

    Returns:
        Tuple of (label, confidence)
    """
    prompt = CLASSIFICATION_PROMPT.format(post_text=post_text or "[no text]")

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low",  # Use low detail to reduce cost
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 10,
        "temperature": 0,  # Deterministic output
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = client.post(OPENAI_API_URL, json=payload, headers=headers, timeout=30.0)
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"].strip().lower()

    # Parse response
    label = "neutral"  # default
    if "bearish" in content:
        label = "bearish"
    elif "bullish" in content:
        label = "bullish"
    elif "neutral" in content:
        label = "neutral"

    # GPT-4o-mini doesn't provide logprobs easily, so estimate confidence
    # based on whether the response was clean
    confidence = 0.85 if content in ["bearish", "neutral", "bullish"] else 0.6

    return label, confidence


def call_gemini(
    image_base64: str,
    post_text: str,
    api_key: str,
    client: httpx.Client,
) -> tuple[str, float]:
    """Call Gemini 2.0 Flash vision API.

    Returns:
        Tuple of (label, confidence)
    """
    prompt = CLASSIFICATION_PROMPT.format(post_text=post_text or "[no text]")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64,
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 10,
        },
    }

    url = f"{GEMINI_API_URL}?key={api_key}"
    response = client.post(url, json=payload, timeout=30.0)
    response.raise_for_status()

    data = response.json()
    content = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()

    # Parse response
    label = "neutral"  # default
    if "bearish" in content:
        label = "bearish"
    elif "bullish" in content:
        label = "bullish"
    elif "neutral" in content:
        label = "neutral"

    confidence = 0.80 if content in ["bearish", "neutral", "bullish"] else 0.55

    return label, confidence


def load_unlabeled_posts(
    db_path: Path,
    exclude_thread_ids: set[int],
    limit: int | None = None,
) -> list[dict]:
    """Load posts from database that haven't been labeled yet.

    Excludes posts already in training_labels (manual) and those in exclude_thread_ids.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT t.thread_id, t.op_text_clean as text, t.thumbnail_url as image_url
        FROM thread_ops t
        LEFT JOIN training_labels l ON t.thread_id = l.thread_id
        WHERE l.id IS NULL
          AND t.thumbnail_url IS NOT NULL
          AND t.thumbnail_url != ''
        ORDER BY t.reply_count DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)
    posts = []
    for row in cursor.fetchall():
        if row["thread_id"] not in exclude_thread_ids:
            posts.append(dict(row))

    conn.close()
    return posts


def load_checkpoint(output_path: Path) -> tuple[list[LabelResult], set[int]]:
    """Load existing results from checkpoint file."""
    if not output_path.exists():
        return [], set()

    results = []
    labeled_ids = set()

    with open(output_path) as f:
        data = json.load(f)
        for item in data:
            results.append(LabelResult(**item))
            labeled_ids.add(item["thread_id"])

    return results, labeled_ids


def save_checkpoint(results: list[LabelResult], output_path: Path):
    """Save results to checkpoint file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Auto-label posts using vision APIs")
    parser.add_argument("--db", required=True, help="Path to posts.db")
    parser.add_argument(
        "--api",
        choices=["openai", "gemini"],
        required=True,
        help="API to use for labeling",
    )
    parser.add_argument(
        "--output",
        default="data/auto_labels.json",
        help="Output JSON file for labels",
    )
    parser.add_argument("--limit", type=int, help="Max posts to label")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Save checkpoint every N posts",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Get API key
    if args.api == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("Set OPENAI_API_KEY environment variable")
            return 1
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("Set GEMINI_API_KEY environment variable")
            return 1

    db_path = Path(args.db)
    output_path = Path(args.output)

    # Load checkpoint if resuming
    results, labeled_ids = [], set()
    if args.resume and output_path.exists():
        results, labeled_ids = load_checkpoint(output_path)
        logger.info(f"Resuming from checkpoint: {len(results)} posts already labeled")

    # Load posts to label
    posts = load_unlabeled_posts(db_path, labeled_ids, args.limit)
    logger.info(f"Found {len(posts)} posts to label")

    if not posts:
        logger.info("No posts to label")
        return 0

    # Label posts
    client = httpx.Client()
    call_api = call_openai if args.api == "openai" else call_gemini

    stats = {"bearish": 0, "neutral": 0, "bullish": 0, "errors": 0}
    start_time = time.time()

    for i, post in enumerate(posts):
        try:
            # Download/encode image
            image_base64 = encode_image_from_url(post["image_url"], client)
            if not image_base64:
                stats["errors"] += 1
                continue

            # Call API
            label, confidence = call_api(
                image_base64,
                post["text"],
                api_key,
                client,
            )

            # Create result
            result = LabelResult(
                thread_id=post["thread_id"],
                label=label,
                confidence=confidence,
                api_used=args.api,
                post_text=post["text"] or "",
                image_url=post["image_url"],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            results.append(result)
            stats[label] += 1

            # Progress
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 3600  # posts/hour
                logger.info(
                    f"Progress: {i + 1}/{len(posts)} "
                    f"({rate:.0f}/hr) - {label}"
                )

            # Checkpoint
            if (i + 1) % args.batch_size == 0:
                save_checkpoint(results, output_path)
                logger.info(f"Checkpoint saved: {len(results)} total")

            # Rate limit
            time.sleep(args.rate_limit)

        except httpx.HTTPStatusError as e:
            logger.error(f"API error for thread {post['thread_id']}: {e}")
            stats["errors"] += 1
            if e.response.status_code == 429:
                logger.warning("Rate limited - waiting 60s")
                time.sleep(60)
        except Exception as e:
            logger.error(f"Error labeling thread {post['thread_id']}: {e}")
            stats["errors"] += 1

    # Final save
    save_checkpoint(results, output_path)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("Auto-labeling complete!")
    print(f"Total labeled: {len(results)}")
    print(f"Distribution: {stats}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Output saved: {output_path}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())

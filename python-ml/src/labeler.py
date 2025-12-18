#!/usr/bin/env python3
"""
BizCharts Sentiment Labeling Tool

A GUI tool for manually labeling thread OP sentiment to create training data.

Usage:
    python -m src.labeler                           # Label all unlabeled posts
    python -m src.labeler --source warosu           # Label only warosu posts
    python -m src.labeler --source live             # Label only live posts
    python -m src.labeler --from 2024-01-01 --to 2024-06-30  # Date range
    python -m src.labeler --session my_session      # Named labeling session
    python -m src.labeler --export labels.csv       # Export labeled data
    python -m src.labeler --active-learning --model models/setfit  # Priority ordering

Controls:
    1       : Very bearish (strong sell signals, panic, despair)
    2       : Somewhat bearish (negative outlook, concerns)
    3       : Neutral/Crab (sideways, uncertain, mixed signals)
    4       : Somewhat bullish (positive outlook, optimism)
    5       : Very bullish (strong buy signals, euphoria, WAGMI)
    0/S     : Not relevant to sentiment (skip)
    Left    : Go to previous post
    Right   : Go to next post
    N       : Add a note to current post
    Q/Esc   : Quit and save progress
"""

import argparse
import io
import sqlite3
import sys
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not installed. Run: pip install Pillow")
    print("Images will not be displayed.")


@dataclass
class ThreadOP:
    """Thread OP data for labeling."""
    thread_id: int
    subject: Optional[str]
    op_text: Optional[str]
    op_text_clean: Optional[str]
    image_url: Optional[str]
    thumbnail_url: Optional[str]
    local_thumbnail_path: Optional[str]  # Local path to downloaded thumbnail
    reply_count: int
    created_at: int
    source: str
    # Existing label if any
    existing_rating: Optional[int] = None
    existing_skipped: bool = False


class LabelingSession:
    """Manages the labeling session and database operations."""

    def __init__(self, db_path: str, labeler_id: str = "default"):
        self.db_path = db_path
        self.labeler_id = labeler_id
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        # Ensure training_labels table exists
        self._ensure_schema()

    def _ensure_schema(self):
        """Create training_labels table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER,
                sentiment_rating INTEGER CHECK(sentiment_rating >= 1 AND sentiment_rating <= 5),
                skipped BOOLEAN DEFAULT FALSE,
                notes TEXT,
                labeler_id TEXT DEFAULT 'default',
                labeled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                text_snapshot TEXT,
                image_url_snapshot TEXT,
                UNIQUE(thread_id, labeler_id)
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_thread ON training_labels(thread_id)")
        self.conn.commit()

    def get_unlabeled_posts(self, limit: int = 1000, source: Optional[str] = None,
                           date_from: Optional[str] = None, date_to: Optional[str] = None) -> list[ThreadOP]:
        """Get posts that haven't been labeled yet by this labeler."""
        # Build query with optional filters
        query = """
            SELECT
                t.thread_id, t.subject, t.op_text, t.op_text_clean,
                t.image_url, t.thumbnail_url, t.local_thumbnail_path,
                t.reply_count, t.created_at, t.source,
                l.sentiment_rating, l.skipped
            FROM thread_ops t
            LEFT JOIN training_labels l ON t.thread_id = l.thread_id AND l.labeler_id = ?
            WHERE l.id IS NULL
        """
        params = [self.labeler_id]

        if source:
            query += " AND t.source = ?"
            params.append(source)

        if date_from:
            # Convert YYYY-MM-DD to Unix timestamp
            try:
                dt = datetime.strptime(date_from, "%Y-%m-%d")
                query += " AND t.created_at >= ?"
                params.append(int(dt.timestamp()))
            except ValueError:
                print(f"Warning: Invalid date format for --from: {date_from}")

        if date_to:
            try:
                dt = datetime.strptime(date_to, "%Y-%m-%d")
                # End of day
                dt = dt.replace(hour=23, minute=59, second=59)
                query += " AND t.created_at <= ?"
                params.append(int(dt.timestamp()))
            except ValueError:
                print(f"Warning: Invalid date format for --to: {date_to}")

        query += " ORDER BY t.reply_count DESC, t.created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)

        posts = []
        for row in cursor:
            posts.append(ThreadOP(
                thread_id=row["thread_id"],
                subject=row["subject"],
                op_text=row["op_text"],
                op_text_clean=row["op_text_clean"],
                image_url=row["image_url"],
                thumbnail_url=row["thumbnail_url"],
                local_thumbnail_path=row["local_thumbnail_path"],
                reply_count=row["reply_count"] or 0,
                created_at=row["created_at"],
                source=row["source"],
                existing_rating=row["sentiment_rating"],
                existing_skipped=bool(row["skipped"]),
            ))
        return posts

    def get_all_posts(self, limit: int = 1000, source: Optional[str] = None,
                     date_from: Optional[str] = None, date_to: Optional[str] = None) -> list[ThreadOP]:
        """Get all posts (for review mode)."""
        query = """
            SELECT
                t.thread_id, t.subject, t.op_text, t.op_text_clean,
                t.image_url, t.thumbnail_url, t.local_thumbnail_path,
                t.reply_count, t.created_at, t.source,
                l.sentiment_rating, l.skipped
            FROM thread_ops t
            LEFT JOIN training_labels l ON t.thread_id = l.thread_id AND l.labeler_id = ?
            WHERE 1=1
        """
        params = [self.labeler_id]

        if source:
            query += " AND t.source = ?"
            params.append(source)

        if date_from:
            try:
                dt = datetime.strptime(date_from, "%Y-%m-%d")
                query += " AND t.created_at >= ?"
                params.append(int(dt.timestamp()))
            except ValueError:
                pass

        if date_to:
            try:
                dt = datetime.strptime(date_to, "%Y-%m-%d")
                dt = dt.replace(hour=23, minute=59, second=59)
                query += " AND t.created_at <= ?"
                params.append(int(dt.timestamp()))
            except ValueError:
                pass

        query += " ORDER BY t.reply_count DESC, t.created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)

        posts = []
        for row in cursor:
            posts.append(ThreadOP(
                thread_id=row["thread_id"],
                subject=row["subject"],
                op_text=row["op_text"],
                op_text_clean=row["op_text_clean"],
                image_url=row["image_url"],
                thumbnail_url=row["thumbnail_url"],
                local_thumbnail_path=row["local_thumbnail_path"],
                reply_count=row["reply_count"] or 0,
                created_at=row["created_at"],
                source=row["source"],
                existing_rating=row["sentiment_rating"],
                existing_skipped=bool(row["skipped"]),
            ))
        return posts

    def save_label(self, thread_id: int, rating: Optional[int], skipped: bool,
                   text_snapshot: str, image_url: str, notes: Optional[str] = None):
        """Save a label for a thread."""
        self.conn.execute("""
            INSERT INTO training_labels
                (thread_id, sentiment_rating, skipped, notes, labeler_id,
                 labeled_at, text_snapshot, image_url_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id, labeler_id) DO UPDATE SET
                sentiment_rating = excluded.sentiment_rating,
                skipped = excluded.skipped,
                notes = excluded.notes,
                labeled_at = excluded.labeled_at
        """, (thread_id, rating, skipped, notes, self.labeler_id,
              datetime.now().isoformat(), text_snapshot, image_url))
        self.conn.commit()

    def get_stats(self, source: Optional[str] = None) -> dict:
        """Get labeling statistics."""
        if source:
            cursor = self.conn.execute("""
                SELECT
                    COUNT(*) as total_labeled,
                    SUM(CASE WHEN l.skipped THEN 1 ELSE 0 END) as skipped,
                    AVG(CASE WHEN NOT l.skipped THEN l.sentiment_rating END) as avg_rating,
                    COUNT(DISTINCT l.thread_id) as unique_threads
                FROM training_labels l
                JOIN thread_ops t ON l.thread_id = t.thread_id
                WHERE l.labeler_id = ? AND t.source = ?
            """, (self.labeler_id, source))
        else:
            cursor = self.conn.execute("""
                SELECT
                    COUNT(*) as total_labeled,
                    SUM(CASE WHEN skipped THEN 1 ELSE 0 END) as skipped,
                    AVG(CASE WHEN NOT skipped THEN sentiment_rating END) as avg_rating,
                    COUNT(DISTINCT thread_id) as unique_threads
                FROM training_labels
                WHERE labeler_id = ?
            """, (self.labeler_id,))
        row = cursor.fetchone()
        return {
            "total_labeled": row["total_labeled"] or 0,
            "skipped": row["skipped"] or 0,
            "avg_rating": round(row["avg_rating"], 2) if row["avg_rating"] else None,
            "unique_threads": row["unique_threads"] or 0,
        }

    def export_labels(self, output_path: str, source: Optional[str] = None):
        """Export labels to CSV."""
        import csv

        query = """
            SELECT
                l.thread_id,
                l.sentiment_rating,
                l.skipped,
                l.notes,
                l.labeler_id,
                l.labeled_at,
                l.text_snapshot,
                l.image_url_snapshot,
                t.subject,
                t.reply_count,
                t.source,
                date(t.created_at, 'unixepoch') as created_date
            FROM training_labels l
            JOIN thread_ops t ON l.thread_id = t.thread_id
        """
        params = []

        if source:
            query += " WHERE t.source = ?"
            params.append(source)

        query += " ORDER BY l.labeled_at"

        cursor = self.conn.execute(query, params)

        count = 0
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "thread_id", "sentiment_rating", "skipped", "notes", "labeler_id",
                "labeled_at", "text_snapshot", "image_url", "subject", "reply_count",
                "source", "created_date"
            ])
            for row in cursor:
                writer.writerow(list(row))
                count += 1

        print(f"Exported {count} labels to {output_path}")


class LabelingGUI:
    """Tkinter GUI for labeling posts."""

    def __init__(self, session: LabelingSession, posts: list[ThreadOP], source_filter: Optional[str] = None):
        self.session = session
        self.posts = posts
        self.source_filter = source_filter
        self.current_index = 0
        self.current_note = ""

        # Cache for performance
        self._stats_cache = None
        self._stats_dirty = True
        self._image_cache = {}  # thread_id -> PhotoImage

        if not posts:
            print("No unlabeled posts found!")
            sys.exit(0)

        self.root = tk.Tk()
        self.root.title("BizCharts Sentiment Labeler")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e1e")

        self._setup_ui()
        self._bind_keys()
        self._preload_images()  # Preload first few images
        self._show_current_post()

    def _setup_ui(self):
        """Set up the UI components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Style configuration
        style = ttk.Style()
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Consolas", 11))
        style.configure("Header.TLabel", font=("Consolas", 14, "bold"))
        style.configure("Rating.TLabel", font=("Consolas", 24, "bold"))

        # Top bar with stats
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        self.stats_label = ttk.Label(top_frame, text="", style="TLabel")
        self.stats_label.pack(side=tk.LEFT)

        self.progress_label = ttk.Label(top_frame, text="", style="TLabel")
        self.progress_label.pack(side=tk.RIGHT)

        # Content area (two columns)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left column: Image
        left_frame = ttk.Frame(content_frame, width=500)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        left_frame.pack_propagate(False)

        self.image_label = ttk.Label(left_frame, text="No Image", style="TLabel")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Right column: Text and controls
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Thread info
        info_frame = ttk.Frame(right_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.thread_id_label = ttk.Label(info_frame, text="", style="Header.TLabel")
        self.thread_id_label.pack(anchor=tk.W)

        self.subject_label = ttk.Label(info_frame, text="", style="TLabel", wraplength=600)
        self.subject_label.pack(anchor=tk.W)

        self.meta_label = ttk.Label(info_frame, text="", style="TLabel")
        self.meta_label.pack(anchor=tk.W)

        # OP Text (scrollable)
        text_frame = ttk.Frame(right_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.text_widget = tk.Text(
            text_frame, wrap=tk.WORD, font=("Consolas", 11),
            bg="#2d2d2d", fg="#ffffff", insertbackground="#ffffff",
            padx=10, pady=10, state=tk.DISABLED
        )
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, command=self.text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.config(yscrollcommand=scrollbar.set)

        # Rating display
        rating_frame = ttk.Frame(right_frame)
        rating_frame.pack(fill=tk.X, pady=10)

        self.rating_label = ttk.Label(rating_frame, text="", style="Rating.TLabel")
        self.rating_label.pack()

        # Controls help
        help_text = """
Controls:
  1 = Very Bearish    2 = Bearish    3 = Neutral/Crab
  4 = Bullish         5 = Very Bullish
  0/S = Not relevant (skip)
  Left/Right = Navigate    N = Add note    Q/Esc = Quit
        """
        self.help_label = ttk.Label(right_frame, text=help_text.strip(), style="TLabel", justify=tk.LEFT)
        self.help_label.pack(anchor=tk.W, pady=(10, 0))

    def _bind_keys(self):
        """Bind keyboard shortcuts."""
        self.root.bind("<Left>", lambda e: self._prev_post())
        self.root.bind("<Right>", lambda e: self._next_post())
        self.root.bind("<Escape>", lambda e: self._quit())
        self.root.bind("q", lambda e: self._quit())
        self.root.bind("Q", lambda e: self._quit())
        self.root.bind("s", lambda e: self._skip_post())
        self.root.bind("S", lambda e: self._skip_post())
        self.root.bind("n", lambda e: self._add_note())
        self.root.bind("N", lambda e: self._add_note())

        # Number keys for ratings (1-5)
        for i in range(1, 6):
            self.root.bind(str(i), lambda e, r=i: self._rate_post(r))
        # 0 for skip
        self.root.bind("0", lambda e: self._skip_post())

    def _get_stats_cached(self):
        """Get stats with caching to avoid DB queries on every navigation."""
        if self._stats_dirty or self._stats_cache is None:
            self._stats_cache = self.session.get_stats(self.source_filter)
            self._stats_dirty = False
        return self._stats_cache

    def _show_current_post(self):
        """Display the current post."""
        if not self.posts:
            return

        post = self.posts[self.current_index]

        # Update stats (cached)
        stats = self._get_stats_cached()
        self.stats_label.config(
            text=f"Labeled: {stats['total_labeled']} | Skipped: {stats['skipped']} | Avg: {stats['avg_rating'] or 'N/A'}"
        )
        self.progress_label.config(
            text=f"{self.current_index + 1} / {len(self.posts)}"
        )

        # Thread info
        self.thread_id_label.config(text=f"Thread #{post.thread_id}")
        self.subject_label.config(text=post.subject or "(no subject)")

        # Format timestamp
        dt = datetime.fromtimestamp(post.created_at) if post.created_at else None
        date_str = dt.strftime("%Y-%m-%d %H:%M") if dt else "Unknown"

        self.meta_label.config(
            text=f"Replies: {post.reply_count} | Source: {post.source} | Date: {date_str}"
        )

        # OP Text
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        text = post.op_text_clean or post.op_text or "(no text)"
        self.text_widget.insert("1.0", text)
        self.text_widget.config(state=tk.DISABLED)

        # Current rating
        if post.existing_skipped:
            self.rating_label.config(text="SKIPPED", foreground="#888888")
        elif post.existing_rating:
            color = self._rating_color(post.existing_rating)
            label = self._rating_label(post.existing_rating)
            self.rating_label.config(text=f"{post.existing_rating}/5 - {label}", foreground=color)
        else:
            self.rating_label.config(text="Not rated", foreground="#666666")

        # Load image (prefer local, fallback to URL)
        self._load_image(post.local_thumbnail_path, post.thumbnail_url or post.image_url)

    def _preload_images(self):
        """Preload images for nearby posts."""
        # Preload current and next few images
        for i in range(min(3, len(self.posts))):
            if i < len(self.posts):
                post = self.posts[i]
                self._get_cached_image(post.thread_id, post.local_thumbnail_path,
                                      post.thumbnail_url or post.image_url)

    def _get_cached_image(self, thread_id: int, local_path: Optional[str], url: Optional[str]):
        """Get image from cache or load it."""
        if thread_id in self._image_cache:
            return self._image_cache[thread_id]

        if not PIL_AVAILABLE:
            return None

        image = None

        # Try local file first
        if local_path:
            local_file = Path(local_path)
            if local_file.exists():
                try:
                    image = Image.open(local_file)
                except Exception:
                    pass

        # Fallback to URL if no local file
        if image is None and url:
            try:
                with urlopen(url, timeout=5) as response:
                    image_data = response.read()
                image = Image.open(io.BytesIO(image_data))
            except Exception:
                pass

        if image is None:
            self._image_cache[thread_id] = None
            return None

        # Scale image to fit display area (scale UP or DOWN as needed)
        target_width = 400
        target_height = 500

        # Calculate scale to fit while maintaining aspect ratio
        width_ratio = target_width / image.width
        height_ratio = target_height / image.height
        scale = min(width_ratio, height_ratio)

        # Apply scale (allows both enlarging and shrinking)
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)

        if new_width > 0 and new_height > 0:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(image)
        self._image_cache[thread_id] = photo
        return photo

    def _load_image(self, local_path: Optional[str], url: Optional[str]):
        """Load and display image from local file or URL."""
        post = self.posts[self.current_index]
        photo = self._get_cached_image(post.thread_id, local_path, url)

        if photo:
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        else:
            self.image_label.config(image="", text="No Image")

        # Preload next image in background
        next_idx = self.current_index + 1
        if next_idx < len(self.posts):
            next_post = self.posts[next_idx]
            if next_post.thread_id not in self._image_cache:
                self.root.after(10, lambda: self._get_cached_image(
                    next_post.thread_id, next_post.local_thumbnail_path,
                    next_post.thumbnail_url or next_post.image_url))

    def _rating_color(self, rating: int) -> str:
        """Get color for rating display."""
        colors = {
            1: "#ff4444",  # Red - Very bearish
            2: "#ff8844",  # Orange - Bearish
            3: "#888888",  # Gray - Neutral
            4: "#88cc44",  # Light green - Bullish
            5: "#44ff44",  # Green - Very bullish
        }
        return colors.get(rating, "#666666")

    def _rating_label(self, rating: int) -> str:
        """Get label for rating."""
        labels = {
            1: "Very Bearish",
            2: "Bearish",
            3: "Neutral",
            4: "Bullish",
            5: "Very Bullish",
        }
        return labels.get(rating, "Unknown")

    def _rate_post(self, rating: int):
        """Rate the current post."""
        post = self.posts[self.current_index]

        self.session.save_label(
            thread_id=post.thread_id,
            rating=rating,
            skipped=False,
            text_snapshot=post.op_text_clean or post.op_text or "",
            image_url=post.image_url or "",
            notes=self.current_note if self.current_note else None,
        )

        # Update post object
        post.existing_rating = rating
        post.existing_skipped = False

        # Mark stats as needing refresh
        self._stats_dirty = True

        # Clear note and move to next
        self.current_note = ""
        self._next_post()

    def _skip_post(self):
        """Mark current post as skipped/irrelevant."""
        post = self.posts[self.current_index]

        self.session.save_label(
            thread_id=post.thread_id,
            rating=None,
            skipped=True,
            text_snapshot=post.op_text_clean or post.op_text or "",
            image_url=post.image_url or "",
            notes=self.current_note if self.current_note else None,
        )

        # Update post object
        post.existing_rating = None
        post.existing_skipped = True

        # Mark stats as needing refresh
        self._stats_dirty = True

        # Clear note and move to next
        self.current_note = ""
        self._next_post()

    def _add_note(self):
        """Add a note to the current post."""
        note = simpledialog.askstring("Add Note", "Enter note for this post:")
        if note:
            self.current_note = note

    def _prev_post(self):
        """Navigate to previous post."""
        if self.current_index > 0:
            self.current_index -= 1
            self._show_current_post()

    def _next_post(self):
        """Navigate to next post."""
        if self.current_index < len(self.posts) - 1:
            self.current_index += 1
            self._show_current_post()

    def _quit(self):
        """Quit the application."""
        stats = self.session.get_stats(self.source_filter)
        if messagebox.askyesno(
            "Quit",
            f"Labeled {stats['total_labeled']} posts this session.\nQuit?"
        ):
            self.root.quit()

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def _load_active_learning_posts(
    session: LabelingSession,
    db_path: str,
    model_path: str,
    limit: int,
    pool_size: int,
    source: Optional[str],
) -> list[ThreadOP]:
    """Load posts ordered by active learning priority.

    Uses hybrid uncertainty-diversity sampling to prioritize posts that will
    provide maximum information gain when labeled.
    """
    from pathlib import Path

    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"Model not found at {model_path}")
        print("Train a model first using: python -m src.training.setfit_trainer --db <db>")
        print("Falling back to default ordering...")
        return session.get_unlabeled_posts(limit, source)

    try:
        from .active_learning.labeler_integration import ActiveLearningLabeler
        from .models.setfit_model import SetFitSentimentModel

        print(f"Loading model from {model_path}...")
        model = SetFitSentimentModel(model_path=model_path)

        print(f"Running active learning acquisition on {pool_size} candidates...")
        al = ActiveLearningLabeler(db_path, model=model)

        # Get posts from database
        all_posts = session.get_unlabeled_posts(pool_size, source)
        if not all_posts:
            return []

        # Get suggested thread IDs in priority order
        thread_ids = [p.thread_id for p in all_posts]
        texts = [p.op_text_clean or p.op_text or "" for p in all_posts]

        # Run acquisition
        probs = model.predict_proba(texts)
        embeddings = model.get_embeddings(texts)

        from .active_learning.acquisition import hybrid_acquisition

        selected_indices = hybrid_acquisition(probs, embeddings, n_select=limit)

        # Reorder posts by selected indices
        posts = [all_posts[i] for i in selected_indices]

        print(f"Active learning: selected {len(posts)} posts from {len(all_posts)} candidates")
        return posts

    except ImportError as e:
        print(f"Active learning not available: {e}")
        print("Falling back to default ordering...")
        return session.get_unlabeled_posts(limit, source)
    except Exception as e:
        print(f"Active learning failed: {e}")
        print("Falling back to default ordering...")
        return session.get_unlabeled_posts(limit, source)


def main():
    parser = argparse.ArgumentParser(description="BizCharts Sentiment Labeling Tool")
    parser.add_argument(
        "--db", default="../data/posts.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--session", default="default",
        help="Labeler ID / session name"
    )
    parser.add_argument(
        "--limit", type=int, default=1000,
        help="Maximum posts to load"
    )
    parser.add_argument(
        "--source", choices=["live", "warosu"],
        help="Filter by data source (live or warosu)"
    )
    parser.add_argument(
        "--from", dest="date_from", metavar="DATE",
        help="Filter posts from this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--to", dest="date_to", metavar="DATE",
        help="Filter posts up to this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--export", metavar="FILE",
        help="Export labels to CSV file and exit"
    )
    parser.add_argument(
        "--review", action="store_true",
        help="Review all posts (including already labeled)"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show labeling statistics and exit"
    )
    parser.add_argument(
        "--active-learning", action="store_true",
        help="Order posts by active learning priority (uncertainty + diversity)"
    )
    parser.add_argument(
        "--model", default="models/setfit",
        help="Path to trained model for active learning (default: models/setfit)"
    )
    parser.add_argument(
        "--pool-size", type=int, default=1000,
        help="Candidate pool size for active learning (default: 1000)"
    )

    args = parser.parse_args()

    # Resolve database path
    db_path = Path(args.db)
    if not db_path.exists():
        # Try relative to project root (rust-scraper/data/posts.db)
        script_dir = Path(__file__).parent.parent  # python-ml/
        project_root = script_dir.parent  # bizcharts/
        db_path = project_root / "rust-scraper" / "data" / "posts.db"

    if not db_path.exists():
        print(f"Database not found: {args.db}")
        print("Run the Rust scraper first to collect data.")
        sys.exit(1)

    session = LabelingSession(str(db_path), args.session)

    if args.stats:
        stats = session.get_stats(args.source)
        print("\nLabeling Statistics")
        print("=" * 40)
        if args.source:
            print(f"Source filter: {args.source}")
        print(f"Total labeled: {stats['total_labeled']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Average rating: {stats['avg_rating'] or 'N/A'}")
        print(f"Unique threads: {stats['unique_threads']}")
        return

    if args.export:
        session.export_labels(args.export, args.source)
        return

    # Load posts
    if args.review:
        posts = session.get_all_posts(args.limit, args.source, args.date_from, args.date_to)
    elif args.active_learning:
        posts = _load_active_learning_posts(
            session=session,
            db_path=str(db_path),
            model_path=args.model,
            limit=args.limit,
            pool_size=args.pool_size,
            source=args.source,
        )
    else:
        posts = session.get_unlabeled_posts(args.limit, args.source, args.date_from, args.date_to)

    if not posts:
        print("No posts to label!")
        if args.source:
            print(f"(filtered by source: {args.source})")
        if args.date_from or args.date_to:
            print(f"(date range: {args.date_from or 'any'} to {args.date_to or 'any'})")
        print("Run the Rust scraper first to collect data, or use --review to review labeled posts.")
        sys.exit(0)

    filter_info = []
    if args.source:
        filter_info.append(f"source={args.source}")
    if args.date_from:
        filter_info.append(f"from={args.date_from}")
    if args.date_to:
        filter_info.append(f"to={args.date_to}")

    if filter_info:
        print(f"Filters: {', '.join(filter_info)}")
    print(f"Loaded {len(posts)} posts for labeling")

    # Start GUI
    gui = LabelingGUI(session, posts, args.source)
    gui.run()


if __name__ == "__main__":
    main()

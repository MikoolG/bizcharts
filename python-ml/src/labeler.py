#!/usr/bin/env python3
"""
BizCharts Sentiment Labeling Tool

A GUI tool for manually labeling thread OP sentiment to create training data.

Usage:
    python -m src.labeler                    # Label from database
    python -m src.labeler --session my_session  # Named labeling session
    python -m src.labeler --export labels.csv   # Export labeled data

Controls:
    1-9, 0  : Rate sentiment (1=extremely bearish, 5=neutral, 0/10=extremely bullish)
    S       : Skip this post (not relevant to sentiment)
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
                sentiment_rating INTEGER CHECK(sentiment_rating >= 1 AND sentiment_rating <= 10),
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

    def get_unlabeled_posts(self, limit: int = 1000) -> list[ThreadOP]:
        """Get posts that haven't been labeled yet by this labeler."""
        cursor = self.conn.execute("""
            SELECT
                t.thread_id, t.subject, t.op_text, t.op_text_clean,
                t.image_url, t.thumbnail_url, t.reply_count, t.created_at, t.source,
                l.sentiment_rating, l.skipped
            FROM thread_ops t
            LEFT JOIN training_labels l ON t.thread_id = l.thread_id AND l.labeler_id = ?
            WHERE l.id IS NULL
            ORDER BY t.reply_count DESC, t.created_at DESC
            LIMIT ?
        """, (self.labeler_id, limit))

        posts = []
        for row in cursor:
            posts.append(ThreadOP(
                thread_id=row["thread_id"],
                subject=row["subject"],
                op_text=row["op_text"],
                op_text_clean=row["op_text_clean"],
                image_url=row["image_url"],
                thumbnail_url=row["thumbnail_url"],
                reply_count=row["reply_count"] or 0,
                created_at=row["created_at"],
                source=row["source"],
                existing_rating=row["sentiment_rating"],
                existing_skipped=bool(row["skipped"]),
            ))
        return posts

    def get_all_posts(self, limit: int = 1000) -> list[ThreadOP]:
        """Get all posts (for review mode)."""
        cursor = self.conn.execute("""
            SELECT
                t.thread_id, t.subject, t.op_text, t.op_text_clean,
                t.image_url, t.thumbnail_url, t.reply_count, t.created_at, t.source,
                l.sentiment_rating, l.skipped
            FROM thread_ops t
            LEFT JOIN training_labels l ON t.thread_id = l.thread_id AND l.labeler_id = ?
            ORDER BY t.reply_count DESC, t.created_at DESC
            LIMIT ?
        """, (self.labeler_id, limit))

        posts = []
        for row in cursor:
            posts.append(ThreadOP(
                thread_id=row["thread_id"],
                subject=row["subject"],
                op_text=row["op_text"],
                op_text_clean=row["op_text_clean"],
                image_url=row["image_url"],
                thumbnail_url=row["thumbnail_url"],
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

    def get_stats(self) -> dict:
        """Get labeling statistics."""
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

    def export_labels(self, output_path: str):
        """Export labels to CSV."""
        import csv

        cursor = self.conn.execute("""
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
                t.source
            FROM training_labels l
            JOIN thread_ops t ON l.thread_id = t.thread_id
            ORDER BY l.labeled_at
        """)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "thread_id", "sentiment_rating", "skipped", "notes", "labeler_id",
                "labeled_at", "text_snapshot", "image_url", "subject", "reply_count", "source"
            ])
            for row in cursor:
                writer.writerow(list(row))

        print(f"Exported {cursor.rowcount} labels to {output_path}")


class LabelingGUI:
    """Tkinter GUI for labeling posts."""

    def __init__(self, session: LabelingSession, posts: list[ThreadOP]):
        self.session = session
        self.posts = posts
        self.current_index = 0
        self.current_note = ""

        if not posts:
            print("No unlabeled posts found!")
            sys.exit(0)

        self.root = tk.Tk()
        self.root.title("BizCharts Sentiment Labeler")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e1e")

        self._setup_ui()
        self._bind_keys()
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
  1-9, 0    Rate (1=bearish ... 5=neutral ... 10=bullish)
  S         Skip (not relevant)
  Left/Right    Navigate posts
  N         Add note
  Q/Esc     Quit
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

        # Number keys for ratings
        for i in range(1, 10):
            self.root.bind(str(i), lambda e, r=i: self._rate_post(r))
        self.root.bind("0", lambda e: self._rate_post(10))

    def _show_current_post(self):
        """Display the current post."""
        if not self.posts:
            return

        post = self.posts[self.current_index]

        # Update stats
        stats = self.session.get_stats()
        self.stats_label.config(
            text=f"Labeled: {stats['total_labeled']} | Skipped: {stats['skipped']} | Avg: {stats['avg_rating'] or 'N/A'}"
        )
        self.progress_label.config(
            text=f"Post {self.current_index + 1} / {len(self.posts)}"
        )

        # Thread info
        self.thread_id_label.config(text=f"Thread #{post.thread_id}")
        self.subject_label.config(text=post.subject or "(no subject)")

        # Format timestamp
        from datetime import datetime
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
            self.rating_label.config(text=f"Rating: {post.existing_rating}/10", foreground=color)
        else:
            self.rating_label.config(text="Not rated", foreground="#666666")

        # Load image
        self._load_image(post.thumbnail_url or post.image_url)

    def _load_image(self, url: Optional[str]):
        """Load and display image from URL."""
        if not url or not PIL_AVAILABLE:
            self.image_label.config(image="", text="No Image" if not url else "PIL not installed")
            return

        try:
            # Download image
            with urlopen(url, timeout=5) as response:
                image_data = response.read()

            # Open with PIL
            image = Image.open(io.BytesIO(image_data))

            # Resize to fit label (max 500x500)
            max_size = (500, 600)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Update label (keep reference to prevent garbage collection)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo

        except (URLError, Exception) as e:
            self.image_label.config(image="", text=f"Failed to load image:\n{str(e)[:50]}")

    def _rating_color(self, rating: int) -> str:
        """Get color for rating display."""
        if rating <= 3:
            return "#ff4444"  # Red (bearish)
        elif rating <= 4:
            return "#ff8844"  # Orange
        elif rating <= 6:
            return "#888888"  # Gray (neutral)
        elif rating <= 7:
            return "#88cc44"  # Light green
        else:
            return "#44ff44"  # Green (bullish)

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
        stats = self.session.get_stats()
        if messagebox.askyesno(
            "Quit",
            f"Labeled {stats['total_labeled']} posts this session.\nQuit?"
        ):
            self.root.quit()

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


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

    args = parser.parse_args()

    # Resolve database path
    db_path = Path(args.db)
    if not db_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        db_path = script_dir / "data" / "posts.db"

    if not db_path.exists():
        print(f"Database not found: {args.db}")
        print("Run the Rust scraper first to collect data.")
        sys.exit(1)

    session = LabelingSession(str(db_path), args.session)

    if args.stats:
        stats = session.get_stats()
        print("\nLabeling Statistics")
        print("=" * 40)
        print(f"Total labeled: {stats['total_labeled']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Average rating: {stats['avg_rating'] or 'N/A'}")
        print(f"Unique threads: {stats['unique_threads']}")
        return

    if args.export:
        session.export_labels(args.export)
        return

    # Load posts
    if args.review:
        posts = session.get_all_posts(args.limit)
    else:
        posts = session.get_unlabeled_posts(args.limit)

    if not posts:
        print("No posts to label!")
        print("Run the Rust scraper first to collect data, or use --review to review labeled posts.")
        sys.exit(0)

    print(f"Loaded {len(posts)} posts for labeling")

    # Start GUI
    gui = LabelingGUI(session, posts)
    gui.run()


if __name__ == "__main__":
    main()

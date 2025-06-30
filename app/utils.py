from __future__ import annotations
import os
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Initialise module logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = STORAGE_DIR / os.getenv("DB_PATH", "articles.db")
LAST_RUN_FILE = STORAGE_DIR / "last_run.txt"
DEFAULT_EPOCH = datetime.fromtimestamp(0, tz=timezone.utc)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    title TEXT,
    url TEXT,
    content TEXT,
    published DATETIME,
    sentiment REAL,
    is_good INTEGER,
    category TEXT,
    reason TEXT,
    source_type TEXT DEFAULT 'ai_generated'
);
"""


@contextmanager
def get_db():
    """Yields an open SQLite connection with the expected schema initialised."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA_SQL)

    # --- lightweight migrations ---
    # If the table was created before new columns were added, ensure they exist.
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(articles)").fetchall()
    }
    if "source_type" not in existing_cols:
        conn.execute("ALTER TABLE articles ADD COLUMN source_type TEXT DEFAULT 'ai_generated'")
    if "reason" not in existing_cols:  # historical safety
        conn.execute("ALTER TABLE articles ADD COLUMN reason TEXT")

    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def get_last_run_time() -> datetime:
    """Reads the last successful run time from a file."""
    if not LAST_RUN_FILE.exists():
        return DEFAULT_EPOCH
    with open(LAST_RUN_FILE, "r") as f:
        try:
            return datetime.fromisoformat(f.read().strip())
        except (ValueError, TypeError):
            return DEFAULT_EPOCH


def set_last_run_time() -> None:
    """Writes the current time to the last run file."""
    with open(LAST_RUN_FILE, "w") as f:
        f.write(utcnow().isoformat())


# ---------------------------------------------------------------------------
# House-keeping: prune old rows
# ---------------------------------------------------------------------------


def prune_old_articles(days_to_keep: int = 7) -> None:
    """Delete rows older than *days_to_keep* to keep the database small."""
    cutoff = utcnow() - timedelta(days=days_to_keep)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM articles WHERE published < ?", (cutoff.isoformat(),))
        pruned = cur.rowcount
    if pruned:
        logger.info("Pruned %s old articles (older than %s)", pruned, days_to_keep) 
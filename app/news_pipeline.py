"""Pipelines that combine fetching, sentiment scoring & LLM good-news filtering."""
from __future__ import annotations

import asyncio
from .fetchers import fetch_latest_articles
from .filters import run_sentiment, filter_good
from .utils import get_db, set_last_run_time, prune_old_articles, get_last_run_time, utcnow
from .trace.logger import logger

async def run_pipeline() -> None:
    """Run the end-to-end pipeline.

    Behaviour differences on first run vs. incremental runs:
    • First run (no last_run.txt) → fetch past 7 days, up to 500 articles (5×100 pages).
    • Subsequent runs       → fetch only since last run, up to 100 articles.
    """
    prune_old_articles(days_to_keep=7)

    last_run = get_last_run_time()

    # A "first run" is when the last run time is at the epoch (or the db is empty).
    first_run = last_run.year < 1971
    if not first_run:
        with get_db() as conn:
            if conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 0:
                first_run = True

    if first_run:
        minutes_back = 24 * 7 * 60  # 1 week
        max_articles = 500
    else:
        delta_minutes = max(5, int((utcnow() - last_run).total_seconds() // 60))
        minutes_back = delta_minutes
        max_articles = 100

    logger.info(f"Fetching articles from {minutes_back} minutes ago")

    set_last_run_time() # Stamp the start time of the run

    await fetch_latest_articles(minutes_back=minutes_back, max_articles=max_articles)
    await asyncio.to_thread(run_sentiment)
    await filter_good()


def get_good_news(
    limit: int = 50, sort_by: str = "published"
) -> list:
    """
    Return latest *good* articles, including their ID and source_type.
    The returned tuple has 9 elements.
    """
    order_clause = "sentiment DESC" if sort_by == "sentiment" else "published DESC"

    # Efficient deduplication directly in SQLite using a window function.
    # `ROW_NUMBER() OVER (PARTITION BY lower(title) ORDER BY published DESC)`
    # assigns rank 1 to the most-recent row of every case-insensitive title group.

    with get_db() as conn:
        rows = conn.execute(
            f"""
            WITH ranked AS (
                SELECT id, title, content, url, published, category, sentiment, reason, is_good, source_type,
                       ROW_NUMBER() OVER (
                           PARTITION BY lower(title)
                           ORDER BY published DESC
                       ) AS rn
                FROM articles
                WHERE is_good = 1
            )
            SELECT id, title, content, url, published, category, sentiment, reason, is_good, source_type
            FROM ranked
            WHERE rn = 1
            ORDER BY {order_clause}
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return rows 
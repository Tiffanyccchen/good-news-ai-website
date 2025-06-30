from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta
from typing import List

import aiohttp
import asyncio

from app.utils import get_db, utcnow
from app.trace.logger import logger

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2/everything"

# A curated list of major, reputable news source IDs from NewsAPI
MAJOR_SOURCE_IDS = [
    "bbc-news",             # global breadth, frequent human-interest features
    "abc-news",             # large U.S. feed with many upbeat "America Strong" segments
    "associated-press",     # massive wire service, daily positive-impact stories
    "reuters",              # high-volume, reliable international reporting
    "usa-today",            # consumer-friendly mix of heart-warming and lifestyle news
    "time",                 # in-depth progress pieces on science, health and society
    "national-geographic",  # conservation wins, exploration, inspiring nature coverage
    "new-scientist",        # scientific breakthroughs and technological improvements
    "techcrunch",           # startup innovations, tech-for-good launches
    "wired",                # forward-looking tech and "better future" features
    "bloomberg",            # business growth stories, sustainable-finance coverage
    "axios",                # concise "good trend" explainers, policy progress trackers
    "espn",                 # uplifting sports moments, athlete philanthropy
    "nbc-news",             # general feed with "Inspiring America" series
    "cbs-news",             # "On The Road" and other feel-good segments
    "independent",          # UK outlet with regular human-interest and culture wins
    "newsweek",             # broad coverage including environment and health positives
    "abc-news-au",          # Australian lens on science, wildlife rescue, community aid
]


def _article_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


async def fetch_latest_articles(minutes_back: int = 60, max_articles: int = 100) -> List[dict]:
    """Fetch latest articles from NewsAPI.

    Args:
        minutes_back: How many minutes of look-back window. Allows finer granularity than hours.
        max_articles: Maximum number of articles to fetch (capped at 500). Uses paging (pageSize=100).
    """
    if not NEWS_API_KEY:
        logger.error("NEWS_API_KEY env var not set. Cannot fetch news.")
        return []

    # NewsAPI free tier delivers data with up to 24-hour delay. Shift both
    # ends of our window back by 24h so we actually receive results.
    delay = timedelta(hours=24)
    window_end = utcnow() - delay
    window_start = window_end - timedelta(minutes=minutes_back)
    from_date = window_start.strftime("%Y-%m-%dT%H:%M:%S")
    to_date = window_end.strftime("%Y-%m-%dT%H:%M:%S")

    params = {
        "sources": ",".join(MAJOR_SOURCE_IDS),
        "from": from_date,
        "to": to_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY,
    }

    page_size = 100  # fixed (API max)
    max_articles = min(max_articles, 100)
    max_pages = min((max_articles + page_size - 1) // page_size, 5)

    parsed: list[dict] = []
    
    async with aiohttp.ClientSession() as session:
        for page in range(1, max_pages + 1):
            params_page = {**params, "page": page}
            logger.info(f"Fetching page {page} of {max_pages} with params {params_page}")

            attempts = 0
            while attempts < 3:
                attempts += 1
                async with session.get(BASE_URL, params=params_page, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:
                        # Respect Retry-After header or exponential backoff
                        retry_after = resp.headers.get("Retry-After")
                        wait_sec = int(retry_after) if retry_after and retry_after.isdigit() else attempts * 2
                        logger.warning("NewsAPI 429 on page %s. Waiting %s s before retry %s/3", page, wait_sec, attempts)
                        await asyncio.sleep(wait_sec)
                        continue

                    if resp.status != 200:
                        logger.error(
                            "NewsAPI request failed (page %s) with status %s: %s",
                            page,
                            resp.status,
                            await resp.text(),
                        )
                        data = None
                    else:
                        data = await resp.json()
                        logger.info(f"Fetched page {page} of {max_pages} with data {data}")
                    break 

            if data is None:
                break

            if data.get("status") != "ok":
                logger.error("NewsAPI returned an error on page %s: %s", page, data.get("message"))
                break

            articles = data.get("articles", [])

            # ---- Parse current page ----
            for art in articles:
                published_str = art.get("publishedAt")
                try:
                    published_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    continue

                parsed.append(
                    {
                        "id": _article_id(art["url"]),
                        "title": art.get("title") or "",
                        "url": art.get("url"),
                        "content": art.get("description") or art.get("content") or "",
                        "published": published_dt,
                    }
                )

            # Stop if we fetched fewer than page_size or reached max_articles
            if len(articles) < page_size or len(parsed) >= max_articles:
                break

    # Persist unique articles
    if parsed:
        unique_articles = {item["id"]: item for item in parsed}.values()
        logger.info("Fetched and parsed %s unique articles from NewsAPI.", len(unique_articles))
        with get_db() as conn:
            cur = conn.cursor()
            for art in unique_articles:
                try:
                    cur.execute(
                        """
                        INSERT INTO articles (id, title, url, content, published)
                        VALUES (?,?,?,?,?)
                        """,
                        (
                            art["id"],
                            art["title"],
                            art["url"],
                            art["content"],
                            art["published"].isoformat(),
                        ),
                    )
                except Exception:
                    # Duplicate, ignore
                    pass
    return parsed 
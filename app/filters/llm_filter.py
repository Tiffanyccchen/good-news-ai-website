"""Second layer filter: call Groq LLM to judge whether an article is good news."""
from __future__ import annotations

import asyncio
import os
from typing import Literal

import instructor
from groq import AsyncGroq
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

from app.utils import get_db
from app.trace.logger import logger

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Hold the patched client; initialise lazily to avoid import-time side-effects
_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq | None:
    """Return singleton patched Groq client (or None when not configured)."""
    global _client
    if _client is not None:
        return _client

    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not found. LLM filter disabled.")
        return None

    _client = instructor.patch(AsyncGroq(api_key=GROQ_API_KEY))
    return _client


# --- Pydantic Schema for Structured Output ---
class NewsJudgement(BaseModel):
    is_good_news: bool = Field(description="True if the article is good news, False otherwise.")
    category: Literal["cute_or_fun", "improvement", "heartwarming", "none"] = Field(
        description="The category of good news."
    )
    reason: str = Field(
        description="A brief justification for the classification, explaining why it fits the category."
    )


class SafetyJudgement(BaseModel):
    is_safe_and_good: bool = Field(description="True if the content is safe, on-topic, and genuinely positive news. False otherwise.")
    reason: str = Field(description="A brief explanation for the decision, especially if it fails the check.")


# --- Core Logic ---
async def validate_user_submission(title: str, content: str) -> SafetyJudgement | None:
    """Uses Llama Guard to check if user-submitted content is safe and positive."""
    client = _get_client()
    if not client:
        return None
    
    try:
        judgement = await client.chat.completions.create(
            model="llama3-8b-8192", # Corrected to a valid model name for this task
            response_model=SafetyJudgement,
            max_retries=1,
            messages=[
                {
                    "role": "user",
                    "content": """
                    You are a content moderator for a 'Good News' website. Your task is to determine if a user's submission is safe AND a genuinely positive, uplifting news story.
                    - Set `is_safe_and_good` to `true` if the submission is BOTH safe for a general audience and a genuinely positive story.It can be be brief. It can be seen as good as long as not a sarcastic comment, a rant, an advertisement, political complaining. It should fit the spirit of the site.
                    - Otherwise, set `is_safe_and_good` to `false`.
                    - The `reason` field should briefly explain your decision, especially if it fails the check.
                    """,
                },
                {
                    "role": "user",
                    "content": f"Please moderate this submission:\nTitle: {title}\n\nStory: {content}",
                },
            ],
        )
        return judgement
    except Exception as e:
        logger.error("Error processing user submission with Llama Guard: %r", e)
        return None


async def _judge(article_title: str, article_content: str) -> NewsJudgement | None:
    """Uses instructor to get a validated Pydantic model from the LLM."""
    if not _get_client():
        return None

    max_retries = 3
    backoff_secs = 2

    for attempt in range(1, max_retries + 1):
        try:
            judgement = await _get_client().chat.completions.create(
                model="llama3-70b-8192",
                response_model=NewsJudgement,
                max_retries=2,
                messages=[
                    {
                        "role": "user",
                        "content": """
                        You are a news classification expert. Your task is to analyze an article and classify it based on the following criteria for "good news".

                        Classification Guide:
                        - **"cute_or_fun"**: genuinely lighthearted, amusing, adorable, or delightfully silly items (e.g. an otter playing piano, a harmless viral meme, a car that honks emojis). The Generic lifestyle trends, celebrity outfits, brand promo, or listicles that read like ads does not count.
                        - **"improvement"**: clear, evidence-based progress that benefits society, the planet, or knowledge (e.g. peer-reviewed medical breakthrough, major poverty drop, verified clean-energy milestone). Pure product marketing, One-off luxury launches does not count.
                        - **heartwarming"**: authentic acts of kindness, courage, inclusion, or community generosity (e.g. strangers rescue a dog, huge donation saves a library, first Deaf pilot licensed). General tips and tricks, or articles that are not about a specific act of kindness, courage, inclusion, or community generosity does not count.
                        - **"none"**: Use this category if the article is neutral, political, tragic, or does not fit any of the above. The `is_good_news` field must be `false` if the category is "none".
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"Please classify this article:\nTitle: {article_title}\n\nContent: {article_content}",
                    },
                ],
            )
            return judgement

        except Exception as e:
            error_str = str(e)
            # rudimentary 429 / rate limit detection
            if "429" in error_str or "rate limit" in error_str.lower():
                wait_for = backoff_secs * attempt
                logger.warning("Groq rate-limited (429). Waiting %s s before retry %s/%s…", wait_for, attempt, max_retries)
                await asyncio.sleep(wait_for)
                continue  # retry outer loop

            logger.error("Error processing article with Groq/instructor: %r", e)
            return None

    logger.error("Exceeded max retries (%s) for article: %s", max_retries, article_title)
    return None


async def filter_good(batch_limit: int = 20, concurrency: int = 3) -> None:
    if not _get_client():
        return

    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, title, content FROM articles WHERE is_good IS NULL LIMIT ?",
            (batch_limit,),
        ).fetchall()
    if not rows:
        logger.info("No articles require LLM classification.")
        return

    sem = asyncio.Semaphore(concurrency)

    async def _task(r):
        article_id, title, content = r
        async with sem:
            judgement = await _judge(title, content or "")
            with get_db() as c:
                if judgement:
                    # Successful classification
                    c.execute(
                        "UPDATE articles SET is_good=?, category=?, reason=? WHERE id=?",
                        (
                            1 if judgement.is_good_news else 0,
                            judgement.category,
                            judgement.reason,
                            article_id,
                        ),
                    )
                    logger.info(
                        "Article '%s' judged as: %s (Category: %s)",
                        title,
                        "Good" if judgement.is_good_news else "Not Good",
                        judgement.category,
                    )
                else:
                    # Mark as not good to prevent endless retries on failures / rate-limits
                    c.execute("UPDATE articles SET is_good=0 WHERE id=?", (article_id,))
                    logger.warning("Failed to get judgement for article '%s'. Marked as not good.", title)

            # Gentle sleep to respect Groq rate limits
            await asyncio.sleep(0.4)

    await tqdm_asyncio.gather(*[_task(r) for r in rows])
    logger.info("Finished LLM classification for %s articles.", len(rows)) 
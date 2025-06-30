"""First-layer filtering using a lightweight Hugging Face sentiment model."""
from __future__ import annotations

from typing import List

from transformers import pipeline

from app.utils import get_db
from app.trace.logger import logger

_classifier = None


def _classifier_instance():
    global _classifier
    if _classifier is None:
        logger.info("Loading sentiment model…")
        _classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    return _classifier


def run_sentiment(max_negative_prob: float = 0.7) -> None:
    clf = _classifier_instance()

    with get_db() as conn:
        rows = conn.execute("SELECT id, title, content FROM articles WHERE sentiment IS NULL").fetchall()
        if not rows:
            return

        texts: List[str] = [f"{r[1]}\n\n{r[2] or ''}" for r in rows]
        logger.info("Running sentiment analysis on %s new articles.", len(texts))
        preds = clf(texts, batch_size=64, truncation=True)

        for (art_id, title, _content), pred in zip(rows, preds):
            label = pred["label"].lower()
            score = pred["score"]

            # Convert label to a 0-1 "positivity" score
            if label == "positive":
                positivity = score
            elif label == "negative":
                positivity = 1 - score
            else:  # Neutral
                positivity = 0.5

            conn.execute("UPDATE articles SET sentiment=? WHERE id=?", (positivity * 100, art_id))

            # A high "negativity" score can disqualify an article from the LLM step.
            negativity = 1 - positivity
            logger.info("Text %s has negativity %.2f", title, negativity)
            if negativity > max_negative_prob:
                conn.execute("UPDATE articles SET is_good=0 WHERE id=?", (art_id,)) 
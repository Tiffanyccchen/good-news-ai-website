#!/usr/bin/env python
"""Streamlit front-end for the Good-News project.
Run with `streamlit run run_frontend.py` (Hugging Face Streamlit Space compatible).
"""
from __future__ import annotations

# Set environment variable to disable tokenizer parallelism. This is a recommended fix
# for a known issue on Windows where the transformers/tokenizers library can cause
# the parent process to hang on shutdown (Ctrl+C) due to how multiprocessing is handled.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import os
import threading
from datetime import datetime, timedelta, timezone

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_autorefresh import st_autorefresh

from app.news_pipeline import get_good_news, run_pipeline
from app.utils import get_last_run_time
from app.trace.logger import logger
from app.ui.sidebar import render_sidebar
from app.ui.tabs import render_article_tabs
from app.ui.submission_form import render_submission_form

if "NEWS_API_KEY" in st.secrets:
    os.environ["NEWS_API_KEY"] = st.secrets["NEWS_API_KEY"]
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
#
#  --- Initialize session state ---
if "favorites" not in st.session_state:
    st.session_state.favorites = []

# --- Page Config ---
st.set_page_config(page_title="Daily Good News", page_icon="🌞", layout="wide")
st.title("🌞 Daily Good News")
st.subheader("An automated good news feed, analyzed and categorized by AI created by Yu Ting Chen.")

# --- API Key Checks ---
if not os.getenv("NEWS_API_KEY"):
    st.warning("`NEWS_API_KEY` is not set. The app cannot fetch new articles.", icon="⚠️")
if not os.getenv("GROQ_API_KEY"):
    st.warning("`GROQ_API_KEY` is not set. The LLM filter will be skipped.", icon="⚠️")


# --- Background Pipeline Logic ---
def background_pipeline_task():
    """Run the news-processing pipeline in a background thread and trigger a UI refresh when done.

    The function is robust: if *anything* inside the pipeline raises, we make sure to
    clear the `pipeline_running` flag so that subsequent cycles are not blocked.
    """
    try:
        asyncio.run(run_pipeline())
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
    finally:
        st.session_state["pipeline_running"] = False
        st.cache_data.clear()
        st.rerun()

def initialize_and_refresh():
    """Handles the logic for running the background pipeline."""
    now = datetime.now(timezone.utc)
    last_run = get_last_run_time()
    if (now - last_run) > timedelta(minutes=15):
        if not st.session_state.get("pipeline_running"):
            st.session_state.pipeline_running = True
            
            t = threading.Thread(target=background_pipeline_task, daemon=True)
            add_script_run_ctx(t)
            t.start()
            st.rerun()

initialize_and_refresh()

# --- Main App Flow ---
num_items, sort_by_option = render_sidebar()
sort_by = "sentiment" if sort_by_option == "Most Positive" else "published"

# --- Data Loading & Auto-Run ---
CACHE_VERSION = 3

@st.cache_data(ttl=1800)
def cached_get_good_news(sort_by: str, v: int = CACHE_VERSION) -> list:
    """
    Wrapper to cache the database query. Fetches the maximum number of articles
    (100) so that the UI slider can operate on a cached list.
    `v` busts the cache when schema changes.
    """
    return get_good_news(limit=100, sort_by=sort_by)

all_rows = cached_get_good_news(sort_by=sort_by)
rows = all_rows[:num_items].copy()

if not rows and not st.session_state.get("pipeline_running"):
    st.info("No news found. Attempting to fetch data in the background...")
    st.session_state.pipeline_running = True

    t = threading.Thread(target=background_pipeline_task, daemon=True)
    add_script_run_ctx(t)
    t.start()
    st.rerun()

# --- Render Main UI ---
render_submission_form()
render_article_tabs(rows)

# Add the auto-refresher and the CSS tweaks at the bottom
st_autorefresh(interval=60_000, key="auto_rerun")

st.markdown("""
<style>
div[data-testid="column"] {
    width: fit-content !important;
    flex: unset;
}
div[data-testid="column"] * {
    width: fit-content !important;
}
</style>
""", unsafe_allow_html=True) 
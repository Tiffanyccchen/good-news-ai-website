from __future__ import annotations
import streamlit as st
from datetime import  timezone

from app.fetchers.newsapi_fetcher import MAJOR_SOURCE_IDS
from app.utils import get_last_run_time

def render_sidebar():
    """Renders the sidebar UI and returns user-selected display options."""
    with st.sidebar:
        st.header("Display Options")
        num_items = st.slider("Articles to show", 100, 500, 300, 50)
        sort_by_option = st.radio(
            "Sort by",
            ["Most Recent", "Most Positive"],
            captions=["Order by publish date", "Order by sentiment score"],
            horizontal=True,
        )
        st.divider()

        last_run_time = get_last_run_time()
        if last_run_time.year < 1971:
            st.caption("Last Updated:\nNever (first run pending...)")
        else:
            last_update_str = last_run_time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            st.caption(f"Last Updated (UTC):\n{last_update_str}")

        # note about timezone
        st.caption("Times shown are in Coordinated Universal Time (UTC).")

        # Persistent status indicator during background processing
        status_box = st.empty()
        if st.session_state.get("pipeline_running"):
            status_box.info("⏳ Updating news feed in the background…")
        else:
            status_box.empty()

        st.divider()
        st.info(
            "🕒 This feed updates automatically every 15 minutes.It will only retain data for the last 7 days."
        )
        st.info(
            "🔖 NewsAPI free tier provides data 24 hours late, so the latest stories are from *yesterday*."
        )
        st.caption("Beta notice: article volume is limited to maximum 100 per call for free tier users.")

        with st.expander("How it Works"):
            st.markdown(
                """
                - **News Source**: We fetch recent articles from a curated list of trusted sources via the [NewsAPI](https://newsapi.org).
                - **AI Classification**: Each article is analyzed by **Llama-3-70B** (running on the [Groq API](https://groq.com/)) to determine if it's positive news and to assign it a category.
                - **User Submissions**: Users can submit their own positive news stories. It will be reviewed by the **Llama-3-8B** (Also on the [Groq API](https://groq.com/)) to ensure it is positive and safe for a general audience.
                """
            )

        with st.expander("View Included Sources"):
            st.markdown(f"_{len(MAJOR_SOURCE_IDS)} sources enabled:_")
            st.markdown(" • ".join(sorted(MAJOR_SOURCE_IDS)))

        return num_items, sort_by_option 
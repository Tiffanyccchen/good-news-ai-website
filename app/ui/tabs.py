from __future__ import annotations

import streamlit as st
from datetime import datetime
from urllib.parse import urlparse

CATEGORY_EMOJIS = {"cute_or_fun": "🥳", "improvement": "🚀", "heartwarming": "❤️", "user_submitted": "💌", "none": "🔹"}

def _toggle_favorite(article_id: str):
    """Adds or removes an article ID from the session's favorite list."""
    if article_id in st.session_state.favorites:
        st.session_state.favorites.remove(article_id)
    else:
        st.session_state.favorites.append(article_id)

def _display_single_article(column, article_data, prefix: str):
    """Renders a single article card in a given column."""
    article_id, title, content, url, published, category, sentiment, reason, is_good, source_type = article_data
    
    # Title is essential.
    if not title:
        return

    is_favorite = article_id in st.session_state.favorites

    with column:
        with st.container(border=True):
            c1, c2, c3 = st.columns([0.1, 0.8, 0.1])
            with c1:
                st.markdown(f"### {CATEGORY_EMOJIS.get(category, '🔹')}")
            with c2:
                # User submissions have no URL, so just display the title as plain text.
                if source_type == 'user_submitted' or not url:
                    st.markdown(f"**{title}**")
                else:
                    st.markdown(f"**[{title}]({url})**")
            with c3:
                st.button(
                    "❤️" if is_favorite else "🤍",
                    key=f"{prefix}_fav_{article_id}",
                    on_click=_toggle_favorite,
                    args=(article_id,),
                    help="Add to your saved articles" if not is_favorite else "Remove from saved articles"
                )

            # Display the content for user submissions, and the reason for AI articles.
            if source_type == 'user_submitted':
                st.markdown(f"> *{content}*") # For user submissions, 'reason' holds the story
            elif reason:
                st.markdown(f"> *{reason}*")

            # Handle caption display based on source
            if source_type == 'user_submitted':
                 date_str = datetime.fromisoformat(published).strftime("%b %d, %Y")
                 st.caption(f"Shared by a user • {date_str}")
            elif url:
                domain = urlparse(url).netloc.replace("www.", "")
                date_str = datetime.fromisoformat(published).strftime("%b %d, %Y")
                st.caption(f"{domain} • {date_str}")

            # Sentiment score is only applicable to AI-analyzed articles
            if sentiment is not None and source_type == 'ai_generated':
                st.markdown(f"✨ **Score: {sentiment:.0f}%**")

def _display_articles_in_tab(tab, articles, prefix: str):
    """Displays a list of articles within a given tab, arranged in two columns."""
    if not articles:
        tab.write("No articles in this category.")
        return

    col1, col2 = tab.columns(2, gap="small")
    for i, article in enumerate(articles):
        _display_single_article(col1 if i % 2 == 0 else col2, article, prefix)


def render_article_tabs(rows: list):
    """Creates the tab layout and renders articles within them."""
    
    ai_rows = [r for r in rows if r[9] == 'ai_generated']
    user_rows = [r for r in rows if r[9] == 'user_submitted']
    
    saved_rows = [r for r in rows if r[0] in st.session_state.favorites]
    heartwarming_rows = [r for r in ai_rows if r[5] == 'heartwarming']
    fun_rows = [r for r in ai_rows if r[5] == 'cute_or_fun']
    improvement_rows = [r for r in ai_rows if r[5] == 'improvement']
    other_rows = [r for r in ai_rows if r[5] not in CATEGORY_EMOJIS.keys()]

    tab_titles = [
        f"All ({len(ai_rows)})",
        f"⭐ Saved ({len(saved_rows)})",
        f"💌 User Submissions ({len(user_rows)})",
        f"{CATEGORY_EMOJIS['heartwarming']} Heartwarming ({len(heartwarming_rows)})",
        f"{CATEGORY_EMOJIS['cute_or_fun']} Cute or Fun ({len(fun_rows)})",
        f"{CATEGORY_EMOJIS['improvement']} Improvement ({len(improvement_rows)})",
        f"{CATEGORY_EMOJIS['none']} Others ({len(other_rows)})",
    ]
    all_tab, saved_tab, user_tab, heart_tab, fun_tab, improv_tab, none_tab = st.tabs(tab_titles)

    with all_tab:
        _display_articles_in_tab(all_tab, ai_rows, "all")
    with saved_tab:
        _display_articles_in_tab(saved_tab, saved_rows, "saved")
    with user_tab:
        _display_articles_in_tab(user_tab, user_rows, "user")
    with heart_tab:
        _display_articles_in_tab(heart_tab, heartwarming_rows, "heart")
    with fun_tab:
        _display_articles_in_tab(fun_tab, fun_rows, "fun")
    with improv_tab:
        _display_articles_in_tab(improv_tab, improvement_rows, "improv")
    with none_tab:
        _display_articles_in_tab(none_tab, other_rows, "other")
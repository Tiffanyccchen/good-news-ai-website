from __future__ import annotations

import streamlit as st
import hashlib
from app.utils import get_db, utcnow
from app.filters.llm_filter import validate_user_submission
import asyncio
import logging

def _article_id(title: str, content: str) -> str:
    """Creates a unique ID for an article based on its title and content."""
    return hashlib.sha256(f"{title}-{content}".encode()).hexdigest()

def render_submission_form():
    """Renders a form for users to submit their own good news article."""
    with st.expander("💌 Share Your Own Good News!"):
        with st.form(key="submission_form", clear_on_submit=True):
            title = st.text_input("Give your story a title")
            content = st.text_area("Your good news story", placeholder="I saw a stranger help someone change a tire in the rain today!")
            
            submitted = st.form_submit_button("Submit for Review")

            if submitted:
                if title and content:
                    with st.spinner("Analyzing your submission..."):
                        judgement = asyncio.run(validate_user_submission(title, content))

                    if judgement and judgement.is_safe_and_good:
                        try:
                            with get_db() as conn:
                                conn.execute(
                                    """
                                    INSERT INTO articles (id, url, title, content, reason, published, is_good, source_type, category)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        _article_id(title, content),
                                        None,  # No URL for user submissions
                                        title,
                                        content, # Store the story in the 'content' field
                                        "Verified by user submission.", # Reason field
                                        utcnow().isoformat(),
                                        1, # Passed check, so it is good
                                        'user_submitted',
                                        'user_submitted'
                                    )
                                )
                            st.toast("Thank you for sharing! Your story has been posted.", icon="🎉")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not submit article. It might already be in our database.")
                            logging.error(f"Error saving user submission: {e}")
                    else:
                        reason = "the AI moderator is currently unavailable. Please try again later."
                        if judgement:
                            reason = judgement.reason
                        st.error(f"Submission rejected: {reason}")
                else:
                    st.warning("Please provide both a title and a story.") 
# 🌞 Daily Good News

**Team:** Created by Yu Ting Chen

## 📖 Project Description

Daily Good News is an AI-powered web application designed to combat the negativity of the modern news cycle. It automatically fetches articles from major news outlets, uses a multi-stage AI pipeline to analyze and filter them, and presents a curated feed of exclusively positive and uplifting stories.

The core of the project is an automated pipeline that:
1.  **Fetches Data**: Periodically retrieves the latest articles from a curated list of reputable sources via the NewsAPI.
2.  **Analyzes Sentiment**: Performs an initial sentiment analysis on all fetched articles.
3.  **Applies LLM Judgement**: Uses a powerful Large Language Model (Llama 3 via the Groq API) to perform a sophisticated classification. The model determines if an article is genuinely "good news" and categorizes it as "Cute/Fun," "Improvement," or "Heartwarming."
4.  **Presents News**: Displays the curated good news in a clean, user-friendly interface built with Streamlit.

The application also allows users to submit their own good news links, which are validated by an AI moderator before being considered for the feed.

## ⚙️ Setup Instructions

Follow these steps to run the project locally.

**1. Clone the Repository**

```bash
git clone https://github.com/your-username/good-news.git
cd good-news
```

**2. Create a Virtual Environment**

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies**

Install all required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

**4. Configure API Keys**

This application uses [Streamlit's secrets management](https://docs.streamlit.io/library/advanced-features/secrets-management). Create a file at `~/.streamlit/secrets.toml` or inside your project at `.streamlit/secrets.toml` and add your API keys to it:

```toml
# .streamlit/secrets.toml

NEWS_API_KEY = "YOUR_NEWSAPI_KEY_HERE"
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```

*   Get a free **NewsAPI key** at [newsapi.org](https://newsapi.org).
*   Get a free **Groq API key** at [console.groq.com/keys](https://console.groq.com/keys).

**5. Run the Application**

Launch the Streamlit application with the following command:

```bash
streamlit run run_frontend.py
```

The application should now be running and accessible in your web browser.

## 🚀 Usage Instructions

Once the application is running:
*   **View Articles**: The main page displays the latest good news articles in tabs with some filter and sorting options available.
*   **Customize Your Feed**: Use the sidebar to select the maximum number of articles to display and sort them by "Most Recent" or "Most Positive." Also able to like and save your own favorite news.
*   **Auto-Refresh**: The feed automatically checks for new articles every minute. The data pipeline itself runs in the background every 15 minutes.
*   **Submit Your Own News**: Use the submission form to enter a title and a story. The AI moderator will review it for safety and positivity.

## 📝 Assumptions & Limitations

*   **API Dependencies**: The application is entirely dependent on the availability and terms of service of the free NewsAPI and Groq APIs. The time and amount are limited. (See https://newsapi.org/pricing and https://console.groq.com/docs/rate-limits)
*   **Subjectivity of "Good News"**: The classification of news is performed by an LLM and is inherently subjective. While the prompts are engineered to be specific, the model's judgment may not align with every user's definition of "good news."
*   **Initial Run**: The first time the pipeline runs, it fetches a larger historical backlog of articles (one week). This initial run may take a few minutes to complete. Subsequent runs are much faster. (But still subject to free-tier Grok TPM and RPM)

## 📊 Data Sources

The primary data source is the **NewsAPI** ([newsapi.org](https://newsapi.org)). The application fetches articles from a pre-defined list of major, reputable news organizations to ensure data quality.

## Architecture

```
               ┌────────────┐
               │  Scheduler │  (cron / HF refresh button)
               └─────┬──────┘
                     ▼
             ┌─────────────────┐
             │  fetcher.py     │  ← RSS feeds / NewsAPI
             └─────────────────┘
                     ▼
             ┌─────────────────┐
             │ sentiment.py    │  ← fast, local model
             └─────────────────┘
                     ▼
             ┌─────────────────┐
             │ llm_filter.py   │  ← Groq (or other LLM)
             └─────────────────┘
                     ▼
             ┌─────────────────┐
             │   SQLite DB     │
             └─────────────────┘
                     ▼
             ┌─────────────────┐
             │   Streamlit UI  │
             └─────────────────┘
```

## Customising the good-news criteria

* Adjust the `min_positive_prob` threshold in `app/sentiment.py`.
* Tweak the prompt inside `app/llm_filter.py` (or even bring your own model).

## Caveats / Todo

* Add more features in the website e.g. Comment section, Specify the type of good news you like for recommendation...
* The user saved articles will only persisted for the current session. That's TBD.
* For scaling up, a more sophisticated system design is needed. This is just a MVP to showcase the idea.
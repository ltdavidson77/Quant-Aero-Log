# File: api_clients/news_feed_handler.py
# --------------------------------------
# Fetches news headlines using NewsAPI.org

import requests
import pandas as pd
import os

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your_news_api_key")
BASE_NEWS_URL = "https://newsapi.org/v2/everything"

def fetch_news_headlines(query="stock market", from_date="2023-01-01", to_date="2023-01-02", language="en"):
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": language,
        "sortBy": "relevancy",
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(BASE_NEWS_URL, params=params)
    if response.status_code != 200:
        raise Exception(f"News API error: {response.text}")

    articles = response.json().get("articles", [])
    df = pd.DataFrame(articles)
    if not df.empty:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"])
        df.set_index("publishedAt", inplace=True)
    return df[["title", "description", "url"]]

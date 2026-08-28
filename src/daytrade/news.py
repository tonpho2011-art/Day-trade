"""News headlines via Google News RSS (no API key needed) with a crude
keyword-based sentiment score. This is NOT real sentiment analysis --
it just counts positive/negative words in headlines as a rough signal."""
from urllib.parse import quote

import feedparser

POSITIVE_WORDS = {
    "surge", "soar", "beat", "beats", "record", "growth", "upgrade", "rally",
    "gain", "gains", "jump", "jumps", "strong", "outperform", "bullish",
    "profit", "wins", "win", "expand", "expansion", "positive",
}
NEGATIVE_WORDS = {
    "plunge", "plummet", "miss", "misses", "downgrade", "sell-off", "selloff",
    "drop", "drops", "fall", "falls", "weak", "underperform", "bearish",
    "loss", "losses", "lawsuit", "investigation", "cut", "cuts", "negative",
    "warning", "recall", "layoffs", "fraud",
}


def get_headlines(symbol: str, limit: int = 6) -> list[str]:
    query = quote(f"{symbol} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:limit]]


def score_sentiment(headlines: list[str]) -> dict:
    pos = neg = 0
    for h in headlines:
        words = set(h.lower().replace("-", " ").split())
        pos += len(words & POSITIVE_WORDS)
        neg += len(words & NEGATIVE_WORDS)

    if pos > neg:
        label = "leaning positive"
    elif neg > pos:
        label = "leaning negative"
    else:
        label = "mixed / neutral"

    return {"positive_hits": pos, "negative_hits": neg, "label": label}

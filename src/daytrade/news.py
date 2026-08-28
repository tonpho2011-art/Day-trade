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


THEMES = {
    "earnings/guidance": ["earnings", "guidance", "beats", "misses", "quarter", "q1", "q2", "q3", "q4", "revenue", "eps"],
    "M&A / partnership": ["acquisition", "acquire", "merger", "partnership", "deal", "expands"],
    "analyst action": ["upgrade", "downgrade", "price target", "analyst", "rating"],
    "legal / regulatory": ["lawsuit", "investigation", "fraud", "recall", "fine", "sec ", "probe"],
    "leadership / restructuring": ["ceo", "layoffs", "restructuring", "resigns", "steps down"],
}


def detect_themes(headlines: list[str]) -> list[str]:
    found = []
    joined = " | ".join(h.lower() for h in headlines)
    for theme, keywords in THEMES.items():
        if any(kw in joined for kw in keywords):
            found.append(theme)
    return found


def summarize_news_impact(headlines: list[str], sentiment: dict, technical: dict) -> str:
    if not headlines:
        return "No recent headlines were found for this ticker, so there's no news angle to weigh in here."

    themes = detect_themes(headlines)
    theme_str = ", ".join(themes) if themes else "no clear recurring theme"

    vr = technical.get("volume_ratio")
    if vr is not None and vr == vr and vr > 1.5:
        conviction = (f"volume running {vr:.1f}x the 20-day average, which suggests the move is "
                      "backed by real conviction rather than just headline noise")
    elif vr is not None and vr == vr:
        conviction = f"volume at {vr:.1f}x average -- not an unusually strong confirmation either way"
    else:
        conviction = "volume data unavailable"

    news_leans_bullish = sentiment["positive_hits"] > sentiment["negative_hits"]
    news_leans_bearish = sentiment["negative_hits"] > sentiment["positive_hits"]
    signal_bullish = technical["signal"] in ("BUY", "STRONG BUY")
    signal_bearish = technical["signal"] in ("SELL", "STRONG SELL")

    if (news_leans_bullish and signal_bullish) or (news_leans_bearish and signal_bearish):
        agreement = f"that lines up with the {technical['signal']} technical read (score {technical['score']})"
    elif (news_leans_bullish and signal_bearish) or (news_leans_bearish and signal_bullish):
        agreement = (f"that actually CONFLICTS with the {technical['signal']} technical read (score "
                     f"{technical['score']}) -- price action and news sentiment are pointing different ways here")
    else:
        agreement = f"the technical read is {technical['signal']} (score {technical['score']}), roughly neutral either way"

    return (f"Headlines this cycle center on {theme_str}, and skew {sentiment['label']} "
            f"({sentiment['positive_hits']} positive / {sentiment['negative_hits']} negative keyword hits). "
            f"{agreement[0].upper()}{agreement[1:]}, and {conviction}.")


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

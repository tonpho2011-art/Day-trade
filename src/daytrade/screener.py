"""Market-wide scans using yfinance's built-in predefined screeners."""
import pandas as pd
import yfinance as yf

SCREENS = {
    "gainers": "day_gainers",
    "losers": "day_losers",
    "active": "most_actives",
}


def top_movers(kind: str = "gainers", count: int = 15) -> pd.DataFrame:
    if kind not in SCREENS:
        raise ValueError(f"kind must be one of {list(SCREENS)}")
    res = yf.screen(SCREENS[kind], count=count)
    quotes = res.get("quotes", [])
    rows = [{
        "symbol": q.get("symbol"),
        "name": q.get("shortName"),
        "price": q.get("regularMarketPrice"),
        "change_%": q.get("regularMarketChangePercent"),
        "volume": q.get("regularMarketVolume"),
    } for q in quotes]
    return pd.DataFrame(rows)

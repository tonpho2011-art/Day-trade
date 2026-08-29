"""Builds a liquid scan universe: today's movers + the S&P 500.

Scanning the full ~8,000-ticker US market live isn't practical with free
data sources -- this covers what actually matters for day-trading
purposes: names already in play, plus every S&P 500 constituent for
breadth. That's a stand-in for "the whole market", not literally it.
"""
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from daytrade.screener import top_movers

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Used only if the Wikipedia fetch fails -- a small basket of large, liquid
# names so the scanner still has something to work with.
FALLBACK_SP500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO",
    "JPM", "V", "UNH", "XOM", "LLY", "MA", "HD", "PG", "COST", "MRK", "ABBV",
    "CVX", "PEP", "KO", "ADBE", "WMT", "BAC", "CRM", "NFLX", "AMD", "DIS",
]

_cache = {"symbols": None, "fetched_at": None}
_CACHE_TTL_SECONDS = 86400


def get_sp500_symbols(force_refresh: bool = False) -> list[str]:
    now = datetime.now(timezone.utc)
    if (
        not force_refresh
        and _cache["symbols"]
        and (now - _cache["fetched_at"]).total_seconds() < _CACHE_TTL_SECONDS
    ):
        return _cache["symbols"]

    try:
        req = urllib.request.Request(SP500_URL, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=15).read()
        table = pd.read_html(io.BytesIO(html))[0]
        symbols = [s.replace(".", "-") for s in table["Symbol"].tolist()]
    except Exception:
        symbols = FALLBACK_SP500

    _cache["symbols"] = symbols
    _cache["fetched_at"] = now
    return symbols


def get_daytrade_universe(max_count: int = 100) -> list[str]:
    """Today's gainers/active/losers first (already in play), then filled
    out with S&P 500 names for breadth, capped at max_count."""
    symbols: list[str] = []
    seen: set[str] = set()
    kinds = ("gainers", "active", "losers")
    per_kind = max(10, max_count // 3)

    def _fetch(kind: str):
        try:
            return top_movers(kind, count=per_kind)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=len(kinds)) as pool:
        dfs = pool.map(_fetch, kinds)

    for df in dfs:
        if df is None:
            continue
        for s in df["symbol"].tolist():
            if s and s not in seen:
                seen.add(s)
                symbols.append(s)

    if len(symbols) < max_count:
        for s in get_sp500_symbols():
            if s not in seen:
                seen.add(s)
                symbols.append(s)
            if len(symbols) >= max_count:
                break

    return symbols[:max_count]

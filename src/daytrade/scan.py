"""Batch scans across many tickers at once.

The per-ticker path in analyze.py is fine for one symbol but far too slow for
a whole candidate list, so these helpers pull OHLCV for every candidate in a
single threaded yfinance download and score them locally.
"""
import pandas as pd
import yfinance as yf

from daytrade.indicators import build_signal
from daytrade.news import get_headlines, score_sentiment
from daytrade.screener import top_movers

MIN_BARS = 40
MAX_CANDIDATES = 40


def _slice_symbol(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Pull one symbol's OHLCV out of a batch download.

    A single-symbol download comes back with flat columns, a multi-symbol one
    with a (symbol, field) MultiIndex, so both shapes are handled here.
    """
    if isinstance(data.columns, pd.MultiIndex):
        if symbol not in data.columns.get_level_values(0):
            return pd.DataFrame()
        df = data[symbol].copy()
    else:
        df = data.copy()
    df = df.rename(columns=str.title)
    return df.dropna()


def _batch_download(symbols: list[str], period: str, interval: str) -> pd.DataFrame | None:
    """Dedup symbols and download them all in one threaded call. Returns
    None if the download failed or came back empty."""
    try:
        data = yf.download(
            tickers=symbols,
            period=period,
            interval=interval,
            group_by="ticker",
            threads=True,
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return None
    return None if data is None or data.empty else data


def batch_signals(symbols: list[str], meta: dict | None = None, period: str = "6mo") -> list[dict]:
    """Download every symbol in one call and run build_signal on each.

    `meta` optionally maps symbol -> extra fields (name, change_%) to merge into
    each result. Symbols that error out or have too little history are skipped.
    """
    symbols = list(dict.fromkeys(s for s in symbols if s))
    if not symbols:
        return []

    meta = meta or {}
    data = _batch_download(symbols, period, "1d")
    if data is None:
        return []

    results = []
    for symbol in symbols:
        try:
            df = _slice_symbol(data, symbol)
            if len(df) < MIN_BARS or "Close" not in df or "Volume" not in df:
                continue
            result = {"symbol": symbol, **meta.get(symbol, {}), **build_signal(df)}
        except Exception:
            continue
        results.append(result)
    return results


def batch_intraday_signals(
    symbols: list[str],
    meta: dict | None = None,
    period: str = "5d",
    interval: str = "5m",
    sma_fast: int = 6,
    sma_slow: int = 24,
) -> list[dict]:
    """Like batch_signals, but on intraday bars.

    Default sma_fast/sma_slow (6/24 bars) on 5-minute bars is roughly a
    30-minute vs 2-hour trend read -- tune via the args if you change
    `interval`. Candlestick patterns are already folded into build_signal's
    score/signal/reasons; `patterns` is still exposed on the result for
    callers that want the raw booleans.
    """
    symbols = list(dict.fromkeys(s for s in symbols if s))
    if not symbols:
        return []

    meta = meta or {}
    data = _batch_download(symbols, period, interval)
    if data is None:
        return []

    min_bars = max(MIN_BARS, sma_slow + 5)
    results = []
    for symbol in symbols:
        try:
            df = _slice_symbol(data, symbol)
            if len(df) < min_bars or "Close" not in df or "Volume" not in df:
                continue

            base = build_signal(df, sma_fast=sma_fast, sma_slow=sma_slow)
            result = {"symbol": symbol, **meta.get(symbol, {}), **base, "df": df}
        except Exception:
            continue
        results.append(result)
    return results


def _candidates(universe: str, count: int) -> tuple[list[str], dict]:
    kinds = ["gainers", "active", "losers"] if universe == "combined" else [universe]
    per_kind = max(5, count // max(1, len(kinds)) + 5)

    symbols, meta = [], {}
    for kind in kinds:
        try:
            df = top_movers(kind, count=per_kind)
        except Exception:
            continue
        for row in df.to_dict("records"):
            symbol = row.get("symbol")
            if not symbol or symbol in meta:
                continue
            symbols.append(symbol)
            meta[symbol] = {
                "name": row.get("name"),
                "candidate_price": row.get("price"),
                "change_%": row.get("change_%"),
            }
    return symbols[:MAX_CANDIDATES], meta


def strong_movers(direction: str, universe: str = "combined", count: int = 25) -> list[dict]:
    """Scan the current movers universe for STRONG BUY / STRONG SELL signals."""
    if direction not in ("buy", "sell"):
        raise ValueError("direction must be 'buy' or 'sell'")

    symbols, meta = _candidates(universe, count)
    scanned = batch_signals(symbols, meta=meta)

    wanted = "STRONG BUY" if direction == "buy" else "STRONG SELL"
    hits = [r for r in scanned if r["signal"] == wanted]
    hits.sort(key=lambda r: r["score"], reverse=direction == "buy")
    return hits[:count]


def news_movers(count: int = 25, max_change_pct: float = 3.0) -> list[dict]:
    """Names in play on volume but not yet priced in, with positive headlines.

    Highly speculative: this is keyword counting on recent headlines, not any
    kind of causal analysis. Treat it as a watchlist starting point only.
    """
    symbols, meta = [], {}
    for kind in ("active", "gainers"):
        try:
            df = top_movers(kind, count=count)
        except Exception:
            continue
        for row in df.to_dict("records"):
            symbol = row.get("symbol")
            if not symbol or symbol in meta:
                continue
            symbols.append(symbol)
            meta[symbol] = row

    results = []
    for symbol in symbols[:MAX_CANDIDATES]:
        change = meta[symbol].get("change_%")
        if change is None or abs(change) >= max_change_pct:
            continue
        try:
            headlines = get_headlines(symbol)
        except Exception:
            continue
        if not headlines:
            continue
        sentiment = score_sentiment(headlines)
        if sentiment["positive_hits"] < 2 or sentiment["positive_hits"] <= sentiment["negative_hits"]:
            continue
        results.append({
            "symbol": symbol,
            "name": meta[symbol].get("name"),
            "change_%": change,
            "positive_hits": sentiment["positive_hits"],
            "negative_hits": sentiment["negative_hits"],
            "top_headline": headlines[0],
        })

    results.sort(key=lambda r: r["positive_hits"], reverse=True)
    return results

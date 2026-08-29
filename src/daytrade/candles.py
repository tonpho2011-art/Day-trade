"""Simple candlestick pattern detection on the most recent bar(s).

These are classic reversal/continuation patterns, checked with plain
OHLC math -- no external TA library needed. Like everything else here,
treat these as one input among several, not a signal on their own.
"""
import pandas as pd


def _body(row) -> float:
    return abs(row["Close"] - row["Open"])


def _range(row) -> float:
    return row["High"] - row["Low"]


def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    prev_red = prev["Close"] < prev["Open"]
    curr_green = curr["Close"] > curr["Open"]
    engulfs = curr["Open"] <= prev["Close"] and curr["Close"] >= prev["Open"]
    return bool(prev_red and curr_green and engulfs)


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    prev_green = prev["Close"] > prev["Open"]
    curr_red = curr["Close"] < curr["Open"]
    engulfs = curr["Open"] >= prev["Close"] and curr["Close"] <= prev["Open"]
    return bool(prev_green and curr_red and engulfs)


def is_hammer(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    row = df.iloc[-1]
    body, rng = _body(row), _range(row)
    if rng <= 0:
        return False
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    return bool(body <= 0.3 * rng and lower_wick >= 2 * body and upper_wick <= 0.15 * rng)


def is_shooting_star(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    row = df.iloc[-1]
    body, rng = _body(row), _range(row)
    if rng <= 0:
        return False
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    return bool(body <= 0.3 * rng and upper_wick >= 2 * body and lower_wick <= 0.15 * rng)


def is_doji(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    row = df.iloc[-1]
    rng = _range(row)
    if rng <= 0:
        return False
    return bool(_body(row) <= 0.1 * rng)


def detect_patterns(df: pd.DataFrame) -> dict:
    return {
        "bullish_engulfing": is_bullish_engulfing(df),
        "bearish_engulfing": is_bearish_engulfing(df),
        "hammer": is_hammer(df),
        "shooting_star": is_shooting_star(df),
        "doji": is_doji(df),
    }

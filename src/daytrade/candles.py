"""Candlestick pattern detection on the most recent bar(s).

Classic reversal/continuation patterns, checked with plain OHLC math -- no
external TA library needed. Like everything else here, treat these as one
input among several, not a signal on their own.
"""
import pandas as pd


def _body(row) -> float:
    return abs(row["Close"] - row["Open"])


def _range(row) -> float:
    return row["High"] - row["Low"]


def _upper_wick(row) -> float:
    return row["High"] - max(row["Open"], row["Close"])


def _lower_wick(row) -> float:
    return min(row["Open"], row["Close"]) - row["Low"]


def _is_green(row) -> bool:
    return row["Close"] > row["Open"]


def _is_red(row) -> bool:
    return row["Close"] < row["Open"]


def _recent_trend(df: pd.DataFrame, lookback: int = 5) -> str:
    """Cheap trend read for the bars leading into a pattern: compares the
    average close of the lookback window before the pattern bar(s) against
    the window before that. Used only to tell a hammer from a hanging man
    (same shape, opposite meaning depending on what preceded it)."""
    closes = df["Close"]
    if len(closes) < lookback * 2 + 1:
        return "flat"
    recent = closes.iloc[-lookback - 1:-1].mean()
    prior = closes.iloc[-lookback * 2 - 1:-lookback - 1].mean()
    if recent > prior * 1.001:
        return "up"
    if recent < prior * 0.999:
        return "down"
    return "flat"


def _long_lower_wick_small_body(row) -> bool:
    body, rng = _body(row), _range(row)
    if rng <= 0:
        return False
    return bool(body <= 0.3 * rng and _lower_wick(row) >= 2 * body and _upper_wick(row) <= 0.15 * rng)


def _long_upper_wick_small_body(row) -> bool:
    body, rng = _body(row), _range(row)
    if rng <= 0:
        return False
    return bool(body <= 0.3 * rng and _upper_wick(row) >= 2 * body and _lower_wick(row) <= 0.15 * rng)


def is_hammer(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    return _long_lower_wick_small_body(df.iloc[-1]) and _recent_trend(df) != "up"


def is_hanging_man(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    return _long_lower_wick_small_body(df.iloc[-1]) and _recent_trend(df) == "up"


def is_inverted_hammer(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    return _long_upper_wick_small_body(df.iloc[-1]) and _recent_trend(df) != "up"


def is_shooting_star(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    return _long_upper_wick_small_body(df.iloc[-1]) and _recent_trend(df) == "up"


def is_doji(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    row = df.iloc[-1]
    rng = _range(row)
    if rng <= 0:
        return False
    return bool(_body(row) <= 0.1 * rng)


def is_spinning_top(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    row = df.iloc[-1]
    body, rng = _body(row), _range(row)
    if rng <= 0:
        return False
    return bool(0.05 * rng < body <= 0.3 * rng and _upper_wick(row) >= body and _lower_wick(row) >= body)


def is_bullish_marubozu(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    row = df.iloc[-1]
    body, rng = _body(row), _range(row)
    if rng <= 0:
        return False
    return bool(_is_green(row) and (rng - body) <= 0.05 * rng)


def is_bearish_marubozu(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    row = df.iloc[-1]
    body, rng = _body(row), _range(row)
    if rng <= 0:
        return False
    return bool(_is_red(row) and (rng - body) <= 0.05 * rng)


def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    engulfs = curr["Open"] <= prev["Close"] and curr["Close"] >= prev["Open"]
    return bool(_is_red(prev) and _is_green(curr) and engulfs)


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    engulfs = curr["Open"] >= prev["Close"] and curr["Close"] <= prev["Open"]
    return bool(_is_green(prev) and _is_red(curr) and engulfs)


def is_bullish_harami(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    if not _is_red(prev):
        return False
    prev_body = _body(prev)
    inside = curr["Open"] <= prev["Open"] and curr["Open"] >= prev["Close"] and \
        curr["Close"] <= prev["Open"] and curr["Close"] >= prev["Close"]
    return bool(_is_green(curr) and inside and _body(curr) < 0.6 * prev_body)


def is_bearish_harami(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    if not _is_green(prev):
        return False
    prev_body = _body(prev)
    inside = curr["Open"] <= prev["Close"] and curr["Open"] >= prev["Open"] and \
        curr["Close"] <= prev["Close"] and curr["Close"] >= prev["Open"]
    return bool(_is_red(curr) and inside and _body(curr) < 0.6 * prev_body)


def is_piercing_line(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    if not (_is_red(prev) and _is_green(curr)):
        return False
    midpoint = (prev["Open"] + prev["Close"]) / 2
    return bool(curr["Open"] < prev["Close"] and midpoint < curr["Close"] < prev["Open"])


def is_dark_cloud_cover(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    if not (_is_green(prev) and _is_red(curr)):
        return False
    midpoint = (prev["Open"] + prev["Close"]) / 2
    return bool(curr["Open"] > prev["Close"] and prev["Open"] < curr["Close"] < midpoint)


def is_tweezer_bottom(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    rng = max(_range(prev), _range(curr))
    if rng <= 0:
        return False
    similar_low = abs(prev["Low"] - curr["Low"]) <= 0.1 * rng
    return bool(similar_low and _is_red(prev) and _is_green(curr))


def is_tweezer_top(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    rng = max(_range(prev), _range(curr))
    if rng <= 0:
        return False
    similar_high = abs(prev["High"] - curr["High"]) <= 0.1 * rng
    return bool(similar_high and _is_green(prev) and _is_red(curr))


def is_morning_star(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    c1_body, c2_body = _body(c1), _body(c2)
    if c1_body <= 0:
        return False
    star = c2_body < 0.5 * c1_body
    gap_down = max(c2["Open"], c2["Close"]) < c1["Close"]
    recovers = c3["Close"] > (c1["Open"] + c1["Close"]) / 2
    return bool(_is_red(c1) and star and gap_down and _is_green(c3) and recovers)


def is_evening_star(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    c1_body, c2_body = _body(c1), _body(c2)
    if c1_body <= 0:
        return False
    star = c2_body < 0.5 * c1_body
    gap_up = min(c2["Open"], c2["Close"]) > c1["Close"]
    breaks_down = c3["Close"] < (c1["Open"] + c1["Close"]) / 2
    return bool(_is_green(c1) and star and gap_up and _is_red(c3) and breaks_down)


def is_three_white_soldiers(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    greens = _is_green(c1) and _is_green(c2) and _is_green(c3)
    rising = c3["Close"] > c2["Close"] > c1["Close"]
    opens_higher = c2["Open"] > c1["Open"] and c3["Open"] > c2["Open"]
    return bool(greens and rising and opens_higher)


def is_three_black_crows(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    reds = _is_red(c1) and _is_red(c2) and _is_red(c3)
    falling = c3["Close"] < c2["Close"] < c1["Close"]
    opens_lower = c2["Open"] < c1["Open"] and c3["Open"] < c2["Open"]
    return bool(reds and falling and opens_lower)


# name -> bias, used by indicators.build_signal to turn a detected pattern
# into a vote without hardcoding the same mapping twice.
PATTERN_BIAS = {
    "hammer": 1,
    "inverted_hammer": 1,
    "bullish_engulfing": 1,
    "piercing_line": 1,
    "morning_star": 1,
    "three_white_soldiers": 1,
    "bullish_harami": 1,
    "tweezer_bottom": 1,
    "bullish_marubozu": 1,
    "hanging_man": -1,
    "shooting_star": -1,
    "bearish_engulfing": -1,
    "dark_cloud_cover": -1,
    "evening_star": -1,
    "three_black_crows": -1,
    "bearish_harami": -1,
    "tweezer_top": -1,
    "bearish_marubozu": -1,
    "doji": 0,
    "spinning_top": 0,
}

_DETECTORS = {
    "hammer": is_hammer,
    "hanging_man": is_hanging_man,
    "inverted_hammer": is_inverted_hammer,
    "shooting_star": is_shooting_star,
    "doji": is_doji,
    "spinning_top": is_spinning_top,
    "bullish_marubozu": is_bullish_marubozu,
    "bearish_marubozu": is_bearish_marubozu,
    "bullish_engulfing": is_bullish_engulfing,
    "bearish_engulfing": is_bearish_engulfing,
    "bullish_harami": is_bullish_harami,
    "bearish_harami": is_bearish_harami,
    "piercing_line": is_piercing_line,
    "dark_cloud_cover": is_dark_cloud_cover,
    "tweezer_bottom": is_tweezer_bottom,
    "tweezer_top": is_tweezer_top,
    "morning_star": is_morning_star,
    "evening_star": is_evening_star,
    "three_white_soldiers": is_three_white_soldiers,
    "three_black_crows": is_three_black_crows,
}


def detect_patterns(df: pd.DataFrame) -> dict:
    return {name: bool(fn(df)) for name, fn in _DETECTORS.items()}

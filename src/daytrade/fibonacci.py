"""Fibonacci retracement/extension levels off the most recent swing.

Finds the highest high and lowest low in a lookback window, figures out
which one happened more recently (that tells us whether we're mid-pullback
in an uptrend or mid-bounce in a downtrend), and derives the standard
retracement (0.236/0.382/0.5/0.618/0.786) and extension (1.272/1.618)
levels off that swing. Used two ways:

  1. As one more vote in indicators.build_signal: price reacting off a
     retracement level in the direction of the larger swing is a classic
     "buy the pullback" / "sell the bounce" read.
  2. As a sanity check on stop-loss/take-profit placement in risk.py --
     an ATR-based target that lands right on top of a real support/
     resistance level is more believable than one that ignores structure.
"""
import pandas as pd

RETRACEMENTS = (0.236, 0.382, 0.5, 0.618, 0.786)
EXTENSIONS = (1.272, 1.618)

# How close price needs to be to a level (as a fraction of the swing range)
# to count as "reacting off" it rather than just passing through.
_PROXIMITY = 0.08


def swing_high_low(df: pd.DataFrame, lookback: int = 40) -> dict | None:
    """Most recent swing high/low within the lookback window, and which
    one printed later (defines the active leg direction)."""
    window = df.iloc[-lookback:] if len(df) > lookback else df
    if len(window) < 5:
        return None

    high_idx = window["High"].idxmax()
    low_idx = window["Low"].idxmin()
    swing_high = float(window.loc[high_idx, "High"])
    swing_low = float(window.loc[low_idx, "Low"])
    if swing_high <= swing_low:
        return None

    # Position within the window tells us recency without relying on the
    # index type (could be a DatetimeIndex or a plain RangeIndex).
    high_pos = window.index.get_loc(high_idx)
    low_pos = window.index.get_loc(low_idx)
    leg = "up" if low_pos < high_pos else "down"

    return {"high": swing_high, "low": swing_low, "leg": leg}


def retracement_levels(swing: dict) -> dict[float, float]:
    """Price at each retracement ratio, measured back from the leg's end."""
    rng = swing["high"] - swing["low"]
    if swing["leg"] == "up":
        # Pullback support levels below the high.
        return {r: swing["high"] - rng * r for r in RETRACEMENTS}
    # Bounce resistance levels above the low.
    return {r: swing["low"] + rng * r for r in RETRACEMENTS}


def extension_levels(swing: dict) -> dict[float, float]:
    """Projected continuation targets beyond the swing, for take-profit
    placement -- the classic "measured move" levels."""
    rng = swing["high"] - swing["low"]
    if swing["leg"] == "up":
        return {e: swing["low"] + rng * e for e in EXTENSIONS}
    return {e: swing["high"] - rng * e for e in EXTENSIONS}


def fib_vote(df: pd.DataFrame, lookback: int = 40) -> tuple[int, str | None]:
    """+1/-1/0 vote from the last bar reacting off a fib level, plus a
    human-readable reason (or None if nothing fired)."""
    swing = swing_high_low(df, lookback)
    if swing is None or len(df) < 2:
        return 0, None

    rng = swing["high"] - swing["low"]
    tol = rng * _PROXIMITY
    last, prev = df.iloc[-1], df.iloc[-2]
    close = float(last["Close"])
    bullish_bar = close > float(prev["Close"])
    bearish_bar = close < float(prev["Close"])

    if swing["leg"] == "up":
        # Broke clean below the swing low -- the pullback failed.
        if close < swing["low"] - tol:
            return -1, f"Price broke below the swing low ({swing['low']:.2f}), pullback structure failed"
        levels = retracement_levels(swing)
        for r, level in levels.items():
            if abs(close - level) <= tol and bullish_bar:
                return 1, f"Bouncing off the {r:.1%} Fibonacci retracement ({level:.2f}) of the recent up-swing"
        return 0, None

    # leg == "down"
    if close > swing["high"] + tol:
        return 1, f"Price broke above the swing high ({swing['high']:.2f}), downtrend structure failed"
    levels = retracement_levels(swing)
    for r, level in levels.items():
        if abs(close - level) <= tol and bearish_bar:
            return -1, f"Rejected at the {r:.1%} Fibonacci retracement ({level:.2f}) of the recent down-swing"
    return 0, None


def nearest_support(df: pd.DataFrame, price: float, lookback: int = 40) -> float | None:
    """Nearest fib level (retracement or the swing low itself) below
    `price`, for tightening a stop-loss onto real structure."""
    swing = swing_high_low(df, lookback)
    if swing is None:
        return None
    candidates = [swing["low"], *retracement_levels(swing).values()]
    below = [c for c in candidates if c < price]
    return max(below) if below else None


def nearest_resistance(df: pd.DataFrame, price: float, lookback: int = 40) -> float | None:
    """Nearest fib level (retracement, swing high, or extension) above
    `price`, for placing a take-profit on real structure instead of a
    round percentage."""
    swing = swing_high_low(df, lookback)
    if swing is None:
        return None
    candidates = [swing["high"], *retracement_levels(swing).values(), *extension_levels(swing).values()]
    above = [c for c in candidates if c > price]
    return min(above) if above else None

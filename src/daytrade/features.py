"""Turns a build_signal()-style dict into a flat numeric feature vector for
the ML ensemble in ml_model.py.

Kept separate from indicators.py so the same feature construction runs
identically at training time (over historical bars via
indicators.signal_series) and at inference time (over the live last bar
via indicators.build_signal) -- any drift between the two would silently
break the model.
"""
from daytrade import candles

# Fixed order matters: this is the column order every model is trained and
# queried with.
FEATURE_COLUMNS = [
    "score",
    "trend_score",
    "momentum_score",
    "volatility_score",
    "volume_score",
    "candlestick_score",
    "rsi",
    "volume_ratio",
    "price_vs_sma_fast",
    "price_vs_sma_slow",
    "sma_fast_vs_slow",
    "atr_pct",
    "bull_pattern_count",
    "bear_pattern_count",
]


def build_features(sig: dict) -> dict:
    price = sig.get("price") or 0.0
    cat = sig.get("category_scores", {})
    patterns = sig.get("patterns", {}) or {}
    bull = sum(1 for name, hit in patterns.items() if hit and candles.PATTERN_BIAS.get(name, 0) > 0)
    bear = sum(1 for name, hit in patterns.items() if hit and candles.PATTERN_BIAS.get(name, 0) < 0)
    atr = sig.get("atr") or 0.0
    sma_fast = sig.get("sma_fast") or price
    sma_slow = sig.get("sma_slow") or price

    return {
        "score": sig.get("score") or 0.0,
        "trend_score": cat.get("trend", 0.0),
        "momentum_score": cat.get("momentum", 0.0),
        "volatility_score": cat.get("volatility", 0.0),
        "volume_score": cat.get("volume", 0.0),
        "candlestick_score": cat.get("candlestick", 0.0),
        "rsi": sig.get("rsi") if sig.get("rsi") is not None else 50.0,
        "volume_ratio": sig.get("volume_ratio") if sig.get("volume_ratio") is not None else 1.0,
        "price_vs_sma_fast": (price - sma_fast) / price if price else 0.0,
        "price_vs_sma_slow": (price - sma_slow) / price if price else 0.0,
        "sma_fast_vs_slow": (sma_fast - sma_slow) / price if price else 0.0,
        "atr_pct": atr / price if price else 0.0,
        "bull_pattern_count": bull,
        "bear_pattern_count": bear,
    }

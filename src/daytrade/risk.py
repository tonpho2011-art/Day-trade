"""Volatility-aware stop-loss / take-profit placement.

The old approach used one flat stop-loss %% and take-profit %% for every
symbol. That's blind to the fact that a small, choppy mover can swing 2%%
in a single 5-minute bar on pure noise, while a large-cap can trend 2%%
over an hour. Sizing the stop off each symbol's own recent volatility
(ATR -- Average True Range) fixes that: a quiet stock gets a tight stop,
a wild one gets room to breathe, and the take-profit target scales with
the stop so the reward:risk ratio stays constant.

Levels are then nudged onto real chart structure using the Fibonacci
swing levels from fibonacci.py: a stop is tightened up to the nearest
support instead of the raw ATR distance, if that support is closer,
because a level everyone else is watching too is more likely to matter.
Same idea for the take-profit against the nearest resistance/extension.
"""
from dataclasses import dataclass

import pandas as pd

from daytrade import fibonacci

DEFAULT_ATR_WINDOW = 14
DEFAULT_STOP_ATR_MULT = 1.5
DEFAULT_TARGET_ATR_MULT = 3.0  # 2:1 reward:risk by default, matching the mult ratio
MIN_STOP_PCT = 0.5   # floor -- never let a near-zero-ATR quiet stock get a hair-trigger stop
MAX_STOP_PCT = 6.0    # ceiling -- never let one wild bar blow the stop out past what's sane on leverage
MIN_REWARD_RISK = 1.5  # floor -- a fib snap that shrinks the target below this isn't worth taking as-is


def atr(df: pd.DataFrame, window: int = DEFAULT_ATR_WINDOW) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


@dataclass
class StopPlan:
    stop_price: float
    take_price: float
    stop_pct: float
    take_pct: float
    atr_value: float
    reason: str


def compute_stops(
    df: pd.DataFrame,
    entry_price: float,
    side: str = "long",
    stop_atr_mult: float = DEFAULT_STOP_ATR_MULT,
    target_atr_mult: float = DEFAULT_TARGET_ATR_MULT,
) -> StopPlan:
    """ATR-sized stop/take-profit around `entry_price`, snapped onto the
    nearest fib support/resistance when that's tighter than the raw ATR
    distance. `side` is "long" (only mode autotrade.py uses today)."""
    if side != "long":
        raise ValueError("only 'long' is supported")

    atr_series = atr(df)
    atr_value = float(atr_series.iloc[-1]) if not atr_series.empty and pd.notna(atr_series.iloc[-1]) else None
    if not atr_value or atr_value <= 0:
        # Fallback: no usable ATR (too little history) -- treat the recent
        # bar range as a rough proxy so we still get a sane, non-zero stop.
        atr_value = float((df["High"] - df["Low"]).tail(DEFAULT_ATR_WINDOW).mean()) or entry_price * 0.01

    stop_price = entry_price - stop_atr_mult * atr_value
    take_price = entry_price + target_atr_mult * atr_value
    reason = f"ATR({DEFAULT_ATR_WINDOW})={atr_value:.4f}, {stop_atr_mult}x/{target_atr_mult}x stop/target"

    support = fibonacci.nearest_support(df, entry_price)
    if support is not None and support > stop_price:
        stop_price = support
        reason += f"; stop snapped up to fib support {support:.2f}"

    resistance = fibonacci.nearest_resistance(df, entry_price)
    if resistance is not None and resistance < take_price:
        take_price = resistance
        reason += f"; target snapped down to fib resistance {resistance:.2f}"

    stop_pct = max(MIN_STOP_PCT, min(MAX_STOP_PCT, (entry_price - stop_price) / entry_price * 100))
    take_pct = max(0.5, (take_price - entry_price) / entry_price * 100)

    # A resistance level can sit closer than the ATR target would, which is
    # useful information (that's a real ceiling) but not a trade worth
    # taking if it wrecks the reward:risk ratio -- fall back to the ATR
    # target's ratio instead of trading a level that's too close to pay for
    # the risk just taken on the stop side.
    if take_pct < stop_pct * MIN_REWARD_RISK:
        take_pct = stop_pct * MIN_REWARD_RISK
        reason += f"; target reverted to {MIN_REWARD_RISK}x stop (fib target was too close to be worth the risk)"

    # Re-derive prices from the clamped percentages so stop_price/take_price
    # stay consistent with stop_pct/take_pct after the floor/ceiling.
    stop_price = entry_price * (1 - stop_pct / 100)
    take_price = entry_price * (1 + take_pct / 100)

    return StopPlan(
        stop_price=stop_price, take_price=take_price,
        stop_pct=stop_pct, take_pct=take_pct,
        atr_value=atr_value, reason=reason,
    )

"""Fibonacci 5-candle retracement: limit at a Fib pullback, SL=START, TP=END.

Pending order is placed on the close that *ends* the 5-candle trend (no
same-bar fill). Later bars fill if they touch the level (gap-through uses
the open). Defaults match the guide: 50% entry, no EMA filter, no min
range, extended-trend off.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

LONG = "LONG"
SHORT = "SHORT"
NY = "America/New_York"
NOTIONAL = 3500.0
RTH_START = 9 * 60 + 30
RTH_END = 16 * 60
NO_ENTRY = 15
FLATTEN = 5


@dataclass
class FibConfig:
    level: float = 0.5
    min_range: float = 0.0
    use_ema: bool = False
    ema_fast: int = 50
    ema_slow: int = 200
    notional: float = NOTIONAL


def fib_price(start: float, end: float, level: float) -> float:
    """START=100%, END=0%. 25% is nearest END; 70% is nearest START."""
    return end + (start - end) * level


def _to_ny(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if getattr(index, "tz", None) is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(NY)


def rth_bars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = _to_ny(out.index)
    minutes = out.index.hour * 60 + out.index.minute
    return out.loc[(minutes >= RTH_START) & (minutes < RTH_END)]


def _minutes(ts) -> int:
    return int(ts.hour) * 60 + int(ts.minute)


def _color(row) -> str | None:
    if row["Close"] > row["Open"]:
        return "green"
    if row["Close"] < row["Open"]:
        return "red"
    return None


def _ema_ok(bars: pd.DataFrame, i: int, side: str, cfg: FibConfig) -> bool:
    if not cfg.use_ema or i + 1 < cfg.ema_slow:
        return not cfg.use_ema
    close = bars["Close"].iloc[: i + 1]
    fast = close.ewm(span=cfg.ema_fast, adjust=False).mean().iloc[-1]
    slow = close.ewm(span=cfg.ema_slow, adjust=False).mean().iloc[-1]
    if side == LONG:
        return fast > slow
    return fast < slow


def _close_trade(trades: list[dict], pos: dict, exit_price: float, reason: str, ts, cfg: FibConfig, symbol: str) -> None:
    entry = pos["entry"]
    side = pos["side"]
    pnl_pct = (exit_price - entry) / entry if side == LONG else (entry - exit_price) / entry
    trades.append({
        "symbol": symbol,
        "side": side,
        "entry_time": pos["entry_time"],
        "exit_time": ts,
        "entry_price": entry,
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "pnl": pnl_pct * cfg.notional,
    })


def simulate_symbol(df: pd.DataFrame, cfg: FibConfig, symbol: str = "") -> list[dict]:
    bars = rth_bars(df)
    if bars.empty or len(bars) < 7:
        return []

    trades: list[dict] = []
    pending = None
    pos = None
    n = len(bars)

    for i in range(n):
        ts = bars.index[i]
        row = bars.iloc[i]
        o, h, low, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        mtc = RTH_END - _minutes(ts)

        if pos is not None and i > pos["entry_i"]:
            side = pos["side"]
            stop, target = pos["stop"], pos["target"]
            exit_price = reason = None
            if side == LONG:
                if o <= stop:
                    exit_price, reason = o, "stop-loss"
                elif o >= target:
                    exit_price, reason = o, "take-profit"
                elif low <= stop and h >= target:
                    exit_price, reason = stop, "stop-loss"
                elif low <= stop:
                    exit_price, reason = stop, "stop-loss"
                elif h >= target:
                    exit_price, reason = target, "take-profit"
            else:
                if o >= stop:
                    exit_price, reason = o, "stop-loss"
                elif o <= target:
                    exit_price, reason = o, "take-profit"
                elif h >= stop and low <= target:
                    exit_price, reason = stop, "stop-loss"
                elif h >= stop:
                    exit_price, reason = stop, "stop-loss"
                elif low <= target:
                    exit_price, reason = target, "take-profit"
            if exit_price is None and mtc <= FLATTEN:
                exit_price, reason = c, "flatten"
            if exit_price is not None:
                _close_trade(trades, pos, exit_price, reason, ts, cfg, symbol)
                pos = None

        if pos is None and pending is not None and i > pending["placed_i"]:
            side, entry = pending["side"], pending["entry"]
            filled = None
            if side == LONG:
                if o <= entry:
                    filled = o
                elif low <= entry:
                    filled = entry
            else:
                if o >= entry:
                    filled = o
                elif h >= entry:
                    filled = entry
            if filled is not None:
                pos = {
                    "side": side,
                    "entry": float(filled),
                    "stop": pending["stop"],
                    "target": pending["target"],
                    "entry_i": i,
                    "entry_time": ts,
                }
                pending = None
                # Allow SL/TP on the fill bar after the open fill.
                stop, target = pos["stop"], pos["target"]
                exit_price = reason = None
                if side == LONG:
                    if low <= stop and h >= target:
                        exit_price, reason = stop, "stop-loss"
                    elif low <= stop:
                        exit_price, reason = stop, "stop-loss"
                    elif h >= target:
                        exit_price, reason = target, "take-profit"
                else:
                    if h >= stop and low <= target:
                        exit_price, reason = stop, "stop-loss"
                    elif h >= stop:
                        exit_price, reason = stop, "stop-loss"
                    elif low <= target:
                        exit_price, reason = target, "take-profit"
                if exit_price is None and mtc <= FLATTEN:
                    exit_price, reason = c, "flatten"
                if exit_price is not None:
                    _close_trade(trades, pos, exit_price, reason, ts, cfg, symbol)
                    pos = None

        if mtc <= FLATTEN:
            pending = None
            continue
        if pos is not None or mtc <= NO_ENTRY:
            continue

        # Streak includes bar i. A completed trend uses a 5+ run *before* i
        # plus an opposite close on i. streak_color is color of i, so the
        # prior run is not streak. Measure the run that just broke:
        ended = _complete_on(bars, i, cfg)
        if ended:
            pending = ended

    if pos is not None:
        last = bars.iloc[-1]
        _close_trade(
            trades, pos, float(last["Close"]), "flatten", bars.index[-1], cfg, symbol,
        )
    return trades


def _complete_on(bars: pd.DataFrame, i: int, cfg: FibConfig):
    if i < 5:
        return None
    prev_open = float(bars.iloc[i - 1]["Open"])
    close = float(bars.iloc[i]["Close"])
    # Walk backward for a same-color run ending at i-1.
    run_color = _color(bars.iloc[i - 1])
    if run_color is None:
        return None
    if run_color == "green" and close >= prev_open:
        return None
    if run_color == "red" and close <= prev_open:
        return None
    j = i - 1
    while j >= 0 and _color(bars.iloc[j]) == run_color:
        j -= 1
    first_i = j + 1
    length = i - first_i
    if length < 5:
        return None
    run = bars.iloc[first_i:i]
    first = run.iloc[0]
    prev = bars.iloc[first_i - 1] if first_i > 0 else first
    if run_color == "green":
        start = min(float(first["Low"]), float(prev["Low"]))
        end = float(run["High"].max())
        side = LONG
    else:
        start = max(float(first["High"]), float(prev["High"]))
        end = float(run["Low"].min())
        side = SHORT
    if abs(start - end) < cfg.min_range:
        return None
    if not _ema_ok(bars, i, side, cfg):
        return None
    entry = fib_price(start, end, cfg.level)
    if side == LONG and not (start < entry < end):
        return None
    if side == SHORT and not (end < entry < start):
        return None
    return {
        "side": side,
        "entry": entry,
        "stop": start,
        "target": end,
        "placed_i": i,
    }


def summarize(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "n_wins": 0, "n_longs": 0, "n_shorts": 0,
            "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
            "profit_factor": None, "max_drawdown": 0.0, "exits": {},
        }
    pnls = [float(t["pnl"]) for t in sorted(trades, key=lambda t: t["exit_time"])]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gp, gl = float(sum(wins)), float(abs(sum(losses)))
    exits: dict[str, int] = {}
    for t in trades:
        exits[t["exit_reason"]] = exits.get(t["exit_reason"], 0) + 1
    equity = peak = max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return {
        "n_trades": n,
        "n_wins": len(wins),
        "n_longs": sum(1 for t in trades if t["side"] == LONG),
        "n_shorts": sum(1 for t in trades if t["side"] == SHORT),
        "win_rate": len(wins) / n,
        "total_pnl": float(sum(pnls)),
        "avg_pnl": float(sum(pnls) / n),
        "profit_factor": (gp / gl) if gl else None,
        "max_drawdown": max_dd,
        "exits": exits,
    }

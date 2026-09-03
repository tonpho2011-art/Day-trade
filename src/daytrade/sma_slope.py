"""SMA 34/89 dual-slope trend follower (CNPS 03), long and short.

Unlike a crossover, this does not wait for SMA(34) to cross SMA(89).
A long is both slopes rising; a short is both slopes falling. Optional
SMA(200) slope filter. Exits: N-point stop, 2N-point target, reversal,
filter flip, session flatten.

The published script is VN futures (09:00–14:30, 3 contracts, 10/20
points). Here points are dollars, contracts are shares, and the default
session is 09:30–14:30 America/New_York so it can run on Alpaca stocks.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

LONG = "LONG"
SHORT = "SHORT"
FLAT = "FLAT"
NY = "America/New_York"


@dataclass
class SlopeConfig:
    fast: int = 34
    slow: int = 89
    filter_len: int = 200
    use_filter: bool = False
    direction: str = "both"
    stop_points: float = 10.0
    take_points: float = 20.0
    take_profit: bool = True
    time_filter: bool = True
    session_start_min: int = 9 * 60 + 30
    session_end_min: int = 14 * 60 + 30
    interval_minutes: int = 2
    qty: float = 3.0


def _to_ny(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if getattr(index, "tz", None) is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(NY)


def rth_bars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = _to_ny(out.index)
    minutes = out.index.hour * 60 + out.index.minute
    mask = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    return out.loc[mask]


def _minutes(ts) -> int:
    return int(ts.hour) * 60 + int(ts.minute)


def in_session(ts, cfg: SlopeConfig) -> bool:
    m = _minutes(ts)
    if cfg.time_filter:
        return cfg.session_start_min <= m < cfg.session_end_min
    return (9 * 60 + 30) <= m < (16 * 60)


def is_flatten_bar(ts, cfg: SlopeConfig) -> bool:
    end = cfg.session_end_min if cfg.time_filter else 16 * 60
    return _minutes(ts) + cfg.interval_minutes >= end


def signal_series(df: pd.DataFrame, cfg: SlopeConfig) -> pd.Series:
    close = df["Close"]
    fast = close.rolling(cfg.fast).mean()
    slow = close.rolling(cfg.slow).mean()
    long_ok = (fast > fast.shift(1)) & (slow > slow.shift(1))
    short_ok = (fast < fast.shift(1)) & (slow < slow.shift(1))
    if cfg.use_filter:
        filt = close.rolling(cfg.filter_len).mean()
        long_ok = long_ok & (filt > filt.shift(1))
        short_ok = short_ok & (filt < filt.shift(1))
    out = pd.Series(FLAT, index=df.index)
    if cfg.direction in ("both", "long"):
        out = out.mask(long_ok.fillna(False), LONG)
    if cfg.direction in ("both", "short"):
        out = out.mask(short_ok.fillna(False), SHORT)
    return out


def filter_against(df: pd.DataFrame, cfg: SlopeConfig, side: str) -> pd.Series:
    """True when the SMA(200) slope filter says to exit `side`."""
    if not cfg.use_filter:
        return pd.Series(False, index=df.index)
    filt = df["Close"].rolling(cfg.filter_len).mean()
    rising = filt > filt.shift(1)
    if side == LONG:
        return (~rising).fillna(False)
    return rising.fillna(False)


def simulate_symbol(df: pd.DataFrame, cfg: SlopeConfig, symbol: str = "") -> list[dict]:
    bars = rth_bars(df)
    if bars.empty or len(bars) < cfg.slow + 2:
        return []
    sig = signal_series(bars, cfg)
    against_long = filter_against(bars, cfg, LONG)
    against_short = filter_against(bars, cfg, SHORT)
    trades: list[dict] = []
    pos = None
    pending = None
    n = len(bars)

    def open_pos(side: str, fill_i: int) -> None:
        nonlocal pos
        entry = float(bars.iloc[fill_i]["Open"])
        if entry <= 0:
            return
        ts = bars.index[fill_i]
        if side == LONG:
            stop, target = entry - cfg.stop_points, entry + cfg.take_points
        else:
            stop, target = entry + cfg.stop_points, entry - cfg.take_points
        pos = {
            "side": side,
            "entry_price": entry,
            "entry_time": ts,
            "entry_i": fill_i,
            "stop": stop,
            "target": target,
        }

    def close_pos(i: int, price: float, reason: str) -> None:
        nonlocal pos
        entry = pos["entry_price"]
        side = pos["side"]
        if side == LONG:
            pnl = (price - entry) * cfg.qty
        else:
            pnl = (entry - price) * cfg.qty
        trades.append({
            "symbol": symbol,
            "side": side,
            "entry_time": pos["entry_time"],
            "exit_time": bars.index[i],
            "entry_price": entry,
            "exit_price": float(price),
            "exit_reason": reason,
            "qty": cfg.qty,
            "pnl": pnl,
        })
        pos = None

    for i in range(n):
        ts = bars.index[i]
        row = bars.iloc[i]
        o, h, low, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])

        if pending is not None and i == pending["i"]:
            if pos is None and in_session(ts, cfg) and not is_flatten_bar(ts, cfg):
                open_pos(pending["side"], i)
            pending = None

        if pos is not None and i >= pos["entry_i"]:
            side = pos["side"]
            stop, target = pos["stop"], pos["target"]
            on_entry = i == pos["entry_i"]
            exit_price, reason = None, None
            if side == LONG:
                if not on_entry and o <= stop:
                    exit_price, reason = o, "stop-loss"
                elif cfg.take_profit and (not on_entry) and o >= target:
                    exit_price, reason = o, "take-profit"
                elif low <= stop and cfg.take_profit and h >= target:
                    exit_price, reason = stop, "stop-loss"
                elif low <= stop:
                    exit_price, reason = stop, "stop-loss"
                elif cfg.take_profit and h >= target:
                    exit_price, reason = target, "take-profit"
            else:
                if not on_entry and o >= stop:
                    exit_price, reason = o, "stop-loss"
                elif cfg.take_profit and (not on_entry) and o <= target:
                    exit_price, reason = o, "take-profit"
                elif h >= stop and cfg.take_profit and low <= target:
                    exit_price, reason = stop, "stop-loss"
                elif h >= stop:
                    exit_price, reason = stop, "stop-loss"
                elif cfg.take_profit and low <= target:
                    exit_price, reason = target, "take-profit"

            if exit_price is None and is_flatten_bar(ts, cfg):
                exit_price, reason = c, "flatten"
            if exit_price is None and (
                (side == LONG and bool(against_long.iloc[i]))
                or (side == SHORT and bool(against_short.iloc[i]))
            ):
                exit_price, reason = c, "filter"
            if exit_price is None and (
                (side == LONG and sig.iloc[i] == SHORT)
                or (side == SHORT and sig.iloc[i] == LONG)
            ):
                exit_price, reason = c, "reversal"
                if i + 1 < n and bars.index[i + 1].date() == ts.date():
                    pending = {"side": sig.iloc[i], "i": i + 1}

            if exit_price is not None:
                close_pos(i, exit_price, reason)

        if pos is not None or i >= n - 1:
            continue
        if pending is not None:
            continue
        if not in_session(ts, cfg):
            continue
        fill_ts = bars.index[i + 1]
        if fill_ts.date() != ts.date() or not in_session(fill_ts, cfg) or is_flatten_bar(fill_ts, cfg):
            continue
        if sig.iloc[i] in (LONG, SHORT):
            pending = {"side": sig.iloc[i], "i": i + 1}

    return trades


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

"""Walk-forward backtest of a single long-only vote function on 5m bars.

Matches autotrade.py mechanical risk: 2% stop, 4% target, no new entries
inside 15 minutes of the close, flatten at 5 minutes before the close.
Signal is taken on a closed bar; fill is the next bar's open (no lookahead).
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import numpy as np
import pandas as pd

from daytrade.agents.votes import BUY

NY = "America/New_York"
STOP_PCT = 0.02
TAKE_PCT = 0.04
NOTIONAL = 3500.0  # --cash-per-trade 500 * --leverage 7
NO_ENTRY_MINUTES_BEFORE_CLOSE = 15
FLATTEN_MINUTES_BEFORE_CLOSE = 5
RTH_START_MIN = 9 * 60 + 30
RTH_END_MIN = 16 * 60


def _to_ny(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if getattr(index, "tz", None) is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(NY)


def rth_bars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = _to_ny(out.index)
    minutes = out.index.hour * 60 + out.index.minute
    mask = (minutes >= RTH_START_MIN) & (minutes < RTH_END_MIN)
    return out.loc[mask]


def _minutes_to_close(ts) -> int:
    return RTH_END_MIN - (int(ts.hour) * 60 + int(ts.minute))


def _vote_prefix(bars: pd.DataFrame, i: int) -> pd.DataFrame:
    """History visible at bar i: current RTH session plus the prior day.

    EMA/BB need ~20-24 bars of warmup; PO3 needs the full opening range
    of the current session. Truncating to 'last 80 bars' would drop the
    open on a full 78-bar day.
    """
    prefix = bars.iloc[: i + 1]
    ts = prefix.index[-1]
    cutoff = pd.Timestamp(ts.date(), tz=ts.tz) - pd.Timedelta(days=1)
    return prefix[prefix.index >= cutoff]


def _session_slices(index: pd.DatetimeIndex) -> list[tuple[object, int, int]]:
    """(date, start, end) half-open index ranges for each RTH session."""
    out = []
    if len(index) == 0:
        return out
    start = 0
    cur = index[0].date()
    for i in range(1, len(index)):
        d = index[i].date()
        if d != cur:
            out.append((cur, start, i))
            cur = d
            start = i
    out.append((cur, start, len(index)))
    return out


def _ema_buy_mask(window: pd.DataFrame) -> np.ndarray:
    n = len(window)
    mask = np.zeros(n, dtype=bool)
    if n < 24 or "Close" not in window or "Volume" not in window:
        return mask
    close = window["Close"].to_numpy(dtype=float)
    open_ = window["Open"].to_numpy(dtype=float)
    vol = window["Volume"].to_numpy(dtype=float)
    ema9 = pd.Series(close).ewm(span=9, adjust=False).mean().to_numpy()
    ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().to_numpy()
    vr = vol / pd.Series(vol).rolling(20).mean().to_numpy()
    aligned = ema9 > ema21
    crossed = np.zeros(n, dtype=bool)
    crossed[1:] = (ema9[1:] > ema21[1:]) & (ema9[:-1] <= ema21[:-1])
    recent = crossed.copy()
    recent[1:] |= crossed[:-1]
    recent[2:] |= crossed[:-2]
    green = close > open_
    ok = np.isfinite(vr) & green & aligned & recent & (vr >= 1.5)
    ok[:23] = False
    return ok.astype(bool)


def _bb_buy_mask(window: pd.DataFrame) -> np.ndarray:
    from daytrade.candles import is_bullish_engulfing, is_hammer

    n = len(window)
    mask = np.zeros(n, dtype=bool)
    if n < 20 or "Close" not in window:
        return mask
    close = window["Close"]
    lower = close.rolling(20).mean() - 2.0 * close.rolling(20).std()
    poke = (window["Low"] < lower) & (close > lower) & lower.notna()
    for i in np.flatnonzero(poke.to_numpy()):
        if i < 19:
            continue
        prefix = window.iloc[: i + 1]
        if is_hammer(prefix) or is_bullish_engulfing(prefix):
            mask[i] = True
    return mask


def _po3_buy_mask(session: pd.DataFrame) -> np.ndarray:
    n = len(session)
    mask = np.zeros(n, dtype=bool)
    if n < 11:
        return mask
    low = session["Low"].to_numpy(dtype=float)
    high = session["High"].to_numpy(dtype=float)
    close = session["Close"].to_numpy(dtype=float)
    range_low = float(low[:6].min())
    po3_from = None
    swept = False
    for i in range(6, n):
        if low[i] < range_low:
            swept = True
        if swept and close[i] >= range_low:
            po3_from = i
            break
    if po3_from is None:
        return mask

    zones: list[list] = []  # [end_i, bottom, top, inverted]
    for i in range(n):
        if i >= 2 and low[i - 2] > high[i]:
            zones.append([i, float(high[i]), float(low[i - 2]), False])
        if i >= 1:
            prev = close[i - 1]
            for z in zones:
                if (not z[3]) and z[0] < i - 1 and prev > z[2]:
                    z[3] = True
        if i < po3_from or i < 4:
            continue
        last_l, last_h = low[i], high[i]
        for z in zones:
            if z[3] and last_l <= z[2] and last_h >= z[1]:
                mask[i] = True
                break
    return mask


def compute_buy_masks(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    """Vectorized equivalents of the three vote functions on already-RTH bars.

    EMA/BB see the current session plus the previous calendar day, matching
    `_vote_prefix`. PO3 sees only the current RTH session.
    """
    n = len(bars)
    out = {
        "po3_ifvg": np.zeros(n, dtype=bool),
        "ema_trend": np.zeros(n, dtype=bool),
        "bb_reversion": np.zeros(n, dtype=bool),
    }
    sessions = _session_slices(bars.index)
    by_date = {d: (a, b) for d, a, b in sessions}
    for d, start, end in sessions:
        prev = d - timedelta(days=1)
        if prev in by_date:
            a, b = by_date[prev]
            window = bars.iloc[a:end]
            offset = start - a
        else:
            window = bars.iloc[start:end]
            offset = 0
        ema = _ema_buy_mask(window)
        bb = _bb_buy_mask(window)
        out["ema_trend"][start:end] = ema[offset:]
        out["bb_reversion"][start:end] = bb[offset:]
        out["po3_ifvg"][start:end] = _po3_buy_mask(bars.iloc[start:end])
    return out


def _simulate_bars(
    bars: pd.DataFrame,
    is_buy,
    *,
    stop_pct: float = STOP_PCT,
    take_pct: float = TAKE_PCT,
    notional: float = NOTIONAL,
    symbol: str = "",
) -> list[dict]:
    if bars.empty or "Close" not in bars.columns:
        return []

    trades: list[dict] = []
    pos = None
    n = len(bars)

    for i in range(n):
        ts = bars.index[i]
        row = bars.iloc[i]
        mtc = _minutes_to_close(ts)

        if pos is not None and i >= pos["entry_i"]:
            stop = pos["stop"]
            target = pos["target"]
            o = float(row["Open"])
            h = float(row["High"])
            low = float(row["Low"])
            c = float(row["Close"])
            on_entry_bar = i == pos["entry_i"]

            exit_price = None
            reason = None
            if not on_entry_bar and o <= stop:
                exit_price, reason = o, "stop-loss"
            elif not on_entry_bar and o >= target:
                exit_price, reason = o, "take-profit"
            elif low <= stop and h >= target:
                exit_price, reason = stop, "stop-loss"
            elif low <= stop:
                exit_price, reason = stop, "stop-loss"
            elif h >= target:
                exit_price, reason = target, "take-profit"
            elif mtc <= FLATTEN_MINUTES_BEFORE_CLOSE:
                exit_price, reason = c, "flatten"

            if exit_price is not None:
                entry = pos["entry_price"]
                pnl_pct = (exit_price - entry) / entry
                trades.append({
                    "symbol": symbol,
                    "entry_time": pos["entry_time"],
                    "exit_time": ts,
                    "entry_price": entry,
                    "exit_price": float(exit_price),
                    "exit_reason": reason,
                    "pnl_pct": pnl_pct,
                    "pnl": pnl_pct * notional,
                })
                pos = None

        if pos is not None or i >= n - 1:
            continue

        fill_ts = bars.index[i + 1]
        if fill_ts.date() != ts.date():
            continue
        if _minutes_to_close(fill_ts) <= NO_ENTRY_MINUTES_BEFORE_CLOSE:
            continue

        try:
            buy = bool(is_buy(i))
        except Exception:
            continue
        if not buy:
            continue

        entry_price = float(bars.iloc[i + 1]["Open"])
        if entry_price <= 0:
            continue
        pos = {
            "entry_price": entry_price,
            "entry_time": fill_ts,
            "entry_i": i + 1,
            "stop": entry_price * (1.0 - stop_pct),
            "target": entry_price * (1.0 + take_pct),
        }

    return trades


def simulate_symbol(
    df: pd.DataFrame,
    vote_fn: Callable,
    *,
    stop_pct: float = STOP_PCT,
    take_pct: float = TAKE_PCT,
    notional: float = NOTIONAL,
    symbol: str = "",
) -> list[dict]:
    bars = rth_bars(df)

    def is_buy(i: int) -> bool:
        return vote_fn(_vote_prefix(bars, i)) == BUY

    return _simulate_bars(
        bars, is_buy, stop_pct=stop_pct, take_pct=take_pct, notional=notional, symbol=symbol,
    )


def summarize(trades: list[dict]) -> dict:
    n = len(trades)
    empty = {
        "n_trades": 0,
        "n_wins": 0,
        "n_losses": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "avg_pnl": 0.0,
        "avg_pnl_pct": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "profit_factor": None,
        "max_drawdown": 0.0,
        "exits": {},
    }
    if n == 0:
        return empty

    pnls = [float(t["pnl"]) for t in sorted(trades, key=lambda t: t.get("exit_time") or 0)]
    pcts = [float(t["pnl_pct"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gp = float(sum(wins))
    gl = float(abs(sum(losses)))
    exits: dict[str, int] = {}
    for t in trades:
        reason = t["exit_reason"]
        exits[reason] = exits.get(reason, 0) + 1

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    return {
        "n_trades": n,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": len(wins) / n,
        "total_pnl": float(sum(pnls)),
        "avg_pnl": float(sum(pnls) / n),
        "avg_pnl_pct": float(sum(pcts) / n),
        "gross_profit": gp,
        "gross_loss": gl,
        "profit_factor": (gp / gl) if gl else None,
        "max_drawdown": max_dd,
        "exits": exits,
    }


def equity_curve(trades: list[dict]) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(trades)
    frame = frame.sort_values("exit_time")
    return frame.set_index("exit_time")["pnl"].cumsum()


def simulate_all_agents(
    df: pd.DataFrame,
    symbol: str = "",
    *,
    stop_pct: float = STOP_PCT,
    take_pct: float = TAKE_PCT,
) -> dict[str, list[dict]]:
    """Run the three specialists independently on the same bars."""
    bars = rth_bars(df)
    masks = compute_buy_masks(bars)
    return {
        name: _simulate_bars(
            bars, (lambda i, m=mask: bool(m[i])),
            stop_pct=stop_pct, take_pct=take_pct, symbol=symbol,
        )
        for name, mask in masks.items()
    }


def simulate_parquet_job(job: tuple) -> tuple[str, dict[str, list[dict]]]:
    """Picklable worker: (parquet_path, symbol, stop_pct, take_pct) -> trades."""
    path, symbol, stop_pct, take_pct = job
    df = pd.read_parquet(path)
    return symbol, simulate_all_agents(
        df, symbol, stop_pct=stop_pct, take_pct=take_pct,
    )

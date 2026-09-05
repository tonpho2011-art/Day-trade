"""Fibonacci 5-candle retracement: limit at a Fib pullback, SL=START, TP=END.

Pending order is placed on the close that *ends* the 5-candle trend (no
same-bar fill). Later bars fill if they touch the level (gap-through uses
the open). Defaults match the guide: 50% entry, no EMA filter, no min
range, extended-trend off.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LONG = "LONG"
SHORT = "SHORT"
NY = "America/New_York"
NOTIONAL = 3500.0
RTH_START = 9 * 60 + 30
RTH_END = 16 * 60
NO_ENTRY = 15
FLATTEN = 5
BAD_BAR = 0.25
SWEEP_LEVELS = (0.25, 0.382, 0.50, 0.618, 0.70)


@dataclass
class FibConfig:
    level: float = 0.618
    min_range: float = 0.0
    use_ema: bool = False
    ema_fast: int = 50
    ema_slow: int = 200
    notional: float = NOTIONAL
    fill_at_limit: bool = False
    htf_minutes: int = 30


def fib_price(start: float, end: float, level: float) -> float:
    """START=100%, END=0%. 25% is nearest END; 70% is nearest START."""
    return end + (start - end) * level


def last_closed_htf_color(df: pd.DataFrame, asof, minutes: int = 30) -> int | None:
    """1=green, -1=red, 0=doji, None if no finished HTF bar yet."""
    if not minutes or df is None or df.empty:
        return None
    bars = df.copy()
    bars.index = _to_ny(pd.DatetimeIndex(bars.index))
    asof = pd.Timestamp(asof)
    if asof.tzinfo is None:
        asof = asof.tz_localize(NY)
    else:
        asof = asof.tz_convert(NY)
    window_start = asof.floor(f"{minutes}min")
    prior = bars.loc[bars.index < window_start]
    if prior.empty:
        return None
    htf = prior.resample(f"{minutes}min", closed="left", label="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna(how="any")
    if htf.empty:
        return None
    expected = max(1, minutes // 5)
    keep = []
    for ts in htf.index:
        end = ts + pd.Timedelta(minutes=minutes)
        piece = prior.loc[(prior.index >= ts) & (prior.index < end)]
        if len(piece) >= expected:
            keep.append(ts)
    if not keep:
        return None
    last = htf.loc[keep[-1]]
    o, c = float(last["Open"]), float(last["Close"])
    if c > o:
        return 1
    if c < o:
        return -1
    return 0


def htf_allows(side: str, color: int | None) -> bool:
    if color is None or color == 0:
        return False
    if side == LONG:
        return color == 1
    return color == -1


def plan_live_tickets(
    *,
    held: set[str],
    setups: list[dict],
    pending: list[dict],
    cap: int = 16,
) -> dict:
    """Keep at most (cap - fills) resting entry limits, ranked like the overlay."""
    held = set(held)
    room = max(0, cap - len(held))
    pending_by_sym = {p["symbol"]: p for p in pending if p["symbol"] not in held}
    fresh = [s for s in setups if s["symbol"] not in held]
    ranked = sorted(
        fresh,
        key=lambda t: (
            -actual_rr(t),
            -abs(float(t["stop"]) - float(t["target"])),
            t["symbol"],
        ),
    )
    chosen = ranked[:room]
    chosen_syms = {t["symbol"] for t in chosen}
    keep = [pending_by_sym[t["symbol"]] for t in chosen if t["symbol"] in pending_by_sym]
    place = [t for t in chosen if t["symbol"] not in pending_by_sym]
    cancel = [p for sym, p in pending_by_sym.items() if sym not in chosen_syms]
    return {"room": room, "place": place, "keep": keep, "cancel": cancel}


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
        "stop": pos["stop"],
        "target": pos["target"],
    })


def _fill_price(side: str, o: float, h: float, low: float, entry: float, fill_at_limit: bool):
    if side == LONG:
        if o <= entry:
            return entry if fill_at_limit else o
        if low <= entry:
            return entry
        return None
    if o >= entry:
        return entry if fill_at_limit else o
    if h >= entry:
        return entry
    return None


def simulate_symbol(df: pd.DataFrame, cfg: FibConfig, symbol: str = "") -> list[dict]:
    bars = rth_bars(df)
    if bars.empty or len(bars) < 7:
        return []
    trades, pending, pos = _walk(bars, cfg, symbol=symbol)
    if pos is not None:
        _close_trade(
            trades, pos, float(bars["Close"].iloc[-1]), "flatten", bars.index[-1], cfg, symbol,
        )
    return trades


def current_setup(df: pd.DataFrame, cfg: FibConfig) -> dict | None:
    """Pending limit after the last closed RTH bar, or None if already filled."""
    bars = rth_bars(df)
    if bars.empty or len(bars) < 7:
        return None
    trades, pending, pos = _walk(bars, cfg, symbol="")
    if pos is not None:
        return None
    return pending


def _walk(bars: pd.DataFrame, cfg: FibConfig, symbol: str = ""):
    opens = bars["Open"].to_numpy(dtype=np.float64, copy=False)
    highs = bars["High"].to_numpy(dtype=np.float64, copy=False)
    lows = bars["Low"].to_numpy(dtype=np.float64, copy=False)
    closes = bars["Close"].to_numpy(dtype=np.float64, copy=False)
    mtc_arr = RTH_END - (bars.index.hour.to_numpy() * 60 + bars.index.minute.to_numpy())
    colors = np.zeros(len(bars), dtype=np.int8)
    colors[closes > opens] = 1
    colors[closes < opens] = -1
    index = bars.index
    trades: list[dict] = []
    pending = None
    pos = None
    n = len(bars)
    for i in range(n):
        ts = index[i]
        o, h, low, c = opens[i], highs[i], lows[i], closes[i]
        mtc = int(mtc_arr[i])
        if i > 0:
            prev_close = float(closes[i - 1])
            if prev_close > 0 and (
                abs(o / prev_close - 1.0) > BAD_BAR
                or (h - low) / prev_close > BAD_BAR
            ):
                if pos is not None:
                    _close_trade(trades, pos, prev_close, "flatten", ts, cfg, symbol)
                    pos = None
                pending = None
                continue
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
            filled = _fill_price(side, o, h, low, entry, cfg.fill_at_limit)
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
        ended = _complete_on_arr(opens, highs, lows, closes, colors, i, cfg, bars)
        if ended:
            pending = ended
    return trades, pending, pos


def _complete_on_arr(opens, highs, lows, closes, colors, i: int, cfg: FibConfig, bars: pd.DataFrame):
    if i < 5:
        return None
    run_color = int(colors[i - 1])
    if run_color == 0:
        return None
    close = float(closes[i])
    prev_open = float(opens[i - 1])
    if run_color == 1 and close >= prev_open:
        return None
    if run_color == -1 and close <= prev_open:
        return None
    j = i - 1
    while j >= 0 and colors[j] == run_color:
        j -= 1
    first_i = j + 1
    if i - first_i < 5:
        return None
    prev_i = first_i - 1 if first_i > 0 else first_i
    if run_color == 1:
        start = float(min(lows[first_i], lows[prev_i]))
        end = float(highs[first_i:i].max())
        side = LONG
    else:
        start = float(max(highs[first_i], highs[prev_i]))
        end = float(lows[first_i:i].min())
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
    if cfg.htf_minutes:
        color = last_closed_htf_color(bars.iloc[: i + 1], bars.index[i], cfg.htf_minutes)
        if not htf_allows(side, color):
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


def apply_roundtrip_bps(trades: list[dict], bps: float, notional: float = NOTIONAL) -> list[dict]:
    cost = notional * bps / 10_000.0
    out = []
    for trade in trades:
        row = dict(trade)
        row["pnl"] = float(trade["pnl"]) - cost
        row["cost"] = cost
        out.append(row)
    return out


def cap_concurrent(trades: list[dict], max_positions: int) -> list[dict]:
    kept: list[dict] = []
    active_exits: list = []
    for trade in sorted(trades, key=lambda t: t["entry_time"]):
        entry = trade["entry_time"]
        active_exits = [ex for ex in active_exits if ex > entry]
        if len(active_exits) < max_positions:
            kept.append(trade)
            active_exits.append(trade["exit_time"])
    return kept


def actual_rr(trade: dict) -> float:
    entry = float(trade["entry_price"])
    stop = float(trade["stop"])
    target = float(trade["target"])
    if trade["side"] == LONG:
        den = entry - stop
        if den <= 0:
            return 0.0
        return (target - entry) / den
    den = stop - entry
    if den <= 0:
        return 0.0
    return (entry - target) / den


def select_portfolio(trades: list[dict], max_positions: int = 16) -> list[dict]:
    by_time: dict = {}
    for trade in trades:
        ts = pd.Timestamp(trade["entry_time"])
        by_time.setdefault(ts, []).append(trade)
    kept: list[dict] = []
    active_exits: list = []
    for ts in sorted(by_time):
        active_exits = [ex for ex in active_exits if ex > ts]
        slots = max_positions - len(active_exits)
        if slots <= 0:
            continue
        ranked = sorted(
            by_time[ts],
            key=lambda t: (
                -actual_rr(t),
                -abs(float(t["stop"]) - float(t["target"])),
                t.get("symbol") or "",
            ),
        )
        for trade in ranked[:slots]:
            kept.append(trade)
            active_exits.append(trade["exit_time"])
    return kept


def median_bar_time(indexes) -> pd.Timestamp | None:
    """Median timestamp across bar indexes (Alpaca parquet is datetime64[us])."""
    stamps = []
    for idx in indexes:
        if idx is None or len(idx) == 0:
            continue
        ts = pd.to_datetime(idx, utc=True)
        stamps.append(ts.asi8)
    if not stamps:
        return None
    med = int(np.median(np.concatenate(stamps)))
    unit = "us" if med < 10**16 else "ns"
    return pd.Timestamp(med, unit=unit, tz="UTC").tz_convert(NY)


def choose_level(rows: list[dict]) -> dict | None:
    eligible = [
        row for row in rows
        if row["first_half_pnl"] >= 0 and row["second_half_pnl"] >= 0
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda row: row["total_pnl"])


def split_by_time(trades: list[dict], cutoff) -> tuple[list[dict], list[dict]]:
    cutoff = pd.Timestamp(cutoff)
    first, second = [], []
    for trade in trades:
        ts = pd.Timestamp(trade["entry_time"])
        if ts.tzinfo is None and cutoff.tzinfo is not None:
            ts = ts.tz_localize(cutoff.tzinfo)
        elif ts.tzinfo is not None and cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(ts.tzinfo)
        elif ts.tzinfo is not None and cutoff.tzinfo is not None:
            ts = ts.tz_convert(cutoff.tzinfo)
        (first if ts < cutoff else second).append(trade)
    return first, second


def _median_entry(trades: list[dict]):
    times = sorted(pd.Timestamp(t["entry_time"]) for t in trades)
    return times[len(times) // 2]


def symbol_pnl(trades: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for trade in trades:
        out[trade["symbol"]] = out.get(trade["symbol"], 0.0) + float(trade["pnl"])
    return out


def monthly_pnl(trades: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for trade in trades:
        ts = pd.Timestamp(trade["exit_time"])
        if ts.tzinfo is not None:
            ts = ts.tz_convert(NY)
        key = f"{ts.year}-{ts.month:02d}"
        out[key] = out.get(key, 0.0) + float(trade["pnl"])
    return dict(sorted(out.items()))


def robustness_report(trades: list[dict], notional: float = NOTIONAL) -> dict:
    if not trades:
        empty = summarize([])
        return {"headline": empty}

    cutoff = _median_entry(trades)
    first, second = split_by_time(trades, cutoff)
    by_sym = symbol_pnl(trades)
    green = sum(1 for v in by_sym.values() if v > 0)
    pnls = np.array([float(t["pnl"]) for t in trades], dtype=np.float64)
    best = max(by_sym.items(), key=lambda kv: kv[1]) if by_sym else ("", 0.0)
    worst = min(by_sym.items(), key=lambda kv: kv[1]) if by_sym else ("", 0.0)
    longs = [t for t in trades if t["side"] == LONG]
    shorts = [t for t in trades if t["side"] == SHORT]
    months = monthly_pnl(trades)
    red_months = sum(1 for v in months.values() if v < 0)
    return {
        "headline": summarize(trades),
        "first_half": summarize(first),
        "second_half": summarize(second),
        "cutoff": str(cutoff),
        "cost_2bps": summarize(apply_roundtrip_bps(trades, 2, notional)),
        "cost_5bps": summarize(apply_roundtrip_bps(trades, 5, notional)),
        "cost_10bps": summarize(apply_roundtrip_bps(trades, 10, notional)),
        "cap_8": summarize(cap_concurrent(trades, 8)),
        "longs": summarize(longs),
        "shorts": summarize(shorts),
        "n_symbols": len(by_sym),
        "n_symbols_green": green,
        "pct_symbols_green": green / len(by_sym) if by_sym else 0.0,
        "best_symbol": {"symbol": best[0], "pnl": best[1]},
        "worst_symbol": {"symbol": worst[0], "pnl": worst[1]},
        "median_symbol_pnl": float(np.median(list(by_sym.values()))) if by_sym else 0.0,
        "median_trade_pnl": float(np.median(pnls)),
        "mean_trade_pnl": float(np.mean(pnls)),
        "months": months,
        "n_red_months": red_months,
    }

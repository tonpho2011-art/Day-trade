"""SMA 34/89 dual-slope signals and walk-forward exits."""
import pandas as pd

from daytrade.sma_slope import (
    FLAT,
    LONG,
    SHORT,
    SlopeConfig,
    signal_series,
    simulate_symbol,
)


def _frame(closes, start="2026-06-01 09:30", freq="2min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="America/New_York")
    rows = []
    for c in closes:
        rows.append({"Open": c, "High": c + 0.2, "Low": c - 0.2, "Close": c, "Volume": 100.0})
    return pd.DataFrame(rows, index=idx)


def test_both_rising_is_long_without_waiting_for_cross():
    # Fast and slow SMAs both climb; they never need to cross.
    closes = [10, 10, 10, 11, 12, 13, 14, 15, 16, 17]
    cfg = SlopeConfig(fast=3, slow=5, use_filter=False)
    sig = signal_series(_frame(closes), cfg)
    assert sig.iloc[-1] == LONG
    assert LONG in set(sig)


def test_both_falling_is_short():
    closes = [20, 20, 20, 19, 18, 17, 16, 15, 14, 13]
    cfg = SlopeConfig(fast=3, slow=5, use_filter=False)
    sig = signal_series(_frame(closes), cfg)
    assert sig.iloc[-1] == SHORT


def test_mixed_slopes_are_flat():
    # Up then down so fast can rise while slow is still falling (or vice versa).
    closes = [10, 11, 12, 13, 12, 11, 10, 9]
    cfg = SlopeConfig(fast=3, slow=5, use_filter=False)
    sig = signal_series(_frame(closes), cfg)
    assert FLAT in set(sig)


def test_sma200_filter_blocks_long_when_filter_falling():
    up = list(range(1, 40))
    down = list(range(39, 20, -1))
    cfg = SlopeConfig(fast=3, slow=5, filter_len=10, use_filter=True)
    sig = signal_series(_frame(up + down), cfg)
    # Late in the drop the filter slope is down, so no longs.
    tail = sig.iloc[-8:]
    assert LONG not in set(tail)


def test_long_hits_10_point_stop():
    closes = [100] * 8 + [101, 102, 103, 104]
    df = _frame(closes)
    # After entry, drive price down 10+ points.
    df.iloc[-1] = {"Open": 103.0, "High": 103.2, "Low": 89.0, "Close": 90.0, "Volume": 100.0}
    cfg = SlopeConfig(fast=3, slow=5, stop_points=10, take_points=20, take_profit=True)
    trades = simulate_symbol(df, cfg)
    stops = [t for t in trades if t["exit_reason"] == "stop-loss"]
    assert stops
    assert stops[0]["side"] == LONG
    assert abs(stops[0]["exit_price"] - (stops[0]["entry_price"] - 10)) < 1e-6


def test_long_hits_20_point_target():
    closes = [100] * 8 + [101, 102, 103, 104]
    df = _frame(closes)
    df.iloc[-1] = {"Open": 104.0, "High": 130.0, "Low": 103.8, "Close": 125.0, "Volume": 100.0}
    cfg = SlopeConfig(fast=3, slow=5, stop_points=10, take_points=20, take_profit=True)
    trades = simulate_symbol(df, cfg)
    tps = [t for t in trades if t["exit_reason"] == "take-profit"]
    assert tps
    assert abs(tps[0]["exit_price"] - (tps[0]["entry_price"] + 20)) < 1e-6


def test_reversal_closes_long_and_opens_short():
    up = [100, 101, 102, 103, 104, 105, 106, 107, 108]
    down = [107, 106, 105, 104, 103, 102, 101, 100, 99, 98]
    df = _frame(up + down)
    cfg = SlopeConfig(fast=3, slow=5, stop_points=50, take_points=50, take_profit=True)
    trades = simulate_symbol(df, cfg)
    sides = [t["side"] for t in trades]
    reasons = [t["exit_reason"] for t in trades]
    assert LONG in sides
    assert "reversal" in reasons


def test_flatten_at_session_end():
    idx = pd.date_range("2026-06-01 14:00", periods=20, freq="2min", tz="America/New_York")
    closes = list(range(100, 120))
    rows = [{"Open": c, "High": c + 0.2, "Low": c - 0.2, "Close": c, "Volume": 100.0} for c in closes]
    df = pd.DataFrame(rows, index=idx)
    cfg = SlopeConfig(
        fast=3, slow=5, stop_points=50, take_points=50,
        session_start_min=9 * 60 + 30, session_end_min=14 * 60 + 30, time_filter=True,
    )
    trades = simulate_symbol(df, cfg)
    assert trades
    assert trades[-1]["exit_reason"] == "flatten"
    assert trades[-1]["exit_time"].hour == 14
    assert trades[-1]["exit_time"].minute == 28


def test_direction_long_only_skips_shorts():
    closes = [20, 20, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    cfg = SlopeConfig(fast=3, slow=5, direction="long", stop_points=50, take_points=50)
    trades = simulate_symbol(_frame(closes), cfg)
    assert all(t["side"] == LONG for t in trades)

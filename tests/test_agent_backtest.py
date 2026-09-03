"""Walk-forward agent simulator: next-bar fill, 2% stop, 4% target, EOD flatten."""
import pandas as pd
import pytest

from daytrade.agent_backtest import simulate_symbol, summarize
from daytrade.agents.votes import BUY, SKIP


def _rth_day(rows, day="2026-06-01"):
    idx = pd.date_range(f"{day} 09:30", periods=len(rows), freq="5min", tz="America/New_York")
    return pd.DataFrame(rows, index=idx)


def _bar(close, high=None, low=None, open_=None, volume=200.0):
    o = close if open_ is None else open_
    return {
        "Open": o,
        "High": close + 0.2 if high is None else high,
        "Low": close - 0.2 if low is None else low,
        "Close": close,
        "Volume": volume,
    }


def test_buy_fills_at_next_bar_open_not_signal_close():
    rows = [_bar(100.0) for _ in range(10)]
    rows.append(_bar(100.0, open_=100.0, high=100.2, low=99.8))  # signal bar
    rows.append(_bar(101.0, open_=100.5, high=101.2, low=100.4))  # fill
    rows.append(_bar(105.0, open_=101.0, high=105.0, low=100.8))  # take-profit so the trade closes
    calls = []

    def vote(df):
        calls.append(len(df))
        return BUY if len(df) == 11 else SKIP

    trades = simulate_symbol(_rth_day(rows), vote)
    assert len(trades) == 1
    assert trades[0]["entry_price"] == 100.5
    assert max(calls) == 11


def test_stop_loss_exits_at_2_percent():
    rows = [_bar(100.0) for _ in range(10)]
    rows.append(_bar(100.0))
    rows.append(_bar(100.0, open_=100.0, high=100.1, low=99.9))
    rows.append(_bar(97.5, open_=99.0, high=99.1, low=97.4))

    def vote(df):
        return BUY if len(df) == 11 else SKIP

    trades = simulate_symbol(_rth_day(rows), vote)
    assert trades[0]["exit_reason"] == "stop-loss"
    assert trades[0]["exit_price"] == 98.0
    assert trades[0]["pnl"] < 0


def test_stop_and_target_scale_to_one_and_two_percent():
    rows = [_bar(100.0) for _ in range(10)]
    rows.append(_bar(100.0))
    rows.append(_bar(100.0, open_=100.0, high=100.1, low=99.9))
    rows.append(_bar(102.5, open_=100.5, high=102.6, low=100.4))

    def vote(df):
        return BUY if len(df) == 11 else SKIP

    trades = simulate_symbol(_rth_day(rows), vote, stop_pct=0.01, take_pct=0.02)
    assert trades[0]["exit_reason"] == "take-profit"
    assert trades[0]["exit_price"] == 102.0


def test_take_profit_exits_at_4_percent():
    rows = [_bar(100.0) for _ in range(10)]
    rows.append(_bar(100.0))
    rows.append(_bar(100.0, open_=100.0, high=100.1, low=99.9))
    rows.append(_bar(104.5, open_=101.0, high=104.6, low=100.9))

    def vote(df):
        return BUY if len(df) == 11 else SKIP

    trades = simulate_symbol(_rth_day(rows), vote)
    assert trades[0]["exit_reason"] == "take-profit"
    assert trades[0]["exit_price"] == 104.0
    assert trades[0]["pnl"] > 0


def test_same_bar_stop_and_target_counts_as_stop():
    rows = [_bar(100.0) for _ in range(10)]
    rows.append(_bar(100.0))
    rows.append(_bar(100.0, open_=100.0, high=100.1, low=99.9))
    rows.append(_bar(100.0, open_=100.0, high=105.0, low=97.0))

    def vote(df):
        return BUY if len(df) == 11 else SKIP

    trades = simulate_symbol(_rth_day(rows), vote)
    assert trades[0]["exit_reason"] == "stop-loss"
    assert trades[0]["exit_price"] == 98.0


def test_flatten_near_close_if_stops_not_hit():
    rows = [_bar(100.0) for _ in range(76)]  # 09:30 through 15:45
    rows[10] = _bar(100.0)
    rows[11] = _bar(100.0, open_=100.0, high=100.2, low=99.8)
    for i in range(12, 76):
        rows[i] = _bar(101.0, open_=101.0, high=101.3, low=100.7)
    rows.append(_bar(101.0, open_=101.0, high=101.2, low=100.8))  # 15:50
    rows.append(_bar(101.2, open_=101.0, high=101.4, low=100.9))  # 15:55

    def vote(df):
        return BUY if len(df) == 11 else SKIP

    trades = simulate_symbol(_rth_day(rows), vote)
    assert trades[0]["exit_reason"] == "flatten"
    assert trades[0]["exit_price"] == 101.2


def test_no_new_entry_inside_last_15_minutes():
    rows = [_bar(100.0) for _ in range(78)]
    signal_i = 75  # 09:30 + 75*5m = 15:45

    def vote(df):
        return BUY if len(df) == signal_i + 1 else SKIP

    trades = simulate_symbol(_rth_day(rows), vote)
    assert trades == []


def test_summarize_win_rate_and_pnl():
    trades = [
        {"pnl": 140.0, "pnl_pct": 0.04, "exit_reason": "take-profit", "symbol": "A"},
        {"pnl": -70.0, "pnl_pct": -0.02, "exit_reason": "stop-loss", "symbol": "A"},
        {"pnl": 10.0, "pnl_pct": 0.002, "exit_reason": "flatten", "symbol": "B"},
    ]
    stats = summarize(trades)
    assert stats["n_trades"] == 3
    assert stats["n_wins"] == 2
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["total_pnl"] == 80.0

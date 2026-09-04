"""Fibonacci 5-candle retracement: range, limit fill, START stop, END target."""
import pandas as pd

from daytrade.fib5 import FibConfig, fib_price, simulate_symbol


def _frame(rows, start="2026-06-01 09:30", freq="5min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="America/New_York")
    return pd.DataFrame(rows, index=idx)


def _ohlc(o, h, l, c):
    return {"Open": o, "High": h, "Low": l, "Close": c, "Volume": 100.0}


def _bullish_setup():
    # Prev red, then 5 greens. START=min(99.0, 99.4)=99, END=105.
    return [
        _ohlc(100.5, 100.6, 99.0, 99.4),
        _ohlc(99.6, 101.0, 99.4, 100.8),
        _ohlc(100.8, 102.0, 100.6, 101.7),
        _ohlc(101.7, 103.0, 101.5, 102.8),
        _ohlc(102.8, 104.0, 102.6, 103.8),
        _ohlc(103.8, 105.0, 103.6, 104.8),
        _ohlc(104.5, 104.6, 103.0, 103.2),  # close < prev open 103.8 → trend done
        _ohlc(103.0, 103.2, 101.8, 102.5),  # touches 50% = 102
        _ohlc(102.6, 105.2, 102.5, 104.8),  # hits END 105
    ]


def test_fib_50_is_midpoint():
    assert fib_price(start=99.0, end=105.0, level=0.5) == 102.0
    assert fib_price(start=120.0, end=110.0, level=0.5) == 115.0


def test_fib_25_is_near_end():
    assert fib_price(start=99.0, end=105.0, level=0.25) == 103.5


def test_long_limit_at_50_fills_and_takes_end():
    trades = simulate_symbol(_frame(_bullish_setup()), FibConfig(level=0.5, min_range=0))
    assert len(trades) == 1
    t = trades[0]
    assert t["side"] == "LONG"
    assert t["entry_price"] == 102.0
    assert t["exit_reason"] == "take-profit"
    assert t["exit_price"] == 105.0
    assert t["pnl"] > 0


def test_long_stop_is_at_start():
    rows = _bullish_setup()
    rows[-1] = _ohlc(102.4, 102.5, 98.5, 99.0)  # through START 99
    trades = simulate_symbol(_frame(rows), FibConfig(level=0.5, min_range=0))
    assert trades[0]["exit_reason"] == "stop-loss"
    assert trades[0]["exit_price"] == 99.0
    assert trades[0]["pnl"] < 0


def test_min_range_blocks_tiny_trend():
    trades = simulate_symbol(_frame(_bullish_setup()), FibConfig(level=0.5, min_range=50))
    assert trades == []


def test_short_from_five_reds():
    rows = [
        _ohlc(100.0, 101.0, 99.8, 100.6),  # green prev
        _ohlc(100.5, 100.6, 99.0, 99.2),
        _ohlc(99.2, 99.3, 98.0, 98.2),
        _ohlc(98.2, 98.3, 97.0, 97.2),
        _ohlc(97.2, 97.3, 96.0, 96.2),
        _ohlc(96.2, 96.3, 95.0, 95.2),
        _ohlc(95.4, 96.5, 95.3, 96.4),  # close > prev open 96.2 → done
        _ohlc(96.0, 98.2, 95.8, 97.5),  # 50% = 98, START=101, END=95
        _ohlc(97.4, 97.5, 94.8, 95.1),  # TP at 95
    ]
    trades = simulate_symbol(_frame(rows), FibConfig(level=0.5, min_range=0))
    assert len(trades) == 1
    assert trades[0]["side"] == "SHORT"
    assert trades[0]["entry_price"] == 98.0
    assert trades[0]["exit_reason"] == "take-profit"

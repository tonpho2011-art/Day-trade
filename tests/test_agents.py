"""Agent vote rules on synthetic 5m OHLCV."""
import pandas as pd

from daytrade.agents.bollinger_reversion import vote_bb_reversion
from daytrade.agents.ema_trend import vote_ema_trend
from daytrade.agents.po3_ifvg import vote_po3_ifvg
from daytrade.committee import BUY, SKIP


def _frame(rows, start="2026-09-01 09:30"):
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="America/New_York")
    return pd.DataFrame(rows, index=idx)


def _flat(n, price=100.0, volume=100.0):
    return [
        {"Open": price, "High": price + 0.4, "Low": price - 0.4, "Close": price, "Volume": volume}
        for _ in range(n)
    ]


def test_ema_trend_buys_fresh_9_21_cross_with_volume():
    rows = _flat(30, price=20.0, volume=100.0)
    for i in range(30):
        px = 20.0 - i * 0.25
        rows[i] = {"Open": px + 0.05, "High": px + 0.2, "Low": px - 0.2, "Close": px, "Volume": 100.0}
    # Last three closes jump so 9 EMA crosses 21 EMA; last bar is green with volume spike.
    jumps = [16.0, 18.5, 21.0]
    for i, px in enumerate(jumps):
        rows[-(3 - i)] = {
            "Open": px - 0.8, "High": px + 0.3, "Low": px - 1.0, "Close": px, "Volume": 100.0,
        }
    rows[-1]["Volume"] = 800.0
    assert vote_ema_trend(_frame(rows)) == BUY


def test_ema_trend_skips_without_volume_spike():
    rows = _flat(30, price=20.0, volume=100.0)
    for i in range(30):
        px = 20.0 - i * 0.25
        rows[i] = {"Open": px + 0.05, "High": px + 0.2, "Low": px - 0.2, "Close": px, "Volume": 100.0}
    for i, px in enumerate([16.0, 18.5, 21.0]):
        rows[-(3 - i)] = {
            "Open": px - 0.8, "High": px + 0.3, "Low": px - 1.0, "Close": px, "Volume": 100.0,
        }
    assert vote_ema_trend(_frame(rows)) == SKIP


def _bb_base():
    """Oscillating 49/51 so the 20,2σ lower band sits near ~48, not on price."""
    rows = []
    for i in range(23):
        px = 51.0 if i % 2 == 0 else 49.0
        rows.append({"Open": px, "High": px + 0.2, "Low": px - 0.2, "Close": px, "Volume": 200.0})
    return rows


def test_bb_reversion_buys_engulfing_rejecting_lower_band():
    rows = _bb_base()
    rows[-1] = {"Open": 51.0, "High": 51.2, "Low": 48.8, "Close": 49.0, "Volume": 200.0}
    rows.append({"Open": 48.9, "High": 51.4, "Low": 46.5, "Close": 51.3, "Volume": 200.0})
    assert vote_bb_reversion(_frame(rows)) == BUY


def test_bb_reversion_skips_close_still_outside_band():
    rows = _bb_base()
    rows.append({"Open": 47.2, "High": 47.4, "Low": 46.0, "Close": 46.8, "Volume": 200.0})
    assert vote_bb_reversion(_frame(rows)) == SKIP


def _po3_ifvg_rows():
    rows = [
        {"Open": 101.0, "High": 102.0, "Low": 100.0, "Close": 101.0, "Volume": 200.0}
        for _ in range(6)
    ]
    rows.append({"Open": 101.0, "High": 101.5, "Low": 99.0, "Close": 100.5, "Volume": 300.0})
    rows.extend(
        {"Open": 101.0, "High": 102.0, "Low": 100.5, "Close": 101.0, "Volume": 200.0}
        for _ in range(4)
    )
    rows.append({"Open": 103.0, "High": 103.0, "Low": 102.5, "Close": 102.6, "Volume": 200.0})
    rows.append({"Open": 102.0, "High": 102.0, "Low": 99.0, "Close": 99.5, "Volume": 200.0})
    rows.append({"Open": 99.2, "High": 100.0, "Low": 98.0, "Close": 98.5, "Volume": 200.0})
    rows.append({"Open": 101.0, "High": 104.0, "Low": 101.0, "Close": 103.5, "Volume": 200.0})
    rows.append({"Open": 102.0, "High": 103.0, "Low": 101.0, "Close": 102.2, "Volume": 200.0})
    return rows


def test_po3_ifvg_buys_when_range_low_swept_and_ifvg_retests():
    assert vote_po3_ifvg(_frame(_po3_ifvg_rows())) == BUY


def test_po3_ifvg_skips_without_opening_range_sweep():
    rows = [
        {"Open": 101.0, "High": 102.0, "Low": 100.5, "Close": 101.0, "Volume": 200.0}
        for _ in range(20)
    ]
    assert vote_po3_ifvg(_frame(rows)) == SKIP

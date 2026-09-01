"""Last-closed-bar helper: drop an in-progress 5m candle."""
import pandas as pd

from daytrade.agents.bars import closed_bars


def _df(last_start, tz="America/New_York"):
    idx = pd.DatetimeIndex([last_start])
    return pd.DataFrame(
        {"Open": [1.0], "High": [1.1], "Low": [0.9], "Close": [1.05], "Volume": [100.0]},
        index=idx,
    )


def test_drops_unclosed_five_minute_bar():
    start = pd.Timestamp("2026-09-01 10:35", tz="America/New_York")
    now = pd.Timestamp("2026-09-01 10:37", tz="America/New_York")
    out = closed_bars(_df(start), interval="5m", now=now)
    assert out.empty


def test_keeps_bar_after_it_has_closed():
    start = pd.Timestamp("2026-09-01 10:30", tz="America/New_York")
    now = pd.Timestamp("2026-09-01 10:36", tz="America/New_York")
    out = closed_bars(_df(start), interval="5m", now=now)
    assert len(out) == 1

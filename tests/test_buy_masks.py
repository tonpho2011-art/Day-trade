"""Buy-mask must match per-bar vote_fn on the same RTH prefixes."""
from pathlib import Path

import pandas as pd
import pytest

from daytrade.agent_backtest import compute_buy_masks, rth_bars, simulate_all_agents, simulate_symbol
from daytrade.agents.bollinger_reversion import vote_bb_reversion
from daytrade.agents.ema_trend import vote_ema_trend
from daytrade.agents.po3_ifvg import vote_po3_ifvg
from daytrade.agents.votes import BUY
from tests.test_agents import _bb_base, _flat, _frame, _po3_ifvg_rows


def _mask_vs_vote(df, vote_fn, key):
    bars = rth_bars(df)
    mask = compute_buy_masks(bars)[key]
    for i in range(len(bars)):
        vote = vote_fn(bars.iloc[: i + 1])
        assert bool(mask[i]) == (vote == BUY), (key, i, bool(mask[i]), vote)


def test_po3_mask_matches_vote_on_fixture():
    _mask_vs_vote(_frame(_po3_ifvg_rows()), vote_po3_ifvg, "po3_ifvg")


def test_ema_mask_matches_vote_on_fixture():
    rows = _flat(30, price=20.0, volume=100.0)
    for i in range(30):
        px = 20.0 - i * 0.25
        rows[i] = {"Open": px + 0.05, "High": px + 0.2, "Low": px - 0.2, "Close": px, "Volume": 100.0}
    jumps = [16.0, 18.5, 21.0]
    for i, px in enumerate(jumps):
        rows[-(3 - i)] = {
            "Open": px - 0.8, "High": px + 0.3, "Low": px - 1.0, "Close": px, "Volume": 100.0,
        }
    rows[-1]["Volume"] = 800.0
    _mask_vs_vote(_frame(rows), vote_ema_trend, "ema_trend")


def test_bb_mask_matches_vote_on_fixture():
    rows = _bb_base()
    rows[-1] = {"Open": 51.0, "High": 51.2, "Low": 48.8, "Close": 49.0, "Volume": 200.0}
    rows.append({"Open": 48.9, "High": 51.4, "Low": 46.5, "Close": 51.3, "Volume": 200.0})
    _mask_vs_vote(_frame(rows), vote_bb_reversion, "bb_reversion")


def test_masked_simulator_matches_vote_simulator_on_aapl_sample():
    path = Path("data/bars_5m_iex/AAPL.parquet")
    if not path.exists():
        pytest.skip("no cached AAPL bars")
    raw = pd.read_parquet(path)
    bars = rth_bars(raw)
    days = []
    seen = set()
    for ts in bars.index:
        d = ts.date()
        if d not in seen:
            seen.add(d)
            days.append(d)
        if len(days) == 3:
            break
    sample = bars[pd.Series(bars.index.date, index=bars.index).isin(days[:3])]
    masked = simulate_all_agents(sample, symbol="AAPL")
    votes = {
        "po3_ifvg": simulate_symbol(sample, vote_po3_ifvg, symbol="AAPL"),
        "ema_trend": simulate_symbol(sample, vote_ema_trend, symbol="AAPL"),
        "bb_reversion": simulate_symbol(sample, vote_bb_reversion, symbol="AAPL"),
    }
    for key in masked:
        assert [(t["entry_time"], t["exit_reason"], round(t["pnl"], 6)) for t in masked[key]] == [
            (t["entry_time"], t["exit_reason"], round(t["pnl"], 6)) for t in votes[key]
        ]

"""Fibonacci 5-candle retracement: range, limit fill, START stop, END target."""
import pandas as pd

from daytrade.fib5 import (
    FibConfig,
    apply_roundtrip_bps,
    cap_concurrent,
    choose_level,
    fib_price,
    select_portfolio,
    simulate_symbol,
    split_by_time,
    summarize,
)


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


def test_fib_default_has_no_htf_filter():
    cfg = FibConfig()
    assert cfg.level == 0.618
    assert cfg.htf_minutes == 0


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


def test_gap_through_fills_at_open_by_default():
    rows = _bullish_setup()
    rows[-2] = _ohlc(101.5, 103.2, 101.4, 102.5)  # opens through 50% = 102
    trades = simulate_symbol(_frame(rows), FibConfig(level=0.5, min_range=0))
    assert trades[0]["entry_price"] == 101.5


def test_fill_at_limit_does_not_improve_entry():
    rows = _bullish_setup()
    rows[-2] = _ohlc(101.5, 103.2, 101.4, 102.5)
    trades = simulate_symbol(
        _frame(rows), FibConfig(level=0.5, min_range=0, fill_at_limit=True)
    )
    assert trades[0]["entry_price"] == 102.0


def test_roundtrip_cost_reduces_pnl():
    trades = [{"pnl": 10.0, "exit_time": pd.Timestamp("2026-01-02", tz="UTC")}]
    out = apply_roundtrip_bps(trades, bps=10, notional=3500)
    assert out[0]["pnl"] == 10.0 - 3.5


def test_cap_concurrent_keeps_first_open_trades():
    tz = "America/New_York"
    trades = [
        {
            "entry_time": pd.Timestamp("2026-01-02 10:00", tz=tz),
            "exit_time": pd.Timestamp("2026-01-02 11:00", tz=tz),
            "pnl": 1.0,
            "symbol": "A",
        },
        {
            "entry_time": pd.Timestamp("2026-01-02 10:05", tz=tz),
            "exit_time": pd.Timestamp("2026-01-02 11:05", tz=tz),
            "pnl": 1.0,
            "symbol": "B",
        },
        {
            "entry_time": pd.Timestamp("2026-01-02 10:10", tz=tz),
            "exit_time": pd.Timestamp("2026-01-02 11:10", tz=tz),
            "pnl": 99.0,
            "symbol": "C",
        },
    ]
    kept = cap_concurrent(trades, max_positions=2)
    assert [t["symbol"] for t in kept] == ["A", "B"]


def test_split_by_time_is_half_and_half():
    tz = "America/New_York"
    trades = [
        {"entry_time": pd.Timestamp("2026-01-01 10:00", tz=tz), "pnl": 1.0, "side": "LONG", "exit_reason": "flatten", "exit_time": pd.Timestamp("2026-01-01 11:00", tz=tz)},
        {"entry_time": pd.Timestamp("2026-06-01 10:00", tz=tz), "pnl": 2.0, "side": "LONG", "exit_reason": "flatten", "exit_time": pd.Timestamp("2026-06-01 11:00", tz=tz)},
    ]
    first, second = split_by_time(trades, pd.Timestamp("2026-03-01", tz=tz))
    assert summarize(first)["total_pnl"] == 1.0
    assert summarize(second)["total_pnl"] == 2.0


def _fill(symbol, entry_time, exit_time, entry, stop, target, side="LONG", pnl=1.0):
    tz = "America/New_York"
    return {
        "symbol": symbol,
        "side": side,
        "entry_time": pd.Timestamp(entry_time, tz=tz),
        "exit_time": pd.Timestamp(exit_time, tz=tz),
        "entry_price": entry,
        "stop": stop,
        "target": target,
        "pnl": pnl,
    }


def test_bad_gap_bar_does_not_fill():
    rows = _bullish_setup()
    # Pending is placed on bar 6. Next bar gaps 50% down — must not fill at 50.
    rows[-2] = _ohlc(50.0, 50.1, 49.0, 50.0)
    trades = simulate_symbol(_frame(rows), FibConfig(level=0.5, min_range=0))
    assert all(t["entry_price"] > 90 for t in trades)
    assert all(t["pnl"] < 500 for t in trades)


def test_select_portfolio_keeps_higher_rr():
    a = _fill("AAA", "2026-01-02 10:00", "2026-01-02 11:00", entry=102, stop=99, target=105, pnl=1)
    b = _fill("BBB", "2026-01-02 10:00", "2026-01-02 11:00", entry=100, stop=99, target=105, pnl=1)
    kept = select_portfolio([a, b], max_positions=1)
    assert [t["symbol"] for t in kept] == ["BBB"]


def test_select_portfolio_tie_breaks_larger_range():
    a = _fill("AAA", "2026-01-02 10:00", "2026-01-02 11:00", entry=102, stop=99, target=105)
    b = _fill("BBB", "2026-01-02 10:00", "2026-01-02 11:00", entry=110, stop=100, target=120)
    kept = select_portfolio([a, b], max_positions=1)
    assert [t["symbol"] for t in kept] == ["BBB"]


def test_select_portfolio_skips_when_full_then_takes_after_exit():
    open16 = [
        _fill(f"S{i:02d}", "2026-01-02 10:00", "2026-01-02 11:00", entry=102, stop=99, target=105)
        for i in range(16)
    ]
    extra_same = _fill("ZZZ", "2026-01-02 10:00", "2026-01-02 11:00", entry=102, stop=99, target=105)
    later = _fill("NEW", "2026-01-02 11:05", "2026-01-02 12:00", entry=102, stop=99, target=105)
    kept = select_portfolio(open16 + [extra_same, later], max_positions=16)
    symbols = {t["symbol"] for t in kept}
    assert "ZZZ" not in symbols
    assert "NEW" in symbols
    assert len(kept) == 17


def test_choose_level_requires_both_halves_green():
    rows = [
        {"level": 0.25, "total_pnl": 100, "first_half_pnl": 50, "second_half_pnl": -1},
        {"level": 0.5, "total_pnl": 40, "first_half_pnl": 10, "second_half_pnl": 30},
        {"level": 0.7, "total_pnl": 90, "first_half_pnl": 40, "second_half_pnl": 50},
    ]
    assert choose_level(rows)["level"] == 0.7


def test_choose_level_none_if_no_half_passes():
    rows = [
        {"level": 0.5, "total_pnl": 10, "first_half_pnl": -1, "second_half_pnl": 20},
    ]
    assert choose_level(rows) is None


def test_current_setup_is_pending_after_trend_ends():
    from daytrade.fib5 import current_setup

    rows = _bullish_setup()[:-2]  # drop fill + TP bars; last bar completes the trend
    setup = current_setup(_frame(rows), FibConfig(level=0.5, min_range=0))
    assert setup is not None
    assert setup["side"] == "LONG"
    assert setup["entry"] == 102.0
    assert setup["stop"] == 99.0
    assert setup["target"] == 105.0


def _ticket(symbol, entry, stop, target, side="LONG", order_id=None):
    row = {
        "symbol": symbol,
        "side": side,
        "entry_price": entry,
        "stop": stop,
        "target": target,
    }
    if order_id is not None:
        row["order_id"] = order_id
    return row


def test_plan_live_tickets_room_is_cap_minus_fills_not_pending():
    from daytrade.fib5 import plan_live_tickets

    held = {"AMD", "BXP"}
    pending = [_ticket(f"P{i:02d}", entry=102, stop=99, target=105, order_id=f"o{i}") for i in range(14)]
    setups = [_ticket("NEW", entry=100, stop=99, target=105)]
    plan = plan_live_tickets(held=held, setups=setups, pending=pending, cap=16)
    assert plan["room"] == 14
    assert any(t["symbol"] == "NEW" for t in plan["place"])


def test_plan_live_tickets_replaces_weak_pending_with_better_rr():
    from daytrade.fib5 import plan_live_tickets

    weak = _ticket("WEAK", entry=102, stop=99, target=105, order_id="w1")
    strong = _ticket("STRG", entry=100, stop=99, target=105)
    plan = plan_live_tickets(held=set(), setups=[strong], pending=[weak], cap=1)
    assert [t["symbol"] for t in plan["place"]] == ["STRG"]
    assert [t["symbol"] for t in plan["cancel"]] == ["WEAK"]


def test_plan_live_tickets_keeps_pending_already_in_top_slots():
    from daytrade.fib5 import plan_live_tickets

    pending = _ticket("KEEP", entry=100, stop=99, target=105, order_id="k1")
    other = _ticket("SKIP", entry=102, stop=99, target=105)
    plan = plan_live_tickets(held=set(), setups=[pending, other], pending=[pending], cap=1)
    assert plan["place"] == []
    assert plan["cancel"] == []
    assert [t["symbol"] for t in plan["keep"]] == ["KEEP"]


def test_plan_live_tickets_full_book_places_nothing():
    from daytrade.fib5 import plan_live_tickets

    held = {f"H{i}" for i in range(16)}
    plan = plan_live_tickets(
        held=held,
        setups=[_ticket("NEW", entry=100, stop=99, target=105)],
        pending=[],
        cap=16,
    )
    assert plan["room"] == 0
    assert plan["place"] == []


def test_entry_parent_ignores_bracket_legs():
    from daytrade.broker import is_entry_parent

    class _O:
        def __init__(self, parent_id):
            self.parent_id = parent_id

    assert is_entry_parent(_O(None)) is True
    assert is_entry_parent(_O("parent-1")) is False


def test_htf_color_series_matches_per_bar_lookup():
    from daytrade.fib5 import last_closed_htf_color, htf_color_series

    rows = [_ohlc(100 + i * 0.1, 101, 99, 100.2 + (i % 3) * 0.3) for i in range(24)]
    df = _frame(rows)
    series = htf_color_series(df, 30)
    for i, ts in enumerate(df.index):
        slow = last_closed_htf_color(df.iloc[: i + 1], ts, 30)
        fast = int(series[i])
        if slow is None or slow == 0:
            assert fast == 0
        else:
            assert fast == slow


def test_last_closed_htf_uses_finished_30m_only():
    from daytrade.fib5 import last_closed_htf_color

    rows = [_ohlc(100, 101, 99, 100.8)] * 6  # 09:30-09:55 green 30m
    rows += [_ohlc(100.8, 100.9, 100.0, 100.2)]  # 10:00 still in next 30m
    df = _frame(rows)
    assert last_closed_htf_color(df, df.index[-1], minutes=30) == 1
    early = _frame(rows[:4])
    assert last_closed_htf_color(early, early.index[-1], minutes=30) is None


def test_htf_allows_only_same_color_as_impulse():
    from daytrade.fib5 import htf_allows

    assert htf_allows("LONG", 1) and not htf_allows("LONG", -1)
    assert htf_allows("SHORT", -1) and not htf_allows("SHORT", 1)
    assert not htf_allows("LONG", None)


def test_aligned_long_setup_passes_closed_30m_filter():
    from daytrade.fib5 import current_setup

    rows = _bullish_setup()[:-2]
    setup = current_setup(_frame(rows), FibConfig(level=0.5, min_range=0, htf_minutes=30))
    assert setup is not None
    assert setup["side"] == "LONG"


def test_htf_blocks_short_when_closed_30m_is_green():
    from daytrade.fib5 import current_setup

    greens = [
        _ohlc(100 + i * 0.2, 100.3 + i * 0.2, 99.8 + i * 0.2, 100.2 + i * 0.2)
        for i in range(6)
    ]
    short_leg = [
        _ohlc(100.5, 100.6, 99.0, 99.2),
        _ohlc(99.2, 99.3, 98.0, 98.2),
        _ohlc(98.2, 98.3, 97.0, 97.2),
        _ohlc(97.2, 97.3, 96.0, 96.2),
        _ohlc(96.2, 96.3, 95.0, 95.2),
        _ohlc(95.4, 96.5, 95.3, 96.4),
    ]
    rows = greens + short_leg
    blocked = current_setup(_frame(rows), FibConfig(level=0.5, min_range=0, htf_minutes=30))
    allowed = current_setup(_frame(rows), FibConfig(level=0.5, min_range=0, htf_minutes=0))
    assert blocked is None
    assert allowed is not None
    assert allowed["side"] == "SHORT"

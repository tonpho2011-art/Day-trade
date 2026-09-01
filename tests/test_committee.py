"""Committee combiner: 2-of-3 opens a buy; STRONG BUY alone does not."""
from daytrade.committee import select_buys, vote_count


def test_vote_count_counts_buy_only():
    assert vote_count({"a": "BUY", "b": "SKIP", "c": "BUY"}) == 2
    assert vote_count({"a": "SKIP", "b": "SKIP", "c": "SKIP"}) == 0


def test_two_of_three_is_enough_to_buy():
    packets = [
        {"symbol": "AAA", "votes": {"po3_ifvg": "BUY", "ema_trend": "BUY", "bb_reversion": "SKIP"},
         "score": 0.1, "price": 10.0, "signal": "HOLD"},
    ]
    chosen = select_buys(packets, positions=set(), max_positions=8, remaining_bp=10_000, notional=2000)
    assert [c["symbol"] for c in chosen] == ["AAA"]


def test_one_vote_does_not_buy_even_on_strong_buy():
    packets = [
        {"symbol": "BBB", "votes": {"po3_ifvg": "SKIP", "ema_trend": "BUY", "bb_reversion": "SKIP"},
         "score": 3.0, "price": 10.0, "signal": "STRONG BUY"},
    ]
    chosen = select_buys(packets, positions=set(), max_positions=8, remaining_bp=10_000, notional=2000)
    assert chosen == []


def test_unanimous_ranks_ahead_of_two_of_three():
    packets = [
        {"symbol": "TWO", "votes": {"po3_ifvg": "BUY", "ema_trend": "BUY", "bb_reversion": "SKIP"},
         "score": 9.0, "price": 10.0, "signal": "STRONG BUY"},
        {"symbol": "THREE", "votes": {"po3_ifvg": "BUY", "ema_trend": "BUY", "bb_reversion": "BUY"},
         "score": 0.1, "price": 10.0, "signal": "HOLD"},
    ]
    chosen = select_buys(packets, positions=set(), max_positions=8, remaining_bp=10_000, notional=2000)
    assert [c["symbol"] for c in chosen] == ["THREE", "TWO"]


def test_skips_held_names_and_respects_buying_power():
    packets = [
        {"symbol": "HELD", "votes": {"po3_ifvg": "BUY", "ema_trend": "BUY", "bb_reversion": "BUY"},
         "score": 5.0, "price": 10.0, "signal": "STRONG BUY"},
        {"symbol": "NEW", "votes": {"po3_ifvg": "BUY", "ema_trend": "BUY", "bb_reversion": "SKIP"},
         "score": 1.0, "price": 10.0, "signal": "BUY"},
        {"symbol": "EXTRA", "votes": {"po3_ifvg": "BUY", "ema_trend": "BUY", "bb_reversion": "SKIP"},
         "score": 0.5, "price": 10.0, "signal": "BUY"},
    ]
    chosen = select_buys(
        packets, positions={"HELD"}, max_positions=8, remaining_bp=2500, notional=2000,
    )
    assert [c["symbol"] for c in chosen] == ["NEW"]


def test_evaluate_agents_returns_three_skips_on_empty_frame():
    import pandas as pd
    from daytrade.committee import evaluate_agents

    votes = evaluate_agents(pd.DataFrame())
    assert set(votes) == {"po3_ifvg", "ema_trend", "bb_reversion"}
    assert set(votes.values()) == {"SKIP"}

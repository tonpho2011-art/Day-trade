"""2-of-3 vote combiner for new long entries."""
from daytrade.agents.votes import BUY, SKIP

__all__ = ["BUY", "SKIP", "vote_count", "select_buys", "evaluate_agents"]


def vote_count(votes: dict) -> int:
    return sum(1 for v in votes.values() if v == BUY)


def select_buys(
    packets: list[dict],
    positions: set[str],
    max_positions: int,
    remaining_bp: float,
    notional: float,
) -> list[dict]:
    """Return packets that may open a new long, ranked and size-capped.

    A packet qualifies only if at least two agents voted BUY. Held names are
    skipped. Fill order is unanimous first, then higher build_signal score.
    Each fill consumes `notional` buying power and one position slot.
    """
    open_count = len(positions)
    slots = max(0, max_positions - open_count)
    if slots == 0 or remaining_bp < notional:
        return []

    ranked = [
        p for p in packets
        if p["symbol"] not in positions and vote_count(p.get("votes") or {}) >= 2
    ]
    ranked.sort(key=lambda p: (-vote_count(p["votes"]), -float(p.get("score") or 0)))

    chosen = []
    bp = remaining_bp
    for packet in ranked:
        if len(chosen) >= slots or bp < notional:
            break
        chosen.append(packet)
        bp -= notional
    return chosen


def evaluate_agents(df, interval: str = "5m") -> dict:
    """Run the three specialists on closed bars of `df`."""
    from daytrade.agents.bars import closed_bars
    from daytrade.agents.bollinger_reversion import vote_bb_reversion
    from daytrade.agents.ema_trend import vote_ema_trend
    from daytrade.agents.po3_ifvg import vote_po3_ifvg

    bars = closed_bars(df, interval=interval)
    return {
        "po3_ifvg": vote_po3_ifvg(bars),
        "ema_trend": vote_ema_trend(bars),
        "bb_reversion": vote_bb_reversion(bars),
    }


def format_votes(votes: dict) -> str:
    return (
        f"po3={votes.get('po3_ifvg', SKIP)} "
        f"ema={votes.get('ema_trend', SKIP)} "
        f"bb={votes.get('bb_reversion', SKIP)}"
    )

"""Backtest each day-trade agent on historical 5-minute bars.

Uses the same mechanical risk as autotrade.py: $3500 notional (500 x 7),
2% stop, 4% target, no new entries inside 15 minutes of the cash close,
flatten at 5 minutes before the close. Each agent is simulated alone.

Sources:
  yahoo  -- 60 days max (Yahoo's 5m limit)
  alpaca -- IEX 5m, ~14 months on a paper key (SIP is not on the free feed)

Usage:
  python backtest_agents.py
  python backtest_agents.py --source alpaca --universe-size 500 --days 400
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from pathlib import Path

sys.path.insert(0, "src")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from daytrade.agent_backtest import (
    NOTIONAL,
    STOP_PCT,
    TAKE_PCT,
    equity_curve,
    simulate_all_agents,
    simulate_parquet_job,
    simulate_symbol,
    summarize,
)
from daytrade.agents.bollinger_reversion import vote_bb_reversion
from daytrade.agents.ema_trend import vote_ema_trend
from daytrade.agents.po3_ifvg import vote_po3_ifvg
from daytrade.agents.votes import BUY, SKIP
from daytrade.committee import evaluate_agents, vote_count
from daytrade.scan import _batch_download, _slice_symbol
from daytrade.universe import FALLBACK_SP500, get_sp500_symbols


AGENTS = {
    "po3_ifvg": vote_po3_ifvg,
    "ema_trend": vote_ema_trend,
    "bb_reversion": vote_bb_reversion,
}


def vote_two_of_three(df):
    votes = evaluate_agents(df, interval="5m")
    return BUY if vote_count(votes) >= 2 else SKIP


def _daily_equity(trades: list[dict]) -> list[dict]:
    curve = equity_curve(trades)
    if curve.empty:
        return []
    idx = curve.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York")
    daily = curve.groupby(idx.date).last()
    return [{"date": str(d), "equity": round(float(v), 2)} for d, v in daily.items()]


def _print_stats(name: str, stats: dict) -> None:
    pf = stats["profit_factor"]
    pf_s = f"{pf:.2f}" if pf is not None else "n/a"
    print(
        f"  {name}: trades={stats['n_trades']}  wins={stats['n_wins']}  "
        f"win_rate={stats['win_rate']:.1%}  pnl=${stats['total_pnl']:.2f}  "
        f"pf={pf_s}  dd=${stats['max_drawdown']:.2f}  exits={stats['exits']}"
    )


def _yahoo_download(symbols: list[str], period: str, interval: str) -> dict:
    frames = {}
    chunk_size = 20
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        print(f"  downloading {chunk[0]}..{chunk[-1]} ({len(chunk)} names)...")
        data = _batch_download(chunk, period, interval)
        if data is None:
            print("    download failed, skipping chunk")
            continue
        for symbol in chunk:
            df = _slice_symbol(data, symbol)
            if df.empty or "Close" not in df or "Volume" not in df:
                print(f"    {symbol}: no bars")
                continue
            frames[symbol] = df
            print(f"    {symbol}: {len(df)} bars")
    return frames


def _universe(size: int) -> list[str]:
    symbols = list(FALLBACK_SP500)
    if size > len(symbols):
        extra = [s for s in get_sp500_symbols() if s not in symbols]
        symbols = symbols + extra
    return symbols[:size]


def _run_yahoo(frames: dict, include_committee: bool, stop_pct: float, take_pct: float) -> dict[str, list]:
    trades = {name: [] for name in AGENTS}
    if include_committee:
        trades["two_of_three"] = []
    n = len(frames)
    for i, (symbol, df) in enumerate(frames.items(), start=1):
        print(f"  [{i}/{n}] {symbol} ({len(df)} bars)")
        got = simulate_all_agents(df, symbol, stop_pct=stop_pct, take_pct=take_pct)
        for name, rows in got.items():
            trades[name].extend(rows)
        if include_committee:
            trades["two_of_three"].extend(
                simulate_symbol(df, vote_two_of_three, symbol=symbol,
                                stop_pct=stop_pct, take_pct=take_pct)
            )
    return trades


def _run_parquet(jobs: list[tuple[str, str]], workers: int, stop_pct: float, take_pct: float) -> dict[str, list]:
    trades = {name: [] for name in AGENTS}
    n = len(jobs)
    workers = max(1, workers)
    print(f"  simulating {n} symbols with {workers} processes...")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(simulate_parquet_job, (str(path), symbol, stop_pct, take_pct)): symbol
                for symbol, path in jobs}
        for fut in as_completed(futs):
            symbol = futs[fut]
            done += 1
            try:
                _sym, got = fut.result()
            except Exception as e:
                print(f"    {symbol} failed: {e}")
                continue
            for name, rows in got.items():
                trades[name].extend(rows)
            if done == n or done % 25 == 0:
                print(f"    {done}/{n} symbols done")
    return trades


def _report(trades_by_agent: dict, meta: dict, out_path: Path) -> None:
    report = dict(meta)
    report["agents"] = {}
    ranked = []
    for name, trades in trades_by_agent.items():
        stats = summarize(trades)
        stats["agent"] = name
        stats["daily_equity"] = _daily_equity(trades)
        stats["symbols_traded"] = sorted({t["symbol"] for t in trades})
        stats["n_symbols_traded"] = len(stats["symbols_traded"])
        report["agents"][name] = {k: v for k, v in stats.items() if k != "agent"}
        _print_stats(name, stats)
        ranked.append(stats)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
    if ranked:
        by_pnl = max(ranked, key=lambda s: s["total_pnl"])
        traded = [s for s in ranked if s["n_trades"] >= 10]
        print(f"Highest total P/L: {by_pnl['agent']} (${by_pnl['total_pnl']:.2f})")
        if traded:
            by_wr = max(traded, key=lambda s: s["win_rate"])
            print(
                f"Highest win rate (>=10 trades): {by_wr['agent']} "
                f"({by_wr['win_rate']:.1%} on {by_wr['n_trades']})"
            )


def main():
    parser = argparse.ArgumentParser(description="Per-agent 5m day-trade backtest")
    parser.add_argument("--source", choices=("yahoo", "alpaca"), default="yahoo")
    parser.add_argument("--period", default="60d", help="Yahoo lookback (5m max is 60d)")
    parser.add_argument("--days", type=int, default=400, help="Alpaca lookback in calendar days")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--universe-size", type=int, default=30)
    parser.add_argument("--include-committee", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--stop-loss-pct", type=float, default=2.0)
    parser.add_argument("--take-profit-pct", type=float, default=4.0)
    parser.add_argument("--out", default="data/agent_backtest_results.json")
    args = parser.parse_args()

    stop_pct = args.stop_loss_pct / 100.0
    take_pct = args.take_profit_pct / 100.0
    symbols = _universe(args.universe_size)
    print(f"Universe {len(symbols)} names | source={args.source} | "
          f"stop={stop_pct:.0%} target={take_pct:.0%} notional=${NOTIONAL:.0f}")

    if args.source == "alpaca":
        from daytrade.alpaca_bars import download_5m

        jobs = download_5m(symbols, days=args.days)
        if not jobs:
            print("No Alpaca bars downloaded.")
            sys.exit(1)
        print(f"Cached bars for {len(jobs)} symbols")
        trades = _run_parquet(jobs, args.workers, stop_pct, take_pct)
        first = __import__("pandas").read_parquet(jobs[0][1])
        meta = {
            "source": "alpaca-iex",
            "period": f"{args.days}d",
            "interval": "5m",
            "stop_pct": stop_pct,
            "take_pct": take_pct,
            "notional": NOTIONAL,
            "symbols": [s for s, _ in jobs],
            "n_symbols": len(jobs),
            "window_start": str(first.index.min()),
            "window_end": str(first.index.max()),
        }
    else:
        frames = _yahoo_download(symbols, args.period, args.interval)
        if not frames:
            print("No data downloaded.")
            sys.exit(1)
        first = next(iter(frames.values()))
        print(f"Window: {first.index.min()} -> {first.index.max()}")
        trades = _run_yahoo(frames, args.include_committee, stop_pct, take_pct)
        meta = {
            "source": "yahoo",
            "period": args.period,
            "interval": args.interval,
            "stop_pct": stop_pct,
            "take_pct": take_pct,
            "notional": NOTIONAL,
            "symbols": sorted(frames),
            "n_symbols": len(frames),
            "window_start": str(first.index.min()),
            "window_end": str(first.index.max()),
        }

    _report(trades, meta, Path(args.out))


if __name__ == "__main__":
    freeze_support()
    main()

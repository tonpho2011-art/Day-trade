"""Fibonacci 5-candle retracement (not the 3-agent committee).

5 same-color candles → pending limit at Fib 61.8% of START→END. SL=START,
TP=END. Fill on a later bar's touch. RTH only, flatten ~15:55 ET.

  python fib5.py --backtest
  python fib5.py --optimize --universe-size 500
  python fib5.py --live-paper
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from daytrade import broker
from daytrade.agents.bars import closed_bars
from daytrade.fib5 import (
    FLATTEN,
    NO_ENTRY,
    SWEEP_LEVELS,
    FibConfig,
    choose_level,
    current_setup,
    median_bar_time,
    plan_live_tickets,
    robustness_report,
    select_portfolio,
    simulate_symbol,
    split_by_time,
    summarize,
)
from daytrade.tradelog import log_trade
from daytrade.universe import FALLBACK_SP500, get_daytrade_universe, get_sp500_symbols


def _universe(size: int) -> list[str]:
    symbols = list(FALLBACK_SP500)
    if size > len(symbols):
        extra = [s for s in get_sp500_symbols() if s not in symbols]
        symbols = symbols + extra
    return symbols[:size]


def _print_stats(label: str, stats: dict) -> None:
    pf = f"{stats['profit_factor']:.2f}" if stats["profit_factor"] is not None else "n/a"
    print(
        f"\n{label}\n"
        f"  trades={stats['n_trades']} longs={stats['n_longs']} shorts={stats['n_shorts']}  "
        f"win_rate={stats['win_rate']:.1%}  pnl=${stats['total_pnl']:.2f}  "
        f"avg=${stats['avg_pnl']:.2f}  pf={pf}  dd=${stats['max_drawdown']:.2f}"
    )
    print(f"  exits={stats['exits']}")


def _print_robust(report: dict) -> None:
    h = report["headline"]
    print("\n--- harder proof ---")
    print(
        f"  this is NOT one account: {h['n_trades']} independent single-stock trades "
        f"at ${h['avg_pnl']:+.2f} average"
    )
    print(
        f"  symbols profitable: {report['n_symbols_green']}/{report['n_symbols']} "
        f"({report['pct_symbols_green']:.1%})  median symbol P/L=${report['median_symbol_pnl']:.2f}"
    )
    print(
        f"  best {report['best_symbol']['symbol']}=${report['best_symbol']['pnl']:.0f}  "
        f"worst {report['worst_symbol']['symbol']}=${report['worst_symbol']['pnl']:.0f}"
    )
    print(
        f"  median trade=${report['median_trade_pnl']:.2f} vs mean=${report['mean_trade_pnl']:.2f} "
        "(if mean >> median, a few outliers are doing the work)"
    )
    _print_stats(f"first half of sample (before {report['cutoff']})", report["first_half"])
    _print_stats("second half of sample", report["second_half"])
    _print_stats("longs only", report["longs"])
    _print_stats("shorts only", report["shorts"])
    _print_stats("after 2 bps round-trip cost (~$0.70 / $3500)", report["cost_2bps"])
    _print_stats("after 5 bps round-trip cost (~$1.75 / $3500)", report["cost_5bps"])
    _print_stats("after 10 bps round-trip cost (~$3.50 / $3500)", report["cost_10bps"])
    _print_stats("max 8 positions at once (first-come, like a real account)", report["cap_8"])
    months = report["months"]
    red = report["n_red_months"]
    print(f"\n  monthly P/L ({red} losing months of {len(months)}):")
    for month, pnl in months.items():
        print(f"    {month}  ${pnl:+.0f}")


def _load_frames(symbols: list[str], days: int):
    from daytrade.alpaca_bars import download_intraday
    import pandas as pd

    jobs = download_intraday(symbols, minutes=5, days=days)
    frames = {symbol: pd.read_parquet(path) for symbol, path in jobs}
    return frames


def _bar_cutoff(frames: dict):
    return median_bar_time(df.index for df in frames.values())


def run_optimize(args) -> None:

    symbols = _universe(args.universe_size)
    cap = args.max_positions or 16
    print(
        f"Fib optimize | 5m | {len(symbols)} names | cap={cap} | "
        f"levels={list(SWEEP_LEVELS)} | no extra cost"
    )
    frames = _load_frames(symbols, args.days)
    if not frames:
        print("No bars in cache. Need data/bars_5m_iex/.")
        sys.exit(1)
    cutoff = _bar_cutoff(frames)
    print(f"  half-sample cutoff: {cutoff}", flush=True)

    rows = []
    for level in SWEEP_LEVELS:
        cfg = FibConfig(
            level=level,
            min_range=args.min_range,
            use_ema=args.ema_filter,
            notional=args.notional,
            htf_minutes=0 if args.no_htf else args.htf_minutes,
        )
        trades = []
        for i, (symbol, df) in enumerate(frames.items(), start=1):
            if i == 1 or i % 100 == 0:
                print(f"  level={level:.3f} [{i}/{len(frames)}] {symbol}", flush=True)
            trades.extend(simulate_symbol(df, cfg, symbol=symbol))
        kept = select_portfolio(trades, max_positions=cap)
        stats = summarize(kept)
        first, second = split_by_time(kept, cutoff)
        row = {
            "level": level,
            "n_trades": stats["n_trades"],
            "n_longs": stats["n_longs"],
            "n_shorts": stats["n_shorts"],
            "win_rate": stats["win_rate"],
            "total_pnl": stats["total_pnl"],
            "first_half_pnl": summarize(first)["total_pnl"],
            "second_half_pnl": summarize(second)["total_pnl"],
            "avg_pnl": stats["avg_pnl"],
            "profit_factor": stats["profit_factor"],
            "max_drawdown": stats["max_drawdown"],
            "exits": stats["exits"],
        }
        rows.append(row)
        print(
            f"  {level:.3f}  trades={row['n_trades']}  wr={row['win_rate']:.1%}  "
            f"pnl=${row['total_pnl']:.2f}  first=${row['first_half_pnl']:.2f}  "
            f"second=${row['second_half_pnl']:.2f}",
            flush=True,
        )

    winner = choose_level(rows)
    print("\nlevel  trades  win   P/L        first half   second half")
    for row in rows:
        mark = "  <-- winner" if winner and row["level"] == winner["level"] else ""
        print(
            f"{row['level']:.3f}  {row['n_trades']:6d}  {row['win_rate']:5.1%}  "
            f"${row['total_pnl']:10.2f}  ${row['first_half_pnl']:10.2f}  "
            f"${row['second_half_pnl']:10.2f}{mark}"
        )
    if winner:
        print(
            f"\nWinner: Fib {winner['level']:.1%}  P/L=${winner['total_pnl']:.2f} "
            f"(both halves >= 0, max 16 R:R overlay, no extra cost)"
        )
    else:
        print("\nNo tradeable winner: no level was green in both halves.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"cutoff": str(cutoff), "max_positions": cap, "rows": rows, "winner": winner},
            indent=2,
            default=str,
        )
    )
    print(f"Wrote {out}")


def run_backtest(args) -> None:
    cfg = FibConfig(
        level=args.level,
        min_range=args.min_range,
        use_ema=args.ema_filter,
        notional=args.notional,
        fill_at_limit=args.fill_at_limit,
        htf_minutes=0 if args.no_htf else args.htf_minutes,
    )
    symbols = _universe(args.universe_size)
    print(
        f"Fib 5-candle | 5m | {len(symbols)} names | level={cfg.level:.0%} "
        f"min_range={cfg.min_range} ema={cfg.use_ema} notional=${cfg.notional:.0f} "
        f"fill_at_limit={cfg.fill_at_limit}"
    )
    frames = _load_frames(symbols, args.days)
    if not frames:
        print("No bars in cache. Need data/bars_5m_iex/.")
        sys.exit(1)

    trades = []
    for i, (symbol, df) in enumerate(frames.items(), start=1):
        if i % 25 == 0 or i == 1:
            print(f"  [{i}/{len(frames)}] {symbol}", flush=True)
        trades.extend(simulate_symbol(df, cfg, symbol=symbol))
    if args.max_positions:
        trades = select_portfolio(trades, max_positions=args.max_positions)
        _print_stats(
            f"after R:R overlay cap={args.max_positions}",
            summarize(trades),
        )
    else:
        stats = summarize(trades)
        _print_stats("headline (independent symbols, no costs, no position cap)", stats)

    stats = summarize(trades)
    payload = {"config": cfg.__dict__, "stats": stats}
    if args.robust:
        report = robustness_report(trades, notional=cfg.notional)
        _print_robust(report)
        print("\nRe-running with fill-at-limit (no better-than-limit gap fills)...")
        conservative = []
        tight = FibConfig(**{**cfg.__dict__, "fill_at_limit": True})
        for i, (symbol, df) in enumerate(frames.items(), start=1):
            if i % 50 == 0 or i == 1:
                print(f"  fill-at-limit [{i}/{len(frames)}] {symbol}", flush=True)
            conservative.extend(simulate_symbol(df, tight, symbol=symbol))
        cons_stats = summarize(conservative)
        _print_stats("fill at limit only", cons_stats)
        payload["robust"] = report
        payload["fill_at_limit"] = cons_stats

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out}")


def _cfg(args) -> FibConfig:
    return FibConfig(
        level=args.level,
        min_range=args.min_range,
        use_ema=args.ema_filter,
        notional=args.notional,
        fill_at_limit=args.fill_at_limit,
        htf_minutes=0 if args.no_htf else args.htf_minutes,
    )


def _log(symbol, signal, price, action, live, extra=None):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "signal": signal,
        "price": price,
        "action": action,
        "mode": "live-paper" if live else "dry-run",
        "strategy": "fib5",
        "level": extra.get("level") if extra else None,
    }
    if extra:
        row.update(extra)
    log_trade(row)


def run_cycle(client, cfg: FibConfig, args) -> None:
    mtc = broker.minutes_to_close(client)
    if mtc is None and not args.ignore_market_hours:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed -- idling.")
        return

    positions = broker.get_open_positions(client)
    cap = args.max_positions or 16

    if mtc is not None and mtc <= FLATTEN:
        if args.live_paper:
            try:
                if positions:
                    print(f"Flatten window -- closing {len(positions)} position(s).")
                    broker.close_all_positions(client)
                broker.cancel_open_orders(client)
            except Exception as e:
                print(f"  flatten failed: {e}")
                return
        elif positions:
            print(f"Flatten window -- closing {len(positions)} position(s).")
        for symbol, pos in positions.items():
            _log(symbol, "flatten", pos["current_price"],
                 "FLATTEN" if args.live_paper else "FLATTEN_DRY_RUN", args.live_paper)
        return

    if mtc is not None and mtc <= NO_ENTRY:
        print(f"No-entry window ({mtc:.0f}m to close) -- not placing new Fib limits.")
        return

    acct = broker.get_account_summary(client)
    pdt_blocked = acct["equity"] < 25000 and acct["pattern_day_trader"] and acct["daytrade_count"] >= 3
    universe = get_daytrade_universe(max_count=args.universe_size)
    scan = list(dict.fromkeys(universe + list(positions.keys())))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fib {cfg.level:.1%} IEX 5m scanning {len(scan)} names...")

    from daytrade.alpaca_bars import fetch_recent_bars

    frames = fetch_recent_bars(scan, minutes=5, days=5)
    if not frames:
        print("  Alpaca IEX download failed")
        return

    pending = []
    if args.live_paper:
        try:
            pending = broker.get_open_entry_orders(client)
        except Exception as e:
            print(f"  open orders lookup failed: {e}")

    held = set(positions)
    setups = []
    for symbol in scan:
        if symbol in held:
            continue
        df = frames.get(symbol)
        if df is None or df.empty:
            continue
        bars = closed_bars(df, interval="5m")
        if bars is None or len(bars) < 7:
            continue
        setup = current_setup(bars, cfg)
        if not setup:
            continue
        setups.append({
            "symbol": symbol,
            "side": setup["side"],
            "entry_price": float(setup["entry"]),
            "stop": float(setup["stop"]),
            "target": float(setup["target"]),
        })

    plan = plan_live_tickets(held=held, setups=setups, pending=pending, cap=cap)
    if pdt_blocked:
        print("  PDT blocked -- no new entries.")
        plan = plan_live_tickets(held=held, setups=[], pending=pending, cap=cap)
    room = plan["room"]

    if args.live_paper:
        for ticket in plan["cancel"]:
            try:
                broker.cancel_order(client, ticket["order_id"])
                print(f"  cancel weaker pending {ticket['symbol']}")
            except Exception as e:
                print(f"  cancel {ticket['symbol']} failed: {e}")

    chosen = plan["place"]
    if not chosen:
        print(
            f"  setups={len(setups)} held={len(held)} pending={len(pending)} "
            f"keep={len(plan['keep'])} room={room}"
        )
        return

    for setup in chosen:
        symbol = setup["symbol"]
        entry = setup["entry_price"]
        qty = max(1, int(cfg.notional / entry))
        print(
            f"  [{symbol}] {setup['side']} limit {entry:.2f} SL {setup['stop']:.2f} "
            f"TP {setup['target']:.2f} qty={qty}"
        )
        extra = {
            "level": cfg.level,
            "stop": setup["stop"],
            "target": setup["target"],
            "qty": qty,
        }
        if args.live_paper:
            ok, err = broker.safe_bracket_limit(
                client, symbol, setup["side"], qty, entry, setup["stop"], setup["target"],
            )
            if not ok:
                print(f"    order failed: {err}")
                _log(symbol, setup["side"], entry, "ORDER_FAILED", True, extra)
                continue
            _log(symbol, setup["side"], entry, "LIMIT", True, extra)
        else:
            _log(symbol, setup["side"], entry, "LIMIT_DRY_RUN", False, extra)


def run_trade(args) -> None:
    cfg = _cfg(args)
    client = broker.get_client()
    acct = broker.get_account_summary(client)
    cap = args.max_positions or 16
    mode = "LIVE PAPER" if args.live_paper else "DRY RUN"
    print(
        f"Fib 5-candle {cfg.level:.1%} | {mode} | notional=${cfg.notional:.0f} | "
        f"cap={cap} fills | 30m={'off' if not cfg.htf_minutes else str(cfg.htf_minutes)+'m'} | "
        f"cash=${acct['cash']:.2f} equity=${acct['equity']:.2f}"
    )
    print("Paper only. Ctrl+C to stop.")
    try:
        while True:
            try:
                run_cycle(client, cfg, args)
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle failed: {e}")
                if args.once:
                    raise
            if args.once:
                break
            print(f"Sleeping {args.loop_minutes} minute(s)...\n")
            time.sleep(args.loop_minutes * 60)
    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    p = argparse.ArgumentParser(description="Fibonacci 5-candle retracement")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--backtest", action="store_true")
    mode.add_argument("--optimize", action="store_true")
    p.add_argument("--live-paper", action="store_true",
                   help="Submit Fib limit+bracket orders to Alpaca PAPER")
    p.add_argument("--once", action="store_true")
    p.add_argument("--ignore-market-hours", action="store_true")
    p.add_argument("--loop-minutes", type=int, default=5)
    p.add_argument("--universe-size", type=int, default=30)
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--level", type=float, default=0.618)
    p.add_argument("--min-range", type=float, default=0.0)
    p.add_argument("--ema-filter", action="store_true")
    p.add_argument("--fill-at-limit", action="store_true")
    p.add_argument("--notional", type=float, default=3500.0)
    p.add_argument("--max-positions", type=int, default=None)
    p.add_argument("--htf-minutes", type=int, default=0)
    p.add_argument("--no-htf", action="store_true",
                   help="Disable the closed higher-timeframe color filter")
    p.add_argument("--out", default="data/fib5_backtest.json")
    p.add_argument(
        "--robust",
        action="store_true",
        help="Time-split, costs, 8-position cap, fill-at-limit re-run",
    )
    args = p.parse_args()
    if args.optimize:
        if args.out == "data/fib5_backtest.json":
            args.out = "data/fib5_optimize.json"
        run_optimize(args)
    elif args.backtest:
        run_backtest(args)
    else:
        run_trade(args)


if __name__ == "__main__":
    main()

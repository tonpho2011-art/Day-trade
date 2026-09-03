"""Separate SMA 34/89 dual-slope bot (not the 3-agent committee).

Long when SMA(34) and SMA(89) are both rising; short when both are falling.
Default: 2-minute bars, $10 stop / $20 target (points), 3 shares, both
directions, SMA(200) filter off, session 09:30–14:30 ET, flatten at 14:30.

  python sma_slope.py --backtest
  python sma_slope.py --backtest --source alpaca --universe-size 30
  python sma_slope.py                    # dry-run paper loop
  python sma_slope.py --live-paper       # Alpaca paper orders
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
from daytrade.scan import _batch_download, _slice_symbol
from daytrade.sma_slope import SlopeConfig, signal_series, simulate_symbol, summarize
from daytrade.tradelog import log_trade
from daytrade.universe import FALLBACK_SP500, get_daytrade_universe, get_sp500_symbols


def _cfg_from_args(args) -> SlopeConfig:
    start_h, start_m = (int(x) for x in args.session_start.split(":"))
    end_h, end_m = (int(x) for x in args.session_end.split(":"))
    return SlopeConfig(
        fast=args.fast,
        slow=args.slow,
        filter_len=args.filter_len,
        use_filter=args.sma_filter,
        direction=args.direction,
        stop_points=args.stop_points,
        take_points=args.take_points,
        take_profit=not args.no_take_profit,
        time_filter=not args.no_time_filter,
        session_start_min=start_h * 60 + start_m,
        session_end_min=end_h * 60 + end_m,
        interval_minutes=args.interval_minutes,
        qty=args.qty,
    )


def _universe(size: int) -> list[str]:
    symbols = list(FALLBACK_SP500)
    if size > len(symbols):
        extra = [s for s in get_sp500_symbols() if s not in symbols]
        symbols = symbols + extra
    return symbols[:size]


def _yahoo_frames(symbols: list[str], period: str, interval: str) -> dict:
    frames = {}
    for i in range(0, len(symbols), 15):
        chunk = symbols[i : i + 15]
        print(f"  downloading {chunk[0]}..{chunk[-1]}")
        data = _batch_download(chunk, period, interval)
        if data is None:
            continue
        for symbol in chunk:
            df = _slice_symbol(data, symbol)
            if not df.empty:
                frames[symbol] = df
                print(f"    {symbol}: {len(df)} bars")
    return frames


def run_backtest(args) -> None:
    cfg = _cfg_from_args(args)
    symbols = _universe(args.universe_size)
    print(
        f"SMA {cfg.fast}/{cfg.slow} dual-slope | {args.interval} | "
        f"{len(symbols)} names | stop={cfg.stop_points}pt target={cfg.take_points}pt "
        f"qty={cfg.qty} dir={cfg.direction} filter={cfg.use_filter}"
    )
    frames = {}
    if args.source == "alpaca":
        from daytrade.alpaca_bars import download_intraday

        jobs = download_intraday(symbols, minutes=args.interval_minutes, days=args.days)
        for symbol, path in jobs:
            frames[symbol] = __import__("pandas").read_parquet(path)
    else:
        frames = _yahoo_frames(symbols, args.period, args.interval)

    if not frames:
        print("No bars downloaded.")
        sys.exit(1)

    trades = []
    for i, (symbol, df) in enumerate(frames.items(), start=1):
        print(f"  [{i}/{len(frames)}] {symbol}")
        trades.extend(simulate_symbol(df, cfg, symbol=symbol))
    stats = summarize(trades)
    pf = f"{stats['profit_factor']:.2f}" if stats["profit_factor"] is not None else "n/a"
    print(
        f"\ntrades={stats['n_trades']} longs={stats['n_longs']} shorts={stats['n_shorts']}  "
        f"win_rate={stats['win_rate']:.1%}  pnl=${stats['total_pnl']:.2f}  "
        f"avg=${stats['avg_pnl']:.2f}  pf={pf}  dd=${stats['max_drawdown']:.2f}"
    )
    print(f"exits={stats['exits']}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg.__dict__, "stats": stats}, indent=2, default=str))
    print(f"Wrote {out}")


def _log(symbol, signal, price, action, live, extra=None):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "signal": signal,
        "price": price,
        "action": action,
        "mode": "live-paper" if live else "dry-run",
        "strategy": "sma_dual_slope",
    }
    if extra:
        row.update(extra)
    log_trade(row)


def run_cycle(client, cfg: SlopeConfig, args) -> None:
    mtc = broker.minutes_to_close(client)
    if mtc is None and not args.ignore_market_hours:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed -- idling.")
        return

    positions = broker.get_open_positions(client)
    import pandas as pd

    ny_now = pd.Timestamp.now(tz="America/New_York")
    mins = int(ny_now.hour) * 60 + int(ny_now.minute)
    if cfg.time_filter and mins >= cfg.session_end_min:
        if positions:
            print(f"Session ended -- flattening {len(positions)} position(s).")
            if args.live_paper:
                try:
                    broker.close_all_positions(client)
                except Exception as e:
                    print(f"  flatten failed: {e}")
                    return
            for symbol, pos in positions.items():
                _log(symbol, "session end", pos["current_price"],
                     "FLATTEN" if args.live_paper else "FLATTEN_DRY_RUN", args.live_paper)
        return

    acct = broker.get_account_summary(client)
    pdt_blocked = acct["equity"] < 25000 and acct["pattern_day_trader"] and acct["daytrade_count"] >= 3
    universe = get_daytrade_universe(max_count=args.universe_size)
    scan = list(dict.fromkeys(universe + list(positions.keys())))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(scan)} names on {args.interval}...")

    data = _batch_download(scan, period="5d", interval=args.interval)
    if data is None:
        print("  download failed")
        return

    held = set(positions)
    actionable = 0
    for symbol in scan:
        df = _slice_symbol(data, symbol)
        if df.empty:
            continue
        bars = closed_bars(df, interval=args.interval)
        if bars is None or len(bars) < cfg.slow + 2:
            continue
        sig = signal_series(bars, cfg).iloc[-1]
        price = float(bars["Close"].iloc[-1])
        pos = positions.get(symbol)
        qty = float(pos["qty"]) if pos else 0.0
        side = "LONG" if qty > 0 else "SHORT" if qty < 0 else None

        if side and (
            (side == "LONG" and sig == "SHORT")
            or (side == "SHORT" and sig == "LONG")
        ):
            actionable += 1
            print(f"  [{symbol}] {side} reverse -> {sig} @ {price:.2f}")
            if args.live_paper:
                ok, err = broker.safe_close_position(client, symbol)
                if not ok:
                    print(f"    close failed: {err}")
                    continue
                held.discard(symbol)
                opener = broker.safe_buy_qty if sig == "LONG" else broker.safe_sell_qty
                ok, err = opener(client, symbol, cfg.qty)
                if ok:
                    held.add(symbol)
                else:
                    print(f"    reverse open failed: {err}")
            else:
                held.discard(symbol)
                held.add(symbol)
            _log(symbol, sig, price, "REVERSE" if args.live_paper else "REVERSE_DRY_RUN",
                 args.live_paper)
            continue

        if side:
            entry = float(pos["avg_entry_price"])
            if side == "LONG":
                if price <= entry - cfg.stop_points:
                    reason = "stop-loss"
                elif cfg.take_profit and price >= entry + cfg.take_points:
                    reason = "take-profit"
                else:
                    continue
            else:
                if price >= entry + cfg.stop_points:
                    reason = "stop-loss"
                elif cfg.take_profit and price <= entry - cfg.take_points:
                    reason = "take-profit"
                else:
                    continue
            actionable += 1
            print(f"  [{symbol}] {side} {reason} @ {price:.2f}")
            if args.live_paper:
                ok, err = broker.safe_close_position(client, symbol)
                if ok:
                    held.discard(symbol)
                else:
                    print(f"    close failed: {err}")
            else:
                held.discard(symbol)
            _log(symbol, reason, price, "CLOSE" if args.live_paper else "CLOSE_DRY_RUN",
                 args.live_paper)
            continue

        if sig not in ("LONG", "SHORT"):
            continue
        if pdt_blocked or len(held) >= args.max_positions:
            continue
        if cfg.time_filter and not (cfg.session_start_min <= mins < cfg.session_end_min):
            continue
        actionable += 1
        print(f"  [{symbol}] {sig} @ {price:.2f} -> {'BUY' if sig == 'LONG' else 'SELL SHORT'} {cfg.qty}")
        if args.live_paper:
            opener = broker.safe_buy_qty if sig == "LONG" else broker.safe_sell_qty
            ok, err = opener(client, symbol, cfg.qty)
            if ok:
                held.add(symbol)
            else:
                print(f"    order failed: {err}")
        else:
            held.add(symbol)
        _log(symbol, sig, price, sig if args.live_paper else f"{sig}_DRY_RUN", args.live_paper)

    if actionable == 0:
        print("  No dual-slope entries or exits this pass.")


def run_trade(args) -> None:
    cfg = _cfg_from_args(args)
    client = broker.get_client()
    acct = broker.get_account_summary(client)
    mode = "LIVE PAPER" if args.live_paper else "DRY RUN"
    print(
        f"SMA {cfg.fast}/{cfg.slow} dual-slope | {mode} | qty={cfg.qty} | "
        f"stop={cfg.stop_points}pt target={cfg.take_points}pt | "
        f"{args.session_start}-{args.session_end} ET | cash=${acct['cash']:.2f}"
    )
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


def main():
    p = argparse.ArgumentParser(description="SMA 34/89 dual-slope bot (separate from autotrade.py)")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--live-paper", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--ignore-market-hours", action="store_true")
    p.add_argument("--source", choices=("yahoo", "alpaca"), default="yahoo")
    p.add_argument("--period", default="60d")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--interval", default="2m")
    p.add_argument("--interval-minutes", type=int, default=2)
    p.add_argument("--loop-minutes", type=int, default=2)
    p.add_argument("--universe-size", type=int, default=30)
    p.add_argument("--max-positions", type=int, default=8)
    p.add_argument("--fast", type=int, default=34)
    p.add_argument("--slow", type=int, default=89)
    p.add_argument("--filter-len", type=int, default=200)
    p.add_argument("--sma-filter", action="store_true", help="Enable SMA(200) slope filter")
    p.add_argument("--direction", choices=("both", "long", "short"), default="both")
    p.add_argument("--qty", type=float, default=3.0, help="Shares (original script: 3 contracts)")
    p.add_argument("--stop-points", type=float, default=10.0)
    p.add_argument("--take-points", type=float, default=20.0)
    p.add_argument("--no-take-profit", action="store_true")
    p.add_argument("--no-time-filter", action="store_true")
    p.add_argument("--session-start", default="09:30")
    p.add_argument("--session-end", default="14:30")
    p.add_argument("--out", default="data/sma_slope_backtest.json")
    args = p.parse_args()
    if args.backtest:
        run_backtest(args)
    else:
        run_trade(args)


if __name__ == "__main__":
    main()

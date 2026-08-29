"""Fully automated intraday day-trading loop -- no per-ticker prompting.

Scans a liquid universe (today's gainers/active/losers + the S&P 500) on
5-minute candles, combining the SMA/RSI/MACD/volume signal engine with
candlestick patterns (engulfing, hammer, shooting star, doji). Opens
positions on STRONG BUY, exits on STRONG SELL, and auto-flattens
(closes) every open position before market close so nothing is held
overnight -- that's what makes this day-trading rather than swing
trading.

SAFETY: by default this is a DRY RUN -- it prints what it would do and
submits nothing. Pass --live-paper to actually place orders, and even
then it only ever talks to Alpaca's paper (fake money) endpoint -- see
src/daytrade/broker.py. It always connects to your Alpaca account
(read-only unless --live-paper) since it needs the market clock and
your current positions to make decisions.

Usage:
  python autotrade.py                        Dry run, loops every 5 min during market hours
  python autotrade.py --live-paper            Actually trade on paper, same loop
  python autotrade.py --live-paper --once     Single pass then exit (for testing)
  python autotrade.py --universe-size 60 --cash-per-trade 300 --max-positions 6

This is a simple rule-based heuristic on top of free delayed/real-time
data. It will generate losing trades -- that's normal, not a bug. Do not
point this at a real-money account.
"""
import argparse
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "src")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from daytrade import broker
from daytrade.scan import batch_intraday_signals
from daytrade.tradelog import log_trade
from daytrade.universe import get_daytrade_universe


def flatten_all(client, positions: dict, live: bool) -> None:
    print(f"Within the close window -- flattening {len(positions)} open position(s).")
    if live:
        try:
            broker.close_all_positions(client)
        except Exception as e:
            print(f"  -> flatten FAILED: {e}")
            return
    for symbol in positions:
        log_trade({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal": "",
            "score": "",
            "price": "",
            "action": "FLATTEN" if live else "FLATTEN_DRY_RUN",
            "mode": "live-paper" if live else "dry-run",
        })


def run_cycle(client, universe_size: int, cash_per_trade: float, max_positions: int,
              flatten_buffer: int, no_entry_buffer: int, live: bool,
              ignore_market_hours: bool = False) -> None:
    mtc = broker.minutes_to_close(client)
    if mtc is None:
        if not ignore_market_hours:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed -- idling.")
            return
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed, but --ignore-market-hours is set -- "
              f"scanning anyway (orders won't fill until the next session).")
        mtc = 999  # treat as "plenty of time", skip flatten/no-entry window logic

    positions = broker.get_open_positions(client)

    if mtc <= flatten_buffer:
        if positions:
            flatten_all(client, positions, live)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Near close ({mtc:.0f} min left), no positions to flatten.")
        return

    allow_new_entries = mtc > no_entry_buffer
    symbols = get_daytrade_universe(max_count=universe_size)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(symbols)} symbols "
          f"({mtc:.0f} min to close, new entries {'allowed' if allow_new_entries else 'paused'})...")

    results = batch_intraday_signals(symbols)
    open_count = len(positions)
    actionable = 0

    for r in results:
        symbol = r["symbol"]
        signal = r["signal"]
        has_position = symbol in positions
        action = "NONE"

        if signal == "STRONG BUY" and not has_position and allow_new_entries and open_count < max_positions:
            action = "BUY"
        elif signal == "STRONG SELL" and has_position:
            action = "SELL"

        if action == "NONE":
            continue
        actionable += 1

        print(f"  [{symbol}] price={r['price']:.2f} signal={signal} (score {r['score']}) -> {action}")
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal": signal,
            "score": r["score"],
            "price": r["price"],
            "action": action,
            "mode": "live-paper" if live else "dry-run",
        }

        if action == "BUY":
            if live:
                try:
                    broker.buy_notional(client, symbol, cash_per_trade)
                    open_count += 1
                    print(f"    -> submitted paper BUY order for ${cash_per_trade:.2f} of {symbol}")
                except Exception as e:
                    print(f"    -> BUY order FAILED: {e}")
                    row["action"] = "BUY_FAILED"
            else:
                print(f"    -> [dry run] would BUY ${cash_per_trade:.2f} of {symbol}")

        elif action == "SELL":
            if live:
                try:
                    broker.close_position(client, symbol)
                    print(f"    -> submitted paper SELL (close position) for {symbol}")
                except Exception as e:
                    print(f"    -> SELL order FAILED: {e}")
                    row["action"] = "SELL_FAILED"
            else:
                print(f"    -> [dry run] would SELL/close position in {symbol}")

        log_trade(row)

    if actionable == 0:
        print("  No actionable signals this pass.")


def main():
    parser = argparse.ArgumentParser(description="Automated intraday scan-and-trade loop")
    parser.add_argument("--live-paper", action="store_true",
                        help="Actually submit orders to your Alpaca PAPER account. "
                             "Without this flag, nothing is ever submitted.")
    parser.add_argument("--once", action="store_true", help="Single pass then exit (default: loop continuously)")
    parser.add_argument("--interval-minutes", type=int, default=5,
                        help="Minutes between scans (default: 5)")
    parser.add_argument("--universe-size", type=int, default=100,
                        help="Max number of symbols to scan each pass (default: 100)")
    parser.add_argument("--cash-per-trade", type=float, default=500.0,
                        help="Dollar amount per new position (default: 500)")
    parser.add_argument("--max-positions", type=int, default=8,
                        help="Max concurrent open positions (default: 8)")
    parser.add_argument("--flatten-minutes-before-close", type=int, default=5,
                        help="Close every open position once this close to market close (default: 5)")
    parser.add_argument("--no-new-entries-minutes-before-close", type=int, default=15,
                        help="Stop opening new positions once this close to market close (default: 15)")
    parser.add_argument("--ignore-market-hours", action="store_true",
                        help="Scan/decide even when the market is closed, for testing the logic "
                             "(orders won't fill until the next session)")
    args = parser.parse_args()

    client = broker.get_client()
    acct = broker.get_account_summary(client)
    mode = "LIVE PAPER TRADING" if args.live_paper else "DRY RUN (no orders will be submitted)"
    print(f"Connected to Alpaca PAPER account -- cash: ${acct['cash']:.2f}  "
          f"equity: ${acct['equity']:.2f}  buying power: ${acct['buying_power']:.2f}")
    print(f"Mode: {mode}\n")

    try:
        while True:
            run_cycle(
                client,
                args.universe_size,
                args.cash_per_trade,
                args.max_positions,
                args.flatten_minutes_before_close,
                args.no_new_entries_minutes_before_close,
                args.live_paper,
                args.ignore_market_hours,
            )
            if args.once:
                break
            print(f"Sleeping {args.interval_minutes} minute(s)...\n")
            time.sleep(args.interval_minutes * 60)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

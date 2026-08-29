"""Signal-driven paper-trading bot on top of Alpaca's paper account.

SAFETY: by default this ALWAYS runs in dry-run mode -- it prints what it
would do and submits nothing. You must pass --live-paper to actually
place orders, and even then it only ever hits Alpaca's paper (fake
money) endpoint -- see src/daytrade/broker.py.

Usage:
  python bot.py AAPL MSFT NVDA                    Dry run, single pass
  python bot.py AAPL MSFT NVDA --live-paper        Actually place paper trades
  python bot.py AAPL MSFT NVDA --live-paper --loop --interval-minutes 15
                                                    Keep running during market hours

Decision rule (deliberately simple):
  - STRONG BUY and no existing position and under --max-positions -> buy
    --cash-per-trade dollars worth (fractional shares via notional order)
  - STRONG SELL and an existing position -> close the position
  - Anything else (HOLD, BUY, SELL, or already positioned) -> no action

This does not do stop-losses, take-profits, or position trimming. It is a
starting point for experimentation on a paper account, not a finished
trading system.
"""
import argparse
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "src")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from daytrade import broker
from daytrade.scan import batch_signals
from daytrade.tradelog import log_trade


def run_once(tickers: list[str], client, cash_per_trade: float, max_positions: int, live: bool) -> None:
    positions = broker.get_open_positions(client) if client else {}
    open_count = len(positions)

    results = batch_signals([t.upper() for t in tickers])
    found = {r["symbol"] for r in results}
    for symbol in [t.upper() for t in tickers]:
        if symbol not in found:
            print(f"[{symbol}] skipped -- couldn't fetch/analyze (not enough history or a bad ticker)")

    for r in results:
        symbol, signal = r["symbol"], r["signal"]
        has_position = symbol in positions
        action = "NONE"

        if signal == "STRONG BUY" and not has_position and open_count < max_positions:
            action = "BUY"
        elif signal == "STRONG SELL" and has_position:
            action = "SELL"

        print(f"[{symbol}] price={r['price']:.2f} signal={signal} (score {r['score']}) "
              f"position={'yes' if has_position else 'no'} -> action={action}")

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
                ok, err = broker.safe_buy_notional(client, symbol, cash_per_trade)
                if ok:
                    open_count += 1
                    print(f"  -> submitted paper BUY order for ${cash_per_trade:.2f} of {symbol}")
                else:
                    print(f"  -> BUY order FAILED: {err}")
                    row["action"] = "BUY_FAILED"
            else:
                print(f"  -> [dry run] would BUY ${cash_per_trade:.2f} of {symbol}")

        elif action == "SELL":
            if live:
                ok, err = broker.safe_close_position(client, symbol)
                if ok:
                    print(f"  -> submitted paper SELL (close position) for {symbol}")
                else:
                    print(f"  -> SELL order FAILED: {err}")
                    row["action"] = "SELL_FAILED"
            else:
                print(f"  -> [dry run] would SELL/close position in {symbol}")

        log_trade(row)


def main():
    parser = argparse.ArgumentParser(description="Signal-driven Alpaca paper-trading bot")
    parser.add_argument("tickers", nargs="+", help="Tickers to watch, e.g. AAPL MSFT NVDA")
    parser.add_argument("--live-paper", action="store_true",
                        help="Actually submit orders to your Alpaca PAPER account. "
                             "Without this flag, nothing is ever submitted.")
    parser.add_argument("--cash-per-trade", type=float, default=1000.0,
                        help="Dollar amount per new position (default: 1000)")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Max concurrent open positions the bot will open (default: 5)")
    parser.add_argument("--loop", action="store_true", help="Keep running instead of a single pass")
    parser.add_argument("--interval-minutes", type=int, default=15,
                        help="Minutes between passes when --loop is set (default: 15)")
    parser.add_argument("--ignore-market-hours", action="store_true",
                        help="Run even when the market is closed (orders won't fill, useful for testing the logic)")
    args = parser.parse_args()

    client = None
    if args.live_paper:
        client = broker.get_client()
        acct = broker.get_account_summary(client)
        print(f"Connected to Alpaca PAPER account -- cash: ${acct['cash']:.2f}  "
              f"equity: ${acct['equity']:.2f}  buying power: ${acct['buying_power']:.2f}")
    else:
        print("DRY RUN -- no orders will be submitted. Pass --live-paper to actually trade on paper.")

    try:
        while True:
            try:
                if client and not args.ignore_market_hours and not broker.is_market_open(client):
                    print("Market is closed -- skipping this pass. Use --ignore-market-hours to override for testing.")
                else:
                    run_once(args.tickers, client, args.cash_per_trade, args.max_positions, args.live_paper)
            except Exception as e:
                print(f"Cycle failed, will retry next interval: {e}")
                if not args.loop:
                    raise

            if not args.loop:
                break
            print(f"\nSleeping {args.interval_minutes} minutes...\n")
            time.sleep(args.interval_minutes * 60)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

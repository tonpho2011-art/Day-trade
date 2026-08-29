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
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from daytrade import broker
from daytrade.data import get_ohlcv
from daytrade.indicators import build_signal

LOG_PATH = Path("data/trade_log.csv")


def log_trade(row: dict) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_once(tickers: list[str], client, cash_per_trade: float, max_positions: int, live: bool) -> None:
    positions = broker.get_open_positions(client) if client else {}
    open_count = len(positions)

    for symbol in tickers:
        symbol = symbol.upper()
        try:
            df = get_ohlcv(symbol, period="6mo", interval="1d")
            tech = build_signal(df)
        except Exception as e:
            print(f"[{symbol}] skipped -- couldn't fetch/analyze: {e}")
            continue

        signal = tech["signal"]
        has_position = symbol in positions
        action = "NONE"

        if signal == "STRONG BUY" and not has_position and open_count < max_positions:
            action = "BUY"
        elif signal == "STRONG SELL" and has_position:
            action = "SELL"

        print(f"[{symbol}] price={tech['price']:.2f} signal={signal} (score {tech['score']}) "
              f"position={'yes' if has_position else 'no'} -> action={action}")

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal": signal,
            "score": tech["score"],
            "price": tech["price"],
            "action": action,
            "mode": "live-paper" if live else "dry-run",
        }

        if action == "BUY":
            if live:
                try:
                    broker.buy_notional(client, symbol, cash_per_trade)
                    open_count += 1
                    print(f"  -> submitted paper BUY order for ${cash_per_trade:.2f} of {symbol}")
                except Exception as e:
                    print(f"  -> BUY order FAILED: {e}")
                    row["action"] = "BUY_FAILED"
            else:
                print(f"  -> [dry run] would BUY ${cash_per_trade:.2f} of {symbol}")

        elif action == "SELL":
            if live:
                try:
                    broker.close_position(client, symbol)
                    print(f"  -> submitted paper SELL (close position) for {symbol}")
                except Exception as e:
                    print(f"  -> SELL order FAILED: {e}")
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

    while True:
        if client and not args.ignore_market_hours and not broker.is_market_open(client):
            print("Market is closed -- skipping this pass. Use --ignore-market-hours to override for testing.")
        else:
            run_once(args.tickers, client, args.cash_per_trade, args.max_positions, args.live_paper)

        if not args.loop:
            break
        print(f"\nSleeping {args.interval_minutes} minutes...\n")
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()

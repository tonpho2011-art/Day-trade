"""Fully automated intraday day-trading loop -- no per-ticker prompting.

Scans a liquid universe (today's gainers/active/losers + the S&P 500) on
5-minute candles, combining the SMA/RSI/MACD/volume signal engine with
candlestick patterns (engulfing, hammer, shooting star, doji). Opens
positions on STRONG BUY, and exits a held position on any of three
independent triggers, whichever comes first:
  1. Stop-loss  -- price drops --stop-loss-pct from entry
  2. Take-profit -- price rises --take-profit-pct from entry
  3. STRONG SELL signal flips on
Every open position is also force-flattened before market close so
nothing is held overnight -- that's what makes this day-trading rather
than swing trading. Held positions are always re-checked for exits every
cycle even if they've dropped off the scanned universe (e.g. a stock
that cooled off and fell out of today's top movers) -- only the entry
side depends on being in the scan.

SAFETY: by default this is a DRY RUN -- it prints what it would do and
submits nothing. Pass --live-paper to actually place orders, and even
then it only ever talks to Alpaca's paper (fake money) endpoint -- see
src/daytrade/broker.py. It always connects to your Alpaca account
(read-only unless --live-paper) since it needs the market clock, your
current positions, and buying power to make decisions.

LEVERAGE: --leverage multiplies --cash-per-trade (e.g. cash-per-trade
500 * leverage 3 = $1500 notional per position). This uses Alpaca's
margin buying power, which is real leverage even on a paper account --
it amplifies both gains AND losses by the same multiple, and the bot
will refuse a buy if it would exceed remaining buying power rather than
partially fill.

Usage:
  python autotrade.py                        Dry run, loops every 5 min during market hours
  python autotrade.py --live-paper            Actually trade on paper, same loop
  python autotrade.py --live-paper --once     Single pass then exit (for testing)
  python autotrade.py --cash-per-trade 500 --leverage 7 --stop-loss-pct 2 --take-profit-pct 4

This is a simple rule-based heuristic on top of free delayed/real-time
data. It will generate losing trades -- that's normal, not a bug. Do not
point this at a real-money account.
"""
import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, "src")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from daytrade import broker
from daytrade.scan import batch_intraday_signals
from daytrade.tradelog import log_trade
from daytrade.universe import get_daytrade_universe


@dataclass
class Config:
    live: bool
    universe_size: int
    cash_per_trade: float
    leverage: float
    max_positions: int
    stop_loss_pct: float
    take_profit_pct: float
    flatten_buffer: int
    no_entry_buffer: int
    ignore_market_hours: bool

    @property
    def notional_per_trade(self) -> float:
        return self.cash_per_trade * self.leverage


def _log(symbol: str, signal, score, price, action: str, live: bool) -> None:
    log_trade({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "signal": signal,
        "score": score,
        "price": price,
        "action": action,
        "mode": "live-paper" if live else "dry-run",
    })


def flatten_all(client, positions: dict, live: bool) -> None:
    print(f"Within the close window -- flattening {len(positions)} open position(s).")
    if live:
        ok, err = True, None
        try:
            broker.close_all_positions(client)
        except Exception as e:
            ok, err = False, str(e)
        if not ok:
            print(f"  -> flatten FAILED: {err}")
            return
    for symbol, pos in positions.items():
        _log(symbol, "flatten (end of day)", "", pos["current_price"],
             "FLATTEN" if live else "FLATTEN_DRY_RUN", live)


def check_risk_exits(client, positions: dict, cfg: Config) -> set:
    """Stop-loss / take-profit checks on every currently held position,
    independent of the scanned universe. Returns the set of symbols
    exited so the main loop doesn't double-process them."""
    exited = set()
    for symbol, pos in positions.items():
        plpc = pos["unrealized_plpc"] * 100  # decimal -> percent
        if plpc <= -cfg.stop_loss_pct:
            reason, action = "stop-loss", "SELL_STOPLOSS"
        elif plpc >= cfg.take_profit_pct:
            reason, action = "take-profit", "SELL_TAKEPROFIT"
        else:
            continue

        print(f"  [{symbol}] unrealized {plpc:+.2f}% hit {reason} -> SELL")
        if cfg.live:
            ok, err = broker.safe_close_position(client, symbol)
            if ok:
                print(f"    -> submitted paper SELL (close position) for {symbol}")
            else:
                print(f"    -> SELL order FAILED: {err}")
                action = "SELL_FAILED"
        else:
            print(f"    -> [dry run] would SELL/close position in {symbol} ({reason})")

        _log(symbol, f"{reason} ({plpc:+.2f}%)", "", pos["current_price"], action, cfg.live)
        exited.add(symbol)
    return exited


def run_cycle(client, cfg: Config) -> None:
    mtc = broker.minutes_to_close(client)
    if mtc is None:
        if not cfg.ignore_market_hours:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed -- idling.")
            return
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed, but --ignore-market-hours is set -- "
              f"scanning anyway (orders won't fill until the next session).")
        mtc = 999  # treat as "plenty of time", skip flatten/no-entry window logic

    positions = broker.get_open_positions(client)

    if mtc <= cfg.flatten_buffer:
        if positions:
            flatten_all(client, positions, cfg.live)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Near close ({mtc:.0f} min left), no positions to flatten.")
        return

    # Risk exits run on every held position first, regardless of whether
    # it's still in today's scanned universe.
    exited = check_risk_exits(client, positions, cfg) if positions else set()
    for symbol in exited:
        del positions[symbol]

    allow_new_entries = mtc > cfg.no_entry_buffer
    universe = get_daytrade_universe(max_count=cfg.universe_size)
    # Always re-check currently held tickers too, so a STRONG SELL exit
    # isn't missed just because a stock fell out of today's movers list.
    scan_symbols = list(dict.fromkeys(universe + list(positions.keys())))

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(scan_symbols)} symbols "
          f"({mtc:.0f} min to close, new entries {'allowed' if allow_new_entries else 'paused'})...")

    acct = broker.get_account_summary(client)
    remaining_bp = acct["buying_power"]

    # PDT protection: a sub-$25k account gets flagged after 4 day trades in
    # 5 business days, and Alpaca starts rejecting orders. Stop opening NEW
    # positions once we're close to that limit rather than let orders fail
    # opaquely; existing positions can still be exited.
    pdt_blocked = (
        acct["equity"] < 25000
        and acct["pattern_day_trader"]
        and acct["daytrade_count"] >= 3
    )
    if pdt_blocked and allow_new_entries:
        print(f"  PDT guard: {acct['daytrade_count']} day trades already this week on a "
              f"sub-$25k account -- pausing new entries to avoid an order rejection/lockout.")
        allow_new_entries = False

    results = batch_intraday_signals(scan_symbols)
    open_count = len(positions)
    actionable = 0

    for r in results:
        symbol = r["symbol"]
        if symbol in exited:
            continue
        signal = r["signal"]
        has_position = symbol in positions
        action = "NONE"

        if (signal == "STRONG BUY" and not has_position and allow_new_entries
                and open_count < cfg.max_positions and remaining_bp >= cfg.notional_per_trade):
            action = "BUY"
        elif signal == "STRONG SELL" and has_position:
            action = "SELL"

        if action == "NONE":
            continue
        actionable += 1

        print(f"  [{symbol}] price={r['price']:.2f} signal={signal} (score {r['score']}) -> {action}")

        if action == "BUY":
            if cfg.live:
                ok, err = broker.safe_buy_notional(client, symbol, cfg.notional_per_trade)
                if ok:
                    open_count += 1
                    remaining_bp -= cfg.notional_per_trade
                    print(f"    -> submitted paper BUY order for ${cfg.notional_per_trade:.2f} "
                          f"of {symbol} ({cfg.leverage}x leverage)")
                else:
                    print(f"    -> BUY order FAILED: {err}")
                    action = "BUY_FAILED"
            else:
                print(f"    -> [dry run] would BUY ${cfg.notional_per_trade:.2f} of {symbol} "
                      f"({cfg.leverage}x leverage)")

        elif action == "SELL":
            if cfg.live:
                ok, err = broker.safe_close_position(client, symbol)
                if ok:
                    print(f"    -> submitted paper SELL (close position) for {symbol}")
                else:
                    print(f"    -> SELL order FAILED: {err}")
                    action = "SELL_FAILED"
            else:
                print(f"    -> [dry run] would SELL/close position in {symbol}")

        _log(symbol, signal, r["score"], r["price"], action, cfg.live)

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
                        help="Base dollar amount per new position before leverage (default: 500)")
    parser.add_argument("--leverage", type=float, default=7.0,
                        help="Multiplies --cash-per-trade using margin buying power (default: 7.0)")
    parser.add_argument("--max-positions", type=int, default=8,
                        help="Max concurrent open positions (default: 8)")
    parser.add_argument("--stop-loss-pct", type=float, default=2.0,
                        help="Close a position if it drops this %% below entry (default: 2.0)")
    parser.add_argument("--take-profit-pct", type=float, default=4.0,
                        help="Close a position if it rises this %% above entry (default: 4.0)")
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
    print(f"Mode: {mode}")
    print(f"Per-trade size: ${args.cash_per_trade:.2f} x {args.leverage}x leverage = "
          f"${args.cash_per_trade * args.leverage:.2f} notional  |  "
          f"stop-loss {args.stop_loss_pct}%  take-profit {args.take_profit_pct}%\n")

    cfg = Config(
        live=args.live_paper,
        universe_size=args.universe_size,
        cash_per_trade=args.cash_per_trade,
        leverage=args.leverage,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        flatten_buffer=args.flatten_minutes_before_close,
        no_entry_buffer=args.no_new_entries_minutes_before_close,
        ignore_market_hours=args.ignore_market_hours,
    )

    try:
        while True:
            try:
                run_cycle(client, cfg)
            except Exception as e:
                # A network blip or a bad API response must not silently
                # kill the loop -- that would stop monitoring any open
                # (possibly leveraged) positions until someone notices.
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle failed, will retry next interval: {e}")
                if args.once:
                    raise
            if args.once:
                break
            print(f"Sleeping {args.interval_minutes} minute(s)...\n")
            time.sleep(args.interval_minutes * 60)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

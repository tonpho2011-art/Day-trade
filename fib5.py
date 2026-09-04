"""Fibonacci 5-candle retracement backtest (not the 3-agent committee).

5 same-color candles → pending limit at Fib 50% of START→END. SL=START,
TP=END. Fill on a later bar's touch. RTH only, flatten ~15:55 ET.
$3500 notional per trade so P/L is comparable to the agent tables.

  python fib5.py --backtest
  python fib5.py --backtest --universe-size 500
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from daytrade.fib5 import FibConfig, simulate_symbol, summarize
from daytrade.universe import FALLBACK_SP500, get_sp500_symbols


def _universe(size: int) -> list[str]:
    symbols = list(FALLBACK_SP500)
    if size > len(symbols):
        extra = [s for s in get_sp500_symbols() if s not in symbols]
        symbols = symbols + extra
    return symbols[:size]


def run_backtest(args) -> None:
    cfg = FibConfig(
        level=args.level,
        min_range=args.min_range,
        use_ema=args.ema_filter,
        notional=args.notional,
    )
    symbols = _universe(args.universe_size)
    print(
        f"Fib 5-candle | 5m | {len(symbols)} names | level={cfg.level:.0%} "
        f"min_range={cfg.min_range} ema={cfg.use_ema} notional=${cfg.notional:.0f}"
    )

    from daytrade.alpaca_bars import download_intraday
    import pandas as pd

    jobs = download_intraday(symbols, minutes=5, days=args.days)
    frames = {symbol: pd.read_parquet(path) for symbol, path in jobs}
    if not frames:
        print("No bars in cache. Need data/bars_5m_iex/.")
        sys.exit(1)

    trades = []
    for i, (symbol, df) in enumerate(frames.items(), start=1):
        print(f"  [{i}/{len(frames)}] {symbol}", flush=True)
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


def main() -> None:
    p = argparse.ArgumentParser(description="Fibonacci 5-candle retracement backtest")
    p.add_argument("--backtest", action="store_true", required=True)
    p.add_argument("--universe-size", type=int, default=30)
    p.add_argument("--days", type=int, default=400)
    p.add_argument("--level", type=float, default=0.5)
    p.add_argument("--min-range", type=float, default=0.0)
    p.add_argument("--ema-filter", action="store_true")
    p.add_argument("--notional", type=float, default=3500.0)
    p.add_argument("--out", default="data/fib5_backtest.json")
    args = p.parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()

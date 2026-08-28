"""CLI entry point.

Usage:
  python screen.py                  Top gainers right now
  python screen.py --losers         Top losers right now
  python screen.py --active         Most active by volume
  python screen.py AAPL             Full signal + news report for AAPL
  python screen.py AAPL MSFT TSLA   Report for multiple tickers
"""
import argparse
import sys

sys.path.insert(0, "src")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from tabulate import tabulate

from daytrade.analyze import analyze, print_report
from daytrade.screener import top_movers


def main():
    parser = argparse.ArgumentParser(description="Stock screener / signal tool")
    parser.add_argument("tickers", nargs="*", help="Tickers to analyze, e.g. AAPL MSFT")
    parser.add_argument("--losers", action="store_true", help="Show top losers instead of gainers")
    parser.add_argument("--active", action="store_true", help="Show most active by volume")
    parser.add_argument("--count", type=int, default=15, help="How many movers to list")
    args = parser.parse_args()

    if args.tickers:
        for symbol in args.tickers:
            report = analyze(symbol)
            print_report(report)
        return

    kind = "losers" if args.losers else "active" if args.active else "gainers"
    df = top_movers(kind, count=args.count)
    print(f"\nTop {kind} right now:\n")
    print(tabulate(df, headers="keys", tablefmt="simple", showindex=False, floatfmt=".2f"))
    print("\nTip: run `python screen.py <TICKER>` on any of these for a full signal + news report.")


if __name__ == "__main__":
    main()

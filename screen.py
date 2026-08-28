"""CLI entry point.

Usage:
  python screen.py                  Top gainers right now
  python screen.py --losers         Top losers right now
  python screen.py --active         Most active by volume
  python screen.py --strongbuy      Scan the movers universe for STRONG BUY signals
  python screen.py --strongsell     Scan the movers universe for STRONG SELL signals
  python screen.py --news-movers    Quiet names with positive headlines (speculative)
  python screen.py --macro          Rates, dollar and FX backdrop
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
from daytrade.macro import get_macro_snapshot, print_macro
from daytrade.scan import news_movers, strong_movers
from daytrade.screener import top_movers


def print_strong_movers(direction: str, count: int) -> None:
    label = "STRONG BUY" if direction == "buy" else "STRONG SELL"
    print(f"\nScanning current movers for {label} signals -- this pulls history for "
          f"many tickers, so it takes longer than a plain screen...")
    hits = strong_movers(direction, count=count)
    if not hits:
        print(f"\nNo {label} signals in the current movers universe.")
        return

    rows = [{
        "symbol": h["symbol"],
        "name": (h.get("name") or "")[:28],
        "price": h["price"],
        "change_%": h.get("change_%"),
        "score": h["score"],
        "why": "; ".join(h["reasons"][:2]),
    } for h in hits]
    print(f"\n{label} candidates ({len(rows)}):\n")
    print(tabulate(rows, headers="keys", tablefmt="simple", floatfmt=".2f"))
    print("\nNote: rule-based heuristic on daily bars, not financial advice.")


def print_news_movers(count: int) -> None:
    print("\nScanning active names that have NOT moved much yet for positive headlines...")
    hits = news_movers(count=count)
    print("\nSPECULATIVE / EXPERIMENTAL: this is keyword counting on recent Google News "
          "headlines,\nnot causal analysis. Nothing here predicts a move -- treat it as a "
          "watchlist starting point.\n")
    if not hits:
        print("No candidates passed the filter right now.")
        return

    rows = [{
        "symbol": h["symbol"],
        "name": (h.get("name") or "")[:24],
        "change_%": h["change_%"],
        "pos": h["positive_hits"],
        "neg": h["negative_hits"],
        "top headline": h["top_headline"][:70],
    } for h in hits]
    print(tabulate(rows, headers="keys", tablefmt="simple", floatfmt=".2f"))


def main():
    parser = argparse.ArgumentParser(description="Stock screener / signal tool")
    parser.add_argument("tickers", nargs="*", help="Tickers to analyze, e.g. AAPL MSFT")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--losers", action="store_true", help="Show top losers instead of gainers")
    mode.add_argument("--active", action="store_true", help="Show most active by volume")
    mode.add_argument("--strongbuy", action="store_true", help="Scan movers for STRONG BUY signals")
    mode.add_argument("--strongsell", action="store_true", help="Scan movers for STRONG SELL signals")
    mode.add_argument("--news-movers", action="store_true",
                      help="Quiet names with positive headlines (speculative)")
    mode.add_argument("--macro", action="store_true", help="Show the macro rates/FX backdrop")
    parser.add_argument("--count", type=int, default=15, help="How many movers to list")
    args = parser.parse_args()

    if args.tickers:
        for symbol in args.tickers:
            report = analyze(symbol)
            print_report(report)
        print("\nBackground context below -- NOT folded into the buy/sell scores above.")
        print_macro(get_macro_snapshot())
        return

    if args.macro:
        print_macro(get_macro_snapshot())
        return

    if args.strongbuy or args.strongsell:
        print_strong_movers("buy" if args.strongbuy else "sell", args.count)
        return

    if args.news_movers:
        print_news_movers(args.count)
        return

    kind = "losers" if args.losers else "active" if args.active else "gainers"
    df = top_movers(kind, count=args.count)
    print(f"\nTop {kind} right now:\n")
    print(tabulate(df, headers="keys", tablefmt="simple", showindex=False, floatfmt=".2f"))
    print("\nTip: run `python screen.py <TICKER>` on any of these for a full signal + news report.")


if __name__ == "__main__":
    main()

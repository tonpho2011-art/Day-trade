"""Train the ML ensemble that autotrade.py cross-checks STRONG BUY signals
against (see src/daytrade/ml_model.py for the labeling/model details).

Downloads intraday history per symbol, labels every bar with the
"triple barrier" method (did an ATR-based take-profit or stop-loss get
hit first), and fits logistic regression / random forest / gradient
boosting on the same indicator feature set the rule-based engine uses.
Saves the three models + scaler + metadata to data/models/.

Usage:
  python train_models.py                                   Default universe, 60 days of 5m bars
  python train_models.py --symbols AAPL,MSFT,NVDA,TSLA      Train on specific tickers
  python train_models.py --period 30d --interval 15m        Different history window/bar size
"""
import argparse
import sys

sys.path.insert(0, "src")

from daytrade import ml_model
from daytrade.universe import FALLBACK_SP500, get_sp500_symbols


def main():
    parser = argparse.ArgumentParser(description="Train the ML ensemble used by autotrade.py")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated tickers to train on (default: ~40 liquid S&P 500 names)")
    parser.add_argument("--symbol-count", type=int, default=40,
                        help="How many S&P 500 symbols to use when --symbols isn't given (default: 40)")
    parser.add_argument("--period", type=str, default="60d",
                        help="History window to download, yfinance format (default: 60d -- the max yfinance allows for 5m bars)")
    parser.add_argument("--interval", type=str, default="5m",
                        help="Bar size, must match what autotrade.py scans with to be meaningful (default: 5m)")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        try:
            symbols = get_sp500_symbols()[: args.symbol_count]
        except Exception:
            symbols = FALLBACK_SP500[: args.symbol_count]

    print(f"Training on {len(symbols)} symbols, period={args.period}, interval={args.interval}")
    print(f"Symbols: {', '.join(symbols)}\n")

    training_df = ml_model.build_training_set(symbols, period=args.period, interval=args.interval)
    if training_df.empty:
        print("\nNo training samples produced -- check network access / symbol validity / that "
              "the period+interval combination is one yfinance actually supports.")
        return

    print(f"\n{len(training_df)} total samples "
          f"({training_df['label'].mean():.1%} take-profit-first base rate)\n")

    trained = ml_model.train(training_df)
    print(f"Train: {trained['n_train']} samples  Test: {trained['n_test']} samples  "
          f"Base rate: {trained['base_rate']:.1%}\n")
    for name, m in trained["metrics"].items():
        print(f"  {name:22s} accuracy={m['accuracy']:.3f}  precision={m['precision']:.3f}  "
              f"auc={m['auc']}  (n={m['n_test']})")

    ml_model.save(trained)
    print(f"\nSaved ensemble to {ml_model.MODEL_DIR}/ -- autotrade.py will pick it up automatically.")


if __name__ == "__main__":
    main()

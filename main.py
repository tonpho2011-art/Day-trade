import sys
sys.path.insert(0, "src")

from daytrade.backtest import run

if __name__ == "__main__":
    run(symbol="AAPL", period="1y", interval="1d")

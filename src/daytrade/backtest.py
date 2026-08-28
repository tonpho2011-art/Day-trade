"""Run a backtest for a given strategy against fetched data."""
from backtesting import Backtest

from daytrade.data import get_ohlcv
from daytrade.strategies.sma_cross import SmaCross


def run(symbol: str = "AAPL", period: str = "1y", interval: str = "1d", cash: int = 10_000):
    data = get_ohlcv(symbol, period=period, interval=interval)
    bt = Backtest(data, SmaCross, cash=cash, commission=0.001)
    stats = bt.run()
    print(stats)
    bt.plot(open_browser=False, filename=f"data/{symbol}_backtest.html")
    return stats


if __name__ == "__main__":
    run()

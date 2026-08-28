"""Market data fetching utilities."""
import yfinance as yf
import pandas as pd


def get_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data for a symbol using yfinance.

    interval examples: 1m, 5m, 15m, 1h, 1d
    period examples: 1d, 5d, 1mo, 6mo, 1y, 5y, max
    """
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df.dropna()

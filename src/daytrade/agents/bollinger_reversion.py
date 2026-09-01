"""Agent 3: fade a poke through the lower Bollinger band (mean reversion)."""
import pandas as pd

from daytrade.candles import is_bullish_engulfing, is_hammer
from daytrade.agents.votes import BUY, SKIP


def vote_bb_reversion(df: pd.DataFrame) -> str:
    if df is None or len(df) < 20 or "Close" not in df:
        return SKIP

    close = df["Close"]
    lower = close.rolling(20).mean() - 2 * close.rolling(20).std()
    last = df.iloc[-1]
    band = lower.iloc[-1]
    if pd.isna(band):
        return SKIP
    if last["Low"] >= band:
        return SKIP
    if last["Close"] <= band:
        return SKIP
    if not (is_hammer(df) or is_bullish_engulfing(df)):
        return SKIP
    return BUY

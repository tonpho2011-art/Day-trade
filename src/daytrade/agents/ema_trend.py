"""Agent 2: 9/21 EMA cross with volume confirmation (trend following)."""
import pandas as pd

from daytrade.agents.votes import BUY, SKIP
from daytrade.indicators import volume_ratio


def vote_ema_trend(df: pd.DataFrame) -> str:
    if df is None or len(df) < 24 or "Close" not in df or "Volume" not in df:
        return SKIP
    last = df.iloc[-1]
    if last["Close"] <= last["Open"]:
        return SKIP

    ema9 = df["Close"].ewm(span=9, adjust=False).mean()
    ema21 = df["Close"].ewm(span=21, adjust=False).mean()
    if ema9.iloc[-1] <= ema21.iloc[-1]:
        return SKIP

    crossed = False
    for i in range(-3, 0):
        if ema9.iloc[i - 1] <= ema21.iloc[i - 1] and ema9.iloc[i] > ema21.iloc[i]:
            crossed = True
            break
    if not crossed:
        return SKIP

    vr = volume_ratio(df["Volume"]).iloc[-1]
    if pd.isna(vr) or vr < 1.5:
        return SKIP
    return BUY

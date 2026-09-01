"""Agent 1: long-only PO3 (opening-range low sweep) + bullish IFVG retest."""
import pandas as pd

from daytrade.agents.votes import BUY, SKIP

NY = "America/New_York"
OPENING_RANGE_BARS = 6


def vote_po3_ifvg(df: pd.DataFrame) -> str:
    if df is None or len(df) < OPENING_RANGE_BARS + 5:
        return SKIP
    session = _rth_session(df)
    if not _bullish_po3(session):
        return SKIP
    if not _bullish_ifvg_retest(session):
        return SKIP
    return BUY


def _rth_session(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    out.index = idx.tz_convert(NY)
    last_day = out.index[-1].date()
    minutes = out.index.hour * 60 + out.index.minute
    mask = (pd.Series(out.index.date, index=out.index) == last_day) & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    return out.loc[mask]


def _bullish_po3(session: pd.DataFrame) -> bool:
    if len(session) <= OPENING_RANGE_BARS:
        return False
    opening = session.iloc[:OPENING_RANGE_BARS]
    range_low = float(opening["Low"].min())
    after = session.iloc[OPENING_RANGE_BARS:]
    swept = False
    for _, row in after.iterrows():
        if row["Low"] < range_low:
            swept = True
        if swept and row["Close"] >= range_low:
            return True
    return False


def _bearish_fvg_zones(df: pd.DataFrame) -> list[tuple[int, float, float]]:
    zones = []
    for i in range(len(df) - 2):
        c1 = df.iloc[i]
        c3 = df.iloc[i + 2]
        if c1["Low"] > c3["High"]:
            zones.append((i + 2, float(c3["High"]), float(c1["Low"])))
    return zones


def _bullish_ifvg_retest(df: pd.DataFrame) -> bool:
    if len(df) < 5:
        return False
    last = df.iloc[-1]
    last_i = len(df) - 1
    for end_i, bottom, top in _bearish_fvg_zones(df):
        inverted = False
        for j in range(end_i + 1, last_i):
            if df.iloc[j]["Close"] > top:
                inverted = True
                break
        if not inverted:
            continue
        if last["Low"] <= top and last["High"] >= bottom:
            return True
    return False

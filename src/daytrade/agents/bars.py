"""Drop the in-progress last candle so agents only vote on closed bars."""
import pandas as pd

_INTERVALS = {
    "1m": pd.Timedelta(minutes=1),
    "2m": pd.Timedelta(minutes=2),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "60m": pd.Timedelta(hours=1),
    "1h": pd.Timedelta(hours=1),
}


def closed_bars(
    df: pd.DataFrame,
    interval: str = "5m",
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    delta = _INTERVALS.get(interval, pd.Timedelta(minutes=5))
    last = df.index[-1]
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    now = _align_tz(now, last)
    if last + delta > now:
        return df.iloc[:-1].copy()
    return df


def _align_tz(now: pd.Timestamp, last) -> pd.Timestamp:
    last_tz = getattr(last, "tzinfo", None) or getattr(last, "tz", None)
    if last_tz is None:
        return now.tz_localize(None) if now.tzinfo is not None else now
    if now.tzinfo is None:
        return now.tz_localize("UTC").tz_convert(last_tz)
    return now.tz_convert(last_tz)

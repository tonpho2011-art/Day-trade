"""Technical indicators and a simple rule-based buy/sell signal."""
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume / volume.rolling(window).mean()


def build_signal(df: pd.DataFrame, sma_fast: int = 10, sma_slow: int = 30) -> dict:
    """Combine trend/momentum/volume into a BUY/SELL/HOLD call with reasons.

    This is a rule-based heuristic, not a prediction -- treat it as one
    input among many, not investment advice.
    """
    close = df["Close"]
    fast = sma(close, sma_fast)
    slow = sma(close, sma_slow)
    r = rsi(close)
    macd_line, signal_line, hist = macd(close)
    vol_ratio = volume_ratio(df["Volume"])

    last = -1
    score = 0
    reasons = []

    if fast.iloc[last] > slow.iloc[last]:
        score += 1
        reasons.append(f"Uptrend: {sma_fast}-day SMA ({fast.iloc[last]:.2f}) above {sma_slow}-day SMA ({slow.iloc[last]:.2f})")
    else:
        score -= 1
        reasons.append(f"Downtrend: {sma_fast}-day SMA ({fast.iloc[last]:.2f}) below {sma_slow}-day SMA ({slow.iloc[last]:.2f})")

    rsi_last = r.iloc[last]
    if rsi_last < 30:
        score += 1
        reasons.append(f"RSI {rsi_last:.1f} is oversold (<30), possible bounce")
    elif rsi_last > 70:
        score -= 1
        reasons.append(f"RSI {rsi_last:.1f} is overbought (>70), possible pullback")
    else:
        reasons.append(f"RSI {rsi_last:.1f} is neutral")

    if macd_line.iloc[last] > signal_line.iloc[last]:
        score += 1
        reasons.append("MACD line above signal line (bullish momentum)")
    else:
        score -= 1
        reasons.append("MACD line below signal line (bearish momentum)")

    vr_last = vol_ratio.iloc[last]
    if pd.notna(vr_last) and vr_last > 1.5:
        reasons.append(f"Volume is {vr_last:.1f}x the 20-day average (move looks confirmed)")
    elif pd.notna(vr_last):
        reasons.append(f"Volume is {vr_last:.1f}x the 20-day average")

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "score": score,
        "reasons": reasons,
        "price": close.iloc[last],
        "rsi": rsi_last,
        "sma_fast": fast.iloc[last],
        "sma_slow": slow.iloc[last],
        "volume_ratio": vr_last,
    }

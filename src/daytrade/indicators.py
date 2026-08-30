"""Technical indicators and a rule-based buy/sell signal.

build_signal() runs ~60 individual indicator/pattern checks (trend,
momentum, volatility, volume, and candlestick categories) and combines
them into one BUY/SELL/HOLD call. Every indicator "votes" -1 (bearish),
0 (neutral), or +1 (bullish); votes are averaged *within* each category
first, then the category averages are combined -- so a category with 20
indicators in it doesn't drown out one with 5. This is still a rule-based
heuristic, not a prediction -- treat it as one input among many, not
investment advice.
"""
import pandas as pd
import ta

from daytrade import candles

# Weight per category in the final composite score. Trend/momentum are
# generally the most reliable timeframe-agnostic reads; volatility
# indicators mostly confirm rather than lead, so they carry less weight.
CATEGORY_WEIGHTS = {
    "trend": 1.2,
    "momentum": 1.0,
    "volume": 0.8,
    "candlestick": 0.8,
    "volatility": 0.6,
}


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


def signal_from_score(score: float) -> str:
    if score >= 2.5:
        return "STRONG BUY"
    if score >= 1.0:
        return "BUY"
    if score <= -2.5:
        return "STRONG SELL"
    if score <= -1.0:
        return "SELL"
    return "HOLD"


def _safe(row, col, default=None):
    val = row.get(col, default)
    return default if pd.isna(val) else val


def _cross_vote(a, b) -> int:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return 0
    return 1 if a > b else -1 if a < b else 0


def _zero_cross_vote(v) -> int:
    if v is None or pd.isna(v):
        return 0
    return 1 if v > 0 else -1 if v < 0 else 0


def _trend_votes(row, sma_fast_val, sma_slow_val) -> tuple[list[int], list[str]]:
    votes, reasons = [], []

    v = _cross_vote(sma_fast_val, sma_slow_val)
    votes.append(v)
    reasons.append(f"{'Uptrend' if v > 0 else 'Downtrend'}: fast SMA ({sma_fast_val:.2f}) "
                   f"vs slow SMA ({sma_slow_val:.2f})")

    v = _cross_vote(_safe(row, "trend_ema_fast"), _safe(row, "trend_ema_slow"))
    votes.append(v)
    if v:
        reasons.append(f"EMA(12/26) {'bullish' if v > 0 else 'bearish'} cross")

    adx = _safe(row, "trend_adx", 0)
    adx_pos, adx_neg = _safe(row, "trend_adx_pos", 0), _safe(row, "trend_adx_neg", 0)
    if adx and adx > 20:
        v = 1 if adx_pos > adx_neg else -1
        votes.append(v)
        reasons.append(f"ADX {adx:.1f} shows a trending market, {'+DI' if v > 0 else '-DI'} in control")
    else:
        votes.append(0)

    aroon = _safe(row, "trend_aroon_ind", 0)
    v = 1 if aroon > 0 else -1 if aroon < 0 else 0
    votes.append(v)

    v = _zero_cross_vote(_safe(row, "trend_vortex_ind_diff"))
    votes.append(v)

    psar_up, psar_down = _safe(row, "trend_psar_up"), _safe(row, "trend_psar_down")
    v = 1 if psar_up is not None else -1 if psar_down is not None else 0
    votes.append(v)
    if v:
        reasons.append(f"Parabolic SAR is {'below' if v > 0 else 'above'} price (trend intact)")

    close = _safe(row, "Close")
    ich_a, ich_b = _safe(row, "trend_ichimoku_a"), _safe(row, "trend_ichimoku_b")
    if close is not None and ich_a is not None and ich_b is not None:
        top, bottom = max(ich_a, ich_b), min(ich_a, ich_b)
        v = 1 if close > top else -1 if close < bottom else 0
        votes.append(v)
        if v:
            reasons.append(f"Price is {'above' if v > 0 else 'below'} the Ichimoku cloud")
    else:
        votes.append(0)

    cci = _safe(row, "trend_cci", 0)
    v = 1 if cci > 100 else -1 if cci < -100 else 0
    votes.append(v)
    if v:
        reasons.append(f"CCI {cci:.1f} is {'above +100 (strong up move)' if v > 0 else 'below -100 (strong down move)'}")

    votes.append(_zero_cross_vote(_safe(row, "trend_dpo")))
    votes.append(_cross_vote(_safe(row, "trend_kst"), _safe(row, "trend_kst_sig")))
    votes.append(_zero_cross_vote(_safe(row, "trend_trix")))

    stc = _safe(row, "trend_stc", 50)
    v = 1 if stc > 75 else -1 if stc < 25 else 0
    votes.append(v)

    return votes, reasons


def _momentum_votes(row) -> tuple[list[int], list[str]]:
    votes, reasons = [], []

    rsi_last = _safe(row, "rsi_custom")
    if rsi_last is not None:
        if rsi_last >= 85:
            votes.append(-1)
            reasons.append(f"RSI {rsi_last:.1f} is extremely overbought (>=85), blow-off/pullback risk")
        elif rsi_last >= 45:
            votes.append(1)
            reasons.append(f"RSI {rsi_last:.1f} shows strong bullish momentum")
        elif rsi_last < 30:
            votes.append(-1)
            reasons.append(f"RSI {rsi_last:.1f} shows bearish momentum / breakdown (<30)")
        else:
            votes.append(0)
            reasons.append(f"RSI {rsi_last:.1f} is neutral / weak")

    macd_v, macd_sig = _safe(row, "macd_custom"), _safe(row, "macd_signal_custom")
    v = _cross_vote(macd_v, macd_sig)
    votes.append(v)
    reasons.append(f"MACD line {'above' if v > 0 else 'below'} signal line "
                   f"({'bullish' if v > 0 else 'bearish'} momentum)")

    votes.append(_cross_vote(_safe(row, "momentum_stoch"), _safe(row, "momentum_stoch_signal")))
    votes.append(_cross_vote(_safe(row, "momentum_stoch_rsi_k"), _safe(row, "momentum_stoch_rsi_d")))

    wr = _safe(row, "momentum_wr")
    if wr is not None:
        v = -1 if wr > -20 else 1 if wr < -80 else 0
        votes.append(v)
        if v:
            reasons.append(f"Williams %R {wr:.1f} is {'overbought' if v < 0 else 'oversold'}")

    votes.append(_zero_cross_vote(_safe(row, "momentum_tsi")))

    uo = _safe(row, "momentum_uo")
    if uo is not None:
        votes.append(-1 if uo > 70 else 1 if uo < 30 else 0)

    votes.append(_zero_cross_vote(_safe(row, "momentum_ao")))
    votes.append(_zero_cross_vote(_safe(row, "momentum_roc")))
    votes.append(_cross_vote(_safe(row, "momentum_ppo"), _safe(row, "momentum_ppo_signal")))
    votes.append(_cross_vote(_safe(row, "momentum_pvo"), _safe(row, "momentum_pvo_signal")))
    votes.append(_cross_vote(_safe(row, "Close"), _safe(row, "momentum_kama")))

    return votes, reasons


def _volatility_votes(row, prior_bbw) -> tuple[list[int], list[str]]:
    votes, reasons = [], []

    bbp = _safe(row, "volatility_bbp")
    if bbp is not None:
        v = 1 if bbp > 1 else -1 if bbp < 0 else 0
        votes.append(v)
        if v:
            reasons.append(f"Bollinger %B {bbp:.2f} shows a {'breakout above the upper band' if v > 0 else 'breakdown below the lower band'}")

    kcp = _safe(row, "volatility_kcp")
    if kcp is not None:
        votes.append(1 if kcp > 1 else -1 if kcp < 0 else 0)

    dcp = _safe(row, "volatility_dcp")
    if dcp is not None:
        votes.append(1 if dcp > 0.8 else -1 if dcp < 0.2 else 0)

    bbw = _safe(row, "volatility_bbw")
    close_chg = _safe(row, "Close") - _safe(row, "Open", _safe(row, "Close"))
    if bbw is not None and prior_bbw is not None and bbw > prior_bbw:
        votes.append(1 if close_chg > 0 else -1 if close_chg < 0 else 0)
    else:
        votes.append(0)

    return votes, reasons


def _volume_votes(row, prior_row, vol_ratio_last) -> tuple[list[int], list[str]]:
    votes, reasons = [], []

    if vol_ratio_last is not None and pd.notna(vol_ratio_last):
        if vol_ratio_last > 1.5:
            votes.append(1)
            reasons.append(f"Volume is {vol_ratio_last:.1f}x the 20-day average (move looks confirmed)")
        else:
            votes.append(0)
            reasons.append(f"Volume is {vol_ratio_last:.1f}x the 20-day average")

    def _trend_vs_prior(col):
        cur, prior = _safe(row, col), _safe(prior_row, col) if prior_row is not None else None
        if cur is None or prior is None:
            return 0
        return 1 if cur > prior else -1 if cur < prior else 0

    votes.append(_trend_vs_prior("volume_obv"))
    votes.append(_trend_vs_prior("volume_vpt"))
    votes.append(_trend_vs_prior("volume_adi"))

    nvi_v = _trend_vs_prior("volume_nvi")
    votes.append(1 if nvi_v > 0 else 0)

    cmf = _safe(row, "volume_cmf")
    if cmf is not None:
        votes.append(1 if cmf > 0.05 else -1 if cmf < -0.05 else 0)

    mfi = _safe(row, "volume_mfi")
    if mfi is not None:
        votes.append(-1 if mfi > 80 else 1 if mfi < 20 else 0)

    votes.append(_zero_cross_vote(_safe(row, "volume_fi")))
    votes.append(_zero_cross_vote(_safe(row, "volume_sma_em")))

    return votes, reasons


def _candlestick_votes(df: pd.DataFrame) -> tuple[list[int], list[str], dict]:
    patterns = candles.detect_patterns(df)
    votes, reasons = [], []
    for name, detected in patterns.items():
        if not detected:
            continue
        bias = candles.PATTERN_BIAS.get(name, 0)
        votes.append(bias)
        label = name.replace("_", " ")
        tag = "bullish" if bias > 0 else "bearish" if bias < 0 else "indecision"
        reasons.append(f"Candlestick: {label} on the latest bar ({tag})")
    return votes, reasons, patterns


def _category_score(votes: list[int]) -> float:
    if not votes:
        return 0.0
    return sum(votes) / len(votes)


def _candlestick_score(votes: list[int]) -> float:
    fired = [v for v in votes if v != 0]
    if not fired:
        return 0.0
    return max(-1.0, min(1.0, sum(fired) / len(fired)))


def build_signal(df: pd.DataFrame, sma_fast: int = 10, sma_slow: int = 30) -> dict:
    """Combine ~60 trend/momentum/volatility/volume/candlestick indicators
    into a BUY/SELL/HOLD call with reasons.

    This is a rule-based heuristic, not a prediction -- treat it as one
    input among many, not investment advice.
    """
    close = df["Close"]
    fast = sma(close, sma_fast)
    slow = sma(close, sma_slow)
    r = rsi(close)
    macd_line, signal_line, _ = macd(close)
    vol_ratio = volume_ratio(df["Volume"])

    ta_df = ta.add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True,
    )
    ta_df["rsi_custom"] = r
    ta_df["macd_custom"] = macd_line
    ta_df["macd_signal_custom"] = signal_line

    last = ta_df.iloc[-1]
    prior_row = ta_df.iloc[-2] if len(ta_df) > 1 else None
    prior_bbw = ta_df["volatility_bbw"].iloc[-2] if len(ta_df) > 1 else None

    trend_votes, trend_reasons = _trend_votes(last, fast.iloc[-1], slow.iloc[-1])
    momentum_votes, momentum_reasons = _momentum_votes(last)
    volatility_votes, volatility_reasons = _volatility_votes(last, prior_bbw)
    volume_votes, volume_reasons = _volume_votes(last, prior_row, vol_ratio.iloc[-1])
    candle_votes, candle_reasons, patterns = _candlestick_votes(df)

    category_scores = {
        "trend": _category_score(trend_votes),
        "momentum": _category_score(momentum_votes),
        "volatility": _category_score(volatility_votes),
        "volume": _category_score(volume_votes),
        "candlestick": _candlestick_score(candle_votes),
    }

    composite = sum(category_scores[cat] * weight for cat, weight in CATEGORY_WEIGHTS.items())
    composite = round(composite, 2)

    total_indicators = (
        len(trend_votes) + len(momentum_votes) + len(volatility_votes)
        + len(volume_votes) + len(patterns)
    )

    return {
        "signal": signal_from_score(composite),
        "score": composite,
        "reasons": trend_reasons + momentum_reasons + volatility_reasons + volume_reasons + candle_reasons,
        "category_scores": category_scores,
        "indicator_count": total_indicators,
        "patterns": patterns,
        "price": close.iloc[-1],
        "rsi": r.iloc[-1],
        "sma_fast": fast.iloc[-1],
        "sma_slow": slow.iloc[-1],
        "volume_ratio": vol_ratio.iloc[-1],
    }

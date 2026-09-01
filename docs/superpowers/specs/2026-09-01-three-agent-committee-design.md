# Three-agent committee (v1)

Locked 2026-09-01. Replaces `STRONG BUY` as the **entry** gate in `autotrade.py`. Mechanical risk is unchanged.

## Pipeline

- 5-minute Yahoo bars; scan interval ≥ 5 minutes.
- Agents see the **last fully closed** 5m bar only (drop an in-progress last bar).
- Shared packet: Alpaca positions/account + scanned universe OHLCV + existing `build_signal()` score (tie-break / STRONG SELL only).
- New BUY: **≥ 2 of 3** agent votes. Rank 3/3 first, then `build_signal` score. Equal notional (`cash-per-trade × leverage`).
- No LLM. No shorts. No 5m-only CRT extra filter. No 2m ribbon. No EMA+RSI mashup on the trend agent.

## Agents (long-only)

1. **PO3 + IFVG** — RTH opening range = first 6 five-minute bars (09:30–10:00 ET). BUY if that range **low** was swept and price closed back inside, **and** a bearish 3-candle FVG was body-closed through to the upside and the last closed bar retests that inverted zone.
2. **EMA trend** — 9 EMA > 21 EMA, that cross occurred within the last 3 closed bars, last bar volume ≥ 1.5× the 20-bar average, last bar is green (close > open).
3. **Bollinger reversion** — last closed bar low below the 20-period 2σ lower band, hammer or bullish engulfing, close back **inside** the band.

## Not voted

Stop-loss 2%, take-profit 4%, flatten, PDT, max positions, buying power. **STRONG SELL** still exits a held name. **STRONG BUY** does not open a trade.

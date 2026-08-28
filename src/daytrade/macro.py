"""Macro backdrop: US rates from FRED plus cross-border FX from yfinance.

Everything here is context only -- it is never folded into the per-ticker
buy/sell score.
"""
import pandas as pd
import yfinance as yf

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

FX_TICKERS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CNY": "USDCNY=X",
}
DOLLAR_INDEX_TICKER = "DX-Y.NYB"


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Download a FRED series as a DataFrame with `date` and `value` columns.

    Returns an empty frame if the download fails or the series has no usable
    observations, so one bad series never breaks the whole snapshot.
    """
    empty = pd.DataFrame(columns=["date", "value"])
    try:
        df = pd.read_csv(FRED_CSV.format(series_id=series_id))
    except Exception:
        return empty

    if df.empty or len(df.columns) < 2:
        return empty

    date_col = df.columns[0]
    value_col = series_id if series_id in df.columns else df.columns[1]
    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna().reset_index(drop=True)


def _latest_observation(series_id: str) -> tuple[float | None, str | None]:
    df = fetch_fred_series(series_id)
    if df.empty:
        return None, None
    row = df.iloc[-1]
    return float(row["value"]), row["date"].strftime("%Y-%m-%d")


def _latest_quote(ticker: str) -> dict:
    """Latest close and % change vs the prior close for a yfinance ticker."""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
    except Exception:
        return {"price": None, "change_pct": None}

    closes = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
    if closes.empty:
        return {"price": None, "change_pct": None}

    price = float(closes.iloc[-1])
    change_pct = None
    if len(closes) >= 2:
        prior = float(closes.iloc[-2])
        if prior:
            change_pct = (price - prior) / prior * 100
    return {"price": price, "change_pct": change_pct}


def get_macro_snapshot() -> dict:
    """Fed funds, 2s/10s Treasuries, the dollar and the major FX crosses.

    Partial failures come back as None rather than raising.
    """
    fed_funds, fed_funds_as_of = _latest_observation("FEDFUNDS")
    ten_year, ten_year_as_of = _latest_observation("DGS10")
    two_year, two_year_as_of = _latest_observation("DGS2")
    broad_dollar, broad_dollar_as_of = _latest_observation("DTWEXBGS")

    spread = None
    note = "yield curve unavailable (missing Treasury data)"
    if ten_year is not None and two_year is not None:
        spread = ten_year - two_year
        if spread < 0:
            note = "yield curve is INVERTED (historically a recession warning signal)"
        else:
            note = "yield curve is normal (upward sloping)"

    fx = {pair: _latest_quote(ticker) for pair, ticker in FX_TICKERS.items()}

    return {
        "fed_funds_rate": fed_funds,
        "fed_funds_as_of": fed_funds_as_of,
        "treasury_10y": ten_year,
        "treasury_10y_as_of": ten_year_as_of,
        "treasury_2y": two_year,
        "treasury_2y_as_of": two_year_as_of,
        "yield_curve_spread": spread,
        "yield_curve_note": note,
        "broad_dollar_index": broad_dollar,
        "broad_dollar_as_of": broad_dollar_as_of,
        "dollar_index": _latest_quote(DOLLAR_INDEX_TICKER),
        "fx": fx,
    }


def _fmt(value, suffix: str = "", decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}{suffix}"


def print_macro(snapshot: dict) -> None:
    print("\n=== Macro Backdrop ===")

    print("\nUS rates:")
    print(f"  - Fed funds rate: {_fmt(snapshot['fed_funds_rate'], '%')} "
          f"(as of {snapshot['fed_funds_as_of'] or 'n/a'})")
    print(f"  - 10-year Treasury: {_fmt(snapshot['treasury_10y'], '%')} "
          f"(as of {snapshot['treasury_10y_as_of'] or 'n/a'})")
    print(f"  - 2-year Treasury: {_fmt(snapshot['treasury_2y'], '%')} "
          f"(as of {snapshot['treasury_2y_as_of'] or 'n/a'})")
    print(f"  - 10Y-2Y spread: {_fmt(snapshot['yield_curve_spread'], '%')} -- {snapshot['yield_curve_note']}")

    dxy = snapshot["dollar_index"]
    print("\nDollar:")
    print(f"  - Dollar index (DX-Y.NYB): {_fmt(dxy['price'])} "
          f"({_fmt(dxy['change_pct'], '%')} vs prior close)")
    print(f"  - Trade-weighted broad dollar index: {_fmt(snapshot['broad_dollar_index'])} "
          f"(as of {snapshot['broad_dollar_as_of'] or 'n/a'})")

    print("\nFX crosses:")
    for pair, quote in snapshot["fx"].items():
        decimals = 2 if quote["price"] and quote["price"] > 10 else 4
        print(f"  - {pair}: {_fmt(quote['price'], decimals=decimals)} "
              f"({_fmt(quote['change_pct'], '%')} vs prior close)")

    print("\nNote: macro context only -- none of this feeds the per-ticker buy/sell score.")

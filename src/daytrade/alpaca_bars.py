"""Download and cache Alpaca IEX 5-minute bars (paper keys work for IEX)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path("data/bars_5m_iex")
MIN_BARS = 200


def yahoo_to_alpaca(symbol: str) -> str:
    return symbol.replace("-", ".")


def _client():
    from alpaca.data.historical import StockHistoricalDataClient

    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY")
    return StockHistoricalDataClient(key, secret)


def _ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.rename(columns=str.title)
    if isinstance(out.index, pd.MultiIndex):
        out = out.copy()
        out.index = out.index.droplevel(0)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in out.columns]
    return out[keep].dropna()


def cache_path(symbol: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{symbol.replace('/', '-')}.parquet"


def _fetch_chunk(
    alpaca_symbols: list[str],
    start: datetime,
    end: datetime,
    minutes: int = 5,
) -> pd.DataFrame:
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    req = StockBarsRequest(
        symbol_or_symbols=alpaca_symbols,
        timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    return _client().get_stock_bars(req).df


def _write_chunk(
    yahoo_symbols: list[str],
    start: datetime,
    end: datetime,
    cache_dir: Path,
    minutes: int = 5,
) -> list[str]:
    alpaca_symbols = [yahoo_to_alpaca(s) for s in yahoo_symbols]
    raw = _fetch_chunk(alpaca_symbols, start, end, minutes=minutes)
    saved = []
    if raw is None or raw.empty:
        return saved
    present = set()
    if isinstance(raw.index, pd.MultiIndex):
        present = set(raw.index.get_level_values(0).unique())
    for yahoo, alpaca in zip(yahoo_symbols, alpaca_symbols):
        try:
            if isinstance(raw.index, pd.MultiIndex):
                if alpaca not in present:
                    continue
                piece = raw.xs(alpaca)
            else:
                piece = raw
            ohlcv = _ohlcv(piece)
            if len(ohlcv) < MIN_BARS:
                continue
            path = cache_path(yahoo, cache_dir)
            ohlcv.to_parquet(path)
            saved.append(yahoo)
        except Exception:
            continue
    return saved


def download_intraday(
    symbols: list[str],
    *,
    minutes: int = 5,
    days: int = 400,
    cache_dir: Path | None = None,
    chunk_size: int = 25,
    workers: int = 4,
) -> list[tuple[str, Path]]:
    cache_dir = cache_dir or Path(f"data/bars_{minutes}m_iex")
    cache_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    have = []
    missing = []
    for symbol in symbols:
        path = cache_path(symbol, cache_dir)
        if path.exists():
            have.append((symbol, path))
        else:
            missing.append(symbol)

    print(f"  cache hit {len(have)} / {len(symbols)}; downloading {len(missing)}")
    if not missing:
        return have

    chunks = [missing[i : i + chunk_size] for i in range(0, len(missing), chunk_size)]
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(_write_chunk, chunk, start, end, cache_dir, minutes): chunk
            for chunk in chunks
        }
        for fut in as_completed(futs):
            chunk = futs[fut]
            try:
                saved = fut.result()
            except Exception as e:
                print(f"    chunk {chunk[0]}..{chunk[-1]} failed ({e}); retrying one-by-one")
                saved = []
                for symbol in chunk:
                    try:
                        saved.extend(_write_chunk([symbol], start, end, cache_dir, minutes))
                    except Exception as e2:
                        print(f"    {symbol}: {e2}")
            done += len(chunk)
            print(f"    downloaded {len(saved)} from chunk ending {chunk[-1]} ({done}/{len(missing)})")
            for symbol in saved:
                have.append((symbol, cache_path(symbol, cache_dir)))

    have.sort(key=lambda x: x[0])
    return have


def download_5m(symbols: list[str], **kwargs) -> list[tuple[str, Path]]:
    kwargs.setdefault("cache_dir", CACHE_DIR)
    return download_intraday(symbols, minutes=5, **kwargs)


def download_2m(symbols: list[str], **kwargs) -> list[tuple[str, Path]]:
    kwargs.setdefault("cache_dir", Path("data/bars_2m_iex"))
    kwargs.setdefault("days", 60)
    return download_intraday(symbols, minutes=2, **kwargs)

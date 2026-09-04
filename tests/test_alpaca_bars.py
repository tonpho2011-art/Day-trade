"""Alpaca IEX bar helpers."""
import pandas as pd

from daytrade.alpaca_bars import frames_from_chunk, yahoo_to_alpaca


def test_yahoo_to_alpaca_dot():
    assert yahoo_to_alpaca("BRK-B") == "BRK.B"


def test_frames_from_chunk_splits_multiindex():
    idx = pd.MultiIndex.from_product(
        [["AAPL", "MSFT"], pd.to_datetime(["2026-01-02 15:00Z"])],
        names=["symbol", "timestamp"],
    )
    raw = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [10.0, 20.0],
        },
        index=idx,
    )
    frames = frames_from_chunk(raw, ["AAPL", "MSFT", "NVDA"])
    assert set(frames) == {"AAPL", "MSFT"}
    assert list(frames["AAPL"].columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert float(frames["AAPL"]["Close"].iloc[0]) == 1.05

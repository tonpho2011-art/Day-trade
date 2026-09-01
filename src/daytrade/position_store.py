"""Persists the ATR-based stop/take-profit plan computed at entry time for
each open position.

Alpaca's position object only carries qty/entry price/P&L -- it has no
concept of "this specific position's stop-loss is 2.7% because that's what
its ATR was when we bought it." Since every position now gets its own
volatility-sized stop instead of one global flat percentage (see risk.py),
that per-symbol plan has to be remembered somewhere between cycles. A flat
JSON file is enough for a single-process bot; this is not a place that
needs a database.
"""
import json
from pathlib import Path

STORE_PATH = Path("data/positions_meta.json")


def _load(path: Path = STORE_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict, path: Path = STORE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def set_plan(symbol: str, plan: dict, path: Path = STORE_PATH) -> None:
    data = _load(path)
    data[symbol] = plan
    _save(data, path)


def get_plan(symbol: str, path: Path = STORE_PATH) -> dict | None:
    return _load(path).get(symbol)


def clear(symbol: str, path: Path = STORE_PATH) -> None:
    data = _load(path)
    if symbol in data:
        del data[symbol]
        _save(data, path)


def sync(open_symbols: set[str], path: Path = STORE_PATH) -> None:
    """Drop any stored plan whose position isn't open anymore -- e.g. it
    was closed manually outside the bot, or a prior run crashed after the
    broker fill but before clear() ran."""
    data = _load(path)
    stale = [s for s in data if s not in open_symbols]
    if not stale:
        return
    for s in stale:
        del data[s]
    _save(data, path)

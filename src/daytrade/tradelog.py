"""Shared CSV trade/decision logging for bot.py and autotrade.py."""
import csv
from pathlib import Path

LOG_PATH = Path("data/trade_log.csv")

_initialized: set[Path] = set()


def log_trade(row: dict, path: Path = LOG_PATH) -> None:
    if path not in _initialized:
        path.parent.mkdir(exist_ok=True)
        _initialized.add(path)

    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

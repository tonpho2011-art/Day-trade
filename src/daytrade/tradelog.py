"""Shared CSV trade/decision logging for bot.py and autotrade.py."""
import csv
from pathlib import Path

LOG_PATH = Path("data/trade_log.csv")


def log_trade(row: dict, path: Path = LOG_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

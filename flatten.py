"""Close every Alpaca paper position and cancel open orders. No new buys."""
import sys

sys.path.insert(0, "src")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from daytrade import broker


def main() -> None:
    client = broker.get_client()
    positions = broker.get_open_positions(client)
    if not positions:
        print("No open positions.")
        return
    print(f"Flattening {len(positions)} paper position(s):")
    for symbol, pos in positions.items():
        print(f"  {symbol}  qty={pos['qty']:.4f}  "
              f"unrealized {pos['unrealized_plpc']*100:+.2f}%")
    broker.close_all_positions(client)
    print("Submitted close_all_positions (Alpaca paper).")


if __name__ == "__main__":
    main()

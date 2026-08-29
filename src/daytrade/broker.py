"""Thin wrapper around Alpaca's paper-trading API.

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY in a .env file (see
.env.example). Always talks to the PAPER endpoint -- this module has no
code path to a live-money account.
"""
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

load_dotenv()


def get_client() -> TradingClient:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. Copy .env.example to .env "
            "and fill in your Alpaca PAPER trading keys from "
            "https://app.alpaca.markets/paper/dashboard/overview"
        )
    return TradingClient(key, secret, paper=True)


def is_market_open(client: TradingClient) -> bool:
    return client.get_clock().is_open


def get_account_summary(client: TradingClient) -> dict:
    acct = client.get_account()
    return {
        "cash": float(acct.cash),
        "equity": float(acct.equity),
        "buying_power": float(acct.buying_power),
    }


def get_open_positions(client: TradingClient) -> dict:
    """symbol -> {'qty': float, 'market_value': float, 'unrealized_pl': float}"""
    positions = {}
    for p in client.get_all_positions():
        positions[p.symbol] = {
            "qty": float(p.qty),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
        }
    return positions


def buy_notional(client: TradingClient, symbol: str, dollars: float):
    order = MarketOrderRequest(
        symbol=symbol,
        notional=round(dollars, 2),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(order)


def close_position(client: TradingClient, symbol: str):
    return client.close_position(symbol)


def close_all_positions(client: TradingClient):
    return client.close_all_positions(cancel_orders=True)


def minutes_to_close(client: TradingClient) -> float | None:
    """Minutes until the market closes, or None if the market is shut."""
    clock = client.get_clock()
    if not clock.is_open:
        return None
    return (clock.next_close - clock.timestamp).total_seconds() / 60

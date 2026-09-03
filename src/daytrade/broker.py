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
        "pattern_day_trader": bool(acct.pattern_day_trader),
        "daytrade_count": int(acct.daytrade_count or 0),
    }


def get_open_positions(client: TradingClient) -> dict:
    """symbol -> qty/market_value/unrealized P&L (dollar and %) /entry price"""
    positions = {}
    for p in client.get_all_positions():
        positions[p.symbol] = {
            "qty": float(p.qty),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
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


def buy_qty(client: TradingClient, symbol: str, qty: float):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=abs(qty),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(order)


def sell_qty(client: TradingClient, symbol: str, qty: float):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=abs(qty),
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(order)


def close_position(client: TradingClient, symbol: str):
    return client.close_position(symbol)


def close_all_positions(client: TradingClient):
    return client.close_all_positions(cancel_orders=True)


def safe_buy_notional(client: TradingClient, symbol: str, dollars: float) -> tuple[bool, str | None]:
    """Submit a BUY, catching broker errors instead of letting them
    propagate. Returns (ok, error_message)."""
    try:
        buy_notional(client, symbol, dollars)
        return True, None
    except Exception as e:
        return False, str(e)


def safe_buy_qty(client: TradingClient, symbol: str, qty: float) -> tuple[bool, str | None]:
    try:
        buy_qty(client, symbol, qty)
        return True, None
    except Exception as e:
        return False, str(e)


def safe_sell_qty(client: TradingClient, symbol: str, qty: float) -> tuple[bool, str | None]:
    try:
        sell_qty(client, symbol, qty)
        return True, None
    except Exception as e:
        return False, str(e)


def safe_close_position(client: TradingClient, symbol: str) -> tuple[bool, str | None]:
    """Close a position, catching broker errors instead of letting them
    propagate. Returns (ok, error_message)."""
    try:
        close_position(client, symbol)
        return True, None
    except Exception as e:
        return False, str(e)


def minutes_to_close(client: TradingClient) -> float | None:
    """Minutes until the market closes, or None if the market is shut."""
    clock = client.get_clock()
    if not clock.is_open:
        return None
    return (clock.next_close - clock.timestamp).total_seconds() / 60

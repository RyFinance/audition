"""Broker adapters (shared with Meridian) and Audition's paper-only wrapper.
No network: adapters run against httpx.MockTransport."""
from __future__ import annotations

import json

import httpx
import pytest

from audition.brokers import REGISTRY, BrokerError, OrderResult
from audition.brokers.alpaca import Alpaca
from audition.brokers.base import Broker, BrokerAccount
from audition.brokers.questrade import Questrade
from audition.brokers.schwab import Schwab
from audition.brokers.trading212 import Trading212, from_t212, to_t212
from audition.brokers.tradier import Tradier


def mock_client(handler):
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


# -- adapters ----------------------------------------------------------------

def test_alpaca_reads_account_and_sends_a_day_market_order():
    sent = {}

    def handler(request: httpx.Request):
        assert request.url.host == "paper-api.alpaca.markets"
        assert request.headers["APCA-API-KEY-ID"] == "PK1"
        if request.url.path == "/v2/account":
            return httpx.Response(200, json={"account_number": "PA123456", "currency": "USD", "cash": "900.5",
                                             "equity": "1000", "buying_power": "2000"})
        if request.url.path == "/v2/positions":
            return httpx.Response(200, json=[{"symbol": "MS", "qty": "3", "avg_entry_price": "180",
                                              "market_value": "540", "unrealized_pl": "-1.5"}])
        if request.url.path == "/v2/orders" and request.method == "POST":
            sent.update(json.loads(request.content))
            return httpx.Response(200, json={"id": "o1", "status": "filled", "filled_qty": "3", "filled_avg_price": "181.2"})
        raise AssertionError(request.url)

    broker = Alpaca({"key_id": "PK1", "secret_key": "s"}, "paper", mock_client(handler))
    acct = broker.account()
    assert (acct.cash, acct.equity, acct.positions[0].symbol, acct.positions[0].qty) == (900.5, 1000.0, "MS", 3.0)
    result = broker.submit_market("MS", "sell_short", 3)
    assert sent == {"symbol": "MS", "qty": "3", "side": "sell", "type": "market", "time_in_force": "day"}
    assert result.filled and result.avg_fill_price == 181.2


def test_tradier_handles_a_single_position_object_and_null():
    def handler(request):
        if request.url.path.endswith("/balances"):
            return httpx.Response(200, json={"balances": {"total_cash": 50, "total_equity": 150, "margin": {"stock_buying_power": 300}}})
        return httpx.Response(200, json={"positions": {"position": {"symbol": "JNJ", "quantity": 2, "cost_basis": 500}}})

    acct = Tradier({"token": "t", "account_id": "VA1"}, "paper", mock_client(handler)).account()
    assert acct.positions[0].avg_price == 250 and acct.buying_power == 300

    def empty(request):
        if request.url.path.endswith("/balances"):
            return httpx.Response(200, json={"balances": {}})
        return httpx.Response(200, json={"positions": "null"})

    assert Tradier({"token": "t", "account_id": "VA1"}, "paper", mock_client(empty)).account().positions == []


def test_bad_credentials_surface_as_a_broker_error():
    broker = Alpaca({"key_id": "x", "secret_key": "y"}, "paper", mock_client(lambda r: httpx.Response(401)))
    with pytest.raises(BrokerError, match="rejected the credentials"):
        broker.account()


def test_capabilities_block_unsupported_orders_before_any_request():
    no_network = mock_client(lambda r: (_ for _ in ()).throw(AssertionError("no request expected")))
    with pytest.raises(BrokerError, match="short"):
        Trading212({"api_key": "k"}, "paper", no_network).submit_market("AAPL", "sell_short", 1)
    with pytest.raises(BrokerError, match="API orders"):
        Questrade({"refresh_token": "r"}, "live", no_network).submit_market("AAPL", "buy", 1)
    with pytest.raises(BrokerError, match="no paper environment"):
        Schwab({"app_key": "a", "app_secret": "b", "refresh_token": "c"}, "paper", no_network)
    with pytest.raises(BrokerError, match="zero shares"):
        Alpaca({"key_id": "x", "secret_key": "y"}, "paper", no_network).submit_market("AAPL", "buy", 0)


def test_trading212_ticker_codes_round_trip():
    assert to_t212("AAPL") == "AAPL_US_EQ" and from_t212("AAPL_US_EQ") == "AAPL"
    assert to_t212("BRK.B") == "BRK_B_US_EQ"


def test_every_registered_broker_describes_itself():
    for broker_id, cls in REGISTRY.items():
        info = cls.describe()
        assert info["id"] == broker_id and info["fields"] and set(info["can_trade"]) <= set(info["envs"])



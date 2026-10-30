"""Audition's paper-only wrapper: environments, read-only brokers, and order legs."""
from __future__ import annotations

import pytest

from audition import callback, sources
from audition.brokers.base import BrokerAccount, BrokerPosition, OrderResult


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "CONFIG", tmp_path / "connections.json")

    def link(key, **vals):
        cfg = sources.load_config()
        cfg["paper_" + key] = vals
        sources.save_config(cfg)
    return link


def test_every_trading_broker_is_built_on_its_paper_environment(config):
    config("alpaca", key_id="PK1", secret_key="s")
    config("tradier", token="t", account_id="VA1")
    config("trading212", api_key="k")
    config("saxo", token="t")
    for key in ("alpaca", "tradier", "trading212", "saxo"):
        b = callback.broker(key)
        assert b.adapter.env == "paper" and not b.read_only


def test_brokers_without_api_paper_trading_are_read_only(config):
    config("schwab", app_key="a", app_secret="b", refresh_token="c")
    b = callback.broker("paper_schwab")
    assert b.read_only
    with pytest.raises(RuntimeError, match="only reads it"):
        b.send([dict(symbol="SPY", shares=1, from_shares=0, to_shares=1, id="x")])


def test_missing_credentials_name_the_field(config):
    with pytest.raises(RuntimeError, match="key_id"):
        callback.broker("alpaca")


def test_a_flip_from_long_to_short_is_sent_as_close_then_open(config, monkeypatch):
    config("alpaca", key_id="PK1", secret_key="s")
    b = callback.broker("alpaca")
    sent = []
    monkeypatch.setattr(b.adapter, "submit_market",
                        lambda sym, side, qty: sent.append((sym, side, qty)) or OrderResult(order_id="1", status="new"))
    out = b.send([dict(symbol="AAPL", shares=-15, from_shares=10, to_shares=-5, id="a"),
                  dict(symbol="MSFT", shares=3, from_shares=-3, to_shares=0, id="b")])
    assert sent == [("AAPL", "sell", 10), ("AAPL", "sell_short", 5), ("MSFT", "buy_to_cover", 3)]
    assert all(o["status"] == "new" for o in out)


def test_positions_and_equity_come_from_the_adapter(config, monkeypatch):
    config("alpaca", key_id="PK1", secret_key="s")
    b = callback.broker("alpaca")
    monkeypatch.setattr(b.adapter, "account", lambda: BrokerAccount(
        account_id="x", equity=25_000, positions=[BrokerPosition("SPY", 10), BrokerPosition("QQQ", 0)]))
    assert b.equity() == 25_000 and b.positions() == {"SPY": 10.0}


def test_every_broker_shows_up_in_the_paper_trading_list():
    assert {"paper_alpaca", "paper_ibkr", "paper_tradier", "paper_saxo", "paper_trading212", "paper_schwab",
            "paper_questrade", "webhook"} <= set(sources.BROKERS)

"""Alpaca (US). Paper and live share one API; only the host differs.
Docs: https://docs.alpaca.markets/reference"""
from __future__ import annotations

from .base import Broker, BrokerAccount, BrokerPosition, CredentialField, OrderResult, Side, mask, num

HOSTS = {"paper": "https://paper-api.alpaca.markets", "live": "https://api.alpaca.markets"}
DATA_HOST = "https://data.alpaca.markets"
SIDES = {"buy": "buy", "buy_to_cover": "buy", "sell": "sell", "sell_short": "sell"}


class Alpaca(Broker):
    id = "alpaca"
    name = "Alpaca"
    regions = ("us",)
    docs_url = "https://alpaca.markets/docs/"
    note = "Separate key pairs for paper and live. Shorting needs a margin account."
    fields = (
        CredentialField("key_id", "API key ID", placeholder="PK… / AK…"),
        CredentialField("secret_key", "Secret key", secret=True),
    )

    @property
    def _headers(self):
        return {"APCA-API-KEY-ID": self.creds["key_id"], "APCA-API-SECRET-KEY": self.creds["secret_key"]}

    def _get(self, path, **params):
        return self._check(self.http.get(f"{HOSTS[self.env]}{path}", headers=self._headers, params=params)).json()

    def identity(self) -> str:
        return mask(self.creds.get("key_id", ""))

    def account(self) -> BrokerAccount:
        acct = self._get("/v2/account")
        positions = [
            BrokerPosition(symbol=p["symbol"], qty=num(p.get("qty")) or 0.0, avg_price=num(p.get("avg_entry_price")),
                           market_value=num(p.get("market_value")), unrealized_pnl=num(p.get("unrealized_pl")))
            for p in self._get("/v2/positions")
        ]
        return BrokerAccount(account_id=mask(acct.get("account_number", "")), currency=acct.get("currency", "USD"),
                             cash=num(acct.get("cash")), equity=num(acct.get("equity")),
                             buying_power=num(acct.get("buying_power")), positions=positions)

    def quote(self, symbol: str):
        resp = self.http.get(f"{DATA_HOST}/v2/stocks/{symbol}/trades/latest", headers=self._headers)
        if resp.status_code != 200:
            return None
        return num(resp.json().get("trade", {}).get("p"))

    def _submit(self, symbol: str, side: Side, qty: int) -> OrderResult:
        body = {"symbol": symbol, "qty": str(qty), "side": SIDES[side], "type": "market", "time_in_force": "day"}
        resp = self._check(self.http.post(f"{HOSTS[self.env]}/v2/orders", headers=self._headers, json=body)).json()
        return self._result(resp)

    def order(self, order_id: str) -> OrderResult:
        return self._result(self._get(f"/v2/orders/{order_id}"))

    @staticmethod
    def _result(o) -> OrderResult:
        return OrderResult(order_id=str(o.get("id")), status=str(o.get("status", "")).lower(),
                           filled_qty=num(o.get("filled_qty")) or 0.0, avg_fill_price=num(o.get("filled_avg_price")),
                           raw=o)

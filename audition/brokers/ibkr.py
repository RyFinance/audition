"""Interactive Brokers (US, Canada, global) via the Client Portal Web API.

Run IBKR's Client Portal Gateway (or IBeam) on this machine and log in with
your paper or live username -- the login decides the environment, so pick
the matching one here. The gateway listens on https://localhost:5000 with a
self-signed certificate; TLS verification is skipped ONLY for localhost.
Docs: https://www.interactivebrokers.com/campus/ibkr-api-page/cpapi-v1/"""
from __future__ import annotations

from urllib.parse import urlparse

import httpx

from .base import HTTP_TIMEOUT_S, Broker, BrokerAccount, BrokerError, BrokerPosition, CredentialField, OrderResult, Side, num

SIDES = {"buy": "BUY", "buy_to_cover": "BUY", "sell": "SELL", "sell_short": "SELL"}
FILLED = {"filled"}


class IBKR(Broker):
    id = "ibkr"
    name = "Interactive Brokers"
    regions = ("us", "ca", "intl")
    docs_url = "https://www.interactivebrokers.com/campus/ibkr-api-page/cpapi-v1/"
    note = "Needs the Client Portal Gateway running locally and logged in. Paper vs live = which login you used."
    fields = (
        CredentialField("gateway_url", "Gateway URL", placeholder="https://localhost:5000", optional=True),
        CredentialField("account_id", "Account ID", placeholder="DU… (paper) / U… (live)", optional=True),
    )

    def __init__(self, creds, env, client_factory=None):
        base = (creds.get("gateway_url") or "https://localhost:5000").rstrip("/")
        host = urlparse(base).hostname or ""
        local = host in ("localhost", "127.0.0.1", "::1")
        super().__init__(creds, env, client_factory or (lambda: httpx.Client(timeout=HTTP_TIMEOUT_S, verify=not local)))
        self.base = f"{base}/v1/api"
        self._acct = creds.get("account_id") or ""

    def _get(self, path, **params):
        return self._check(self.http.get(f"{self.base}{path}", params=params)).json()

    def _account_id(self) -> str:
        if not self._acct:
            rows = self._get("/portfolio/accounts")
            if not rows:
                raise BrokerError("IBKR gateway is not logged in or returned no accounts")
            self._acct = rows[0]["id"]
        if self.env == "live" and self._acct.startswith("D"):
            raise BrokerError("The gateway is logged in to a paper account (DU…) but live was selected")
        if self.env == "paper" and not self._acct.startswith("D"):
            raise BrokerError("The gateway is logged in to a live account but paper was selected")
        return self._acct

    def identity(self) -> str:
        return self._acct or "gateway"

    def account(self) -> BrokerAccount:
        acct = self._account_id()
        summary = self._get(f"/portfolio/{acct}/summary")
        amount = lambda key: num((summary.get(key) or {}).get("amount"))  # noqa: E731
        positions = [
            BrokerPosition(symbol=p.get("contractDesc") or str(p.get("conid")), qty=num(p.get("position")) or 0.0,
                           avg_price=num(p.get("avgCost")), market_value=num(p.get("mktValue")),
                           unrealized_pnl=num(p.get("unrealizedPnl")))
            for p in self._get(f"/portfolio/{acct}/positions/0") or []
        ]
        currency = (summary.get("netliquidation") or {}).get("currency") or "USD"
        return BrokerAccount(account_id=acct, currency=currency, cash=amount("totalcashvalue"),
                             equity=amount("netliquidation"), buying_power=amount("buyingpower"), positions=positions)

    def _conid(self, symbol: str) -> int:
        rows = self._get("/iserver/secdef/search", symbol=symbol, secType="STK")
        for row in rows or []:
            if str(row.get("symbol", "")).upper() == symbol.upper() and row.get("conid"):
                return int(row["conid"])
        raise BrokerError(f"IBKR has no stock contract for {symbol}")

    def _submit(self, symbol: str, side: Side, qty: int) -> OrderResult:
        acct = self._account_id()
        body = {"orders": [{"conid": self._conid(symbol), "orderType": "MKT", "side": SIDES[side],
                            "quantity": qty, "tif": "DAY"}]}
        resp = self._check(self.http.post(f"{self.base}/iserver/account/{acct}/orders", json=body)).json()
        # IBKR answers with precautionary questions (e.g. "order size exceeds…")
        # that must be confirmed one by one before the order is transmitted.
        for _ in range(5):
            first = resp[0] if isinstance(resp, list) and resp else resp
            if isinstance(first, dict) and first.get("order_id"):
                return self.order(str(first["order_id"]))
            if isinstance(first, dict) and first.get("id") and first.get("message"):
                resp = self._check(self.http.post(f"{self.base}/iserver/reply/{first['id']}",
                                                  json={"confirmed": True})).json()
                continue
            break
        raise BrokerError(f"IBKR did not accept the order: {str(resp)[:200]}")

    def order(self, order_id: str) -> OrderResult:
        o = self._get(f"/iserver/account/order/status/{order_id}")
        status = str(o.get("order_status", "")).lower()
        return OrderResult(order_id=str(order_id), status=status,
                           filled_qty=num(o.get("cum_fill")) or (num(o.get("size")) if status in FILLED else 0.0) or 0.0,
                           avg_fill_price=num(o.get("average_price")), raw=o)

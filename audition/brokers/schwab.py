"""Charles Schwab Trader API (US). No paper environment: use readonly or live.
Access tokens last 30 minutes and are refreshed with the app key/secret and a
refresh token (valid 7 days; re-authorise in the Schwab developer portal).
Docs: https://developer.schwab.com/products/trader-api--individual"""
from __future__ import annotations

import base64
import time

from .base import Broker, BrokerAccount, BrokerError, BrokerPosition, CredentialField, OrderResult, Side, mask, num

API = "https://api.schwabapi.com/trader/v1"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
INSTRUCTIONS = {"buy": "BUY", "sell": "SELL", "sell_short": "SELL_SHORT", "buy_to_cover": "BUY_TO_COVER"}


class Schwab(Broker):
    id = "schwab"
    name = "Charles Schwab"
    regions = ("us",)
    envs = ("live",)
    can_trade = ("live",)
    docs_url = "https://developer.schwab.com/"
    note = "No paper environment. Pick read-only, or live once armed. Refresh tokens expire after 7 days."
    fields = (
        CredentialField("app_key", "App key"),
        CredentialField("app_secret", "App secret", secret=True),
        CredentialField("refresh_token", "Refresh token", secret=True),
    )

    _token: str = ""
    _token_expiry: float = 0.0
    _hash: str = ""

    def _headers(self):
        if not self._token or time.time() > self._token_expiry - 60:
            basic = base64.b64encode(f"{self.creds['app_key']}:{self.creds['app_secret']}".encode()).decode()
            resp = self._check(self.http.post(TOKEN_URL, headers={"Authorization": f"Basic {basic}"},
                                              data={"grant_type": "refresh_token", "refresh_token": self.creds["refresh_token"]}))
            body = resp.json()
            self._token = body["access_token"]
            self._token_expiry = time.time() + float(body.get("expires_in", 1800))
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}

    def _account_hash(self) -> str:
        if not self._hash:
            rows = self._check(self.http.get(f"{API}/accounts/accountNumbers", headers=self._headers())).json()
            if not rows:
                raise BrokerError("Schwab returned no linked accounts")
            self._hash = rows[0]["hashValue"]
        return self._hash

    def identity(self) -> str:
        return mask(self.creds.get("app_key", ""))

    def account(self) -> BrokerAccount:
        data = self._check(self.http.get(f"{API}/accounts/{self._account_hash()}", headers=self._headers(),
                                         params={"fields": "positions"})).json()["securitiesAccount"]
        bal = data.get("currentBalances", {})
        positions = []
        for p in data.get("positions", []):
            qty = (num(p.get("longQuantity")) or 0.0) - (num(p.get("shortQuantity")) or 0.0)
            positions.append(BrokerPosition(
                symbol=p.get("instrument", {}).get("symbol", "?"), qty=qty, avg_price=num(p.get("averagePrice")),
                market_value=num(p.get("marketValue")),
                unrealized_pnl=num(p.get("longOpenProfitLoss")) if qty >= 0 else num(p.get("shortOpenProfitLoss")),
            ))
        return BrokerAccount(account_id=mask(data.get("accountNumber", ""), 2), cash=num(bal.get("cashBalance")),
                             equity=num(bal.get("liquidationValue")), buying_power=num(bal.get("buyingPower")),
                             positions=positions)

    def quote(self, symbol: str):
        resp = self.http.get("https://api.schwabapi.com/marketdata/v1/quotes", headers=self._headers(),
                             params={"symbols": symbol})
        if resp.status_code != 200:
            return None
        return num(resp.json().get(symbol, {}).get("quote", {}).get("lastPrice"))

    def _submit(self, symbol: str, side: Side, qty: int) -> OrderResult:
        body = {
            "orderType": "MARKET", "session": "NORMAL", "duration": "DAY", "orderStrategyType": "SINGLE",
            "orderLegCollection": [{"instruction": INSTRUCTIONS[side], "quantity": qty,
                                    "instrument": {"symbol": symbol, "assetType": "EQUITY"}}],
        }
        resp = self._check(self.http.post(f"{API}/accounts/{self._account_hash()}/orders",
                                          headers=self._headers(), json=body))
        order_id = resp.headers.get("Location", "").rstrip("/").split("/")[-1]
        if not order_id:
            raise BrokerError("Schwab accepted the order but returned no order id")
        return self.order(order_id)

    def order(self, order_id: str) -> OrderResult:
        o = self._check(self.http.get(f"{API}/accounts/{self._account_hash()}/orders/{order_id}",
                                      headers=self._headers())).json()
        legs = [leg for act in o.get("orderActivityCollection", []) for leg in act.get("executionLegs", [])]
        qty = sum(num(leg.get("quantity")) or 0.0 for leg in legs)
        price = (sum((num(leg.get("price")) or 0.0) * (num(leg.get("quantity")) or 0.0) for leg in legs) / qty) if qty else None
        return OrderResult(order_id=str(order_id), status=str(o.get("status", "")).lower(),
                           filled_qty=num(o.get("filledQuantity")) or qty, avg_fill_price=price, raw=o)

"""Trading 212 (international, Invest / ISA accounts). Practice (demo) for paper.
Long-only: the API has no short selling. Tickers use Trading 212's own codes;
US stocks map as AAPL -> AAPL_US_EQ.
Docs: https://t212public-api-docs.redoc.ly/"""
from __future__ import annotations

import base64

from .base import Broker, BrokerAccount, BrokerError, BrokerPosition, CredentialField, OrderResult, Side, mask, num

HOSTS = {"paper": "https://demo.trading212.com/api/v0", "live": "https://live.trading212.com/api/v0"}


def to_t212(symbol: str) -> str:
    return symbol if "_" in symbol else f"{symbol.upper().replace('.', '_')}_US_EQ"


def from_t212(ticker: str) -> str:
    return ticker.split("_")[0]


class Trading212(Broker):
    id = "trading212"
    name = "Trading 212"
    regions = ("intl",)
    supports_short = False
    docs_url = "https://helpcentre.trading212.com/hc/en-us/articles/14584770928157"
    note = "Long-only. Generate the key in Settings → API (Practice and Real have separate keys)."
    fields = (
        CredentialField("api_key", "API key", secret=True),
        CredentialField("api_secret", "API secret", secret=True, optional=True,
                        placeholder="only for keys issued with a secret"),
    )

    @property
    def _headers(self):
        key, secret = self.creds["api_key"], self.creds.get("api_secret")
        if secret:
            return {"Authorization": "Basic " + base64.b64encode(f"{key}:{secret}".encode()).decode()}
        return {"Authorization": key}

    def _get(self, path):
        return self._check(self.http.get(f"{HOSTS[self.env]}{path}", headers=self._headers)).json()

    def identity(self) -> str:
        return mask(self.creds.get("api_key", ""))

    def account(self) -> BrokerAccount:
        cash = self._get("/equity/account/cash")
        info = self._get("/equity/account/info")
        positions = []
        for p in self._get("/equity/portfolio"):
            qty, price = num(p.get("quantity")) or 0.0, num(p.get("currentPrice"))
            positions.append(BrokerPosition(symbol=from_t212(p.get("ticker", "?")), qty=qty, avg_price=num(p.get("averagePrice")),
                                            market_value=qty * price if price is not None else None,
                                            unrealized_pnl=num(p.get("ppl"))))
        return BrokerAccount(account_id=str(info.get("id", "")), currency=info.get("currencyCode", "USD"),
                             cash=num(cash.get("free")), equity=num(cash.get("total")), buying_power=num(cash.get("free")),
                             positions=positions)

    def _submit(self, symbol: str, side: Side, qty: int) -> OrderResult:
        signed = qty if side in ("buy", "buy_to_cover") else -qty
        resp = self._check(self.http.post(f"{HOSTS[self.env]}/equity/orders/market", headers=self._headers,
                                          json={"ticker": to_t212(symbol), "quantity": signed})).json()
        if not resp.get("id"):
            raise BrokerError(f"Trading 212 did not accept the order: {str(resp)[:200]}")
        return self.order(str(resp["id"]))

    def order(self, order_id: str) -> OrderResult:
        o = self._get(f"/equity/orders/{order_id}")
        filled_qty = abs(num(o.get("filledQuantity")) or 0.0)
        value = num(o.get("filledValue"))
        price = (value / filled_qty) if value and filled_qty else None
        return OrderResult(order_id=str(order_id), status=str(o.get("status", "")).lower(),
                           filled_qty=filled_qty, avg_fill_price=price, raw=o)

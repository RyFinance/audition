"""Tradier (US). Sandbox for paper, production for live.
Docs: https://documentation.tradier.com/brokerage-api"""
from __future__ import annotations

from .base import Broker, BrokerAccount, BrokerError, BrokerPosition, CredentialField, OrderResult, Side, mask, num

HOSTS = {"paper": "https://sandbox.tradier.com/v1", "live": "https://api.tradier.com/v1"}


def _as_list(value):
    if value in (None, "null"):
        return []
    return value if isinstance(value, list) else [value]


class Tradier(Broker):
    id = "tradier"
    name = "Tradier"
    regions = ("us",)
    docs_url = "https://documentation.tradier.com/"
    note = "Sandbox tokens come from the developer dashboard; live tokens from the brokerage account."
    fields = (
        CredentialField("token", "Access token", secret=True),
        CredentialField("account_id", "Account number", placeholder="VA… / 6YA…"),
    )

    @property
    def _headers(self):
        return {"Authorization": f"Bearer {self.creds['token']}", "Accept": "application/json"}

    @property
    def _acct(self):
        return f"{HOSTS[self.env]}/accounts/{self.creds['account_id']}"

    def identity(self) -> str:
        return mask(self.creds.get("account_id", ""), 3)

    def account(self) -> BrokerAccount:
        bal = self._check(self.http.get(f"{self._acct}/balances", headers=self._headers)).json().get("balances", {})
        raw = self._check(self.http.get(f"{self._acct}/positions", headers=self._headers)).json().get("positions")
        positions = []
        for p in _as_list(raw.get("position") if isinstance(raw, dict) else raw):
            qty = num(p.get("quantity")) or 0.0
            cost = num(p.get("cost_basis"))
            positions.append(BrokerPosition(symbol=p["symbol"], qty=qty,
                                            avg_price=(cost / qty) if cost is not None and qty else None))
        margin = bal.get("margin") or {}
        cash = bal.get("cash") or {}
        return BrokerAccount(
            account_id=self.identity(), cash=num(bal.get("total_cash")), equity=num(bal.get("total_equity")),
            buying_power=num(margin.get("stock_buying_power")) or num(cash.get("cash_available")), positions=positions,
        )

    def quote(self, symbol: str):
        resp = self.http.get(f"{HOSTS[self.env]}/markets/quotes", headers=self._headers, params={"symbols": symbol})
        if resp.status_code != 200:
            return None
        quote = (resp.json().get("quotes") or {}).get("quote") or {}
        return num(quote.get("last"))

    def _submit(self, symbol: str, side: Side, qty: int) -> OrderResult:
        data = {"class": "equity", "symbol": symbol, "side": side, "quantity": str(qty), "type": "market", "duration": "day"}
        resp = self._check(self.http.post(f"{self._acct}/orders", headers=self._headers, data=data)).json()
        order = resp.get("order") or {}
        if order.get("status") != "ok" or not order.get("id"):
            raise BrokerError(f"Tradier rejected the order: {str(resp.get('errors') or order)[:200]}")
        return self.order(str(order.get("id")))

    def order(self, order_id: str) -> OrderResult:
        o = self._check(self.http.get(f"{self._acct}/orders/{order_id}", headers=self._headers)).json().get("order", {})
        return OrderResult(order_id=str(o.get("id", order_id)), status=str(o.get("status", "")).lower(),
                           filled_qty=num(o.get("exec_quantity")) or 0.0, avg_fill_price=num(o.get("avg_fill_price")) or None,
                           raw=o)


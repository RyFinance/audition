"""Saxo Bank (international). SIM environment for paper, LIVE for live.
Use a 24-hour developer token for testing, or your app's OAuth access token.
Docs: https://www.developer.saxo/openapi/learn"""
from __future__ import annotations

from .base import Broker, BrokerAccount, BrokerError, BrokerPosition, CredentialField, OrderResult, Side, mask, num

HOSTS = {"paper": "https://gateway.saxobank.com/sim/openapi", "live": "https://gateway.saxobank.com/openapi"}
BUYSELL = {"buy": "Buy", "buy_to_cover": "Buy", "sell": "Sell", "sell_short": "Sell"}


class Saxo(Broker):
    id = "saxo"
    name = "Saxo Bank"
    regions = ("intl",)
    docs_url = "https://www.developer.saxo/"
    note = "Developer tokens last 24 hours. Fill price is read from Saxo's quote at fill time."
    fields = (CredentialField("token", "Access token", secret=True, placeholder="24h token from developer.saxo"),)

    _account_key = ""

    @property
    def _headers(self):
        return {"Authorization": f"Bearer {self.creds['token']}"}

    def _get(self, path, **params):
        return self._check(self.http.get(f"{HOSTS[self.env]}{path}", headers=self._headers, params=params)).json()

    def identity(self) -> str:
        return mask(self.creds.get("token", ""), 3)

    def _key(self) -> str:
        if not self._account_key:
            rows = self._get("/port/v1/accounts/me").get("Data", [])
            if not rows:
                raise BrokerError("Saxo returned no accounts")
            self._account_key = rows[0]["AccountKey"]
        return self._account_key

    def account(self) -> BrokerAccount:
        bal = self._get("/port/v1/balances/me")
        rows = self._get("/port/v1/netpositions/me",
                         FieldGroups="NetPositionBase,NetPositionView,DisplayAndFormat").get("Data", [])
        positions = []
        for row in rows:
            base, view, fmt = row.get("NetPositionBase", {}), row.get("NetPositionView", {}), row.get("DisplayAndFormat", {})
            positions.append(BrokerPosition(symbol=str(fmt.get("Symbol", "?")).split(":")[0], qty=num(base.get("Amount")) or 0.0,
                                            avg_price=num(view.get("AverageOpenPrice")), market_value=num(view.get("MarketValue")),
                                            unrealized_pnl=num(view.get("ProfitLossOnTrade"))))
        return BrokerAccount(account_id=mask(self._key(), 3), currency=bal.get("Currency", "USD"),
                             cash=num(bal.get("CashBalance")), equity=num(bal.get("TotalValue")),
                             buying_power=num(bal.get("MarginAvailableForTrading")), positions=positions)

    def _uic(self, symbol: str) -> int:
        rows = self._get("/ref/v1/instruments", Keywords=symbol, AssetTypes="Stock").get("Data", [])
        for row in rows:
            if str(row.get("Symbol", "")).split(":")[0].upper() == symbol.upper():
                return int(row["Identifier"])
        raise BrokerError(f"Saxo has no stock instrument for {symbol}")

    def quote(self, symbol: str):
        try:
            q = self._get("/trade/v1/infoprices", Uic=self._uic(symbol), AssetType="Stock").get("Quote", {})
        except BrokerError:
            return None
        bid, ask = num(q.get("Bid")), num(q.get("Ask"))
        return (bid + ask) / 2 if bid and ask else (ask or bid)

    def _submit(self, symbol: str, side: Side, qty: int) -> OrderResult:
        body = {"AccountKey": self._key(), "Uic": self._uic(symbol), "AssetType": "Stock", "BuySell": BUYSELL[side],
                "Amount": qty, "OrderType": "Market", "OrderDuration": {"DurationType": "DayOrder"}, "ManualOrder": False}
        resp = self._check(self.http.post(f"{HOSTS[self.env]}/trade/v2/orders", headers=self._headers, json=body)).json()
        if not resp.get("OrderId"):
            raise BrokerError(f"Saxo did not accept the order: {str(resp)[:200]}")
        price = self.quote(symbol)
        # Saxo market orders fill immediately in market hours and leave the open-orders list,
        # so the fill is recorded at the quote taken right after submission.
        return OrderResult(order_id=str(resp["OrderId"]), status="filled", filled_qty=float(qty),
                           avg_fill_price=price, raw=resp)

    def order(self, order_id: str) -> OrderResult:
        rows = self._get("/port/v1/orders/me").get("Data", [])
        for row in rows:
            if str(row.get("OrderId")) == str(order_id):
                return OrderResult(order_id=order_id, status=str(row.get("Status", "working")).lower(), raw=row)
        return OrderResult(order_id=order_id, status="filled")

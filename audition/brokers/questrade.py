"""Questrade (Canada). Read-only.

Questrade's API lets personal apps read accounts, balances and positions, but
order placement is reserved for approved partner apps, so Meridian can show
the account and keep trading on its own simulator. Practice accounts use the
practice login host. Refresh tokens are single-use: every refresh returns a
new one, which is written back through `on_refresh`.
Docs: https://www.questrade.com/api/documentation"""
from __future__ import annotations

from typing import Callable, Optional

from .base import Broker, BrokerAccount, BrokerError, BrokerPosition, CredentialField, OrderResult, Side, mask, num

LOGIN = {"paper": "https://practicelogin.questrade.com", "live": "https://login.questrade.com"}


class Questrade(Broker):
    id = "questrade"
    name = "Questrade"
    regions = ("ca",)
    can_trade = ()
    supports_short = False
    docs_url = "https://www.questrade.com/api"
    note = "Read-only: Questrade only allows order placement for approved partner apps."
    fields = (
        CredentialField("refresh_token", "Refresh token", secret=True,
                        placeholder="App Hub → generate new token"),
        CredentialField("account_number", "Account number", optional=True),
    )

    on_refresh: Optional[Callable[[str], None]] = None
    _token = ""
    _server = ""

    def _auth(self):
        if not self._token:
            resp = self.http.get(f"{LOGIN[self.env]}/oauth2/token",
                                 params={"grant_type": "refresh_token", "refresh_token": self.creds["refresh_token"]})
            if resp.status_code >= 400:
                raise BrokerError("Questrade refused the refresh token (they are single-use; generate a new one)")
            body = resp.json()
            self._token, self._server = body["access_token"], body["api_server"].rstrip("/")
            self.creds["refresh_token"] = body["refresh_token"]
            if self.on_refresh:
                self.on_refresh(body["refresh_token"])
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, path):
        headers = self._auth()
        return self._check(self.http.get(f"{self._server}/v1{path}", headers=headers)).json()

    def identity(self) -> str:
        return mask(self.creds.get("account_number", ""), 2) or "questrade"

    def account(self) -> BrokerAccount:
        number = self.creds.get("account_number")
        if not number:
            accounts = self._get("/accounts").get("accounts", [])
            if not accounts:
                raise BrokerError("Questrade returned no accounts")
            number = accounts[0]["number"]
        balances = self._get(f"/accounts/{number}/balances").get("combinedBalances", [])
        bal = next((b for b in balances if b.get("currency") == "CAD"), balances[0] if balances else {})
        positions = [
            BrokerPosition(symbol=p.get("symbol", "?"), qty=num(p.get("openQuantity")) or 0.0,
                           avg_price=num(p.get("averageEntryPrice")), market_value=num(p.get("currentMarketValue")),
                           unrealized_pnl=num(p.get("openPnl")))
            for p in self._get(f"/accounts/{number}/positions").get("positions", [])
        ]
        return BrokerAccount(account_id=mask(str(number), 2), currency=bal.get("currency", "CAD"),
                             cash=num(bal.get("cash")), equity=num(bal.get("totalEquity")),
                             buying_power=num(bal.get("buyingPower")), positions=positions)

    def _submit(self, symbol: str, side: Side, qty: int) -> OrderResult:
        raise BrokerError("Questrade does not allow order placement from personal API apps")

    def order(self, order_id: str) -> OrderResult:
        raise BrokerError("Questrade orders are not available")

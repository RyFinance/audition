"""Broker adapter contract.

Every broker implements the same small surface: read the account, read
positions, submit a market order, read an order back. The bot never talks to
a broker directly -- bot/broker_execution.py does, and only after the risk
gate has approved the trade.

Modes (chosen per person in the desk, see bot/broker_config.py):
  readonly  account and positions are read; the bot keeps its own simulator
  paper     orders go to the broker's paper / sandbox / demo environment
  live      orders go to the real account; needs MERIDIAN_ALLOW_LIVE=yes in
            the environment AND an arm confirmation typed in the desk
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import httpx

Side = Literal["buy", "sell", "sell_short", "buy_to_cover"]
Env = Literal["paper", "live"]

HTTP_TIMEOUT_S = 15.0


class BrokerError(RuntimeError):
    """A broker rejected or failed a request. Message is safe to show."""


@dataclass
class CredentialField:
    name: str
    label: str
    secret: bool = False
    placeholder: str = ""
    optional: bool = False


@dataclass
class BrokerPosition:
    symbol: str
    qty: float
    avg_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None


@dataclass
class BrokerAccount:
    account_id: str
    currency: str = "USD"
    cash: Optional[float] = None
    equity: Optional[float] = None
    buying_power: Optional[float] = None
    positions: List[BrokerPosition] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderResult:
    order_id: str
    status: str
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def filled(self) -> bool:
        return self.filled_qty > 0 and self.avg_fill_price is not None


def mask(value: str, keep: int = 4) -> str:
    value = str(value or "")
    return value if len(value) <= keep * 2 else f"{value[:keep]}…{value[-keep:]}"


def num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


class Broker:
    """Base adapter. Subclasses set the class attributes and implement the
    five request methods. `client_factory` exists so tests can inject an
    httpx.MockTransport; production uses a plain httpx.Client."""

    id = ""
    name = ""
    regions: tuple[str, ...] = ()
    envs: tuple[Env, ...] = ("paper", "live")  # environments the broker offers
    can_trade: tuple[Env, ...] = ("paper", "live")  # environments where API orders are allowed
    supports_short = True
    docs_url = ""
    note = ""
    fields: tuple[CredentialField, ...] = ()

    def __init__(self, creds: Dict[str, str], env: Env,
                 client_factory: Optional[Callable[[], httpx.Client]] = None):
        if env not in self.envs:
            raise BrokerError(f"{self.name} has no {env} environment")
        self.creds = creds
        self.env = env
        self._client_factory = client_factory or (lambda: httpx.Client(timeout=HTTP_TIMEOUT_S))
        self._http: Optional[httpx.Client] = None

    # -- plumbing ----------------------------------------------------------
    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            self._http = self._client_factory()
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def _check(self, resp: httpx.Response) -> httpx.Response:
        if resp.status_code in (401, 403):
            raise BrokerError(f"{self.name} rejected the credentials ({resp.status_code})")
        if resp.status_code >= 400:
            detail = resp.text[:200].replace("\n", " ")
            raise BrokerError(f"{self.name} {resp.request.method} {resp.request.url.path} -> {resp.status_code}: {detail}")
        return resp

    def identity(self) -> str:
        return ""

    @classmethod
    def describe(cls) -> Dict[str, Any]:
        return {
            "id": cls.id, "name": cls.name, "regions": list(cls.regions),
            "envs": list(cls.envs), "can_trade": list(cls.can_trade),
            "supports_short": cls.supports_short, "docs_url": cls.docs_url, "note": cls.note,
            "fields": [asdict(f) for f in cls.fields],
        }

    # -- the contract --------------------------------------------------------
    def account(self) -> BrokerAccount:
        raise NotImplementedError

    def quote(self, symbol: str) -> Optional[float]:
        return None

    def _submit(self, symbol: str, side: Side, qty: int) -> OrderResult:
        raise NotImplementedError

    def order(self, order_id: str) -> OrderResult:
        raise NotImplementedError

    # -- shared order flow ---------------------------------------------------
    def submit_market(self, symbol: str, side: Side, qty: int) -> OrderResult:
        if self.env not in self.can_trade:
            raise BrokerError(f"{self.name} does not accept API orders in its {self.env} environment")
        if side in ("sell_short",) and not self.supports_short:
            raise BrokerError(f"{self.name} does not support short selling through its API")
        if qty < 1:
            raise BrokerError("order quantity rounds to zero shares")
        return self._submit(symbol, side, int(qty))

    def wait_for_fill(self, order_id: str, timeout_s: float = 30.0, poll_s: float = 1.0) -> OrderResult:
        deadline = time.time() + timeout_s
        result = self.order(order_id)
        while not result.filled and result.status not in ("rejected", "canceled", "cancelled", "expired") \
                and time.time() < deadline:
            time.sleep(poll_s)
            result = self.order(order_id)
        return result

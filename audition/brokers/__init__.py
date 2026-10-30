"""Broker adapters, shared with Meridian (RyFinance/meridian). Audition builds them on their
paper environment only (see audition/callback.py: PaperBroker). Add one by subclassing
brokers.base.Broker and listing it in REGISTRY."""
from .alpaca import Alpaca
from .base import Broker, BrokerAccount, BrokerError, BrokerPosition, OrderResult
from .ibkr import IBKR
from .questrade import Questrade
from .saxo import Saxo
from .schwab import Schwab
from .trading212 import Trading212
from .tradier import Tradier

REGISTRY: dict[str, type[Broker]] = {b.id: b for b in (Alpaca, Schwab, IBKR, Tradier, Questrade, Saxo, Trading212)}

# Shown in the desk, grouped by where people bank. IBKR covers all three.
REGIONS = {
    "us": ("United States", ["alpaca", "schwab", "ibkr", "tradier"]),
    "ca": ("Canada", ["questrade", "ibkr"]),
    "intl": ("International", ["ibkr", "saxo", "trading212"]),
}

# Large brokers people ask about that have no public trading API today.
NO_PUBLIC_API = {
    "us": ["Fidelity", "Vanguard", "Robinhood (equities)"],
    "ca": ["Wealthsimple", "TD Direct Investing", "RBC Direct Investing", "BMO InvestorLine"],
    "intl": ["eToro", "Hargreaves Lansdown"],
}

__all__ = ["Broker", "BrokerAccount", "BrokerError", "BrokerPosition", "OrderResult", "REGISTRY", "REGIONS", "NO_PUBLIC_API"]

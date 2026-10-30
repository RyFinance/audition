"""Callback: the paper stage, on your own paper broker.

Each run: compute today's target weights from the strategy file on the latest data, read the
paper account, turn the difference into order tickets (whole shares, a cash buffer, tiny
trades skipped), and optionally send them as market-on-open orders, which is exactly how
Audition's backtest fills. Every run also snapshots the account so the live track record can be
compared with the backtest's own distribution (the cone).

Brokers: Alpaca paper (paper-api.alpaca.markets only; there is no live endpoint in this file),
a webhook to your own engine, or "manual" (no broker: you type equity and positions)."""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod, strategy as strat_mod, universes
from .regions import REGIONS, infer_region
from .sources import creds

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "callback"
PAPER = "https://paper-api.alpaca.markets"


def _http(method, url, headers=None, body=None, form=False, insecure=False, timeout=20):
    data = None
    h = {"User-Agent": "audition/0.2", **(headers or {})}
    if body is not None:
        if form:
            data = urllib.parse.urlencode(body).encode()
            h.setdefault("Content-Type", "application/x-www-form-urlencoded")
        else:
            data = json.dumps(body).encode()
            h.setdefault("Content-Type", "application/json")
    ctx = None
    if insecure:  # IBKR's local gateway uses a self-signed certificate on localhost only
        import ssl
        ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, method=method, headers=h, data=data)
    try:
        raw = urllib.request.urlopen(req, timeout=timeout, context=ctx).read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{e.code} {e.reason}: {e.read()[:300].decode('utf-8', 'ignore')}") from None
    return json.loads(raw) if raw.strip() else {}


def _crypto_pair(sym: str) -> str:
    return sym.replace("-USD", "USDT") if sym.endswith("-USD") else sym


def _yahoo_from_pair(pair: str) -> str:
    return pair[:-4] + "-USD" if pair.endswith("USDT") else pair


def _split_flip(t):
    """An order that crosses zero (long to short) as two legs: close, then open."""
    a, b = t["from_shares"], t["to_shares"]
    if a * b >= 0:
        return [(t["shares"], a)]
    return [(-a, a), (b, 0.0)]


class Broker:
    key = ""
    fractional = False
    required = ()

    def __init__(self):
        from .sources import BROKERS
        self.cfg = creds(self.key)
        self.label = BROKERS[self.key]["label"]
        missing = [f for f in self.required if not self.cfg.get(f)]
        if missing:
            raise RuntimeError(f"{self.label}: add {', '.join(missing)} under Paper trading → Your paper account")

    def describe(self) -> str:
        return f"{self.label}: equity ${self.equity():,.0f}, {len(self.positions())} open position(s)"

    def fills(self, after: str) -> list:
        return []


class AlpacaPaper(Broker):
    key = "alpaca_paper"
    required = ("key_id", "secret")

    def _req(self, method, path, body=None):
        return _http(method, PAPER + path, {"APCA-API-KEY-ID": self.cfg["key_id"], "APCA-API-SECRET-KEY": self.cfg["secret"]}, body)

    def account(self):
        return self._req("GET", "/v2/account")

    def equity(self):
        return float(self.account()["equity"])

    def positions(self) -> dict:
        return {p["symbol"]: float(p["qty"]) for p in self._req("GET", "/v2/positions")}

    def fills(self, after: str) -> list:
        return self._req("GET", f"/v2/orders?status=closed&after={after}T00:00:00Z&limit=500&direction=asc")

    def send(self, tickets: list[dict]) -> list[dict]:
        out = []
        for t in tickets:
            body = dict(symbol=t["symbol"], qty=str(abs(int(t["shares"]))), side="buy" if t["shares"] > 0 else "sell",
                        type="market", time_in_force="opg", client_order_id=t["id"])
            try:
                o = self._req("POST", "/v2/orders", body)
                out.append(dict(symbol=t["symbol"], status=o.get("status"), id=o.get("id")))
            except Exception as e:
                out.append(dict(symbol=t["symbol"], status="rejected", error=str(e)[:200]))
        return out


class TradierSandbox(Broker):
    key = "tradier_sandbox"
    required = ("token", "account_id")
    BASE = "https://sandbox.tradier.com/v1"

    def _req(self, method, path, body=None):
        return _http(method, self.BASE + path, {"Authorization": f"Bearer {self.cfg['token']}", "Accept": "application/json"},
                     body, form=True)

    def equity(self):
        b = self._req("GET", f"/accounts/{self.cfg['account_id']}/balances")["balances"]
        return float(b.get("total_equity") or 0)

    def positions(self) -> dict:
        p = (self._req("GET", f"/accounts/{self.cfg['account_id']}/positions").get("positions") or {})
        p = p.get("position", []) if isinstance(p, dict) else []
        p = [p] if isinstance(p, dict) else p
        return {x["symbol"]: float(x["quantity"]) for x in p}

    def send(self, tickets):
        out = []
        for t in tickets:
            for qty, held in _split_flip(t):
                side = ("buy_to_cover" if held < 0 else "buy") if qty > 0 else ("sell" if held > 0 else "sell_short")
                try:
                    o = self._req("POST", f"/accounts/{self.cfg['account_id']}/orders",
                                  dict(**{"class": "equity"}, symbol=t["symbol"], side=side, quantity=int(abs(qty)),
                                       type="market", duration="day", tag=t["id"][:40].replace("_", "-")))
                    out.append(dict(symbol=t["symbol"], side=side, status=(o.get("order") or {}).get("status"),
                                    id=(o.get("order") or {}).get("id")))
                except Exception as e:
                    out.append(dict(symbol=t["symbol"], side=side, status="rejected", error=str(e)[:200]))
        return out


class IBKRPaper(Broker):
    key = "ibkr_paper"
    required = ()

    def __init__(self):
        super().__init__()
        gw = (self.cfg.get("gateway") or "https://localhost:5000").rstrip("/")
        host = urllib.parse.urlparse(gw).hostname
        if host not in ("localhost", "127.0.0.1"):
            raise RuntimeError("the IBKR gateway must run on this computer (localhost)")
        self.base = gw + "/v1/api"
        self.acct = self.cfg.get("account_id")

    def _req(self, method, path, body=None):
        return _http(method, self.base + path, body=body, insecure=True)

    def _account(self):
        if not self.acct:
            accts = self._req("GET", "/portfolio/accounts")
            self.acct = accts[0]["id"]
        if not str(self.acct).startswith("D"):
            raise RuntimeError(f"account {self.acct} isn't a paper account (paper ids start with D); Audition only trades paper")
        return self.acct

    def equity(self):
        s = self._req("GET", f"/portfolio/{self._account()}/summary")
        return float((s.get("netliquidation") or {}).get("amount") or 0)

    def positions(self) -> dict:
        out, page = {}, 0
        while True:
            rows = self._req("GET", f"/portfolio/{self._account()}/positions/{page}")
            if not rows:
                break
            for r in rows:
                out[r.get("ticker") or r.get("contractDesc")] = float(r["position"])
            if len(rows) < 100:
                break
            page += 1
        return out

    def _conid(self, sym):
        r = self._req("GET", f"/iserver/secdef/search?symbol={urllib.parse.quote(sym)}")
        return int(r[0]["conid"])

    def send(self, tickets):
        acct, out = self._account(), []
        for t in tickets:
            try:
                o = self._req("POST", f"/iserver/account/{acct}/orders", dict(orders=[dict(
                    conid=self._conid(t["symbol"]), orderType="MKT", side="BUY" if t["shares"] > 0 else "SELL",
                    quantity=abs(int(t["shares"])), tif="OPG", cOID=t["id"])]))
                # IBKR asks to confirm warnings ("are you sure?") before it accepts an order
                for _ in range(4):
                    if isinstance(o, list) and o and "id" in o[0] and "order_id" not in o[0]:
                        o = self._req("POST", f"/iserver/reply/{o[0]['id']}", dict(confirmed=True))
                    else:
                        break
                first = o[0] if isinstance(o, list) and o else o
                out.append(dict(symbol=t["symbol"], status=first.get("order_status"), id=first.get("order_id")))
            except Exception as e:
                out.append(dict(symbol=t["symbol"], status="rejected", error=str(e)[:200]))
        return out


class OandaPractice(Broker):
    key = "oanda_practice"
    required = ("token", "account_id")
    BASE = "https://api-fxpractice.oanda.com/v3"

    @staticmethod
    def inst(sym):
        s = sym[:-2] if sym.endswith("=X") else sym
        return f"{s[:3]}_{s[3:6]}" if len(s) == 6 else s

    @staticmethod
    def yahoo(inst):
        return inst.replace("_", "") + "=X" if len(inst) == 7 else inst

    def _req(self, method, path, body=None):
        return _http(method, self.BASE + path, {"Authorization": f"Bearer {self.cfg['token']}"}, body)

    def equity(self):
        return float(self._req("GET", f"/accounts/{self.cfg['account_id']}/summary")["account"]["NAV"])

    def positions(self) -> dict:
        out = {}
        for p in self._req("GET", f"/accounts/{self.cfg['account_id']}/openPositions").get("positions", []):
            out[self.yahoo(p["instrument"])] = float(p["long"]["units"]) + float(p["short"]["units"])
        return out

    def send(self, tickets):
        out = []
        for t in tickets:
            try:
                o = self._req("POST", f"/accounts/{self.cfg['account_id']}/orders", dict(order=dict(
                    type="MARKET", instrument=self.inst(t["symbol"]), units=str(int(t["shares"])), timeInForce="FOK",
                    positionFill="DEFAULT", clientExtensions=dict(id=t["id"][:40]))))
                tx = o.get("orderFillTransaction") or o.get("orderCreateTransaction") or {}
                out.append(dict(symbol=t["symbol"], status="filled" if "orderFillTransaction" in o else "created", id=tx.get("id")))
            except Exception as e:
                out.append(dict(symbol=t["symbol"], status="rejected", error=str(e)[:200]))
        return out


class BinanceTestnet(Broker):
    key = "binance_testnet"
    required = ("key", "secret")
    fractional = True
    BASE = "https://testnet.binance.vision/api"

    def _signed(self, method, path, params=None):
        import time as _t
        q = dict(params or {}, timestamp=int(_t.time() * 1000), recvWindow=5000)
        qs = urllib.parse.urlencode(q)
        sig = hmac.new(self.cfg["secret"].encode(), qs.encode(), hashlib.sha256).hexdigest()
        return _http(method, f"{self.BASE}{path}?{qs}&signature={sig}", {"X-MBX-APIKEY": self.cfg["key"]})

    def _prices(self):
        return {p["symbol"]: float(p["price"]) for p in _http("GET", self.BASE + "/v3/ticker/price")}

    def _balances(self):
        return {b["asset"]: float(b["free"]) + float(b["locked"]) for b in self._signed("GET", "/v3/account")["balances"]}

    def equity(self):
        px, total = self._prices(), 0.0
        for asset, q in self._balances().items():
            if q <= 0:
                continue
            total += q if asset in ("USDT", "USDC", "FDUSD", "BUSD") else q * px.get(asset + "USDT", 0.0)
        return total

    def positions(self) -> dict:
        return {a + "-USD": q for a, q in self._balances().items() if q > 0 and a not in ("USDT", "USDC", "FDUSD", "BUSD")}

    def send(self, tickets):
        info = _http("GET", self.BASE + "/v3/exchangeInfo")
        steps = {}
        for sy in info.get("symbols", []):
            for f in sy["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    steps[sy["symbol"]] = float(f["stepSize"])
        out = []
        for t in tickets:
            pair = _crypto_pair(t["symbol"])
            step = steps.get(pair, 1e-6)
            qty = math.floor(abs(t["shares"]) / step) * step
            if qty <= 0:
                continue
            try:
                o = self._signed("POST", "/v3/order", dict(symbol=pair, side="BUY" if t["shares"] > 0 else "SELL",
                                                            type="MARKET", quantity=f"{qty:.8f}".rstrip("0").rstrip("."),
                                                            newClientOrderId=t["id"][:36]))
                out.append(dict(symbol=t["symbol"], status=o.get("status"), id=o.get("orderId")))
            except Exception as e:
                out.append(dict(symbol=t["symbol"], status="rejected", error=str(e)[:200]))
        return out


class BybitTestnet(Broker):
    key = "bybit_testnet"
    required = ("key", "secret")
    fractional = True
    BASE = "https://api-testnet.bybit.com"

    def _req(self, method, path, params=None, body=None):
        import time as _t
        ts, rw = str(int(_t.time() * 1000)), "5000"
        payload = json.dumps(body) if body is not None else urllib.parse.urlencode(params or {})
        sig = hmac.new(self.cfg["secret"].encode(), (ts + self.cfg["key"] + rw + payload).encode(), hashlib.sha256).hexdigest()
        h = {"X-BAPI-API-KEY": self.cfg["key"], "X-BAPI-TIMESTAMP": ts, "X-BAPI-RECV-WINDOW": rw, "X-BAPI-SIGN": sig}
        url = self.BASE + path + (f"?{payload}" if body is None and payload else "")
        js = _http(method, url, h, body)
        if js.get("retCode") not in (0, None):
            raise RuntimeError(f"Bybit {js.get('retCode')}: {js.get('retMsg')}")
        return js.get("result", {})

    def _wallet(self):
        return self._req("GET", "/v5/account/wallet-balance", dict(accountType="UNIFIED"))["list"][0]

    def equity(self):
        return float(self._wallet().get("totalEquity") or 0)

    def positions(self) -> dict:
        return {c["coin"] + "-USD": float(c["walletBalance"]) for c in self._wallet().get("coin", [])
                if c["coin"] not in ("USDT", "USDC") and float(c["walletBalance"] or 0) > 0}

    def send(self, tickets):
        out = []
        for t in tickets:
            qty = round(abs(t["shares"]), 6)
            if qty <= 0:
                continue
            try:
                o = self._req("POST", "/v5/order/create", body=dict(
                    category="spot", symbol=_crypto_pair(t["symbol"]), side="Buy" if t["shares"] > 0 else "Sell",
                    orderType="Market", qty=f"{qty}", marketUnit="baseCoin", orderLinkId=t["id"][:36]))
                out.append(dict(symbol=t["symbol"], status="submitted", id=o.get("orderId")))
            except Exception as e:
                out.append(dict(symbol=t["symbol"], status="rejected", error=str(e)[:200]))
        return out


class Webhook(Broker):
    key = "webhook"
    required = ("url",)

    def describe(self):
        return f"Will POST orders to {self.cfg['url']}"

    def equity(self):
        raise RuntimeError("a webhook can't report equity; Audition uses the equity you type")

    def positions(self):
        return {}

    def send(self, tickets, meta=None):
        return _webhook_send(dict(meta or {}, orders=[dict(symbol=t["symbol"], qty=t["shares"], type="market_on_open",
                                                          client_order_id=t["id"]) for t in tickets]))


BROKER_CLASSES = {c.key: c for c in (AlpacaPaper, TradierSandbox, IBKRPaper, OandaPractice, BinanceTestnet, BybitTestnet, Webhook)}


def broker(name: str) -> Broker:
    if name not in BROKER_CLASSES:
        raise RuntimeError(f"unknown broker {name}")
    return BROKER_CLASSES[name]()


def _webhook_send(payload: dict) -> dict:
    c = creds("webhook")
    if not c.get("url"):
        raise RuntimeError("no webhook URL; add it under Paper trading → Your paper account")
    body = json.dumps(payload).encode()
    h = {"Content-Type": "application/json", "User-Agent": "audition/0.1"}
    if c.get("secret"):
        h["X-Audition-Signature"] = hmac.new(c["secret"].encode(), body, hashlib.sha256).hexdigest()
    r = urllib.request.urlopen(urllib.request.Request(c["url"], data=body, headers=h, method="POST"), timeout=20)
    return dict(status=r.status)


def targets(path: str) -> dict:
    """Run the strategy on data through the latest close; the last row is tomorrow's open order."""
    strat = strat_mod.load(path)
    uni = strat.universe
    start = (pd.Timestamp.today() - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
    elig = None
    if uni == "sp500":
        tick = sorted(universes.sp500().current())
        region = "us"
    elif isinstance(uni, str) and uni.startswith("byo:"):
        raise RuntimeError("Callback needs live prices; uploaded datasets don't update")
    elif isinstance(uni, dict):
        region = next(iter(uni))
        tick = list(uni[region])
    elif isinstance(uni, (list, tuple)):
        tick = list(uni)
        region = infer_region(tick)
    else:
        region = "us"
        tick = REGIONS["us"]["universe"]
    src = strat.data_source if not strat.data_source.startswith("byo:") else "yahoo"
    bars = data_mod.load(tick, start, source=src)
    if uni == "sp500":
        elig = pd.DataFrame(True, index=bars.index, columns=bars.tickers)
        bars = bars.replace(eligible=elig)
    cfg = REGIONS[region]
    bench = data_mod.load([cfg["bench"]], start).close.iloc[:, 0].reindex(bars.index).ffill()
    w = strat_mod.run(strat, bars, bench, region)
    last = w.iloc[-1].fillna(0.0)
    return dict(strategy=strat.name, sha=strat.sha256[:12], asof=str(w.index[-1].date()), region=region,
                weights={t: float(x) for t, x in last.items() if abs(x) > 1e-6},
                prices={t: float(bars.close[t].iloc[-1]) for t in bars.tickers if np.isfinite(bars.close[t].iloc[-1])},
                raw_prices={t: float(bars.raw_close[t].iloc[-1]) for t in bars.tickers
                            if np.isfinite(bars.raw_close[t].iloc[-1])})


def tickets(tg: dict, equity: float, positions: dict, buffer: float = 0.02, min_trade: float = 0.0025,
            fractional: bool = False) -> list[dict]:
    """Orders (whole shares, or fractional units for crypto) that move `positions` to the target weights."""
    px = tg["raw_prices"]
    out = []
    names = set(tg["weights"]) | {s for s, q in positions.items() if q}
    invest = equity * (1 - buffer)
    for s in sorted(names):
        p = px.get(s)
        if not p:
            continue
        raw = abs(tg["weights"].get(s, 0.0)) * invest / p
        want = (math.floor(raw * 1e6) / 1e6 if fractional else math.floor(raw)) * (1 if tg["weights"].get(s, 0.0) >= 0 else -1)
        have = positions.get(s, 0.0)
        d = round(want - have, 6) if fractional else int(want - have)
        if d == 0 or abs(d * p) < min_trade * equity and want != 0:
            continue
        out.append(dict(symbol=s, shares=d, price=p, value=d * p, target_weight=tg["weights"].get(s, 0.0),
                        from_shares=have, to_shares=want,
                        id=f"aud-{tg['sha'][:6]}-{tg['asof']}-{s}"[:48]))
    return out


def _state_path(name: str) -> Path:
    import re
    return STATE / (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".json")


def load_state(name: str) -> dict:
    p = _state_path(name)
    return json.loads(p.read_text()) if p.exists() else dict(name=name, snapshots=[], sends=[])


def _save_state(st: dict):
    STATE.mkdir(exist_ok=True)
    _state_path(st["name"]).write_text(json.dumps(st, indent=2, default=str))


def run(path: str, broker_name: str = "manual", send: bool = False, equity: float | None = None,
        positions: dict | None = None) -> dict:
    tg = targets(path)
    b = None
    if broker_name not in ("manual", "webhook"):
        b = broker(broker_name)
        equity = b.equity()
        positions = b.positions()
    equity = float(equity or 100_000)
    positions = positions or {}
    tix = tickets(tg, equity, positions, fractional=bool(b and b.fractional))
    st = load_state(tg["strategy"])
    today = dt.date.today().isoformat()
    if b is not None:
        snap = dict(date=today, equity=equity, positions=positions, broker=broker_name)
        st["snapshots"] = [x for x in st["snapshots"] if x["date"] != today] + [snap]
    result = dict(targets=tg, equity=equity, positions=positions, tickets=tix, broker=broker_name, sent=None)
    if send:
        if broker_name == "manual":
            raise RuntimeError("manual mode has no broker to send to")
        if broker_name == "webhook":
            result["sent"] = Webhook().send(tix, dict(strategy=tg["strategy"], asof=tg["asof"], equity=equity))
        else:
            result["sent"] = b.send(tix)
        st["sends"].append(dict(date=today, asof=tg["asof"], broker=broker_name, orders=len(tix), result=result["sent"]))
    _save_state(st)
    result["track"] = track(tg["strategy"], path)
    return result


def track(name: str, path: str | None = None) -> dict:
    """Live equity against the backtest's bootstrap cone, plus fill drift versus the official open."""
    st = load_state(name)
    snaps = st["snapshots"]
    out = dict(snapshots=len(snaps), sends=len(st["sends"]))
    if len(snaps) < 2:
        out["message"] = "The cone appears after two snapshots (two trading days of Callback runs)."
        return out
    eq = pd.Series({pd.Timestamp(s["date"]): s["equity"] for s in snaps}).sort_index()
    live = eq / eq.iloc[0] - 1
    out["live"] = [[str(d.date()), float(x)] for d, x in live.items()]
    # cone from the saved audit's daily returns, if this strategy has a report
    rep = ROOT / "reports"
    hist = None
    for f in rep.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("strategy") == name and d.get("daily_returns"):
            hist = np.array(d["daily_returns"], dtype=float)
            break
    if hist is not None and len(hist) > 60:
        n = len(live)
        rng = np.random.default_rng(1)
        paths = np.cumprod(1 + rng.choice(hist, size=(4000, n - 1)), axis=1) - 1
        q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        out["cone"] = {k: [0.0] + q[i].tolist() for i, k in enumerate(("p5", "p25", "p50", "p75", "p95"))}
        last = float(live.iloc[-1])
        pct = float((paths[:, -1] <= last).mean())
        out["percentile"] = pct
        out["message"] = (f"Live {last:+.1%} over {n - 1} trading days sits at the {pct:.0%} percentile of the backtest's "
                          "own outcomes." + (" Below the 5th: the live strategy isn't behaving like its backtest."
                                             if pct < 0.05 else ""))
    else:
        out["message"] = "Re-audit this strategy to draw the backtest cone."
    try:
        if snaps[-1].get("broker") == "alpaca_paper":
            fills = broker("alpaca_paper").fills(snaps[0]["date"])
            drift = []
            for o in fills:
                if not o.get("filled_avg_price") or not o.get("filled_at"):
                    continue
                d = pd.Timestamp(o["filled_at"]).tz_convert("America/New_York").tz_localize(None).normalize()
                df = data_mod._load_one(o["symbol"], "2020-01-01", max_age_h=4)
                if df is None or d not in df.index:
                    continue
                opn = float(df.loc[d, "open"])
                side = 1 if o["side"] == "buy" else -1
                drift.append(side * (float(o["filled_avg_price"]) / opn - 1) * 1e4)
            if drift:
                out["fill_drift_bps"] = float(np.mean(drift))
                out["fills_checked"] = len(drift)
    except Exception as e:
        out["fill_error"] = str(e)[:200]
    return out

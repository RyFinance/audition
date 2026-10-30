"""Callback: the paper stage, on your own paper broker.

Each run: compute today's target weights from the strategy file on the latest data, read the
paper account, turn the difference into order tickets (whole shares, a cash buffer, tiny
trades skipped), and optionally send them as market orders to the paper account. Run it after
the close and they queue for the next open, which is how Audition's backtest fills. Every run also snapshots the account so the live track record can be
compared with the backtest's own distribution (the cone).

Brokers: the adapters in audition/brokers (shared with Meridian): Alpaca, Interactive Brokers, Tradier,
Saxo and Trading 212 on their paper / sandbox / demo environments; Schwab and Questrade read-only.
Plus a webhook to your own engine, or "manual" (no broker: you type equity and positions)."""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod, strategy as strat_mod, universes
from .regions import REGIONS, infer_region
from .sources import creds, load_config, save_config

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "callback"
PAPER = "https://paper-api.alpaca.markets"


def _split_flip(t):
    """An order that crosses zero (long to short) as two legs: close, then open."""
    a, b = t["from_shares"], t["to_shares"]
    if a * b >= 0:
        return [(t["shares"], a)]
    return [(-a, a), (b, 0.0)]


def _side(qty, held):
    if qty > 0:
        return "buy_to_cover" if held < 0 else "buy"
    return "sell" if held > 0 else "sell_short"


class PaperBroker:
    """Audition's wrapper around the broker adapters (the same ones Meridian uses).

    Orders only ever go to a paper / sandbox / demo environment. A broker without one
    (Schwab) or without API orders (Questrade) is read-only: its positions and equity
    feed the order preview, and send() refuses."""

    fractional = False

    def __init__(self, key: str):
        from .brokers import REGISTRY
        if key not in REGISTRY:
            raise RuntimeError(f"unknown broker {key}")
        cls = REGISTRY[key]
        self.key, self.label = key, cls.name
        creds = {k: v for k, v in (load_config().get("paper_" + key) or {}).items() if v}
        missing = [f.name for f in cls.fields if not f.optional and not creds.get(f.name)]
        if missing:
            raise RuntimeError(f"{cls.name}: add {', '.join(missing)} under Paper trading → Your paper account")
        env = "paper" if "paper" in cls.envs else cls.envs[0]
        self.read_only = env != "paper" or "paper" not in cls.can_trade
        self.adapter = cls(creds, env)
        if key == "questrade":  # refresh tokens are single-use: keep the new one
            def keep(tok):
                cfg = load_config()
                cfg.setdefault("paper_questrade", {})["refresh_token"] = tok
                save_config(cfg)
            self.adapter.on_refresh = keep
        self._acct = None

    def _account(self):
        if self._acct is None:
            self._acct = self.adapter.account()
        return self._acct

    def equity(self) -> float:
        return float(self._account().equity or 0.0)

    def positions(self) -> dict:
        return {p.symbol: float(p.qty) for p in self._account().positions if p.qty}

    def describe(self) -> str:
        a = self._account()
        mode = "read-only" if self.read_only else "paper"
        return f"{self.label} ({mode}): equity {a.currency} {float(a.equity or 0):,.0f}, {len(a.positions)} position(s)"

    def fills(self, after: str) -> list:
        if self.key != "alpaca":
            return []
        return self.adapter._get("/v2/orders", status="closed", after=f"{after}T00:00:00Z", limit=500, direction="asc")

    def send(self, tickets: list[dict]) -> list[dict]:
        if self.read_only:
            raise RuntimeError(f"{self.label} has no paper trading through its API; Audition only reads it")
        out = []
        for t in tickets:
            for qty, held in _split_flip(t):
                side = _side(qty, held)
                try:
                    r = self.adapter.submit_market(t["symbol"], side, int(abs(qty)))
                    out.append(dict(symbol=t["symbol"], side=side, qty=int(abs(qty)), status=r.status, id=r.order_id))
                except Exception as e:
                    out.append(dict(symbol=t["symbol"], side=side, qty=int(abs(qty)), status="rejected", error=str(e)[:200]))
        return out


class Webhook:
    key, label, fractional, read_only = "webhook", "Webhook", False, False

    def __init__(self):
        self.cfg = creds("webhook")
        if not self.cfg.get("url"):
            raise RuntimeError("Webhook: add a URL under Paper trading → Your paper account")

    def describe(self):
        return f"Will POST orders to {self.cfg['url']}"

    def send(self, tickets, meta=None):
        return _webhook_send(dict(meta or {}, orders=[dict(symbol=t["symbol"], qty=t["shares"], type="market",
                                                          client_order_id=t["id"]) for t in tickets]))


def broker(name: str):
    """name: "webhook" or a broker id ("alpaca", "ibkr", ...); the "paper_" config prefix is accepted too."""
    return Webhook() if name == "webhook" else PaperBroker(name.removeprefix("paper_"))


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
        if snaps[-1].get("broker", "").removeprefix("paper_") == "alpaca":
            fills = broker("alpaca").fills(snaps[0]["date"])
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

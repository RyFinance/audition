"""Where prices come from: Yahoo (default), your own uploaded files (BYO), or a data provider
you have a key for (Polygon, Tiingo, Alpaca).

A strategy picks its source with DATA = "polygon" (or "tiingo", "alpaca", "yahoo"), or
UNIVERSE = "byo:<name>" to run on an uploaded dataset. Keys live in config/connections.json
(chmod 600, git-ignored) and are never copied into reports.

Every fetcher returns the same frame as the Yahoo path: daily rows, columns
open/high/low/close/adj_close/volume, where close is split-adjusted and adj_close also
includes dividends."""

from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "connections.json"
BYO = ROOT / "data" / "byo"
PROVIDERS = {
    "yahoo": dict(label="Yahoo Finance", fields=[], note="Free, no key. Adjusted daily bars back to the 1990s."),
    "polygon": dict(label="Polygon.io", fields=["key"], note="Split-adjusted daily aggregates; dividends not included."),
    "tiingo": dict(label="Tiingo", fields=["key"], note="Split- and dividend-adjusted EOD prices back to the 1960s."),
    "alpaca": dict(label="Alpaca market data", fields=["key_id", "secret", "feed"],
                   note="Daily bars from 2016; feed 'iex' is free, 'sip' needs a subscription."),
    "alphavantage": dict(label="Alpha Vantage", fields=["key"],
                         note="Free key at alphavantage.co. Adjusted daily series; free keys are rate-limited (25 calls a day)."),
    "fmp": dict(label="Financial Modeling Prep", fields=["key"],
                note="Split- and dividend-adjusted EOD prices for US and global listings."),
    "eodhd": dict(label="EODHD", fields=["key"],
                  note="70+ exchanges; Yahoo-style suffixes are mapped (AAPL -> AAPL.US, VOD.L -> VOD.LSE)."),
    "twelvedata": dict(label="Twelve Data", fields=["key"],
                       note="Stocks, ETFs, FX and crypto; up to 5,000 daily bars per symbol on the free plan."),
    "marketstack": dict(label="Marketstack", fields=["key"],
                        note="EOD prices for 70+ exchanges, with split- and dividend-adjusted closes."),
    "binance": dict(label="Binance (crypto)", fields=[],
                    note="Free, no key. Daily candles for USDT pairs; BTC-USD maps to BTCUSDT."),
}
NOTES = {  # Audition-specific setup notes (the adapters' own notes describe Meridian's live mode)
    "alpaca": "Use your paper key pair (starts with PK). Shorting needs margin on the paper account.",
    "schwab": "No paper environment, so Audition only reads balances and positions. Refresh tokens expire after 7 days.",
    "ibkr": "Run IBKR's Client Portal Gateway on this computer and log in with your paper username (account DU…).",
    "tradier": "Use the sandbox token from Tradier's developer dashboard and your sandbox account number.",
    "questrade": "Read-only: Questrade only allows order placement for approved partner apps. Practice-account token.",
    "saxo": "A 24-hour SIM token from developer.saxo. Orders go to Saxo's simulation environment.",
    "trading212": "Long-only. Use the API key from your Practice account (Settings → API).",
}


def _brokers() -> dict:
    """Paper brokers: Meridian's adapters (audition/brokers) plus a webhook. Config keys are paper_<id>."""
    from .brokers import REGISTRY
    region = {"us": "US", "ca": "Canada", "intl": "International"}
    out = {}
    for b in REGISTRY.values():
        paper = "paper" in b.envs and "paper" in b.can_trade
        mode = "paper orders" if paper else "read-only (no API paper trading)"
        out["paper_" + b.id] = dict(label=b.name, fields=[f.name for f in b.fields],
                                    placeholders={f.name: f.label + ("" if not f.optional else " (optional)") for f in b.fields},
                                    secret=[f.name for f in b.fields if f.secret],
                                    kind=" · ".join(region.get(r, r) for r in b.regions) + " · " + mode, note=NOTES.get(b.id, b.note))
    out["webhook"] = dict(label="Webhook", fields=["url", "secret"], kind="anything · paper orders",
                          placeholders={"url": "https://your-engine/orders", "secret": "signing secret (optional)"},
                          secret=["secret"],
                          note="POSTs the order list as JSON (HMAC-signed if you set a secret) to your own bot or paper engine.")
    return out


BROKERS = _brokers()


# --------------------------------------------------------------------------- config

def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except Exception:
        return {}


def save_config(cfg: dict):
    CONFIG.parent.mkdir(exist_ok=True)
    gi = CONFIG.parent / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n")
    CONFIG.write_text(json.dumps(cfg, indent=2))
    os.chmod(CONFIG, 0o600)


def masked() -> dict:
    """The config with secrets shown as ••••last4, safe to send to the page."""
    cfg = load_config()
    out = {}
    for name, vals in cfg.items():
        if not isinstance(vals, dict):
            out[name] = vals
            continue
        secret = set(BROKERS.get(name, {}).get("secret", [])) | {"key", "secret", "key_id", "token"}
        out[name] = {k: ("••••" + str(v)[-4:] if v and k in secret else v) for k, v in vals.items()}
    return out


def update_config(name: str, values: dict) -> dict:
    if name not in PROVIDERS and name not in BROKERS and name != "default_source":
        raise ValueError(f"unknown connection {name}")
    cfg = load_config()
    if name == "default_source":
        if values.get("value") not in PROVIDERS:
            raise ValueError("unknown provider")
        cfg["default_source"] = values["value"]
    else:
        cur = cfg.get(name, {})
        for k, v in values.items():
            if v is None or str(v).startswith("••••"):
                continue  # untouched masked field
            cur[k] = str(v).strip()
        if not any(cur.values()):
            cfg.pop(name, None)
        else:
            cfg[name] = cur
    save_config(cfg)
    return masked()


def creds(name: str) -> dict:
    c = load_config().get(name) or {}
    env = {"polygon": {"key": "POLYGON_API_KEY"}, "tiingo": {"key": "TIINGO_API_KEY"},
           "alpaca": {"key_id": "APCA_API_KEY_ID", "secret": "APCA_API_SECRET_KEY"}}.get(name, {})
    for k, var in env.items():
        c.setdefault(k, os.environ.get(var, ""))
    return c


def default_source() -> str:
    return load_config().get("default_source", "yahoo")


# --------------------------------------------------------------------------- fetchers

def _json(url: str, headers: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "audition/0.1", **(headers or {})})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _frame(rows, date_key, o, h, l, c, v, adj_c=None, adj_o=None, adj_h=None, adj_l=None) -> pd.DataFrame | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df[date_key], unit="ms" if np.issubdtype(df[date_key].dtype, np.number) else None, utc=True)
    out = pd.DataFrame({"open": df[o], "high": df[h], "low": df[l], "close": df[c], "volume": df[v]}).astype(float)
    out.index = pd.DatetimeIndex(idx.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize())
    if adj_c:
        # provider gives fully adjusted prices: keep raw close as close, adjusted as adj_close
        out["adj_close"] = df[adj_c].astype(float).to_numpy()
    else:
        out["adj_close"] = out["close"]
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_polygon(ticker: str, start: str = "1990-01-01") -> pd.DataFrame | None:
    key = creds("polygon").get("key")
    if not key:
        raise RuntimeError("no Polygon key; add it under Data")
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    url = (f"https://api.polygon.io/v2/aggs/ticker/{urllib.parse.quote(ticker)}/range/1/day/{start}/{end}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={key}")
    rows = []
    while url:
        js = _json(url)
        rows += js.get("results") or []
        nxt = js.get("next_url")
        url = f"{nxt}&apiKey={key}" if nxt else None
    return _frame(rows, "t", "o", "h", "l", "c", "v")


def fetch_tiingo(ticker: str, start: str = "1990-01-01") -> pd.DataFrame | None:
    key = creds("tiingo").get("key")
    if not key:
        raise RuntimeError("no Tiingo key; add it under Data")
    url = f"https://api.tiingo.com/tiingo/daily/{urllib.parse.quote(ticker.lower())}/prices?startDate={start}&token={key}"
    rows = _json(url)
    if not rows:
        return None
    df = _frame(rows, "date", "open", "high", "low", "close", "volume", adj_c="adjClose")
    # Audition wants split-adjusted (not dividend-adjusted) OHLC plus a fully adjusted close:
    # divide raw prices by every split that happened after each day
    raw = pd.DataFrame(rows)
    split = raw["splitFactor"].astype(float).iloc[::-1].cumprod().iloc[::-1].shift(-1, fill_value=1.0).to_numpy()
    for f in ("open", "high", "low", "close"):
        df[f] = df[f].to_numpy() / split
    df["volume"] = df["volume"].to_numpy() * split
    return df


def fetch_alpaca(ticker: str, start: str = "2016-01-01") -> pd.DataFrame | None:
    c = creds("alpaca")
    if not c.get("key_id") or not c.get("secret"):
        raise RuntimeError("no Alpaca keys; add them under Data")
    feed = c.get("feed") or "iex"
    hdr = {"APCA-API-KEY-ID": c["key_id"], "APCA-API-SECRET-KEY": c["secret"]}
    rows, tok = [], None
    base = (f"https://data.alpaca.markets/v2/stocks/{urllib.parse.quote(ticker)}/bars?timeframe=1Day"
            f"&adjustment=all&start={max(start, '2016-01-01')}&limit=10000&feed={feed}")
    while True:
        js = _json(base + (f"&page_token={tok}" if tok else ""), hdr)
        rows += js.get("bars") or []
        tok = js.get("next_page_token")
        if not tok:
            break
    return _frame(rows, "t", "o", "h", "l", "c", "v")


def _need(name, field="key"):
    v = creds(name).get(field)
    if not v:
        raise RuntimeError(f"no {PROVIDERS[name]['label']} key; add it under Price data")
    return v


def fetch_alphavantage(ticker: str, start: str = "1990-01-01") -> pd.DataFrame | None:
    key = _need("alphavantage")
    base = "https://www.alphavantage.co/query?outputsize=full&symbol=" + urllib.parse.quote(ticker) + "&apikey=" + key
    js = _json(base + "&function=TIME_SERIES_DAILY_ADJUSTED")
    series = js.get("Time Series (Daily)")
    adjusted = True
    if not series:  # adjusted series is premium on some plans: fall back to raw daily
        js = _json(base + "&function=TIME_SERIES_DAILY")
        series = js.get("Time Series (Daily)")
        adjusted = False
    if not series:
        raise RuntimeError(js.get("Note") or js.get("Information") or js.get("Error Message") or "no data")
    rows = [dict(date=d, o=float(v["1. open"]), h=float(v["2. high"]), l=float(v["3. low"]), c=float(v["4. close"]),
                 a=float(v["5. adjusted close"]) if adjusted else float(v["4. close"]),
                 v=float(v["6. volume"] if adjusted else v["5. volume"])) for d, v in series.items()]
    return _frame(rows, "date", "o", "h", "l", "c", "v", adj_c="a")


def fetch_fmp(ticker: str, start: str = "1990-01-01") -> pd.DataFrame | None:
    key = _need("fmp")
    q = f"symbol={urllib.parse.quote(ticker)}&from={start}&apikey={key}"
    raw = _json(f"https://financialmodelingprep.com/stable/historical-price-eod/full?{q}")
    adj = _json(f"https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted?{q}")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(str(raw)[:200] if raw else "no data")
    a = {r["date"]: r.get("adjClose") for r in adj} if isinstance(adj, list) else {}
    for r in raw:
        r["adj"] = a.get(r["date"]) or r["close"]
    return _frame(raw, "date", "open", "high", "low", "close", "volume", adj_c="adj")


EODHD_SUFFIX = {"L": "LSE", "T": "TSE", "DE": "XETRA", "PA": "PA", "AS": "AS", "MI": "MI", "MC": "MC", "SW": "SW",
                "TO": "TO", "HK": "HK", "AX": "AU"}


def _eodhd_symbol(t: str) -> str:
    if t.startswith("^"):
        return t[1:] + ".INDX"
    if t.endswith("-USD"):
        return t + ".CC"
    if t.endswith("=X"):
        return t[:-2] + ".FOREX"
    if "." in t:
        base, suf = t.rsplit(".", 1)
        return f"{base}.{EODHD_SUFFIX.get(suf, suf)}"
    return t + ".US"


def fetch_eodhd(ticker: str, start: str = "1990-01-01") -> pd.DataFrame | None:
    key = _need("eodhd")
    rows = _json(f"https://eodhd.com/api/eod/{urllib.parse.quote(_eodhd_symbol(ticker))}?api_token={key}&fmt=json&from={start}")
    if not isinstance(rows, list):
        raise RuntimeError(str(rows)[:200])
    df = _frame(rows, "date", "open", "high", "low", "close", "volume", adj_c="adjusted_close")
    return _split_adjust(df)


def _split_adjust(df):
    """Raw OHLC to split-adjusted using the adjusted/raw ratio's split-sized steps only."""
    if df is None or df.empty:
        return df
    ratio = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], np.nan).ffill()
    step = ratio / ratio.shift(-1)
    split = step.where((step - 1).abs() > 0.2, 1.0).fillna(1.0)   # dividends move it < 20%, splits much more
    fac = split[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
    for f in ("open", "high", "low", "close"):
        df[f] = df[f] * fac
    df["volume"] = df["volume"] / fac
    return df


def fetch_twelvedata(ticker: str, start: str = "1990-01-01") -> pd.DataFrame | None:
    key = _need("twelvedata")
    sym = ticker[:-2][:3] + "/" + ticker[:-2][3:] if ticker.endswith("=X") else ticker.replace("-", "/")
    base = (f"https://api.twelvedata.com/time_series?symbol={urllib.parse.quote(sym)}&interval=1day&outputsize=5000"
            f"&start_date={start}&order=asc&apikey={key}")
    split = _json(base + "&adjust=splits")
    full = _json(base + "&adjust=all")
    if split.get("status") == "error":
        raise RuntimeError(split.get("message", "error"))
    a = {v["datetime"]: v["close"] for v in full.get("values", [])}
    rows = split.get("values", [])
    for r in rows:
        r["adj"] = float(a.get(r["datetime"], r["close"]))
        r.setdefault("volume", 0)
    return _frame(rows, "datetime", "open", "high", "low", "close", "volume", adj_c="adj")


def fetch_marketstack(ticker: str, start: str = "1990-01-01") -> pd.DataFrame | None:
    key = _need("marketstack")
    rows, offset = [], 0
    while True:
        js = _json(f"https://api.marketstack.com/v2/eod?access_key={key}&symbols={urllib.parse.quote(ticker)}"
                   f"&date_from={start}&limit=1000&offset={offset}")
        if "error" in js:
            raise RuntimeError(js["error"].get("message", "error"))
        rows += js.get("data") or []
        pg = js.get("pagination") or {}
        offset += pg.get("count", 0)
        if not pg.get("count") or offset >= pg.get("total", 0):
            break
    for r in rows:
        r["adj"] = r.get("adj_close") or r["close"]
    df = _frame(rows, "date", "open", "high", "low", "close", "volume", adj_c="adj")
    return _split_adjust(df)


def _binance_symbol(t: str) -> str:
    return t.replace("-USD", "USDT").replace("-", "") if t.endswith("-USD") else t


def fetch_binance(ticker: str, start: str = "2017-01-01") -> pd.DataFrame | None:
    sym = _binance_symbol(ticker)
    t0 = int(pd.Timestamp(max(start, "2017-01-01")).timestamp() * 1000)
    rows = []
    while True:
        k = _json(f"https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval=1d&startTime={t0}&limit=1000")
        if not k:
            break
        rows += [dict(t=x[0], o=float(x[1]), h=float(x[2]), l=float(x[3]), c=float(x[4]), v=float(x[7])) for x in k]
        if len(k) < 1000:
            break
        t0 = k[-1][0] + 86_400_000
    df = _frame(rows, "t", "o", "h", "l", "c", "v")
    if df is not None:  # crypto days are UTC; _frame shifted them to New York
        df.index = pd.DatetimeIndex([pd.Timestamp(r["t"], unit="ms").normalize() for r in rows])
    return df


FETCHERS = {"polygon": fetch_polygon, "tiingo": fetch_tiingo, "alpaca": fetch_alpaca, "alphavantage": fetch_alphavantage,
            "fmp": fetch_fmp, "eodhd": fetch_eodhd, "twelvedata": fetch_twelvedata, "marketstack": fetch_marketstack,
            "binance": fetch_binance}


def test_connection(name: str) -> dict:
    t0 = time.time()
    try:
        if name in FETCHERS:
            sym = "BTC-USD" if name == "binance" else "SPY"
            df = FETCHERS[name](sym, (pd.Timestamp.today() - pd.Timedelta(days=20)).strftime("%Y-%m-%d"))
            if df is None or df.empty:
                return dict(ok=False, message=f"Connected, but no {sym} bars came back.")
            return dict(ok=True, message=f"{sym} last close {df['close'].iloc[-1]:,.2f} on {df.index[-1].date()} "
                                         f"({time.time() - t0:.1f}s)")
        if name in BROKERS:
            from .callback import broker
            b = broker(name)
            return dict(ok=True, message=b.describe())
        return dict(ok=True, message="Yahoo needs no key.")
    except Exception as e:
        return dict(ok=False, message=f"{type(e).__name__}: {e}"[:300])


# --------------------------------------------------------------------------- bring your own data

ALIASES = {
    "date": ["date", "datetime", "time", "timestamp", "day", "dt"],
    "ticker": ["ticker", "symbol", "sym", "asset", "instrument", "code", "name"],
    "open": ["open", "o"], "high": ["high", "h"], "low": ["low", "l"],
    "close": ["close", "c", "last", "price", "px_last"],
    "adj_close": ["adj close", "adj_close", "adjclose", "adjusted close", "adjusted_close", "close_adj"],
    "volume": ["volume", "vol", "v"],
}


def _norm_cols(df: pd.DataFrame) -> dict:
    m = {}
    low = {c: str(c).strip().lower() for c in df.columns}
    for want, names in ALIASES.items():
        for c, l in low.items():
            if l in names and c not in m.values():
                m[want] = c
                break
    return m


def parse_upload(name: str, content: bytes, filename: str = "") -> pd.DataFrame:
    """Long (date, ticker, ohlcv) or wide (date + one close column per ticker) CSV/Parquet/JSON.
    Returns a long frame: date, ticker, open, high, low, close, adj_close, volume."""
    fn = filename.lower()
    if fn.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(content))
    elif fn.endswith(".json"):
        df = pd.read_json(io.BytesIO(content))
    elif fn.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    if df.index.name and str(df.index.name).lower() in ALIASES["date"]:
        df = df.reset_index()
    m = _norm_cols(df)
    if "date" not in m:
        first = df.columns[0]
        try:
            pd.to_datetime(df[first].head(20))
            m["date"] = first
        except Exception:
            raise ValueError("couldn't find a date column")
    if "close" in m or "adj_close" in m:
        out = pd.DataFrame({"date": pd.to_datetime(df[m["date"]], utc=True).dt.tz_localize(None).dt.normalize()})
        out["ticker"] = df[m["ticker"]].astype(str).str.upper().str.strip() if "ticker" in m else re.sub(
            r"[^A-Z0-9.=^-]", "", Path(filename or name).stem.upper()) or "ASSET"
        for f in ("open", "high", "low", "close", "adj_close", "volume"):
            src = m.get(f)
            out[f] = pd.to_numeric(df[src], errors="coerce") if src else np.nan
        if out["close"].isna().all():
            out["close"] = out["adj_close"]
        if out["adj_close"].isna().all():
            out["adj_close"] = out["close"]
        for f in ("open", "high", "low"):
            out[f] = out[f].fillna(out["close"])
    else:
        # wide: every non-date numeric column is one ticker's close
        d = pd.to_datetime(df[m["date"]], utc=True).dt.tz_localize(None).dt.normalize()
        vals = df.drop(columns=[m["date"]]).apply(pd.to_numeric, errors="coerce")
        vals = vals.loc[:, vals.notna().any()]
        if vals.empty:
            raise ValueError("no price columns found")
        vals.index = d
        out = vals.stack().rename("close").reset_index()
        out.columns = ["date", "ticker", "close"]
        out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
        for f in ("open", "high", "low", "adj_close"):
            out[f] = out["close"]
        out["volume"] = np.nan
    out = out.dropna(subset=["date", "close"]).sort_values(["ticker", "date"])
    return out.reset_index(drop=True)


def quality(df: pd.DataFrame) -> dict:
    """A data checker for uploads: the problems that silently wreck a backtest."""
    issues = []
    per = []
    for t, g in df.groupby("ticker"):
        g = g.set_index("date").sort_index()
        n = len(g)
        dup = int(g.index.duplicated().sum())
        nonpos = int((g[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
        hl = int(((g["high"] < g["low"]) | (g["close"] > g["high"] * 1.0001) | (g["close"] < g["low"] * 0.9999)).sum())
        gi = g[~g.index.duplicated()]
        bd = np.diff(gi.index.values).astype("timedelta64[D]").astype(int) if n > 1 else np.array([])
        gaps = int((bd > 7).sum())
        r = gi["adj_close"].pct_change(fill_method=None)
        jumps = r[r.abs() > 0.4]
        split_like = []
        for d, x in jumps.items():
            ratio = 1 / (1 + x) if x < 0 else 1 + x
            for k in (2, 3, 4, 5, 8, 10, 20, 1.5):
                if abs(ratio - k) / k < 0.04:
                    split_like.append(str(d.date()))
                    break
        stale = int((gi["close"].diff() == 0).rolling(5).sum().ge(5).sum())
        no_vol = bool(gi["volume"].isna().all())
        row = dict(ticker=t, rows=n, start=str(gi.index[0].date()), end=str(gi.index[-1].date()),
                   duplicates=dup, nonpositive=nonpos, bad_hl=hl, gaps=gaps, jumps=len(jumps),
                   split_like=split_like[:5], stale_runs=stale, no_volume=no_vol)
        per.append(row)
    def tally(k, msg, level="warn"):
        bad = [p["ticker"] for p in per if (p[k] if not isinstance(p[k], list) else len(p[k]))]
        if bad:
            issues.append(dict(level=level, msg=msg.format(n=len(bad), names=", ".join(bad[:6]) + ("…" if len(bad) > 6 else ""))))
    tally("duplicates", "{n} ticker(s) have duplicate dates (kept the last): {names}")
    tally("nonpositive", "{n} ticker(s) have zero or negative prices: {names}", "err")
    tally("bad_hl", "{n} ticker(s) have high < low or a close outside the day's range: {names}", "err")
    tally("split_like", "{n} ticker(s) jump by a split-like ratio, so the series may be unadjusted: {names}", "err")
    tally("gaps", "{n} ticker(s) have gaps longer than a week: {names}")
    tally("stale_runs", "{n} ticker(s) repeat the same close 5+ days running (stale or filled data): {names}")
    tally("no_volume", "{n} ticker(s) have no volume, so liquidity costs fall back to the region floor: {names}")
    if df["open"].equals(df["close"]):
        issues.append(dict(level="warn", msg="Close prices only: fills at the next open use the close instead."))
    return dict(tickers=len(per), rows=int(len(df)), start=str(df["date"].min().date()),
                end=str(df["date"].max().date()), issues=issues, per_ticker=per[:400])


def save_byo(name: str, content: bytes, filename: str) -> dict:
    name = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_")[:40] or "dataset"
    df = parse_upload(name, content, filename)
    df = df.drop_duplicates(["ticker", "date"], keep="last")
    q = quality(df)
    BYO.mkdir(parents=True, exist_ok=True)
    df.to_pickle(BYO / f"{name}.pkl")
    meta = dict(name=name, file=filename, uploaded=time.strftime("%Y-%m-%d %H:%M"), **{k: q[k] for k in (
        "tickers", "rows", "start", "end", "issues")}, symbols=sorted(df["ticker"].unique().tolist())[:2000])
    (BYO / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return dict(meta=meta, quality=q)


def list_byo() -> list[dict]:
    out = []
    for p in sorted(BYO.glob("*.json")):
        try:
            m = json.loads(p.read_text())
            m.pop("symbols", None)
            out.append(m)
        except Exception:
            pass
    return out


def delete_byo(name: str):
    for ext in (".pkl", ".json"):
        p = BYO / f"{re.sub(r'[^a-z0-9_-]+', '_', name.lower())}{ext}"
        if p.exists():
            p.unlink()


def byo_frames(name: str) -> dict[str, pd.DataFrame]:
    p = BYO / f"{name}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"no uploaded dataset called {name!r}; upload it under Data")
    df = pd.read_pickle(p)
    out = {}
    for t, g in df.groupby("ticker"):
        g = g.set_index("date").sort_index()[["open", "high", "low", "close", "adj_close", "volume"]]
        out[t] = g[~g.index.duplicated(keep="last")]
    return out


def byo_symbols(name: str) -> list[str]:
    return sorted(byo_frames(name))

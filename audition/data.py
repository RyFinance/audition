"""Daily bars from Yahoo, cached per ticker. Returns split- and dividend-adjusted OHLCV
plus the raw (split-adjusted, not dividend-adjusted) close for the revision check."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "cache"
FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class Bars:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    raw_close: pd.DataFrame
    missing: list = field(default_factory=list)
    eligible: pd.DataFrame | None = None   # point-in-time membership (dates x tickers), if known

    @property
    def index(self):
        return self.close.index

    @property
    def tickers(self):
        return list(self.close.columns)

    def frames(self):
        return {f: getattr(self, f) for f in FIELDS}

    def slice(self, end_pos: int) -> "Bars":
        """Bars up to and including row end_pos."""
        s = slice(0, end_pos + 1)
        return Bars(**{f: getattr(self, f).iloc[s] for f in FIELDS},
                    raw_close=self.raw_close.iloc[s], missing=self.missing,
                    eligible=None if self.eligible is None else self.eligible.iloc[s])

    def until(self, end) -> "Bars":
        """Bars on or before `end` (the vault's rehearsal cut)."""
        n = int(self.index.searchsorted(pd.Timestamp(end), side="right")) - 1
        return self.slice(max(n, 0))

    def replace(self, **frames) -> "Bars":
        raw_close = frames.pop("raw_close", self.raw_close)
        eligible = frames.pop("eligible", self.eligible)
        d = {f: getattr(self, f) for f in FIELDS}
        d.update(frames)
        if eligible is not None:
            eligible = eligible.reindex(index=d["close"].index, columns=d["close"].columns, fill_value=False)
        return Bars(**d, raw_close=raw_close, missing=self.missing, eligible=eligible)


def _fetch(ticker: str, start: str) -> pd.DataFrame | None:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(ticker, start=start, auto_adjust=False, progress=False,
                         multi_level_index=False)
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")]


NO_DATA = CACHE / "_no_data.json"


def _no_data() -> dict:
    import json
    try:
        d = json.loads(NO_DATA.read_text())
    except Exception:
        return {}
    return {k: v for k, v in d.items() if time.time() - v < 7 * 86400}


def _mark_no_data(tickers):
    import json
    d = _no_data()
    d.update({t: time.time() for t in tickers})
    CACHE.mkdir(exist_ok=True)
    NO_DATA.write_text(json.dumps(d))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df[~df.index.duplicated(keep="last")]


def _load_one(ticker: str, start: str, max_age_h: float = 18.0, source: str = "yahoo") -> pd.DataFrame | None:
    CACHE.mkdir(exist_ok=True)
    folder = CACHE if source == "yahoo" else CACHE / source
    folder.mkdir(exist_ok=True)
    path = folder / f"{ticker.replace('^', '_').replace('/', '_')}.pkl"
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_h * 3600:
        return _clean(pd.read_pickle(path))
    if source != "yahoo":
        from .sources import FETCHERS
        df = FETCHERS[source](ticker)
        if df is not None and not df.empty:
            df = _clean(df)
            df.to_pickle(path)
        return df
    if ticker in _no_data():
        return None
    df = _fetch(ticker, "1990-01-01")
    if df is not None and not df.empty:
        df = _clean(df)
        df.to_pickle(path)
    else:
        _mark_no_data([ticker])
    return df


def _prefetch(tickers, max_age_h: float = 18.0, chunk: int = 100):
    """Download every uncached ticker in a few threaded batch calls instead of one call each."""
    import yfinance as yf

    CACHE.mkdir(exist_ok=True)
    todo = []
    skip = _no_data()
    for t in tickers:
        path = CACHE / f"{t.replace('^', '_')}.pkl"
        if not (path.exists() and (time.time() - path.stat().st_mtime) < max_age_h * 3600) and t not in skip:
            todo.append(t)
    got = set()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(part, start="1990-01-01", auto_adjust=False, progress=False,
                             group_by="ticker", threads=True)
        if df is None or df.empty:
            continue
        for t in part:
            if t not in df.columns.get_level_values(0):
                continue
            one = df[t].dropna(how="all")
            if one.empty:
                continue
            one = one.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
            one.index = pd.DatetimeIndex(one.index).tz_localize(None).normalize()
            _clean(one).to_pickle(CACHE / f"{t.replace('^', '_')}.pkl")
            got.add(t)
    _mark_no_data([t for t in todo if t not in got])


def load(tickers, start: str, end: str | None = None, source: str = "yahoo") -> Bars:
    """source: "yahoo", a provider name ("polygon", "tiingo", "alpaca") or "byo:<dataset>"."""
    raw = {}
    missing = []
    byo = None
    if source.startswith("byo:"):
        from .sources import byo_frames
        byo = byo_frames(source[4:])
    elif source == "yahoo" and len(tickers) > 25:
        _prefetch(list(dict.fromkeys(tickers)))
    for t in dict.fromkeys(tickers):
        if byo is not None:
            df = byo.get(t)
            if df is None:  # benchmark or FX not in the upload: fall back to Yahoo
                df = _load_one(t, start)
        else:
            try:
                df = _load_one(t, start, source=source)
            except Exception:
                df = None
            if df is None and source != "yahoo" and (t.startswith("^") or "=" in t):
                df = _load_one(t, start)  # indices, futures and FX stay on Yahoo
        if df is None or df.empty:
            missing.append(t)
            continue
        df = df.loc[start:end] if end else df.loc[start:]
        if len(df) < 30:
            missing.append(t)
            continue
        raw[t] = df
    if not raw:
        raise RuntimeError(f"no data for any of {list(tickers)}")

    idx = sorted(set().union(*[d.index for d in raw.values()]))
    idx = pd.DatetimeIndex(idx)

    def panel(col):
        return pd.DataFrame({t: d[col].reindex(idx) for t, d in raw.items()}, index=idx).astype(float)

    close_raw = panel("close")
    adj = panel("adj_close")
    factor = (adj / close_raw).replace([np.inf, -np.inf], np.nan)
    o, h, l = panel("open") * factor, panel("high") * factor, panel("low") * factor
    vol = panel("volume")
    # Yahoo marks some non-trading days with zero volume and a flat bar.
    return Bars(open=o, high=h, low=l, close=adj, volume=vol, raw_close=close_raw, missing=missing)


def fx_rate(pair: str, index: pd.DatetimeIndex) -> pd.Series | None:
    """USD per unit of local currency on the given calendar."""
    df = _load_one(pair, "2000-01-01")
    if df is None:
        return None
    return df["close"].reindex(index, method="ffill").ffill().bfill()


def risk_free(index: pd.DatetimeIndex, ppy: int) -> pd.Series:
    """Daily T-bill return on the given calendar (annual yield compounded per period)."""
    from .regions import RISK_FREE

    df = _load_one(RISK_FREE, "1990-01-01")
    if df is None:
        return pd.Series(0.0, index=index)
    y = (df["close"] / 100.0).reindex(index, method="ffill").ffill().fillna(0.0)
    return (1 + y) ** (1 / ppy) - 1

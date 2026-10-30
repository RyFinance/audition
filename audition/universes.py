"""Point-in-time S&P 500 membership, rebuilt from Wikipedia's constituents list and its dated
table of index changes (walked backwards from today), plus prices for every name that was a
member during the sample, including the ones removed since, wherever Yahoo still has them.

Two guards keep a recycled ticker symbol from importing the wrong company's prices:
- a removed name's prices are only used if Yahoo's history starts before the name joined
  (or before the sample starts), i.e. the series predates any later reuse of the symbol;
- they are cut off 90 days after the name left the index.
Names Yahoo can't price (bankruptcies, most acquisitions) stay in the membership count and are
reported as uncovered, so the point-in-time result is an upper bound, never silently clean.
"""

from __future__ import annotations

import io
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod

CACHE = data_mod.CACHE
SOURCES = {
    "constituents": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "changes": "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500",
}
UA = {"User-Agent": "audition-strategy-audit/0.1 (research tool)"}


def _sym(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    import re
    s = re.sub(r"[^A-Z0-9-]", "", str(x).strip().upper().replace(".", "-"))
    return s or None


def _get(name: str, max_age_days: float = 7.0) -> bytes:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"sp500_{name}.html"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        return path.read_bytes()
    try:
        req = urllib.request.Request(SOURCES[name], headers=UA)
        body = urllib.request.urlopen(req, timeout=45).read()
        path.write_bytes(body)
        return body
    except Exception:
        if path.exists():  # stale beats nothing
            return path.read_bytes()
        raise


def _tables():
    cons = changes = None
    for t in pd.read_html(io.BytesIO(_get("constituents"))):
        if "Symbol" in t.columns:
            cons = t
            break
    for t in pd.read_html(io.BytesIO(_get("changes"))):
        if any("Added" in str(c) for c in t.columns):
            changes = t
            break
    if cons is None or changes is None:
        raise RuntimeError("couldn't find the S&P 500 tables on Wikipedia")
    changes = changes.copy()
    changes.columns = ["_".join(dict.fromkeys(c)) if isinstance(c, tuple) else c for c in changes.columns]
    return cons, changes


@dataclass
class Membership:
    snapshots: dict                       # effective date -> set of members from that date on
    names: dict = field(default_factory=dict)    # ticker -> company name
    first_change: pd.Timestamp | None = None

    def on(self, date) -> set:
        keys = [k for k in self.snapshots if k <= date]
        return self.snapshots[max(keys)] if keys else self.snapshots[min(self.snapshots)]

    def frame(self, index: pd.DatetimeIndex, tickers) -> pd.DataFrame:
        dates = sorted(self.snapshots)
        pos = np.searchsorted(np.array(dates, dtype="datetime64[ns]"), index.values, side="right") - 1
        pos = np.clip(pos, 0, len(dates) - 1)
        cols = list(tickers)
        col_idx = {t: i for i, t in enumerate(cols)}
        snap_rows = []
        for d in dates:
            row = np.zeros(len(cols), dtype=bool)
            for t in self.snapshots[d]:
                i = col_idx.get(t)
                if i is not None:
                    row[i] = True
            snap_rows.append(row)
        mat = np.array(snap_rows)[pos]
        return pd.DataFrame(mat, index=index, columns=cols)

    def ever(self, start, end=None) -> set:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end) if end else pd.Timestamp.today()
        out = set(self.on(start))
        for d, s in self.snapshots.items():
            if start <= d <= end:
                out |= s
        return out

    def current(self) -> set:
        return self.snapshots[max(self.snapshots)]


def sp500() -> Membership:
    cons, ch = _tables()
    current = {_sym(s) for s in cons["Symbol"]}
    names = {_sym(r["Symbol"]): r["Security"] for _, r in cons.iterrows()}
    date_col = [c for c in ch.columns if "Date" in c][0]
    ch["date"] = pd.to_datetime(ch[date_col], errors="coerce", format="mixed")
    ch = ch.dropna(subset=["date"]).sort_values("date", ascending=False)
    today = pd.Timestamp.today().normalize()
    snaps = {today: set(current)}
    members = set(current)
    for _, row in ch.iterrows():
        add, rem = _sym(row.get("Added_Ticker")), _sym(row.get("Removed_Ticker"))
        if rem and row.get("Removed_Security") is not None:
            names.setdefault(rem, row.get("Removed_Security"))
        # state before this change = state after it, minus the addition, plus the removal
        # several changes can share a date: the first one seen (latest in walk order) holds the after-state
        snaps.setdefault(row["date"], set(members))
        if add:
            members.discard(add)
        if rem:
            members.add(rem)
        snaps[row["date"] - pd.Timedelta(days=1)] = set(members)
    m = Membership(snapshots=dict(sorted(snaps.items())), names=names, first_change=ch["date"].min())
    return m


@dataclass
class PITUniverse:
    bars: data_mod.Bars
    membership: Membership
    coverage: pd.DataFrame       # per year: members, priced members, uncovered share
    rejected_reused: list        # removed tickers whose Yahoo series looks like a different company
    unpriced: list               # members Yahoo has nothing for


def load_sp500(start: str) -> PITUniverse:
    m = sp500()
    start_ts = pd.Timestamp(start)
    ever = sorted(m.ever(start_ts))
    current = m.current()
    bars = data_mod.load(ever, start)
    idx = bars.index
    elig = m.frame(idx, bars.tickers)

    rejected, keep = [], []
    for t in bars.tickers:
        if t in current:
            keep.append(t)
            continue
        member_days = elig[t]
        if not member_days.any():
            continue
        first_member = member_days.idxmax()
        last_member = member_days[::-1].idxmax()
        first_px = bars.close[t].first_valid_index()
        # the series must already exist when the name joined (or when the sample starts)
        if first_px is None or first_px > max(first_member, idx[0]) + pd.Timedelta(days=10):
            rejected.append(t)
            continue
        cut = last_member + pd.Timedelta(days=90)
        for f in ("open", "high", "low", "close", "volume", "raw_close"):
            df = getattr(bars, f)
            df.loc[df.index > cut, t] = np.nan
        keep.append(t)
    bars = bars.replace(**{f: getattr(bars, f)[keep] for f in ("open", "high", "low", "close", "volume")},
                        raw_close=bars.raw_close[keep])
    bars.eligible = elig[keep]

    # coverage: of the members on each first trading day of a year, how many have a price that day
    full = m.frame(idx, sorted(set(ever)))
    rows = []
    years = pd.Series(idx.year, index=idx)
    for y, d in years.groupby(years).apply(lambda s: s.index[0]).items():
        mem = full.loc[d]
        members = set(mem[mem].index)
        priced = {t for t in members if t in bars.close.columns and pd.notna(bars.close.at[d, t])}
        rows.append(dict(year=int(y), members=len(members), priced=len(priced),
                         uncovered=1 - len(priced) / len(members) if members else 0.0))
    cov = pd.DataFrame(rows).set_index("year")
    unpriced = sorted(set(ever) - set(keep) - set(rejected))
    return PITUniverse(bars=bars, membership=m, coverage=cov, rejected_reused=rejected, unpriced=unpriced)


def looks_like_sp500(tickers) -> bool:
    """A static list that is (nearly) today's S&P 500: candidates for an automatic point-in-time re-run."""
    t = set(tickers)
    if len(t) < 100:
        return False
    try:
        cur = sp500().current()
    except Exception:
        return False
    return len(t & cur) / len(t) >= 0.8

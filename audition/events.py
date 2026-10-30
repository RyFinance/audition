"""Event collision map: which scheduled events your positions were holding through, and how
much of the P&L came from those days.

Each event is mapped to the open-to-open holding period that absorbs it:
  - FOMC decisions (2 pm ET) and options expiry happen during day d's session -> period d
  - CPI (8:30 am ET) lands before day d's open -> period d-1 (held overnight into it)
  - earnings before the open -> period d-1, after the close -> period d
  - ex-dividend: the drop happens at day d's open -> period d-1
  - index rebalance (quarterly triple-witching Friday) -> period d
  - turn of month: last trading day through the 3rd trading day of the next month

Macro events touch the whole book that period; earnings and ex-dividend only that ticker."""

from __future__ import annotations

import gzip
import http.client
import json
import re
import subprocess
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd

from .data import CACHE
from . import stats

EV = CACHE / "events"
UA = {"User-Agent": "Mozilla/5.0 (audition research tool)"}

PLURAL = {
    "earnings": "earnings reports", "exdiv": "ex-dividend days", "fomc": "FOMC decisions", "cpi": "CPI releases",
    "opex": "options expiries", "rebalance": "index rebalances", "tom": "month turns",
}
LABELS = {
    "earnings": "Earnings", "exdiv": "Ex-dividend", "fomc": "FOMC decision", "cpi": "CPI release",
    "opex": "Options expiry", "rebalance": "Index rebalance", "tom": "Turn of month",
}


def _get(url: str, timeout: int = 30) -> str:
    """curl first (it copes with the Fed's flaky chunked responses), urllib as the fallback."""
    for attempt in range(3):
        try:
            p = subprocess.run(["curl", "-sfL", "--compressed", "--max-time", str(timeout), "-A", UA["User-Agent"], url],
                               capture_output=True, timeout=timeout + 5)
            if p.returncode == 0 and p.stdout:
                return p.stdout.decode("utf-8", "ignore")
        except Exception:
            pass
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "Accept-Encoding": "gzip"}), timeout=timeout)
            body = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            return body.decode("utf-8", "ignore")
        except Exception:
            time.sleep(1 + attempt)
    return ""


def _cached(name: str, max_age_days: float, build):
    EV.mkdir(parents=True, exist_ok=True)
    p = EV / f"{name}.json"
    if p.exists() and time.time() - p.stat().st_mtime < max_age_days * 86400:
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    val = build()
    if val:
        p.write_text(json.dumps(val))
    elif p.exists():
        return json.loads(p.read_text())
    return val


def _fomc_page(url: str, general: bool = False) -> list[str]:
    t = _get(url)
    found = set()
    pats = [r"monetary(\d{8})a", r"press/monetary/(\d{8})", r"press/monetary/\d{4}/(\d{8})"]
    if general:  # before 2003 statements were filed as general press releases
        pats.append(r"press/general/\d{4}/(\d{8})")
    for pat in pats:
        found |= set(re.findall(pat, t))
    return sorted(f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in found if "1994" <= d[:4] <= "2035")


def fomc_dates() -> list[str]:
    """Statement days, scheduled and unscheduled, from the Fed's calendar pages (cached per year)."""
    out = set(_cached("fomc_current", 7, lambda: _fomc_page(
        "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")))
    for y in range(1994, 2021):
        out |= set(_cached(f"fomc_{y}", 3650, lambda y=y: _fomc_page(
            f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm", general=y <= 2002)))
    return sorted(out)


def cpi_dates() -> list[str]:
    """CPI release days: the vintage dates of CPIAUCSL in ALFRED."""
    def build():
        t = _get("https://alfred.stlouisfed.org/series/downloaddata?seid=CPIAUCSL")
        return sorted(set(re.findall(r'<option value="(\d{4}-\d{2}-\d{2})"', t)))
    return _cached("cpi", 14, build)


def earnings_dates(ticker: str) -> list[list]:
    """[[date, 'pre'|'post'], ...] from Yahoo's earnings calendar (about 25 years for large caps)."""
    def build():
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                df = yf.Ticker(ticker).get_earnings_dates(limit=100)
            except Exception:
                return []
        if df is None or df.empty:
            return []
        out = []
        for ts in df.index:
            ts = pd.Timestamp(ts)
            loc = ts.tz_convert("America/New_York") if ts.tzinfo else ts
            out.append([str(loc.date()), "pre" if loc.hour < 12 else "post"])
        return sorted(out)
    safe = ticker.replace("^", "_").replace("/", "_")
    return _cached(f"earn_{safe}", 30, build)


def _third_fridays(idx: pd.DatetimeIndex, months=None):
    out = []
    for y in range(idx[0].year, idx[-1].year + 1):
        for m in range(1, 13):
            if months and m not in months:
                continue
            d = pd.Timestamp(y, m, 1)
            fri = d + pd.Timedelta(days=(4 - d.weekday()) % 7 + 14)
            out.append(fri)
    return out


def _period_of(day: pd.Timestamp, idx: pd.DatetimeIndex, before_open: bool):
    """The holding period (index label) that absorbs an event on `day`."""
    pos = idx.searchsorted(day)
    if pos >= len(idx):
        return None
    on_day = idx[pos] == day
    if before_open or not on_day:
        # pre-open, or on a non-trading day: absorbed overnight by the previous period
        pos -= 1
    return idx[pos] if 0 <= pos < len(idx) else None


def collision_map(rr, max_earn_tickers: int = 60, log=None) -> dict:
    sim = rr.sim
    contrib = sim.contrib
    idx = contrib.index
    if len(idx) < 30:
        return dict(error="too few periods")
    gross = contrib.sum(axis=1)
    net = sim.returns
    ppy = rr.cfg["ppy"]
    total = float(gross.sum())
    exposure = sim.held.abs()
    tot_expo = float(exposure.to_numpy().sum()) or 1.0

    macro_mask = {}
    us_cal = rr.cfg.get("bench") in ("SPY", "^GSPC") or rr.code in ("home", "sp500", "survivors", "us")
    crypto = rr.code == "crypto"

    def mark(days, before_open):
        m = pd.Series(False, index=idx)
        for d in days:
            p = _period_of(pd.Timestamp(d), idx, before_open)
            if p is not None and p >= idx[0]:
                m[p] = True
        return m

    notes = []
    try:
        macro_mask["fomc"] = mark(fomc_dates(), False)
    except Exception as e:
        notes.append(f"FOMC dates unavailable ({e})")
    try:
        macro_mask["cpi"] = mark(cpi_dates(), True)
    except Exception as e:
        notes.append(f"CPI dates unavailable ({e})")
    if not crypto:
        macro_mask["opex"] = mark(_third_fridays(idx), False)
        macro_mask["rebalance"] = mark(_third_fridays(idx, (3, 6, 9, 12)), False)
    s = pd.Series(idx, index=idx)
    month = idx.to_period("M")
    first = s.groupby(month).cumcount()
    last = s.groupby(month).cumcount(ascending=False)
    macro_mask["tom"] = pd.Series(((first <= 2) | (last == 0)).to_numpy(), index=idx)
    if not us_cal and not crypto:
        notes.append("FOMC, CPI and expiry dates are US events, mapped onto this market's calendar")

    # ticker events: only for the names that matter, by absolute P&L contribution
    tick_mask = {}
    if not crypto:
        held_names = contrib.abs().sum().sort_values(ascending=False)
        held_names = held_names[held_names > 0]
        names = list(held_names.index[:max_earn_tickers])
        cover = float(held_names.loc[names].sum() / held_names.sum()) if held_names.sum() > 0 else 0.0
        earn = pd.DataFrame(False, index=idx, columns=contrib.columns)
        exdiv = pd.DataFrame(False, index=idx, columns=contrib.columns)
        got = 0
        for t in names:
            if "=" in t or t.startswith("^"):
                continue
            try:
                ev = earnings_dates(t)
            except Exception:
                ev = []
            if ev:
                got += 1
            for d, when in ev:
                p = _period_of(pd.Timestamp(d), idx, when == "pre")
                if p is not None:
                    earn.at[p, t] = True
        # ex-dividend from the raw vs adjusted close: a step in the adjustment factor is an ex-date
        raw = rr.bars.raw_close.reindex(idx)
        adj = rr.bars.close.reindex(idx)
        fac = (adj / raw).replace([np.inf, -np.inf], np.nan)
        step = fac.pct_change(fill_method=None).abs() > 1e-6
        # the factor changes on the ex day d; the drop is absorbed by period d-1
        exdiv = step.shift(-1, fill_value=False).reindex(columns=contrib.columns, fill_value=False)
        tick_mask["earnings"] = earn
        tick_mask["exdiv"] = exdiv
        notes.append(f"earnings dates for the {len(names)} biggest P&L names ({got} found), "
                     f"covering {cover:.0%} of absolute P&L")

    rows = []
    any_event = pd.DataFrame(False, index=idx, columns=contrib.columns)
    for key, m in macro_mask.items():
        p = float(gross[m].sum())
        n_ev = int(m.sum())
        off = net[~m]
        rows.append(dict(key=key, label=LABELS[key], scope="whole book", events=n_ev,
                         share_time=float(m.mean()), share_expo=float(exposure[m].to_numpy().sum() / tot_expo),
                         pnl=p, share_pnl=p / total if total else float("nan"),
                         per_period_on=float(gross[m].mean()) if n_ev else float("nan"),
                         per_period_off=float(gross[~m].mean()),
                         sharpe_without=stats.sharpe((off - sim.rf[~m]), ppy) if len(off) > 20 else float("nan")))
        any_event |= pd.DataFrame(np.repeat(m.to_numpy()[:, None], contrib.shape[1], axis=1), index=idx, columns=contrib.columns)
    for key, m in tick_mask.items():
        m = m.reindex(index=idx, columns=contrib.columns, fill_value=False).astype(bool)
        c = contrib.where(m, 0.0)
        p = float(c.to_numpy().sum())
        n_ev = int((m & (exposure > 0)).to_numpy().sum())
        stripped = net - c.sum(axis=1)
        rows.append(dict(key=key, label=LABELS[key], scope="that ticker", events=n_ev,
                         share_time=float((m & (exposure > 0)).any(axis=1).mean()),
                         share_expo=float(exposure.where(m, 0).to_numpy().sum() / tot_expo),
                         pnl=p, share_pnl=p / total if total else float("nan"),
                         per_period_on=float(c.to_numpy().sum() / max(n_ev, 1)),
                         per_period_off=float("nan"),
                         sharpe_without=stats.sharpe(stripped - sim.rf, ppy)))
        any_event |= m
    # everything stripped: zero the contribution of every position on an event period
    clean = net - contrib.where(any_event, 0.0).sum(axis=1)
    base_sr = stats.sharpe(net - sim.rf, ppy)
    clean_sr = stats.sharpe(clean - sim.rf, ppy)
    rows.sort(key=lambda r: -abs(r["share_pnl"]) if np.isfinite(r["share_pnl"]) else 0)

    # verdict
    top = next((r for r in rows if np.isfinite(r["share_pnl"])), None)
    if top is None or not total:
        verdict = "Not enough P&L to attribute."
    else:
        conc = top["share_pnl"] / max(top["share_expo"], 1e-9)
        what = PLURAL[top["key"]]
        if top["share_pnl"] >= 0:
            verdict = (f"{top['share_pnl']:.0%} of gross P&L came from positions held through {what}, "
                       f"which took only {top['share_expo']:.0%} of exposure ({conc:.1f}x their share). ")
        else:
            verdict = (f"Positions held through {what} gave back {-top['share_pnl']:.0%} of gross P&L "
                       f"on {top['share_expo']:.0%} of exposure. ")
        if clean_sr < base_sr * 0.5:
            verdict += (f"Strip every event and the Sharpe drops from {base_sr:.2f} to {clean_sr:.2f}: much of this "
                        "strategy is an event bet, whether or not it was designed as one.")
        elif clean_sr > base_sr:
            verdict += (f"Without event periods the Sharpe rises from {base_sr:.2f} to {clean_sr:.2f}; events are a "
                        "drag, so sitting them out is worth testing.")
        else:
            verdict += f"Strip every event and the Sharpe goes from {base_sr:.2f} to {clean_sr:.2f}: the edge isn't hiding in events."
    return dict(rows=rows, base_sharpe=base_sr, clean_sharpe=clean_sr, total_gross=total, verdict=verdict, notes=notes)

"""Claim checker: test posted trades ("bought NVDA at 112.30 on 4 March, sold 120.10 on the 6th")
against what the market actually printed.

Per claim: was it a trading day, did the price trade that day (or in that 5-minute / hourly bar
when a time is given and Yahoo still has intraday data), does the stated return match the
prices, and where in the day's range was the fill. Across claims: a hindsight score. Real fills
land all over the day's range (about 0.5 on average); someone who keeps "buying the low and
selling the high" is reporting with hindsight."""

from __future__ import annotations

import io
import math
import re
import warnings

import numpy as np
import pandas as pd

from . import data as data_mod

SIDE_BUY = {"buy", "bought", "long", "b", "bto", "entered long", "add"}
SIDE_SELL = {"sell", "sold", "short", "s", "sto", "shorted"}

LINE = re.compile(
    r"(?P<side>long|short|buy|bought|sell|sold|shorted)\s+\$?(?P<t>[A-Za-z][A-Za-z0-9.\-=^]{0,11})\s+"
    r"(?:on\s+)?(?P<d1>\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})(?:[ T](?P<t1>\d{1,2}:\d{2}))?\s*(?:@|at)?\s*\$?(?P<p1>[\d.,]+)"
    r"\s*(?:->|→|to|,|;|then|exit|out)\s*(?:on\s+)?(?P<d2>\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})(?:[ T](?P<t2>\d{1,2}:\d{2}))?"
    r"\s*(?:@|at)?\s*\$?(?P<p2>[\d.,]+)(?:.*?(?P<ret>[-+]?\d+(?:\.\d+)?)\s*%)?", re.I)


def parse(text: str) -> tuple[list[dict], list[str]]:
    """CSV with a header (ticker, side, entry_date, entry_price, exit_date, exit_price[, entry_time,
    exit_time, claimed_return]) or one trade per line in loose English."""
    text = text.strip()
    claims, bad = [], []
    if not text:
        return claims, bad
    first = text.splitlines()[0].lower()
    if "ticker" in first and ("entry" in first or "price" in first):
        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip().lower() for c in df.columns]
        for _, r in df.iterrows():
            claims.append(dict(ticker=str(r["ticker"]).upper().strip(), side=str(r.get("side", "long")).lower().strip(),
                               d1=str(r["entry_date"]), t1=_s(r.get("entry_time")), p1=float(r["entry_price"]),
                               d2=str(r["exit_date"]), t2=_s(r.get("exit_time")), p2=float(r["exit_price"]),
                               ret=_f(r.get("claimed_return")), raw=", ".join(str(x) for x in r.values)))
        return claims, bad
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE.search(line)
        if not m:
            bad.append(line)
            continue
        g = m.groupdict()
        claims.append(dict(ticker=g["t"].upper().lstrip("$"), side=g["side"].lower(), d1=g["d1"], t1=g["t1"],
                           p1=float(g["p1"].replace(",", "")), d2=g["d2"], t2=g["t2"], p2=float(g["p2"].replace(",", "")),
                           ret=float(g["ret"]) / 100 if g["ret"] else None, raw=line))
    return claims, bad


def _s(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip() in ("", "nan") else str(x).strip()


def _f(x):
    try:
        v = float(str(x).replace("%", ""))
        return v / 100 if abs(v) > 1.5 else v
    except Exception:
        return None


def _split_factor(ticker: str, day: pd.Timestamp) -> float:
    """How many shares one share on `day` became through later splits (Yahoo's split history)."""
    import yfinance as yf
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = yf.Ticker(ticker).splits
    except Exception:
        return 1.0
    if sp is None or sp.empty:
        return 1.0
    sp.index = pd.DatetimeIndex(sp.index).tz_localize(None).normalize()
    later = sp[sp.index > day]
    return float(later.prod()) if len(later) else 1.0


def _intraday(ticker: str, day: pd.Timestamp, hhmm: str):
    """(low, high, resolution) of the bar around a stated time, if Yahoo still has it."""
    import yfinance as yf
    age = (pd.Timestamp.today().normalize() - day).days
    if age <= 55:
        interval, pad = "5m", 10
    elif age <= 700:
        interval, pad = "60m", 60
    else:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start=day.strftime("%Y-%m-%d"), end=(day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                             interval=interval, progress=False, auto_adjust=False, multi_level_index=False, prepost=True)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    df.index = idx
    h, mi = (int(x) for x in hhmm.split(":"))
    at = day + pd.Timedelta(hours=h, minutes=mi)
    win = df[(df.index >= at - pd.Timedelta(minutes=pad)) & (df.index <= at + pd.Timedelta(minutes=pad))]
    if win.empty:
        return None
    return float(win["Low"].min()), float(win["High"].max()), interval


def check(claims: list[dict], tol: float = 0.003) -> dict:
    tickers = sorted({c["ticker"] for c in claims})
    bars = {}
    for t in tickers:
        try:
            df = data_mod._load_one(t, "1990-01-01")
            bars[t] = None if df is None or df.empty else df
        except Exception:
            bars[t] = None
    out = []
    positions = []
    for c in claims:
        res = dict(c, problems=[], verdict="plausible")
        df = bars.get(c["ticker"])
        short = c["side"] in SIDE_SELL   # the side is the entry leg: "short"/"sold" opens a short
        try:
            d1, d2 = pd.Timestamp(c["d1"]), pd.Timestamp(c["d2"])
        except Exception:
            res["problems"].append("unreadable date")
            res["verdict"] = "unreadable"
            out.append(res)
            continue
        if df is None:
            res["problems"].append(f"no price history for {c['ticker']} on Yahoo")
            res["verdict"] = "unverifiable"
            out.append(res)
            continue
        if d2 < d1:
            res["problems"].append("exit is before entry")
        if d1 > pd.Timestamp.today() or d2 > pd.Timestamp.today():
            res["problems"].append("date is in the future")
        legs = []
        for leg, d, p, tm, is_buy in (("entry", d1, c["p1"], c["t1"], not short), ("exit", d2, c["p2"], c["t2"], short)):
            info = dict(leg=leg, date=str(d.date()), price=p)
            if d not in df.index:
                info["trading_day"] = False
                res["problems"].append(f"{leg} {d.date()} wasn't a trading day for {c['ticker']}")
                legs.append(info)
                continue
            row = df.loc[d]
            f = _split_factor(c["ticker"], d) if row["close"] > 0 else 1.0
            lo, hi = float(row["low"]) * f, float(row["high"]) * f
            info.update(trading_day=True, low=lo, high=hi, open=float(row["open"]) * f, close=float(row["close"]) * f,
                        split_factor=f)
            inside = lo * (1 - tol) <= p <= hi * (1 + tol)
            info["inside"] = inside
            if not inside:
                gap = (p / hi - 1) if p > hi else (p / lo - 1)
                res["problems"].append(f"{leg} price {p:g} never traded on {d.date()} (range {lo:.2f}-{hi:.2f}, "
                                       f"{gap:+.1%} outside)")
            if hi > lo:
                pos = (p - lo) / (hi - lo)
                pos = min(max(pos, 0.0), 1.0)
                # how good was the fill: 1 = bought the exact low / sold the exact high
                info["fill_quality"] = 1 - pos if is_buy else pos
                positions.append(info["fill_quality"])
            if tm:
                bar = _intraday(c["ticker"], d, tm)
                if bar:
                    blo, bhi, res_ = bar
                    blo, bhi = blo * f, bhi * f
                    info["bar"] = dict(low=blo, high=bhi, interval=res_, time=tm)
                    if not (blo * (1 - tol) <= p <= bhi * (1 + tol)):
                        res["problems"].append(f"{leg} price {p:g} isn't in the {res_} bar around {tm} ET "
                                               f"({blo:.2f}-{bhi:.2f})")
                else:
                    info["bar"] = None
            legs.append(info)
        res["legs"] = legs
        actual = (c["p2"] / c["p1"] - 1) * (-1 if short else 1) if c["p1"] else float("nan")
        res["return_from_prices"] = actual
        if c.get("ret") is not None and np.isfinite(actual) and abs(c["ret"] - actual) > max(0.005, abs(actual) * 0.1):
            res["problems"].append(f"claimed {c['ret']:+.1%} but those prices give {actual:+.1%}")
        # best and worst possible outcome between the two days, using daily closes
        if all(l.get("trading_day") for l in legs):
            seg = df.loc[d1:d2, "close"]
            if len(seg) > 1:
                res["close_to_close"] = float((seg.iloc[-1] / seg.iloc[0] - 1) * (-1 if short else 1))
        hard = [p for p in res["problems"] if "never traded" in p or "wasn't a trading day" in p or "future" in p
                or "before entry" in p or "isn't in the" in p]
        fq = [l.get("fill_quality") for l in legs if l.get("fill_quality") is not None]
        if hard:
            res["verdict"] = "impossible"
        elif res["problems"]:
            res["verdict"] = "inconsistent"
        elif fq and min(fq) >= 0.97:
            res["verdict"] = "suspicious"
            res["problems"].append("both fills within 3% of the day's best possible price")
        out.append(res)

    n = len(positions)
    hind = float(np.mean(positions)) if n else float("nan")
    z = (hind - 0.5) / math.sqrt(1 / 12 / n) if n else float("nan")
    # P(mean of n uniform fills >= observed), normal approximation
    p = 0.5 * math.erfc(z / math.sqrt(2)) if n else float("nan")
    counts = {k: sum(1 for r in out if r["verdict"] == k) for k in
              ("plausible", "suspicious", "inconsistent", "impossible", "unverifiable", "unreadable")}
    rets = [r["return_from_prices"] for r in out if np.isfinite(r.get("return_from_prices", np.nan))]
    compounded = float(np.prod([1 + x for x in rets]) - 1) if rets else float("nan")
    if counts["impossible"]:
        verdict = (f"{counts['impossible']} of {len(out)} claimed trades could not have happened at the stated "
                   "prices or dates.")
    elif n >= 6 and p < 0.01:
        verdict = (f"Every fill traded, but the fills are too good: average fill quality {hind:.2f} "
                   f"(random timing averages 0.50; odds of this by luck about {p:.1g}). That's what hindsight looks like.")
    elif n >= 6 and hind > 0.7:
        verdict = f"All prices traded. Fill quality {hind:.2f} is better than typical but not conclusive."
    else:
        verdict = f"Nothing impossible in {len(out)} claim(s). Fill quality {hind:.2f} on {n} fills." if n else \
            "Nothing could be checked."
    verdict += " A claim checker can't see trades that weren't posted; selective posting is invisible here."
    return dict(claims=out, counts=counts, hindsight=hind, hindsight_p=p, fills=n, compounded=compounded, verdict=verdict)

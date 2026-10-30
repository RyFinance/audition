"""Twin finder: compare a strategy with a library of well-known strategies and factors, so you
know whether you found something new or rebuilt momentum (or a cheap ETF).

Twins are close-to-close; Audition's track records are open-to-open, so everything is compared
on weekly (Friday) returns, which washes out the half-day offset."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import data as data_mod, factors, stats
from .data import CACHE

ETFS = {
    "SPY": ("S&P 500", "market"), "QQQ": ("Nasdaq 100", "market"), "IWM": ("Russell 2000", "market"),
    "EFA": ("Developed ex-US", "market"), "EEM": ("Emerging markets", "market"),
    "TLT": ("Long Treasuries", "bonds"), "IEF": ("7-10y Treasuries", "bonds"), "HYG": ("High yield", "credit"),
    "GLD": ("Gold", "commodity"), "DBC": ("Commodities", "commodity"), "VNQ": ("REITs", "real estate"),
    "MTUM": ("Momentum ETF", "style ETF"), "QUAL": ("Quality ETF", "style ETF"), "VLUE": ("Value ETF", "style ETF"),
    "USMV": ("Min-vol ETF", "style ETF"), "BTC-USD": ("Bitcoin", "crypto"),
}
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
FF_NAMES = {"Mkt-RF": "US market excess (FF)", "SMB": "Size: small minus big", "HML": "Value: high minus low B/M",
            "RMW": "Profitability", "CMA": "Investment: conservative minus aggressive", "UMD": "Momentum 12-1 (FF)",
            "ST_Rev": "Short-term reversal", "LT_Rev": "Long-term reversal"}


def _rules(px: pd.DataFrame, op: pd.DataFrame) -> dict:
    """Simple rule-based classics built from ETF prices. Each: (daily returns, name, family)."""
    r = px.pct_change(fill_method=None)
    out = {}
    spy = px["SPY"]
    sr = r["SPY"]
    out["spy_trend"] = ((spy > spy.rolling(200).mean()).shift(1).astype(float) * sr, "SPY 200-day trend", "trend")
    assets = [a for a in ("SPY", "EFA", "EEM", "TLT", "GLD", "DBC") if a in px]
    sig = np.sign(px[assets] / px[assets].shift(252) - 1)
    m_sig = sig.resample("ME").last().reindex(px.index, method="ffill").shift(1)
    out["tsmom"] = ((m_sig * r[assets]).mean(axis=1), "Time-series momentum, 6 assets", "trend")
    if "IEF" in px:
        out["6040"] = (0.6 * sr + 0.4 * r["IEF"], "60/40 stocks/bonds", "allocation")
    rp = [a for a in ("SPY", "TLT", "GLD", "DBC") if a in px]
    iv = 1 / r[rp].rolling(60).std()
    w = iv.div(iv.sum(axis=1), axis=0).shift(1)
    out["riskparity"] = ((w * r[rp]).sum(axis=1), "Risk parity (inverse vol)", "allocation")
    out["overnight"] = (op["SPY"] / px["SPY"].shift(1) - 1, "SPY overnight (close to open)", "anomaly")
    out["intraday"] = (px["SPY"] / op["SPY"] - 1, "SPY intraday (open to close)", "anomaly")
    s = pd.Series(px.index, index=px.index)
    mo = px.index.to_period("M")
    tom = (s.groupby(mo).cumcount() <= 2) | (s.groupby(mo).cumcount(ascending=False) == 0)
    out["tom"] = (sr.where(tom, 0.0), "Turn of the month (SPY)", "calendar")
    out["reversal1d"] = (sr.where(sr.shift(1) < 0, 0.0), "Buy SPY after a down day", "mean reversion")
    may = ~px.index.month.isin([5, 6, 7, 8, 9, 10])
    out["sellmay"] = (sr.where(may, 0.0), "Sell in May (Nov-Apr only)", "calendar")
    sec = [x for x in SECTORS if x in px]
    if len(sec) >= 6:
        mom = px[sec] / px[sec].shift(126) - 1
        rank = mom.resample("ME").last().rank(axis=1, ascending=False)
        pick = (rank <= 3).astype(float).reindex(px.index, method="ffill").shift(1) / 3
        out["sectormom"] = ((pick * r[sec]).sum(axis=1), "Sector momentum, top 3 of 9", "momentum")
    if "BTC-USD" in px:
        b = px["BTC-USD"]
        out["btctrend"] = ((b > b.rolling(50).mean()).shift(1).astype(float) * r["BTC-USD"], "Bitcoin 50-day trend", "trend")
    return out


def library(start: str = "2000-01-01") -> pd.DataFrame:
    """Daily returns of every twin (columns), cached for a day. Attrs carry names/families."""
    path = CACHE / "twins.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < 86400:
        return pd.read_pickle(path)
    tick = list(ETFS) + SECTORS
    bars = data_mod.load(tick, start)
    px, op = bars.close, bars.open
    cols, meta = {}, {}
    for t, (name, fam) in ETFS.items():
        if t in px:
            cols[t] = px[t].pct_change(fill_method=None)
            meta[t] = (name, fam, "buy and hold")
    for k, (ser, name, fam) in _rules(px, op).items():
        cols[k] = ser
        meta[k] = (name, fam, "rule")
    try:
        ff = factors.load_all()
        for c in ff.columns:
            if c in FF_NAMES and c != "RF":
                cols[c] = ff[c]
                meta[c] = (FF_NAMES[c], "factor", "long-short factor")
    except Exception:
        pass
    df = pd.DataFrame(cols)
    df.attrs["meta"] = meta
    df.to_pickle(path)
    return df


def _weekly(x: pd.Series) -> pd.Series:
    return (1 + x.fillna(0)).resample("W-FRI").prod() - 1


def find(rr, top: int = 12) -> dict:
    lib = library()
    meta = lib.attrs.get("meta", {})
    strat = _weekly(rr.sim.returns - rr.sim.rf)
    rf_w = _weekly(rr.sim.rf)
    rows = []
    W = {}
    for c in lib.columns:
        tw = _weekly(lib[c].dropna())
        fam = meta.get(c, ("", "", ""))
        if fam[2] != "long-short factor":
            tw = tw - rf_w.reindex(tw.index).fillna(0)  # compare excess with excess
        j = pd.concat([strat, tw], axis=1, join="inner").dropna()
        j = j[(j.iloc[:, 0] != 0) | (j.iloc[:, 1] != 0)]
        if len(j) < 52:
            continue
        W[c] = tw
        a, b = j.iloc[:, 0], j.iloc[:, 1]
        corr = float(a.corr(b))
        beta = float(np.cov(a, b)[0, 1] / b.var()) if b.var() > 0 else float("nan")
        rows.append(dict(key=c, name=fam[0] or c, family=fam[1], kind=fam[2], corr=corr, beta=beta,
                         weeks=len(j), twin_sharpe=stats.sharpe(b, 52), same_sign=float((np.sign(a) == np.sign(b)).mean())))
    if not rows:
        return dict(error="not enough overlap with any twin")
    rows.sort(key=lambda r: -abs(r["corr"]))

    # forward stepwise: best mix of up to 3 twins
    chosen, best_r2 = [], 0.0
    y_full = strat
    for _ in range(3):
        cand = None
        for r in rows[:25]:
            k = r["key"]
            if k in chosen:
                continue
            X = pd.concat([W[x] for x in chosen + [k]], axis=1, join="inner")
            j = pd.concat([y_full, X], axis=1, join="inner").dropna()
            if len(j) < 52:
                continue
            y, Xm = j.iloc[:, 0].to_numpy(), np.column_stack([np.ones(len(j)), j.iloc[:, 1:].to_numpy()])
            coef, *_ = np.linalg.lstsq(Xm, y, rcond=None)
            res = y - Xm @ coef
            r2 = 1 - res.var() / y.var() if y.var() > 0 else 0.0
            if cand is None or r2 > cand[1]:
                cand = (k, r2)
        if cand is None or cand[1] - best_r2 < 0.02:
            break
        chosen.append(cand[0])
        best_r2 = cand[1]
    mix = None
    if chosen:
        X = pd.concat([W[x] for x in chosen], axis=1, join="inner")
        j = pd.concat([y_full, X], axis=1, join="inner").dropna()
        y = j.iloc[:, 0]
        Xm = np.column_stack([np.ones(len(j)), j.iloc[:, 1:].to_numpy()])
        coef, *_ = np.linalg.lstsq(Xm, y.to_numpy(), rcond=None)
        resid = y.to_numpy() - Xm[:, 1:] @ coef[1:]       # alpha + noise: what the twins don't explain
        beta_nw, se_nw, _ = factors.newey_west(y.to_numpy(), Xm, lags=4)
        tstat = beta_nw / se_nw
        mix = dict(twins=[dict(key=k, name=meta.get(k, (k,))[0], weight=float(w)) for k, w in zip(chosen, coef[1:])],
                   r2=float(best_r2), alpha_ann=float(coef[0] * 52), alpha_t=float(tstat[0]),
                   residual_sharpe=stats.sharpe(pd.Series(resid), 52), strat_sharpe=stats.sharpe(y, 52))

    t1 = rows[0]
    if abs(t1["corr"]) >= 0.8:
        verdict = (f"Your strategy is {t1['corr']:.2f} correlated with {t1['name']}: largely a rediscovery of it. ")
    elif mix and mix["r2"] >= 0.6:
        verdict = (f"{mix['r2']:.0%} of its weekly moves are explained by a mix of "
                   + ", ".join(t["name"] for t in mix["twins"]) + ". ")
    elif mix and mix["r2"] < 0.25:
        verdict = (f"Nothing in the library explains it well (best mix R² {mix['r2']:.0%}, closest twin "
                   f"{t1['name']} at {t1['corr']:.2f}). Original, or noise; the other tapes decide which. ")
    else:
        verdict = (f"Partly familiar: closest twin is {t1['name']} ({t1['corr']:.2f}); the best mix explains "
                   f"{mix['r2'] if mix else 0:.0%}. ")
    if mix:
        verdict += (f"What's left after the twins: alpha {mix['alpha_ann']:+.1%} a year (t {mix['alpha_t']:.1f}), "
                    f"Sharpe {mix['residual_sharpe']:.2f} versus {mix['strat_sharpe']:.2f} in total.")
    cheaper = [r for r in rows[:5] if r["kind"] == "buy and hold" and r["corr"] >= 0.6
               and r["twin_sharpe"] >= (mix["strat_sharpe"] if mix else 0)]
    if cheaper:
        c = cheaper[0]
        verdict += f" {c['name']} ({c['key']}) did much the same with Sharpe {c['twin_sharpe']:.2f} and no work."
    return dict(rows=rows[:top], all=len(rows), mix=mix, verdict=verdict, library_size=len(lib.columns))

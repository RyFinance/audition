"""Fama-French five factors plus momentum (daily, Ken French data library) and a
Newey-West regression of a strategy's excess returns on them."""

from __future__ import annotations

import io
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from .data import CACHE

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {"ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "umd": "F-F_Momentum_Factor_daily_CSV.zip",
         "st_rev": "F-F_ST_Reversal_Factor_daily_CSV.zip", "lt_rev": "F-F_LT_Reversal_Factor_daily_CSV.zip"}


def _csv(name: str) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / FILES[name]
    if not (path.exists() and time.time() - path.stat().st_mtime < 7 * 86400):
        err = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(BASE + FILES[name], headers={"User-Agent": "audition/0.1"})
                body = urllib.request.urlopen(req, timeout=60).read()
                zipfile.ZipFile(io.BytesIO(body)).testzip()
                path.write_bytes(body)
                err = None
                break
            except Exception as e:
                err = e
                time.sleep(1.5 * (attempt + 1))
        if err is not None and not path.exists():
            raise err
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    head = next(i for i, l in enumerate(lines) if l.strip().startswith(",") or l.lower().startswith(",mkt"))
    rows = []
    for l in lines[head + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break
            continue
        rows.append(parts)
    cols = [c.strip() for c in lines[head].split(",")][1:]
    df = pd.DataFrame([r[1:] for r in rows], index=pd.to_datetime([r[0] for r in rows], format="%Y%m%d"),
                      columns=cols).astype(float) / 100.0
    return df


def load() -> pd.DataFrame:
    ff = _csv("ff5")
    umd = _csv("umd")
    umd.columns = ["UMD"]
    return ff.join(umd, how="inner")


def load_all() -> pd.DataFrame:
    """FF5 + momentum + short- and long-term reversal (reversal files are optional)."""
    df = load()
    for key, col in (("st_rev", "ST_Rev"), ("lt_rev", "LT_Rev")):
        try:
            x = _csv(key)
            x.columns = [col]
            df = df.join(x, how="left")
        except Exception:
            pass
    return df


def newey_west(y: np.ndarray, X: np.ndarray, lags: int = 10):
    """OLS coefficients with Newey-West (Bartlett) standard errors."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2


def attribute(returns: pd.Series, ppy: int = 252) -> dict | None:
    """Weekly regression. Audition's returns run open to open while the factors run close to
    close, so daily rows are half a day out of step; compounding both to weeks removes that."""
    f = load()
    names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    end = min(returns.index[-1], f.index[-1])
    r = returns.loc[:end]
    fx = f.loc[r.index[0]:end]
    wk_r = (1 + r).resample("W-FRI").prod() - 1
    wk_f = (1 + fx).resample("W-FRI").prod() - 1
    df = pd.concat([wk_r.rename("r"), wk_f], axis=1, join="inner").dropna()
    df = df[(df != 0).any(axis=1)]
    if len(df) < 52:
        return None
    y = (df["r"] - df["RF"]).to_numpy()
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in names])
    beta, se, r2 = newey_west(y, X, lags=4)
    return dict(
        alpha=float(beta[0] * 52), alpha_t=float(beta[0] / se[0]), alpha_se=float(se[0] * 52),
        betas={n: float(b) for n, b in zip(names, beta[1:])},
        t={n: float(b / s) for n, b, s in zip(names, beta[1:], se[1:])},
        r2=float(r2), n=len(df), start=str(df.index[0].date()), end=str(df.index[-1].date()),
    )

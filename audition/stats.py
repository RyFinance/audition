"""Performance and significance statistics. Sharpe is always on excess returns over T-bills."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as st

EULER = 0.5772156649


def equity(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()


def cagr(r: pd.Series, ppy: int) -> float:
    if len(r) == 0:
        return float("nan")
    g = float((1 + r).prod())
    return g ** (ppy / len(r)) - 1 if g > 0 else -1.0


def sharpe(excess: pd.Series, ppy: int) -> float:
    sd = excess.std()
    return float(excess.mean() / sd * math.sqrt(ppy)) if sd > 0 else 0.0


def sortino(excess: pd.Series, ppy: int) -> float:
    dn = excess.clip(upper=0)
    dd = math.sqrt((dn ** 2).mean())
    return float(excess.mean() / dd * math.sqrt(ppy)) if dd > 0 else 0.0


def drawdown(r: pd.Series) -> pd.Series:
    eq = equity(r)
    return eq / eq.cummax() - 1


def max_dd_duration(r: pd.Series) -> int:
    """Longest run of periods spent below a prior equity high."""
    under = (drawdown(r) < -1e-12).to_numpy()
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return best


def lo_sharpe(excess: pd.Series, ppy: int, lags: int = 10) -> tuple[float, list]:
    """Lo (2002) autocorrelation-adjusted annual Sharpe, autocorrelations truncated at `lags`."""
    sr = excess.mean() / excess.std() if excess.std() > 0 else 0.0
    rho = [float(excess.autocorr(k)) for k in range(1, lags + 1)]
    rho = [0.0 if not np.isfinite(x) else x for x in rho]
    q = ppy
    denom = q + 2 * sum((q - k) * rho[k - 1] for k in range(1, lags + 1))
    eta = q / math.sqrt(denom) if denom > 0 else math.sqrt(q)
    return float(sr * eta), rho


def psr(excess: pd.Series, sr0_period: float = 0.0) -> float:
    """Probabilistic Sharpe ratio: P(true per-period SR > sr0) given skew and kurtosis."""
    n = len(excess)
    if n < 3 or excess.std() == 0:
        return 0.5
    sr = excess.mean() / excess.std()
    g3 = float(st.skew(excess))
    g4 = float(st.kurtosis(excess, fisher=False))
    v = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if v <= 0:
        v = 1e-9
    return float(st.norm.cdf((sr - sr0_period) * math.sqrt(n - 1) / math.sqrt(v)))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """Expected maximum per-period Sharpe among n_trials skill-less strategies (Bailey & Lopez de Prado)."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z1 = st.norm.ppf(1 - 1 / n_trials)
    z2 = st.norm.ppf(1 - 1 / (n_trials * math.e))
    return float(math.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2))


def min_track_record(excess: pd.Series, sr0_period: float = 0.0, conf: float = 0.95) -> float:
    n = len(excess)
    if excess.std() == 0:
        return float("inf")
    sr = excess.mean() / excess.std()
    if sr <= sr0_period:
        return float("inf")
    g3 = float(st.skew(excess))
    g4 = float(st.kurtosis(excess, fisher=False))
    z = st.norm.ppf(conf)
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (z / (sr - sr0_period)) ** 2)


def summary(r: pd.Series, rf: pd.Series, ppy: int) -> dict:
    ex = (r - rf.reindex(r.index).fillna(0.0))
    dd = drawdown(r)
    mdd = float(dd.min()) if len(dd) else 0.0
    c = cagr(r, ppy)
    m = (1 + r).resample("ME").prod() - 1
    wk = (1 + r).resample("W").prod() - 1
    q05, q01 = r.quantile(0.05), r.quantile(0.01)
    lo, rho = lo_sharpe(ex, ppy)
    return dict(
        cagr=c, vol=float(r.std() * math.sqrt(ppy)), sharpe=sharpe(ex, ppy), sortino=sortino(ex, ppy),
        lo_sharpe=lo, rho=rho, max_dd=mdd, calmar=(c / abs(mdd)) if mdd < 0 else float("nan"),
        dd_duration=max_dd_duration(r), worst_day=float(r.min()), worst_week=float(wk.min()),
        worst_month=float(m.min()), best_month=float(m.max()),
        es95=float(r[r <= q05].mean()), es99=float(r[r <= q01].mean()),
        skew=float(st.skew(r)), kurt=float(st.kurtosis(r)), hit=float((r > 0).mean()),
        n=len(r), years=len(r) / ppy, t_stat=float(ex.mean() / ex.std() * math.sqrt(len(ex))) if ex.std() > 0 else 0.0,
        psr=psr(ex), min_trl_years=min_track_record(ex) / ppy,
    )


def yearly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


def beta_alpha(r: pd.Series, bench: pd.Series, rf: pd.Series, ppy: int) -> dict:
    df = pd.concat([r - rf, bench - rf], axis=1, keys=["s", "b"]).dropna()
    if len(df) < 30 or df["b"].var() == 0:
        return dict(beta=float("nan"), alpha=float("nan"), corr=float("nan"), alpha_t=float("nan"))
    res = st.linregress(df["b"], df["s"])
    resid = df["s"] - (res.intercept + res.slope * df["b"])
    se = resid.std() / math.sqrt(len(df))
    return dict(beta=float(res.slope), alpha=float(res.intercept * ppy), corr=float(res.rvalue),
                alpha_t=float(res.intercept / se) if se > 0 else 0.0)

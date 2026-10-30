"""Bootstrap resampling of a return series. The stationary block bootstrap keeps
volatility clustering and autocorrelation; the iid shuffle is shown beside it
because it usually understates drawdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stationary_indices(n: int, length: int, paths: int, mean_block: float, rng) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty((paths, length), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, paths)
    jump = rng.random((paths, length)) < p
    fresh = rng.integers(0, n, (paths, length))
    for t in range(1, length):
        nxt = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(jump[:, t], fresh[:, t], nxt)
    return idx


def iid_indices(n: int, length: int, paths: int, rng) -> np.ndarray:
    return rng.integers(0, n, (paths, length))


def path_stats(r: np.ndarray, ppy: int) -> dict:
    eq = np.cumprod(1 + r, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    mdd = (eq / peak - 1).min(axis=1)
    final = eq[:, -1]
    years = r.shape[1] / ppy
    cagr = np.where(final > 0, final ** (1 / years) - 1, -1.0)
    return dict(cagr=cagr, mdd=mdd, final=final, eq=eq)


def run(r: pd.Series, ppy: int, years: float = 3.0, paths: int = 2000, mean_block: float = 20.0,
        seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    x = r.to_numpy()
    n = len(x)
    length = int(min(years * ppy, max(n, ppy)))
    out = {}
    for name, idx in (
        ("block", stationary_indices(n, length, paths, mean_block, rng)),
        ("iid", iid_indices(n, length, paths, rng)),
    ):
        s = path_stats(x[idx], ppy)
        out[name] = dict(
            cagr_p=np.percentile(s["cagr"], [5, 25, 50, 75, 95]).tolist(),
            mdd_p=np.percentile(s["mdd"], [5, 25, 50, 75, 95]).tolist(),
            p_loss=float((s["final"] < 1).mean()),
            p_dd25=float((s["mdd"] <= -0.25).mean()),
            p_dd50=float((s["mdd"] <= -0.50).mean()),
            fan=np.percentile(s["eq"], [5, 25, 50, 75, 95], axis=0),
        )
    out["years"] = length / ppy
    out["paths"] = paths
    out["mean_block"] = mean_block
    return out


def block_sharpe_ci(excess: pd.Series, ppy: int, paths: int = 2000, mean_block: float = 20.0, seed: int = 11):
    rng = np.random.default_rng(seed)
    x = excess.to_numpy()
    idx = stationary_indices(len(x), len(x), paths, mean_block, rng)
    s = x[idx]
    sr = s.mean(axis=1) / s.std(axis=1) * np.sqrt(ppy)
    return np.percentile(sr, [2.5, 50, 97.5]).tolist(), float((sr <= 0).mean())


def effective_n(excess: pd.Series, lags: int = 10) -> float:
    """Effective number of independent observations given autocorrelation."""
    n = len(excess)
    rho = [excess.autocorr(k) for k in range(1, lags + 1)]
    rho = [0.0 if not np.isfinite(x) else x for x in rho]
    denom = 1 + 2 * sum((1 - k / n) * rho[k - 1] for k in range(1, lags + 1))
    return float(min(n, n / denom)) if denom > 0 else float(n)

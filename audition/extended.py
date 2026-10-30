"""Checks 61-70 (beyond the review template), the full metrics table and the full Monte Carlo study."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import engine, montecarlo, stats
from .catalog import ADDRESSED, FAILED, NOT, PARTIAL


def _fp(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def _f(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


# --------------------------------------------------------------------------- helpers

def position_episodes(held: pd.DataFrame, contrib: pd.DataFrame) -> pd.Series:
    """P&L (as fraction of equity) of each contiguous holding episode per asset."""
    out = []
    s = np.sign(held.round(9))
    for t in held.columns:
        st = s[t].to_numpy()
        c = contrib[t].to_numpy()
        i, n = 0, len(st)
        while i < n:
            if st[i] == 0:
                i += 1
                continue
            j = i
            while j + 1 < n and st[j + 1] == st[i]:
                j += 1
            out.append(float(c[i:j + 1].sum()))
            i = j + 1
    return pd.Series(out, dtype=float)


def equal_weight_universe(rr) -> pd.Series:
    """Daily-rebalanced equal weight of every asset trading that day, open to open, no costs."""
    px = rr.bars.open.ffill()
    fwd = (px.shift(-1) / px - 1).iloc[:-1]
    live = rr.bars.close.notna()
    if rr.bars.eligible is not None:  # the real index: members on each date only
        live = live & rr.bars.eligible.reindex_like(live).fillna(False).astype(bool)
    live = live.iloc[:-1]
    r = fwd.where(live).mean(axis=1).fillna(0.0)
    return r.reindex(rr.sim.returns.index).fillna(0.0)


# --------------------------------------------------------------------------- checks

def run_checks(strat, rr, runs, ppy, quick, Check, _sim, _sharpe, _try_run, survivors=False):
    out = []
    r, rf = rr.sim.returns, rr.sim.rf
    ex = r - rf
    base = rr.summary["sharpe"]
    years = len(r) / ppy

    # 61 turnover and holding period
    annual_turn = float(rr.sim.turnover.sum() / years) if years > 0 else float("nan")
    gross = rr.sim.held.abs().sum(axis=1).mean()
    hold_days = float(2 * gross / rr.sim.turnover.mean()) if rr.sim.turnover.mean() > 0 else float("inf")
    c = Check(61, evidence=dict(annual_turnover=annual_turn, avg_holding_periods=hold_days))
    c.finding = (f"Trades {annual_turn:.1f}x its capital a year; a position is held about {hold_days:.0f} periods "
                 f"on average. Costs scale with this.")
    c.status = PARTIAL if annual_turn > 50 else ADDRESSED
    if c.status == PARTIAL:
        c.finding += " Very high turnover for daily bars: execution quality decides the result."
    out.append(c)

    # 62 cross-market replication
    others = [x for code, x in runs.items() if x is not rr and not x.error and code != "survivors"]
    c = Check(62)
    if not others:
        c.status = NOT
        c.finding = "Only one market was run. Add --regions to replay the file elsewhere."
    else:
        pos = [x for x in others if x.summary["sharpe"] > 0]
        c.evidence = dict({x.code: x.summary["sharpe"] for x in others})
        c.finding = (f"Positive Sharpe in {len(pos)} of {len(others)} other markets: " +
                     ", ".join(f"{x.cfg['name']} {x.summary['sharpe']:.2f}" for x in others) + ".")
        c.status = ADDRESSED if len(pos) >= max(2, math.ceil(len(others) / 2)) else FAILED if not pos else PARTIAL
    out.append(c)

    # 63 rolling stability
    roll = ex.rolling(ppy).mean() / ex.rolling(ppy).std() * math.sqrt(ppy)
    roll = roll.dropna()
    frac = float((roll > 0).mean()) if len(roll) else float("nan")
    c = Check(63, evidence=dict(frac_positive=frac, worst=float(roll.min()) if len(roll) else None,
                                best=float(roll.max()) if len(roll) else None))
    c.finding = (f"Rolling one-year Sharpe is positive {frac:.0%} of the time (range {_f(roll.min())} to "
                 f"{_f(roll.max())}).")
    c.status = ADDRESSED if frac >= 0.7 else PARTIAL if frac >= 0.5 else FAILED
    out.append(c)

    # 64 timing luck
    lag = {k: _sharpe(_sim(rr, delay=k), ppy) for k in range(1, 6)}
    vals = np.array(list(lag.values()))
    c = Check(64, evidence=dict(sharpe_by_delay=lag))
    c.finding = ("Sharpe trading 1-5 days after the signal: " + ", ".join(f"{v:.2f}" for v in vals) +
                 f". Spread {vals.max() - vals.min():.2f}.")
    c.status = PARTIAL if base > 0 and vals.min() < 0.5 * base else ADDRESSED
    out.append(c)

    # 65 start-date sensitivity
    yrs = sorted(set(r.index.year))
    rows = {}
    for y in yrs[:-2]:
        sub = r[r.index.year >= y]
        rows[y] = (stats.cagr(sub, ppy), stats.sharpe(sub - rf.reindex(sub.index), ppy))
    c = Check(65, evidence=dict(by_start_year={y: dict(cagr=a, sharpe=b) for y, (a, b) in rows.items()}))
    if rows:
        sh = [b for _, b in rows.values()]
        neg = [y for y, (_, b) in rows.items() if b <= 0]
        c.finding = (f"Starting in any year from {yrs[0]} to {yrs[-3]}: Sharpe {min(sh):.2f} to {max(sh):.2f}, "
                     f"CAGR {_fp(min(a for a, _ in rows.values()))} to {_fp(max(a for a, _ in rows.values()))}.")
        c.status = FAILED if len(neg) > len(rows) / 2 else PARTIAL if neg else ADDRESSED
        if neg:
            c.finding += f" Negative from {', '.join(map(str, neg[:4]))}."
    else:
        c.status = NOT
        c.finding = "Under three years of data."
    out.append(c)

    # 66 universe sensitivity
    tick = rr.bars.tickers
    c = Check(66)
    if len(tick) < 10:
        c.status = NOT
        c.finding = f"Only {len(tick)} assets; too few to subsample."
    else:
        rng = np.random.default_rng(21)
        srs = []
        for _ in range(5 if quick else 15):
            keep = sorted(rng.choice(tick, size=int(len(tick) * 0.8), replace=False))
            b = rr.bars.replace(**{f: getattr(rr.bars, f)[keep] for f in ("open", "high", "low", "close", "volume")},
                                raw_close=rr.bars.raw_close[keep])
            w = _try_run(strat, rr, bars=b)
            if w is None:
                continue
            sim = engine.simulate(w, b, rr.rf, rr.cm, ppy, eval_start=getattr(rr, "eval_start", None))
            srs.append(stats.sharpe(sim.returns - sim.rf, ppy))
        if srs:
            a = np.array(srs)
            c.evidence = dict(sharpes=srs, p10=float(np.percentile(a, 10)), median=float(np.median(a)))
            c.finding = (f"Dropping a random 20% of the universe {len(a)} times: Sharpe {a.min():.2f} to {a.max():.2f}, "
                         f"median {np.median(a):.2f} vs {base:.2f}.")
            c.status = (FAILED if base > 0 and np.median(a) < 0.5 * base else
                        PARTIAL if base > 0 and np.percentile(a, 10) < 0.5 * base else ADDRESSED)
        else:
            c.status = PARTIAL
            c.finding = "The strategy errored on reduced universes."
    out.append(c)

    # 67 best-day dependence
    k = 10
    top = r.nlargest(k).index
    r2 = r.copy()
    r2.loc[top] = rf.reindex(top)
    c_all, c_wo = stats.cagr(r, ppy), stats.cagr(r2, ppy)
    rf_cagr = stats.cagr(rf, ppy)
    c = Check(67, evidence=dict(cagr=c_all, cagr_without_best10=c_wo))
    c.finding = (f"CAGR {_fp(c_all)}; without its 10 best days ({k / len(r):.2%} of the sample) {_fp(c_wo)}. "
                 f"T-bills returned {_fp(rf_cagr)}.")
    c.status = FAILED if c_all > rf_cagr and c_wo <= rf_cagr else PARTIAL if c_wo < 0.5 * c_all else ADDRESSED
    out.append(c)

    # 68 own-universe equal weight
    ew = equal_weight_universe(rr)
    ew_sh, ew_cg = stats.sharpe(ew - rf, ppy), stats.cagr(ew, ppy)
    c = Check(68, evidence=dict(ew_sharpe=ew_sh, ew_cagr=ew_cg, sharpe=base, cagr=c_all))
    c.finding = (f"Equal-weighting every asset in the same universe (no costs) gives Sharpe {ew_sh:.2f}, CAGR "
                 f"{_fp(ew_cg)}; the strategy gives {base:.2f} and {_fp(c_all)} after costs.")
    if len(tick) > 1 and base < ew_sh:
        c.status = FAILED
        c.finding += " Holding the whole universe equally had the better risk-adjusted return; the selection isn't adding Sharpe."
    elif len(tick) > 1 and base < ew_sh + 0.15:
        c.status = PARTIAL
        c.finding += " Most of the result is the universe, not the selection."
    else:
        c.status = ADDRESSED
    if survivors:
        c.finding += " This universe is all survivors, so its equal weight is inflated too."
    out.append(c)

    # 69 win/loss
    ep = position_episodes(rr.sim.held, rr.sim.contrib)
    c = Check(69)
    if len(ep) == 0:
        c.status = NOT
        c.finding = "No positions."
    else:
        wins, losses = ep[ep > 0], ep[ep < 0]
        pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf")
        hit = float((ep > 0).mean())
        payoff = float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else float("nan")
        c.evidence = dict(positions=len(ep), profit_factor=pf, hit_rate=hit, payoff=payoff,
                          best=float(ep.max()), worst=float(ep.min()))
        c.finding = (f"{len(ep)} positions: {hit:.0%} winners, profit factor {_f(pf)}, average win "
                     f"{_f(payoff)}x the average loss. Best position {_fp(ep.max(), 2)} of equity, worst {_fp(ep.min(), 2)}.")
        c.status = PARTIAL if pf < 1.1 else ADDRESSED
    out.append(c)

    # 70 drawdown pain
    dd = stats.drawdown(r)
    ulcer = float(np.sqrt((dd ** 2).mean()))
    under = float((dd < -0.10).mean())
    total = float((1 + r).prod() - 1)
    rec = total / abs(dd.min()) if dd.min() < 0 else float("inf")
    c = Check(70, evidence=dict(ulcer=ulcer, time_under_water=under, recovery_factor=rec))
    c.finding = (f"Ulcer index {ulcer:.1%}; more than 10% below the high {under:.0%} of the time; recovery factor (total return / "
                 f"max drawdown) {_f(rec, 1)}.")
    c.status = PARTIAL if under > 0.5 or ulcer > 0.2 else ADDRESSED
    out.append(c)

    # 71 factor attribution
    c = Check(71)
    try:
        from . import factors
        fa = factors.attribute(r, ppy) if ppy == 252 else None
    except Exception as e:
        fa = None
        c.finding = f"Factor data unavailable ({type(e).__name__})."
    if fa:
        c.evidence = fa
        b = fa["betas"]
        big = sorted(b.items(), key=lambda kv: -abs(kv[1]))[:3]
        c.finding = (f"Against the Fama-French five factors plus momentum (weekly, {fa['start']} to {fa['end']}, Newey-West): "
                     f"alpha {fa['alpha'] * 100:+.1f}%/yr (t {fa['alpha_t']:.2f}), R² {fa['r2']:.2f}. Main exposures: "
                     + ", ".join(f"{k} {v:+.2f} (t {fa['t'][k]:.1f})" for k, v in big) + ".")
        if rr.code not in ("us", "sp500", "survivors", "home") or (rr.code == "home" and rr.cfg.get("bench") != "SPY"):
            c.finding += " These are US factors; for other markets treat this as indicative."
        if fa["alpha_t"] >= 2:
            c.status = ADDRESSED
        elif fa["alpha"] > 0:
            c.status = PARTIAL
            c.finding += " Alpha isn't distinguishable from zero: the return is mostly factor exposure you could buy cheaply."
        else:
            c.status = FAILED
            c.finding += " No alpha after factors: a factor ETF mix would have done the same job."
    else:
        c.status = NOT
        c.finding = c.finding or "Daily factor data only covers US equities at 252 periods a year."
    out.append(c)
    return out, ew


# --------------------------------------------------------------------------- full metrics

def full_metrics(rr, bench_r: pd.Series, ew: pd.Series, ppy: int):
    r, rf = rr.sim.returns, rr.sim.rf
    ex = r - rf
    s = rr.summary
    dd = stats.drawdown(r)
    m = (1 + r).resample("ME").prod() - 1
    y = stats.yearly(r)
    total = float((1 + r).prod() - 1)
    gain = ex[ex > 0].sum()
    loss = -ex[ex < 0].sum()
    omega = float(gain / loss) if loss > 0 else float("inf")
    ba = stats.beta_alpha(r, bench_r, rf, ppy)
    act = (r - bench_r).dropna()
    te = float(act.std() * math.sqrt(ppy))
    ir = float(act.mean() * ppy / te) if te > 0 else float("nan")
    up = bench_r > 0
    upcap = float(r[up].mean() / bench_r[up].mean()) if bench_r[up].mean() != 0 else float("nan")
    dncap = float(r[~up].mean() / bench_r[~up].mean()) if bench_r[~up].mean() != 0 else float("nan")
    var95, var99 = float(r.quantile(0.05)), float(r.quantile(0.01))
    gross = rr.sim.held.abs().sum(axis=1)
    ep = position_episodes(rr.sim.held, rr.sim.contrib)
    wins, losses = ep[ep > 0], ep[ep < 0]
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    trough = dd.idxmin() if len(dd) else None
    bsum = stats.summary(bench_r, rf, ppy)
    esum = stats.summary(ew, rf, ppy)

    rows = [
        ("Returns", [
            ("Total return", _fp(total), _fp((1 + bench_r).prod() - 1)),
            ("CAGR", _fp(s["cagr"]), _fp(bsum["cagr"])),
            ("Average month", _fp(float(m.mean()), 2), ""),
            ("Best year", _fp(float(y.max())), ""),
            ("Worst year", _fp(float(y.min())), ""),
            ("Positive months", _fp(float((m > 0).mean()), 0), ""),
            ("Positive years", _fp(float((y > 0).mean()), 0), ""),
            ("Hit rate (days)", _fp(s["hit"], 0), _fp(bsum["hit"], 0)),
        ]),
        ("Risk-adjusted", [
            ("Sharpe (excess of T-bills)", _f(s["sharpe"]), _f(bsum["sharpe"])),
            ("Sortino", _f(s["sortino"]), _f(bsum["sortino"])),
            ("Calmar", _f(s["calmar"]), _f(bsum["calmar"])),
            ("Omega (0)", _f(omega), ""),
            ("Lo-adjusted Sharpe", _f(s["lo_sharpe"]), _f(bsum["lo_sharpe"])),
            ("Probabilistic Sharpe", _fp(s["psr"], 0), _fp(bsum["psr"], 0)),
            ("t-statistic", _f(s["t_stat"]), _f(bsum["t_stat"])),
            ("Information ratio", _f(ir), ""),
        ]),
        ("Risk", [
            ("Volatility", _fp(s["vol"]), _fp(bsum["vol"])),
            ("Max drawdown", _fp(s["max_dd"]), _fp(bsum["max_dd"])),
            ("Max drawdown trough", str(trough.date()) if trough is not None else "n/a", ""),
            ("Longest drawdown", f"{s['dd_duration']} periods", f"{bsum['dd_duration']} periods"),
            ("Average drawdown", _fp(avg_dd), ""),
            ("Time more than 10% below high", _fp(float((dd < -0.10).mean()), 0), ""),
            ("Ulcer index", _fp(float(np.sqrt((dd ** 2).mean()))), ""),
            ("VaR 95% / 99% (day)", f"{_fp(var95, 2)} / {_fp(var99, 2)}", ""),
            ("Expected shortfall 95% / 99%", f"{_fp(s['es95'], 2)} / {_fp(s['es99'], 2)}", ""),
            ("Worst day / week / month", f"{_fp(s['worst_day'])} / {_fp(s['worst_week'])} / {_fp(s['worst_month'])}",
             f"{_fp(bsum['worst_day'])} / {_fp(bsum['worst_week'])} / {_fp(bsum['worst_month'])}"),
            ("Skew / excess kurtosis", f"{s['skew']:.2f} / {s['kurt']:.1f}", f"{bsum['skew']:.2f} / {bsum['kurt']:.1f}"),
        ]),
        ("Versus the market", [
            ("Beta", _f(ba["beta"]), "1.00"),
            ("Alpha (annual)", _fp(ba["alpha"]), ""),
            ("Alpha t-stat", _f(ba["alpha_t"]), ""),
            ("Correlation", _f(ba["corr"]), ""),
            ("Tracking error", _fp(te), ""),
            ("Up capture / down capture", f"{_f(upcap)} / {_f(dncap)}", ""),
            ("Own universe, equal weight: Sharpe / CAGR", f"{esum['sharpe']:.2f} / {_fp(esum['cagr'])}", ""),
        ]),
        ("Trading", [
            ("Average gross exposure", f"{gross.mean():.2f}x", ""),
            ("Time in market", _fp(float((gross > 1e-9).mean()), 0), ""),
            ("Annual turnover", f"{rr.sim.turnover.sum() / (len(r) / ppy):.1f}x", ""),
            ("Positions taken", str(len(ep)), ""),
            ("Position hit rate", _fp(float((ep > 0).mean()), 0) if len(ep) else "n/a", ""),
            ("Profit factor", _f(float(wins.sum() / -losses.sum())) if len(losses) and losses.sum() < 0 else "n/a", ""),
            ("Payoff (avg win / avg loss)", _f(float(wins.mean() / -losses.mean())) if len(wins) and len(losses) else "n/a", ""),
            ("Cost per unit traded", f"{rr.sim.cost_bps_per_turnover:.1f} bp", ""),
            ("Total costs paid", _fp(float(rr.sim.costs.sum())), ""),
        ]),
    ]
    monthly = m.to_frame("r")
    monthly["y"], monthly["m"] = monthly.index.year, monthly.index.month
    grid = monthly.pivot(index="y", columns="m", values="r")
    return rows, grid, y


# --------------------------------------------------------------------------- monte carlo study

def mc_study(rr, ppy, sens, quick):
    r = rr.sim.returns
    paths = 1000 if quick else 5000
    horizons = {}
    for yrs in (1, 3, 5):
        res = montecarlo.run(r, ppy, years=yrs, paths=paths)
        horizons[yrs] = {k: {kk: vv for kk, vv in res[k].items() if kk != "fan"} for k in ("block", "iid")}
        if yrs == 3:
            fan3 = res["block"]["fan"]
    # 2x costs
    extra_cost = rr.sim.costs
    r2 = r - extra_cost
    stress = montecarlo.run(r2, ppy, years=3, paths=paths)["block"]
    stress.pop("fan", None)
    # parameter uncertainty: each path draws one nearby parameter set, then block-bootstraps it
    param = None
    if sens:
        rng = np.random.default_rng(31)
        series = [s["returns"].reindex(r.index).fillna(0.0).to_numpy() for s in sens]
        L = 3 * ppy
        res = []
        for _ in range(paths // 5):
            x = series[rng.integers(len(series))]
            idx = montecarlo.stationary_indices(len(x), L, 1, 20.0, rng)[0]
            res.append(x[idx])
        st = montecarlo.path_stats(np.array(res), ppy)
        param = dict(cagr_p=np.percentile(st["cagr"], [5, 25, 50, 75, 95]).tolist(),
                     mdd_p=np.percentile(st["mdd"], [5, 25, 50, 75, 95]).tolist(),
                     p_loss=float((st["final"] < 1).mean()), p_dd50=float((st["mdd"] <= -0.5).mean()))
    return dict(horizons=horizons, stress_2x=stress, param=param, paths=paths, fan3=fan3)

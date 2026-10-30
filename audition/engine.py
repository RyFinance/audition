"""Audition's own execution simulator. The strategy only supplies target weights;
fills, costs, financing and capacity are modelled here, independent of the author's code.

Timing: weights decided at the close of day t are filled at the open of day t+delay
and held open-to-open. fill="close" instead fills at the close of t+delay and holds
close-to-close, so fill="close", delay=0 is the impossible "trade at the close you
just observed".
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .data import Bars
from .regions import buy_tax_bps


@dataclass
class CostModel:
    fee_bps: float
    spread_bps: pd.Series          # full quoted spread estimate per ticker
    buy_tax_bps: pd.Series
    capital: float = 1_000_000.0
    borrow_bps_yr: float = 50.0
    impact_coef: float = 0.5       # square-root law: cost = coef * sigma * sqrt(trade / ADV); 0.5 is mid-range of published fits
    spread_detail: dict = None     # per ticker: abdi_ranaldo, liquidity, used, dollar_volume
    impact: bool = True
    cost_mult: float = 1.0
    volume_in_dollars: bool = False
    usd_per_unit: pd.Series | float = 1.0   # converts price x volume into dollars

    def scaled(self, **kw) -> "CostModel":
        return replace(self, **kw)


@dataclass
class SimResult:
    returns: pd.Series        # net, per period
    gross: pd.Series          # before costs, financing and borrow (excluding cash interest)
    costs: pd.Series
    borrow: pd.Series
    cash_return: pd.Series
    turnover: pd.Series
    held: pd.DataFrame        # weights held during each period
    contrib: pd.DataFrame     # held * asset return
    participation: pd.DataFrame
    rf: pd.Series
    cost_bps_per_turnover: float

    @property
    def excess(self) -> pd.Series:
        return self.returns - self.rf


def abdi_ranaldo_spread_bps(bars: Bars, floor_bps: float) -> pd.Series:
    """Close-high-low effective spread estimator (Abdi & Ranaldo 2017), median of monthly values."""
    c = np.log(bars.close)
    eta = (np.log(bars.high) + np.log(bars.low)) / 2
    s2 = 4 * (c - eta) * (c - eta.shift(-1))
    monthly = s2.resample("ME").mean().clip(lower=0) ** 0.5
    est = (monthly.median() * 1e4).fillna(floor_bps)
    return est.clip(lower=floor_bps, upper=500.0)


# Typical quoted spread (bps) by median daily dollar volume. High/low estimators like
# Abdi-Ranaldo are unreliable for liquid names (intraday volatility swamps a one-tick spread,
# and negative months clip to zero), so liquid names use the tier and only names trading
# under $2M a day fall back to the high/low estimate. Never below the region floor.
LIQUIDITY_TIERS = [(1e9, 1.0), (2e8, 2.0), (5e7, 4.0), (1e7, 10.0), (2e6, 25.0)]


def dollar_volume_usd(bars: Bars, usd, volume_in_dollars: bool) -> pd.DataFrame:
    dv = bars.volume if volume_in_dollars else bars.volume * bars.close
    if isinstance(usd, pd.Series):
        return dv.mul(usd.reindex(dv.index).ffill().bfill(), axis=0)
    return dv * usd


def liquidity_spread_bps(median_dv: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=median_dv.index)
    for t, v in median_dv.items():
        for threshold, bps in LIQUIDITY_TIERS:
            if np.isfinite(v) and v >= threshold:
                out[t] = bps
                break
    return out


def estimate_spreads(bars: Bars, region_cfg: dict, usd) -> tuple[pd.Series, dict]:
    floor = region_cfg["spread_floor_bps"]
    ar = abdi_ranaldo_spread_bps(bars, floor)
    dv = dollar_volume_usd(bars, usd, region_cfg.get("volume_in_dollars", False))
    med = dv.tail(504).median()  # last two years: spreads today, not in 2008
    liq = liquidity_spread_bps(med)
    used = liq.fillna(ar).clip(lower=floor)
    detail = {t: dict(abdi_ranaldo=float(ar.get(t, np.nan)), liquidity=float(liq.get(t, np.nan)),
                      used=float(used[t]), dollar_volume=float(med.get(t, np.nan))) for t in used.index}
    return used, detail


def cost_model(bars: Bars, region_cfg: dict, capital: float, fx: pd.Series | None = None,
               overrides: dict | None = None) -> CostModel:
    tickers = bars.tickers
    usd = region_cfg.get("price_scale", 1.0) * (fx if fx is not None else 1.0)
    spreads, detail = estimate_spreads(bars, region_cfg, usd)
    o = overrides or {}
    if "spread_bps" in o:  # your own broker's numbers beat any estimate
        sp = o["spread_bps"]
        spreads = (pd.Series(sp, dtype=float).reindex(spreads.index).fillna(spreads) if isinstance(sp, dict)
                   else pd.Series(float(sp), index=spreads.index))
        for t in detail:
            detail[t]["used"] = float(spreads[t])
            detail[t]["declared"] = True
    cm = CostModel(
        fee_bps=float(o.get("fee_bps", region_cfg["fee_bps"])),
        spread_bps=spreads, spread_detail=detail,
        buy_tax_bps=pd.Series({t: buy_tax_bps(t) for t in tickers}),
        capital=capital,
        volume_in_dollars=region_cfg.get("volume_in_dollars", False),
        usd_per_unit=usd,
    )
    if "impact_coef" in o:
        cm.impact_coef = float(o["impact_coef"])
    if "borrow_bps_yr" in o:
        cm.borrow_bps_yr = float(o["borrow_bps_yr"])
    return cm


def simulate(w: pd.DataFrame, bars: Bars, rf: pd.Series, cm: CostModel, ppy: int,
             delay: int = 1, fill: str = "open", dtype=np.float64, eval_start=None) -> SimResult:
    px = (bars.open if fill == "open" else bars.close).ffill()
    held = w.shift(delay).fillna(0.0).astype(dtype)
    fwd = (px.shift(-1) / px - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(dtype)
    # drop the final row, which has no forward return
    held, fwd = held.iloc[:-1], fwd.iloc[:-1]
    idx = held.index

    contrib = held * fwd
    gross = contrib.sum(axis=1)

    trades = held.diff().fillna(held)
    turn_i = trades.abs()
    buys = trades.clip(lower=0)

    close = bars.close.ffill().reindex(idx)
    dollar_vol = bars.volume.reindex(idx) if cm.volume_in_dollars else bars.volume.reindex(idx) * close
    fx = cm.usd_per_unit
    if isinstance(fx, pd.Series):
        fx = fx.reindex(idx).ffill().bfill()
        dollar_vol = dollar_vol.mul(fx, axis=0)
    else:
        dollar_vol = dollar_vol * fx
    adv = dollar_vol.rolling(20, min_periods=5).mean().shift(1)
    sigma = bars.close.pct_change(fill_method=None).rolling(20, min_periods=5).std().shift(1).reindex(idx)
    trade_dollars = turn_i * cm.capital
    participation = (trade_dollars / adv).replace([np.inf, -np.inf], np.nan)

    half_spread = (cm.spread_bps.reindex(w.columns).fillna(10.0) / 2.0) / 1e4
    per_side = cm.fee_bps / 1e4 + half_spread
    cost_i = turn_i * per_side + buys * (cm.buy_tax_bps.reindex(w.columns).fillna(0.0) / 1e4)
    if cm.impact:
        impact = (cm.impact_coef * sigma.fillna(sigma.stack().median()) *
                  np.sqrt(participation.fillna(1.0).clip(lower=0)))
        cost_i = cost_i + turn_i * impact.clip(upper=0.2)
    costs = cost_i.sum(axis=1) * cm.cost_mult

    borrow = (-held.clip(upper=0)).sum(axis=1) * (cm.borrow_bps_yr / 1e4 / ppy)
    rf_i = rf.reindex(idx).ffill().fillna(0.0)
    cash = 1.0 - held.sum(axis=1)
    # idle cash earns T-bills; borrowed cash (leverage) pays T-bills + 50bp
    cash_ret = np.where(cash >= 0, cash * rf_i, cash * (rf_i + 0.005 / ppy))
    cash_ret = pd.Series(cash_ret, index=idx)

    net = gross + cash_ret - costs - borrow
    if eval_start is not None:
        # the warm-up before the strategy's first position isn't part of the track record
        keep = idx >= eval_start
        net, gross, costs, borrow, cash_ret, rf_i = (x[keep] for x in (net, gross, costs, borrow, cash_ret, rf_i))
        held, contrib, participation, turn_i = held[keep], contrib[keep], participation[keep], turn_i[keep]
    tot_turn = turn_i.sum(axis=1)
    cbpt = float(costs.sum() / tot_turn.sum() * 1e4) if tot_turn.sum() > 0 else 0.0
    return SimResult(
        returns=net.astype(float), gross=gross.astype(float), costs=costs, borrow=borrow,
        cash_return=cash_ret, turnover=tot_turn, held=held, contrib=contrib,
        participation=participation, rf=rf_i, cost_bps_per_turnover=cbpt,
    )


def count_trades(held: pd.DataFrame) -> int:
    """Position openings: a ticker going from flat (or the other side) to a position."""
    s = np.sign(held.round(6))
    opened = (s != 0) & (s != s.shift(1).fillna(0))
    return int(opened.to_numpy().sum())

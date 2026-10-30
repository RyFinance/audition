"""Loads a user strategy file and runs it under Audition's control.

The contract (see examples/ and README):

    NAME     = "My strategy"                       # optional
    PARAMS   = {"lookback": 60, "top_n": 3}        # numbers here get sensitivity-tested
    UNIVERSE = {"us": ["AAPL", ...]}               # optional; a plain list means "home region"
    DECLARE  = {...}                               # optional self-reported facts, see README

    def weights(data, **params) -> pd.DataFrame:
        # data.open/high/low/close/volume: DataFrames (dates x tickers), adjusted
        # data.benchmark: Series, data.region: str
        # return target weights decided at each date's close (dates x tickers).
        # Audition fills them at the NEXT bar's open; 1.0 = 100% of equity long.
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Bars


@dataclass
class MarketData:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.Series
    region: str
    eligible: pd.DataFrame = None   # True where the asset was in the index on that date


    @property
    def tickers(self):
        return list(self.close.columns)


@dataclass
class Strategy:
    path: Path
    name: str
    params: dict
    universe: dict | list | None
    declare: dict
    fn: object
    sha256: str
    source: str = field(repr=False, default="")
    data_source: str = "yahoo"     # DATA = "polygon" | "tiingo" | "alpaca" | "yahoo", or from UNIVERSE = "byo:..."

    def numeric_params(self):
        return {k: v for k, v in self.params.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}


def load(path: str | Path) -> Strategy:
    path = Path(path).resolve()
    src = path.read_text()
    spec = importlib.util.spec_from_file_location(f"audition_user_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "weights", None)):
        raise ValueError(f"{path.name} must define weights(data, **params)")
    return Strategy(
        path=path,
        name=getattr(mod, "NAME", path.stem),
        params=dict(getattr(mod, "PARAMS", {})),
        universe=getattr(mod, "UNIVERSE", None),
        declare=dict(getattr(mod, "DECLARE", {})),
        fn=mod.weights,
        sha256=hashlib.sha256(src.encode()).hexdigest(),
        source=src,
        data_source=_data_source(mod),
    )


def _data_source(mod) -> str:
    uni = getattr(mod, "UNIVERSE", None)
    if isinstance(uni, str) and uni.startswith("byo:"):
        return uni
    d = getattr(mod, "DATA", None)
    if d:
        return str(d).lower()
    from .sources import default_source
    return default_source()


def run(strat: Strategy, bars: Bars, bench: pd.Series, region: str, params: dict | None = None) -> pd.DataFrame:
    """Call the strategy on a defensive copy of the data and normalize its output."""
    p = dict(strat.params)
    if params:
        p.update(params)
    data = MarketData(
        open=bars.open.copy(), high=bars.high.copy(), low=bars.low.copy(),
        close=bars.close.copy(), volume=bars.volume.copy(),
        benchmark=bench.reindex(bars.index).copy(), region=region,
        eligible=(bars.eligible.copy() if bars.eligible is not None
                  else pd.DataFrame(True, index=bars.index, columns=bars.tickers)),
    )
    w = strat.fn(data, **p)
    if isinstance(w, pd.Series):
        w = w.to_frame(bars.tickers[0]) if len(bars.tickers) == 1 else None
    if not isinstance(w, pd.DataFrame):
        raise TypeError("weights() must return a DataFrame of dates x tickers")
    unknown = set(w.columns) - set(bars.tickers)
    if unknown:
        raise ValueError(f"weights() returned tickers not in the universe: {sorted(unknown)[:5]}")
    w = w.reindex(index=bars.index, columns=bars.tickers).astype(float)
    if np.isinf(w.to_numpy()).any():
        raise ValueError("weights() returned inf")
    w = w.fillna(0.0)
    if bars.eligible is not None:
        w = enforce_eligibility(w, bars.eligible)
    return w


def enforce_eligibility(w: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Drop every position that was opened while the asset wasn't in the index.

    A position opened while eligible may be held after the asset leaves (the strategy sells it
    at its next rebalance), matching how an index-tracking rule really trades. The count of
    dropped position-days is kept in w.attrs for the audit."""
    held = w.ne(0.0)
    start = held & ~held.shift(1, fill_value=False)
    elig = eligible.reindex(index=w.index, columns=w.columns, fill_value=False)
    # eligibility on each position's opening day, carried through the rest of that position
    opened_ok = elig.where(start).ffill().fillna(False).astype(bool)
    bad = held & ~opened_ok
    n_bad = int(bad.to_numpy().sum())
    n_open_bad = int((start & ~elig).to_numpy().sum())
    out = w.mask(bad, 0.0)
    out.attrs["ineligible_position_days"] = n_bad
    out.attrs["ineligible_openings"] = n_open_bad
    return out

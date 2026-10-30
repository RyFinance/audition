"""Cross-sectional momentum: hold the top names by 12-1 month return, rebalanced monthly,
only while each name is above its 200-day average. An honest, unremarkable strategy."""

import pandas as pd

NAME = "12-1 momentum, top 5"
PARAMS = {"lookback": 252, "skip": 21, "top_n": 5, "trend": 200}
DECLARE = {
    "variants_tried": 12,
    "oos_start": "2022-01-01",
    "oos_looks": 1,
}


def weights(data, lookback, skip, top_n, trend):
    close = data.close
    mom = close.shift(skip) / close.shift(lookback) - 1
    above = close > close.rolling(trend).mean()
    # first trading day of each month (knowable on the day, unlike "last day of the month")
    months = close.index.to_series().dt.month
    first_day = (months != months.shift(1)).to_numpy()
    rank = mom.where(above).rank(axis=1, ascending=False)
    w = (rank <= top_n).astype(float) / top_n
    # trade once a month, hold in between
    w.loc[~first_day] = float("nan")
    return w.ffill().fillna(0.0)

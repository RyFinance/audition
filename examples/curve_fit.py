"""Short-term mean reversion on a single ETF with suspiciously precise parameters,
picked as the best of many tries. Audition should flag selection and sensitivity."""

UNIVERSE = ["QQQ"]
NAME = "QQQ dip buyer"
PARAMS = {"lookback": 3, "entry": -0.0137, "hold": 4}
DECLARE = {
    "variants_tried": 400,
    "reported": {"cagr": 0.31, "sharpe": 1.9, "max_dd": -0.12},
}


def weights(data, lookback, entry, hold):
    c = data.close.iloc[:, 0]
    r = c / c.shift(lookback) - 1
    signal = (r < entry).astype(float)
    held = signal.rolling(hold, min_periods=1).max()
    return held.to_frame(data.close.columns[0])

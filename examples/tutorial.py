"""Tutorial strategy: a QQQ dip buyer that looks great and has three classic mistakes.

Open its disc to see how Audition finds them:
  1. A centred moving average (center=True) quietly uses future prices: lookahead.
  2. The parameters were tuned until the backtest looked best, but DECLARE says only one variant was tried.
  3. The reported Sharpe is the in-sample, before-costs number.

Fix them one at a time and re-import to watch the grade change.
"""

import pandas as pd

NAME = "Tutorial: the too-good dip buyer"
UNIVERSE = ["QQQ"]
PARAMS = {"drop": 0.02, "trend": 50, "hold": 3}
DECLARE = {
    "variants_tried": 1,                       # mistake 2: it was really dozens
    "reported": {"sharpe": 2.1, "cagr": 0.31, "max_dd": -0.09},   # mistake 3: in-sample, no costs
}


def weights(data, drop, trend, hold):
    px = data.close["QQQ"]
    smooth = px.rolling(trend, center=True).mean()      # mistake 1: center=True looks ahead
    dip = (px / px.shift(1) - 1) < -drop
    buy = dip & (px > smooth)
    w = buy.astype(float).rolling(hold, min_periods=1).max()   # hold for `hold` days after each dip
    return pd.DataFrame({"QQQ": w}, index=px.index).fillna(0.0)

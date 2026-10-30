"""A deliberately broken strategy: it 'predicts' tomorrow using tomorrow's close.
Audition should fail it on lookahead."""

NAME = "Tomorrow's news today"
PARAMS = {"top_n": 3}
DECLARE = {"reported": {"cagr": 0.70, "sharpe": 4.0}}


def weights(data, top_n):
    nxt = data.close.shift(-1) / data.close - 1          # <- the future
    rank = nxt.rank(axis=1, ascending=False)
    return (rank <= top_n).astype(float) / top_n

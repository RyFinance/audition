"""The 60 audit questions, grouped as in the review template, and the top-10 priorities."""

ADDRESSED, PARTIAL, FAILED, NOT = "addressed", "partial", "failed", "not_addressed"
LABEL = {ADDRESSED: "Addressed", PARTIAL: "Partial", FAILED: "Failed", NOT: "Not addressed"}
# lower is worse
RANK = {FAILED: 0, NOT: 1, PARTIAL: 2, ADDRESSED: 3}

CATEGORIES = [
    ("research", "Research methodology"),
    ("data", "Historical data integrity"),
    ("execution", "Backtesting and execution realism"),
    ("metrics", "Performance metrics"),
    ("montecarlo", "Monte Carlo validation"),
    ("independence", "Independent validation"),
    ("forward", "Forward paper trading and live deployment"),
    ("extended", "Beyond the 60"),
]

# id: (category, title, what must be established, scope)
# scope "backtest" = Audition can measure it; "process" = depends on how you worked;
# "live" = needs forward or live trading, outside any backtest.
CHECKS = {
    1: ("research", "Lookahead bias", "No decision uses information that wasn't available at the decision timestamp.", "backtest"),
    2: ("research", "Data leakage", "No future information enters features, normalization, training, ranking or signals.", "backtest"),
    3: ("research", "Survivorship bias", "Historical universes include companies that later delisted or failed.", "backtest"),
    4: ("research", "Multiple testing", "The winner wasn't picked from many attempts without correcting for selection.", "backtest"),
    5: ("research", "Overfitting", "Parameters aren't tuned to historical noise.", "backtest"),
    6: ("research", "Out-of-sample contamination", "Out-of-sample data wasn't repeatedly inspected during development.", "process"),
    7: ("research", "Regime dependency", "The edge isn't explained by one market regime.", "backtest"),
    8: ("research", "Parameter sensitivity", "Small reasonable parameter changes don't destroy profitability.", "backtest"),
    9: ("research", "Sample size", "Enough independent observations to support the statistics.", "backtest"),
    10: ("research", "Statistical significance", "Performance isn't a lucky draw from a weak strategy.", "backtest"),
    11: ("data", "Price adjustments", "Splits, dividends and corporate actions are handled correctly.", "backtest"),
    12: ("data", "Historical constituents", "The universe at each date is reconstructed correctly.", "backtest"),
    13: ("data", "Timestamp alignment", "Data across exchanges, timezones and intervals is synchronized.", "backtest"),
    14: ("data", "Missing data", "Missing observations aren't turned into favorable signals or returns.", "backtest"),
    15: ("data", "Data revisions", "Revised history doesn't leak future information.", "backtest"),
    16: ("data", "Exchange calendars", "Holidays, short sessions and closures are handled.", "backtest"),
    17: ("data", "Bad ticks", "Erroneous extreme prices don't create artificial entries or exits.", "backtest"),
    18: ("data", "Data source independence", "The data itself isn't a common error across every test.", "backtest"),
    19: ("execution", "Fill assumptions", "Orders aren't filled at prices that were impossible to get.", "backtest"),
    20: ("execution", "Slippage", "Slippage reflects execution conditions, not a fixed optimistic number.", "backtest"),
    21: ("execution", "Bid-ask spread", "Spread costs are accounted for.", "backtest"),
    22: ("execution", "Partial fills", "Orders that can't execute entirely are accounted for.", "backtest"),
    23: ("execution", "Market impact", "The simulation doesn't assume unlimited liquidity.", "backtest"),
    24: ("execution", "Latency", "Signal and execution delays are realistic.", "backtest"),
    25: ("execution", "Trading fees", "Commissions, exchange fees and taxes are included.", "backtest"),
    26: ("execution", "Shorting costs", "Borrow availability, fees and recalls are accounted for.", "backtest"),
    27: ("execution", "Order execution", "Order types, gaps and intrabar logic are modelled correctly.", "backtest"),
    28: ("execution", "Capacity", "The strategy works at the intended capital without destroying the edge.", "backtest"),
    29: ("metrics", "CAGR calculation", "Correct annualization, compounding and idle-capital treatment.", "backtest"),
    30: ("metrics", "Sharpe calculation", "Correct frequency, annualization and risk-free rate.", "backtest"),
    31: ("metrics", "Serial correlation", "Return smoothing isn't inflating the Sharpe.", "backtest"),
    32: ("metrics", "Maximum drawdown", "Peak-to-trough losses are measured accurately.", "backtest"),
    33: ("metrics", "Drawdown duration", "Recovery periods aren't ignored.", "backtest"),
    34: ("metrics", "Tail risk", "Rare large losses aren't hidden by a good average.", "backtest"),
    35: ("metrics", "Leverage", "Returns and risk account for leverage.", "backtest"),
    36: ("metrics", "Concentration", "Performance doesn't hinge on one position or asset.", "backtest"),
    37: ("metrics", "Correlation", "Positions aren't one large correlated bet.", "backtest"),
    38: ("metrics", "Benchmark comparison", "Compared with passive alternatives and risk exposures.", "backtest"),
    39: ("montecarlo", "Resampling methodology", "Simulations keep temporal and cross-asset dependence.", "backtest"),
    40: ("montecarlo", "Tail distributions", "Simulations don't understate extreme losses.", "backtest"),
    41: ("montecarlo", "Parameter uncertainty", "Results don't assume the estimated parameters are exact.", "backtest"),
    42: ("montecarlo", "Cost stress testing", "Performance survives materially worse costs.", "backtest"),
    43: ("montecarlo", "Regime changes", "Simulations account for changing conditions.", "backtest"),
    44: ("montecarlo", "Sample independence", "Resampling doesn't multiply the independent information.", "backtest"),
    45: ("montecarlo", "Ruin probability", "Probability of severe capital impairment is quantified.", "backtest"),
    46: ("independence", "Shared tool dependency", "Validations aren't all reproducing one tool's error.", "process"),
    47: ("independence", "Shared specifications", "Validators aren't inheriting the same wrong assumptions.", "process"),
    48: ("independence", "Independent calculations", "Returns and risk are recomputed, not taken from the tool.", "backtest"),
    49: ("independence", "Implementation equivalence", "Different implementations follow the same rules.", "backtest"),
    50: ("independence", "Reproducibility", "Inputs, outputs and configs are preserved.", "backtest"),
    51: ("independence", "Numerical accuracy", "Floating point and rounding don't move results.", "backtest"),
    52: ("independence", "Independent verification", "An implementation outside the original tool verifies the numbers.", "backtest"),
    53: ("forward", "Forward performance", "Results continue outside the research dataset.", "live"),
    54: ("forward", "Paper/live discrepancy", "Paper fills resemble achievable real fills.", "live"),
    55: ("forward", "Execution reliability", "No missed signals, duplicate orders or bad position accounting.", "live"),
    56: ("forward", "Risk controls", "Position limits, loss limits and kill switches work.", "live"),
    57: ("forward", "Data outages", "Behaves safely when data is missing or corrupted.", "live"),
    58: ("forward", "Strategy degradation", "Deterioration is detected before much capital is lost.", "live"),
    59: ("forward", "Capital allocation", "Initial capital is sized conservatively.", "live"),
    60: ("forward", "Operational resilience", "Restarts, API failures and rejects are handled safely.", "live"),
    61: ("extended", "Turnover and holding period", "Trading frequency and holding periods are what the strategy claims.", "backtest"),
    62: ("extended", "Cross-market replication", "The edge shows up in markets it wasn't built on.", "backtest"),
    63: ("extended", "Rolling stability", "Performance is spread through time, not one stretch.", "backtest"),
    64: ("extended", "Timing luck", "Trading a few days later doesn't change the result.", "backtest"),
    65: ("extended", "Start-date sensitivity", "Starting in a different year gives a similar outcome.", "backtest"),
    66: ("extended", "Universe sensitivity", "Dropping random names from the universe doesn't break it.", "backtest"),
    67: ("extended", "Best-day dependence", "Return doesn't hinge on a handful of days.", "backtest"),
    68: ("extended", "Own-universe benchmark", "Beats simply equal-weighting its own universe.", "backtest"),
    69: ("extended", "Win/loss profile", "Profit factor, hit rate and payoff per position are plausible.", "backtest"),
    70: ("extended", "Drawdown pain", "Ulcer index, recovery and time under water are tolerable.", "backtest"),
    71: ("extended", "Factor attribution", "Returns aren't just known factor exposure (market, size, value, quality, momentum).", "backtest"),
}

TOP10 = [
    ("Lookahead bias and data leakage", [1, 2]),
    ("Multiple testing and selection bias", [4]),
    ("Untouched out-of-sample validation", [6]),
    ("Independent verification of the numbers", [48, 49, 52]),
    ("Fills, slippage, spreads and market impact", [19, 20, 21, 23]),
    ("Survivorship and universe reconstruction", [3, 12]),
    ("Sharpe inflation from serial correlation", [31]),
    ("Monte Carlo method and tail risk", [39, 40, 45]),
    ("Parameter sensitivity and regime robustness", [8, 7]),
    ("Forward paper trading and execution", [53, 55]),
]
TOP10_IDS = {i for _, ids in TOP10 for i in ids}


def worst(statuses):
    return min(statuses, key=lambda s: RANK[s]) if statuses else NOT

"""Markets Audition can replay a strategy in, with their default universes and cost rules.

Default universes are today's large caps. They are survivors by construction, so
replication on them is evidence of a portable edge, not a survivorship-free test.
"""

REGIONS = {
    "us": dict(
        name="United States", bench="SPY", ppy=252, start="2010-01-01",
        fee_bps=0.5, spread_floor_bps=1.0,
        universe=["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "PG", "KO", "WMT",
                  "HD", "CVX", "PFE", "CSCO", "INTC", "BAC", "MRK", "PEP", "DIS", "MCD"],
    ),
    "uk": dict(
        name="United Kingdom", bench="ISF.L", fx="GBPUSD=X", price_scale=0.01, ppy=252, start="2010-01-01",
        fee_bps=0.5, spread_floor_bps=3.0,
        universe=["HSBA.L", "SHEL.L", "AZN.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "BATS.L",
                  "DGE.L", "LLOY.L", "BARC.L", "VOD.L", "REL.L", "NG.L", "TSCO.L"],
    ),
    "eu": dict(
        name="Europe", bench="EXS1.DE", fx="EURUSD=X", ppy=252, start="2010-01-01",
        fee_bps=1.0, spread_floor_bps=3.0,
        universe=["SAP.DE", "SIE.DE", "ALV.DE", "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "ASML.AS",
                  "BAYN.DE", "BAS.DE", "DTE.DE", "AIR.PA", "BNP.PA", "ENEL.MI", "ISP.MI"],
    ),
    "jp": dict(
        name="Japan", bench="1306.T", fx="JPYUSD=X", ppy=252, start="2010-01-01",
        fee_bps=1.0, spread_floor_bps=3.0,
        universe=["7203.T", "6758.T", "9984.T", "8306.T", "6861.T", "9432.T", "8035.T", "4063.T",
                  "6501.T", "7974.T", "8058.T", "4502.T", "6098.T", "9433.T", "8316.T"],
    ),
    "crypto": dict(
        name="Crypto", bench="BTC-USD", ppy=365, start="2018-01-01",
        fee_bps=10.0, spread_floor_bps=2.0, volume_in_dollars=True,
        universe=["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
                  "LTC-USD", "LINK-USD", "SOL-USD", "AVAX-USD"],
    ),
}

# Transaction taxes on buys, by listing suffix (bps). UK stamp duty, French and Italian FTT.
BUY_TAX_BPS = {".L": 50.0, ".PA": 40.0, ".MI": 10.0}
# Funds are exempt from UK stamp duty; the benchmark ETFs above are all funds.
TAX_EXEMPT = {r["bench"] for r in REGIONS.values()}

RISK_FREE = "^IRX"  # 13-week T-bill yield, used for every region (USD)


def buy_tax_bps(ticker: str) -> float:
    if ticker in TAX_EXEMPT:
        return 0.0
    for suffix, bps in BUY_TAX_BPS.items():
        if ticker.endswith(suffix):
            return bps
    return 0.0


def infer_region(tickers) -> str:
    """Pick a cost region for a custom universe from its listing suffixes."""
    t = list(tickers)
    if t and all(x.endswith("-USD") for x in t):
        return "crypto"
    if t and all(x.endswith(".L") for x in t):
        return "uk"
    if t and all(x.endswith(".T") for x in t):
        return "jp"
    if t and all(any(x.endswith(s) for s in (".DE", ".PA", ".AS", ".MI", ".MC")) for x in t):
        return "eu"
    return "us"

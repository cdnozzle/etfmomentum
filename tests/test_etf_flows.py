import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import etf_flows as ef  # noqa: E402


def fund(name, us=0, non_us=0, bond=0, other=0, cash=0, regions=None, sectors=None, holdings=None):
    alloc = {"Stock US": us, "Stock non-US": non_us, "Bond": bond, "Other": other, "Cash": cash}
    return {
        "Name": name,
        "Asset_Allocation": {k: {"Net_Assets_%": v} for k, v in alloc.items()},
        "World_Regions": {k: {"Equity_%": v} for k, v in (regions or {}).items()},
        "Sector_Weights": {k: {"Equity_%": v} for k, v in (sectors or {}).items()},
        "Top_10_Holdings": holdings or {},
    }


@pytest.mark.parametrize("f, expected", [
    (fund("SPDR S&P 500 ETF Trust", us=99.8, sectors={"Technology": 33}), ("Equity", "United States", "Broad Market")),
    (fund("Technology Select Sector SPDR Fund", us=99.9), ("Equity", "United States", "Technology")),
    (fund("VanEck Semiconductor ETF", us=80, non_us=20), ("Equity", "United States", "Semiconductors")),
    (fund("iShares MSCI South Korea ETF", non_us=100, regions={"Asia Developed": 100}, sectors={"Technology": 58}),
     ("Equity", "South Korea", "Broad Market")),
    (fund("Vanguard FTSE Europe ETF", non_us=99), ("Equity", "Europe", "Broad Market")),
    (fund("iShares MSCI Europe Financials ETF", non_us=99), ("Equity", "Europe", "Financials")),
    (fund("iShares Global Clean Energy ETF", us=40, non_us=60), ("Equity", "Global", "Clean Energy")),
    (fund("VanEck Gold Miners ETF", us=15, non_us=85), ("Equity", "Developed ex-US", "Materials & Mining")),
    (fund("iShares 20+ Year Treasury Bond ETF", bond=97), ("Fixed Income", "United States", "Treasury")),
    (fund("iShares 0-3 Month Treasury Bond ETF", bond=99), ("Fixed Income", "United States", "T-Bills / Ultra-Short")),
    (fund("JPMorgan Ultra-Short Income ETF", bond=90, cash=10), ("Fixed Income", "United States", "T-Bills / Ultra-Short")),
    (fund("iShares iBoxx $ High Yield Corporate Bond ETF", bond=98), ("Fixed Income", "United States", "High Yield")),
    (fund("iShares J.P. Morgan USD Emerging Markets Bond ETF", bond=98), ("Fixed Income", "Emerging Markets", "EM Debt")),
    (fund("iShares National Muni Bond ETF", bond=99), ("Fixed Income", "United States", "Municipal")),
    (fund("Vanguard Total International Bond ETF", bond=99), ("Fixed Income", "Developed ex-US", "Aggregate / Core")),
    (fund("SPDR Gold Shares", other=100), ("Commodity", "Global", "Gold")),
    (fund("United States Oil Fund LP", other=100), ("Commodity", "Global", "Oil")),
    (fund("iShares Bitcoin Trust", other=104.9), ("Crypto", "Global", "Bitcoin")),
    (fund("Grayscale Ethereum Trust", other=100), ("Crypto", "Global", "Ether")),
    (fund("ProShares UltraPro QQQ", us=100), ("Leveraged / Inverse", "United States", "Index / Single Stock")),
    (fund("ProShares Short S&P500", cash=100), ("Leveraged / Inverse", "United States", "Index / Single Stock")),
    (fund("Direxion Daily 20+ Year Treasury Bull 3X Shares", bond=100), ("Leveraged / Inverse", "United States", "Treasury")),
    (fund("iShares Core 60/40 Balanced Allocation ETF", us=40, non_us=20, bond=40), ("Multi-Asset", "Global", "Allocation")),
    (fund("iMGP DBi Managed Futures Strategy ETF", cash=100), ("Alternatives", "Global", "Alternatives")),
    (fund("Communication Services Select Sector SPDR Fund", us=100), ("Equity", "United States", "Communication Services")),
])
def test_classify(f, expected):
    assert ef.classify_etf(f) == expected


def test_country_from_holdings():
    holdings = {f"H{i}": {"Country": "Japan", "Assets_%": 5} for i in range(10)}
    f = fund("WisdomTree Hedged Equity Fund", non_us=100, holdings=holdings)
    assert ef.classify_etf(f)[1] == "Japan"


def snaps(rows):
    df = pd.DataFrame(rows, columns=ef.SNAPSHOT_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_flows_strip_out_price_moves():
    s = snaps([
        ("2026-01-02", "AAA", 100.0, 10.0, 10.0),
        ("2026-02-02", "AAA", 110.0, 11.0, 11.0),   # +10% price, no flow
        ("2026-03-02", "AAA", 130.0, 11.0, 11.0),   # +20 inflow
        ("2026-04-02", "AAA", 117.0, 9.9, 9.9),     # -10% price on 130 -> 117, no flow
    ])
    fs = ef.compute_flow_series(s)
    assert fs["flow"].round(6).tolist() == [0.0, 20.0, 0.0]


def test_distribution_is_not_an_outflow():
    # $1 dividend: close drops 10 -> 9, adjusted close flat, AUM drops by the cash paid out
    s = snaps([("2026-01-02", "AAA", 100.0, 10.0, 9.0), ("2026-01-03", "AAA", 90.0, 9.0, 9.0)])
    assert ef.compute_flow_series(s)["flow"].round(6).tolist() == [0.0]


def test_split_falls_back_to_adjusted_ratio():
    # 2:1 split, no flow: raw close halves, adjusted close unchanged, AUM unchanged
    s = snaps([("2026-01-02", "AAA", 100.0, 20.0, 10.0), ("2026-01-03", "AAA", 100.0, 10.0, 10.0)])
    assert ef.compute_flow_series(s)["flow"].round(6).tolist() == [0.0]


def test_window_flows_and_coverage():
    s = snaps([
        ("2025-12-31", "AAA", 100.0, 10.0, 10.0),
        ("2026-06-30", "AAA", 150.0, 10.0, 10.0),   # +50
        ("2026-09-30", "AAA", 160.0, 10.0, 10.0),   # +10
        ("2026-08-01", "BBB", 50.0, 5.0, 5.0),
        ("2026-09-30", "BBB", 45.0, 5.0, 5.0),      # -5
    ])
    fs = ef.compute_flow_series(s)
    ytd = ef.window_flows(fs, s, "2025-12-31")
    assert ytd.loc["AAA", "flow"] == pytest.approx(60.0)
    assert bool(ytd.loc["AAA", "full_coverage"]) and not bool(ytd.loc["BBB", "full_coverage"])
    q = ef.window_flows(fs, s, "2026-07-11")
    assert q.loc["AAA", "flow"] == pytest.approx(10.0)   # anchored on the 2026-06-30 snapshot
    assert q.loc["BBB", "flow"] == pytest.approx(-5.0)


def test_merge_snapshots_new_wins():
    a = snaps([("2026-01-02", "AAA", 100.0, 10.0, 10.0)])
    b = snaps([("2026-01-02", "AAA", 105.0, 10.0, 10.0)])
    m = ef.merge_snapshots(a, b)
    assert len(m) == 1 and m["aum"].iloc[0] == 105.0


def test_aggregate_flow_pct():
    t = pd.DataFrame({"asset_class": ["Equity", "Equity"], "aum": [110.0, 90.0],
                      "flow_ytd": [10.0, -10.0], "flow_3m": [0.0, 0.0]})
    agg = ef.aggregate_flows(t, ["asset_class"])
    assert agg["flow_ytd_pct"].iloc[0] == 0.0 and agg["n_etfs"].iloc[0] == 2

## etf 1yr -1m momentum strategy visualization

Streamlit app (EODHD data) with two pages:

- **ETF Momentum** (`etf_momentum.py`): 12-1 month and vol-adjusted momentum backtests on an ETF universe.
- **ETF Flows** (`pages/1_ETF_Flows.py`): US ETF net flows YTD and over the last 3 months, grouped
  asset class → country / region → sector / segment (treemap, grouped tables, top movers, flow vs. return).

```bash
pip install -r requirements.txt
streamlit run etf_momentum.py   # ETF Flows appears in the sidebar page list
```

### How flows are estimated

EODHD has no fund-flow, historical AUM or ETF shares-outstanding endpoint; its fundamentals API only
reports each ETF's *current* `TotalAssets`. Flows are therefore estimated from a history of AUM
snapshots stored in `data/aum_snapshots.csv`:

```
flow_t = AUM_t - AUM_(t-1) * (close_t / close_(t-1))
```

i.e. the AUM change that the price change does not explain (distributions are not counted as
outflows; splits are detected and handled). Window flows are the sum of per-snapshot flows.

The history is built by:

1. **Daily GitHub Action** (`.github/workflows/aum_snapshot.yml`): add an `EODHD_API_KEY` repository
   secret and it commits a snapshot every weekday after the close. Run manually with
   `EODHD_API_KEY=... python scripts/snapshot_aum.py [--all] [--min-aum 1e8]`.
2. **The page itself**: every load adds today's snapshot.
3. **Importing history**: upload a CSV with `date,ticker,aum` (e.g. from issuer websites) to backfill
   so YTD / 3M flows are available right away. Prices for those dates are looked up automatically.

Until the history covers a window, the page flags which ETFs only have partial coverage.

### Classification

`etf_flows.classify_etf` uses the fund name plus EODHD's asset allocation, region, sector and top-holdings
data. Asset classes: Equity, Fixed Income, Commodity, Crypto, Multi-Asset, Alternatives, Currency,
Leveraged / Inverse. The third level is the equity sector, or a segment for other classes
(e.g. Treasury / High Yield / T-Bills for bonds, Gold / Oil for commodities).

Tests: `pytest tests`

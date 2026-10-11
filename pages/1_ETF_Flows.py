import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import etf_flows as ef  # noqa: E402

st.set_page_config(page_title="US ETF Flows", layout="wide")
st.title("US ETF Flows by Asset Class, Country & Sector")

INFLOW, OUTFLOW, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
LEVELS = {"Asset class": "asset_class", "Country / region": "geography", "Sector / segment": "sector"}


# --- Cached data access ---
@st.cache_data(ttl=12 * 3600, show_spinner="Fetching ETF fundamentals from EODHD...")
def load_universe(tickers, api_key):
    return ef.build_universe_table(ef.fetch_many_fundamentals(list(tickers), api_key))


@st.cache_data(ttl=24 * 3600, show_spinner="Listing all US ETFs...")
def load_all_us_etfs(api_key):
    return ef.fetch_us_etf_list(api_key)


@st.cache_data(ttl=3600, show_spinner="Fetching latest prices...")
def load_latest_prices(api_key):
    return ef.fetch_bulk_prices(api_key)


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching window returns...")
def load_window_returns(api_key, tickers, start):
    return ef.window_returns(api_key, list(tickers), start)


def fmt_usd(x):
    if pd.isna(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(x)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if x >= div:
            return f"{sign}${x / div:,.1f}{unit}"
    return f"{sign}${x:,.0f}"


def flow_bar(df, label_col, value_col, title):
    df = df.dropna(subset=[value_col]).sort_values(value_col)
    fig = go.Figure(go.Bar(
        y=df[label_col], x=df[value_col] / 1e9, orientation="h",
        marker_color=np.where(df[value_col] >= 0, INFLOW, OUTFLOW),
        hovertemplate="%{y}: $%{x:,.2f}B<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Estimated net flow ($B)", template="plotly_white",
                      height=max(320, 28 * len(df) + 120), margin=dict(l=10, r=10, t=50, b=40), bargap=0.25)
    return fig


# --- Sidebar ---
with st.sidebar:
    st.header("Flow Screen Settings")
    api_key = st.text_input("EODHD API Key", value=os.environ.get("EODHD_API_KEY", ""), type="password")
    universe_mode = st.radio("Universe", ["Curated (~200 large ETFs)", "Curated + custom tickers", "All US-listed ETFs"],
                             help="'All US-listed ETFs' pulls fundamentals for ~4,000 ETFs (about 10 API calls each).")
    custom = ""
    if universe_mode == "Curated + custom tickers":
        custom = st.text_area("Extra tickers (comma separated)", placeholder="e.g. URNM, BUFR, DFAC")
    min_aum_bn = st.number_input("Minimum AUM ($B)", min_value=0.0, value=0.5, step=0.5)

    st.markdown("---")
    snapshot_path = st.text_input("Snapshot file", value=ef.DEFAULT_SNAPSHOT_PATH)
    save_today = st.checkbox("Save snapshot history to file", value=True,
                             help="Flows are estimated from the AUM history in this file; every run adds today's "
                                  "AUM (and any imported history) to it.")
    seed_file = st.file_uploader("Import historical AUM (CSV: date,ticker,aum)", type="csv",
                                 help="Seed the history from another source (issuer sites, Bloomberg, etc.) "
                                      "so YTD / 3M flows are available immediately.")
    run_button = st.button("Load Flows", type="primary", width="stretch")

if not run_button and "flow_table" not in st.session_state:
    st.info(
        "Configure the screen in the sidebar and click **Load Flows**.\n\n"
        "**How flows are estimated.** EODHD does not publish fund flows or historical AUM, only each "
        "ETF's current total assets. This page records an AUM snapshot each time it runs (and a GitHub "
        "Action does so daily), then estimates flows as the change in AUM that price moves do not explain: "
        "`flow = AUM_t - AUM_(t-1) x (price_t / price_(t-1))`. Import historical AUM to backfill."
    )
    st.stop()

if run_button:
    if not api_key:
        st.error("Please enter your EODHD API key to proceed.")
        st.stop()

    if universe_mode == "All US-listed ETFs":
        tickers = load_all_us_etfs(api_key)
    else:
        extra = [t.strip().upper().removesuffix(".US") for t in custom.split(",") if t.strip()]
        tickers = sorted(set(ef.CURATED_UNIVERSE + extra))

    universe = load_universe(tuple(tickers), api_key)
    if universe.empty:
        st.error("No ETF fundamentals returned. Check the API key and that your plan includes fundamentals.")
        st.stop()
    universe = universe[universe["aum"] >= min_aum_bn * 1e9].reset_index(drop=True)

    # Update the snapshot history
    snapshots = ef.load_snapshots(snapshot_path)
    if seed_file is not None:
        seed = pd.read_csv(seed_file)
        seed.columns = [c.strip().lower() for c in seed.columns]
        seed["ticker"] = seed["ticker"].str.upper().str.removesuffix(".US")
        with st.spinner("Looking up prices for imported AUM history..."):
            seed = ef.fill_missing_prices(ef.merge_snapshots(pd.DataFrame(columns=ef.SNAPSHOT_COLUMNS), seed), api_key)
        snapshots = ef.merge_snapshots(snapshots, seed)
    snapshots = ef.merge_snapshots(snapshots, ef.make_snapshot(universe, load_latest_prices(api_key)))
    if save_today:
        try:
            ef.save_snapshots(snapshots, snapshot_path)
        except OSError as exc:
            st.warning(f"Could not write snapshot file ({exc}); flows use this session's data only.")

    # Flows and returns for both windows
    flow_series = ef.compute_flow_series(snapshots)
    starts = ef.window_starts()
    table = universe.set_index("ticker")
    for key, start in starts.items():
        wf = ef.window_flows(flow_series, snapshots, start)
        suffix = key.lower()
        table[f"flow_{suffix}"] = wf["flow"]
        table[f"coverage_{suffix}"] = wf["coverage_start"]
        table[f"full_{suffix}"] = wf["full_coverage"]
        table[f"ret_{suffix}"] = load_window_returns(api_key, tuple(table.index), start.date())
    table = table.reset_index()
    for suffix in ("ytd", "3m"):
        table[f"flow_{suffix}_pct"] = table[f"flow_{suffix}"] / (table["aum"] - table[f"flow_{suffix}"])

    st.session_state["flow_table"] = table
    st.session_state["snapshot_range"] = (snapshots["date"].min(), snapshots["date"].max(), snapshots["date"].nunique())
    st.session_state["starts"] = starts

table = st.session_state["flow_table"]
first_snap, last_snap, n_snap_dates = st.session_state["snapshot_range"]
starts = st.session_state["starts"]

# --- Data coverage ---
coverage_msgs = []
for key, start in starts.items():
    suffix = key.lower()
    full = table[f"full_{suffix}"].fillna(False)
    if not full.all():
        coverage_msgs.append(f"**{key}** (window starts {start:%Y-%m-%d}): {(~full).sum()} of {len(table)} ETFs "
                             f"only have history since {table.loc[~full, f'coverage_{suffix}'].min():%Y-%m-%d}")
st.caption(f"AUM snapshot history: {n_snap_dates} dates, {first_snap:%Y-%m-%d} to {last_snap:%Y-%m-%d}. "
           "Flows = AUM change not explained by price change.")
if n_snap_dates < 2:
    st.warning("Only one snapshot date so far, so no flows can be estimated yet. Flows appear after the next "
               "snapshot (run again tomorrow or enable the daily GitHub Action), or import historical AUM in the sidebar. "
               "AUM and returns below are current.")
elif coverage_msgs:
    st.warning("Partial history; flows cover less than the full window:\n\n- " + "\n- ".join(coverage_msgs))

# --- KPIs ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("ETFs screened", f"{len(table):,}")
k2.metric("Total AUM", fmt_usd(table["aum"].sum()))
k3.metric("Net flows YTD", fmt_usd(table["flow_ytd"].sum(min_count=1)))
k4.metric("Net flows 3M", fmt_usd(table["flow_3m"].sum(min_count=1)))

window = st.radio("Flow window", ["YTD", "3M"], horizontal=True)
sfx = window.lower()
flow_col, pct_col, ret_col = f"flow_{sfx}", f"flow_{sfx}_pct", f"ret_{sfx}"

tab_tree, tab_class, tab_geo, tab_sector, tab_movers, tab_all = st.tabs(
    ["Hierarchy map", "By asset class", "By country / region", "By sector / segment", "Top movers", "All ETFs"])

with tab_tree:
    st.markdown(f"Box size = AUM, colour = {window} net flow as % of starting AUM "
                "(blue inflow, red outflow, grey no data). Click a box to drill into it.")
    tree = table.copy()
    tree["color"] = tree[pct_col].clip(-0.5, 0.5).fillna(0) * 100
    tree["flow_label"] = tree[flow_col].map(fmt_usd)
    fig = px.treemap(tree, path=[px.Constant("US ETFs"), "asset_class", "geography", "sector", "ticker"],
                     values="aum", color="color", color_continuous_scale=[OUTFLOW, NEUTRAL, INFLOW],
                     color_continuous_midpoint=0, custom_data=["name", "flow_label"])
    fig.update_traces(hovertemplate="<b>%{label}</b><br>AUM $%{value:,.0f}<br>Flow % of AUM: %{color:.1f}%<extra></extra>")
    fig.update_layout(height=700, margin=dict(l=0, r=0, t=10, b=0), coloraxis_colorbar_title="Flow %")
    st.plotly_chart(fig, width="stretch")


def level_view(df, levels, label):
    agg = ef.aggregate_flows(df, levels)
    agg["label"] = agg[levels].astype(str).agg(" / ".join, axis=1)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(flow_bar(agg.nlargest(30, "aum"), "label", flow_col, f"{window} net flows by {label}"),
                        width="stretch")
    with c2:
        show = agg[levels + ["n_etfs", "aum", "flow_ytd", "flow_ytd_pct", "flow_3m", "flow_3m_pct"]].sort_values(flow_col, ascending=False)
        st.dataframe(
            show, hide_index=True, width="stretch",
            column_config={
                "asset_class": "Asset class", "geography": "Country / region", "sector": "Sector / segment",
                "n_etfs": st.column_config.NumberColumn("# ETFs"),
                "aum": st.column_config.NumberColumn("AUM ($)", format="compact"),
                "flow_ytd": st.column_config.NumberColumn("Flow YTD ($)", format="compact"),
                "flow_ytd_pct": st.column_config.NumberColumn("Flow YTD % AUM", format="percent"),
                "flow_3m": st.column_config.NumberColumn("Flow 3M ($)", format="compact"),
                "flow_3m_pct": st.column_config.NumberColumn("Flow 3M % AUM", format="percent"),
            })


with tab_class:
    level_view(table, ["asset_class"], "asset class")
with tab_geo:
    asset_classes = sorted(table["asset_class"].unique())
    pick = st.multiselect("Asset classes", asset_classes, default=[a for a in ["Equity"] if a in asset_classes] or asset_classes, key="geo_ac")
    sub = table[table["asset_class"].isin(pick)]
    if not sub.empty:
        level_view(sub, ["asset_class", "geography"], "country / region")
with tab_sector:
    geos = sorted(table["geography"].unique())
    c_ac, c_geo = st.columns(2)
    pick_ac = c_ac.multiselect("Asset classes", sorted(table["asset_class"].unique()), default=["Equity"] if "Equity" in set(table["asset_class"]) else None, key="sec_ac")
    pick_geo = c_geo.multiselect("Countries / regions (empty = all)", geos, key="sec_geo")
    sub = table[table["asset_class"].isin(pick_ac or table["asset_class"].unique())]
    if pick_geo:
        sub = sub[sub["geography"].isin(pick_geo)]
    if not sub.empty:
        level_view(sub, ["asset_class", "sector"], "sector / segment")

with tab_movers:
    n = st.slider("Show top N", 5, 30, 15)
    ranked = table.dropna(subset=[flow_col])
    ranked = ranked.assign(label=ranked["ticker"] + " - " + ranked["sector"])
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(flow_bar(ranked.nlargest(n, flow_col), "label", flow_col, f"Largest {window} inflows"), width="stretch")
    with c2:
        st.plotly_chart(flow_bar(ranked.nsmallest(n, flow_col), "label", flow_col, f"Largest {window} outflows"), width="stretch")
    st.markdown("**Flow vs. performance**: inflows chasing returns (top-right) vs. contrarian buying (top-left).")
    sc = ranked.dropna(subset=[ret_col, pct_col])
    if not sc.empty:
        fig = px.scatter(sc, x=ret_col, y=pct_col, size="aum", color="asset_class", hover_name="ticker",
                         hover_data={"name": True, "aum": ":,.0f", ret_col: ":.1%", pct_col: ":.1%"},
                         color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"])
        fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", template="plotly_white", height=520,
                          xaxis_title=f"{window} total return", yaxis_title=f"{window} flow % of AUM")
        fig.add_hline(y=0, line_color="#999", line_width=1)
        fig.add_vline(x=0, line_color="#999", line_width=1)
        st.plotly_chart(fig, width="stretch")

with tab_all:
    cols = ["ticker", "name", "asset_class", "geography", "sector", "aum", "flow_ytd", "flow_ytd_pct", "ret_ytd",
            "flow_3m", "flow_3m_pct", "ret_3m", "expense_ratio", "issuer"]
    show = table[cols].sort_values(flow_col, ascending=False)
    st.dataframe(show, hide_index=True, width="stretch", height=700, column_config={
        "asset_class": "Asset class", "geography": "Country / region", "sector": "Sector / segment",
        "aum": st.column_config.NumberColumn("AUM ($)", format="compact"),
        "flow_ytd": st.column_config.NumberColumn("Flow YTD ($)", format="compact"),
        "flow_ytd_pct": st.column_config.NumberColumn("Flow YTD %", format="percent"),
        "ret_ytd": st.column_config.NumberColumn("Return YTD", format="percent"),
        "flow_3m": st.column_config.NumberColumn("Flow 3M ($)", format="compact"),
        "flow_3m_pct": st.column_config.NumberColumn("Flow 3M %", format="percent"),
        "ret_3m": st.column_config.NumberColumn("Return 3M", format="percent"),
        "expense_ratio": st.column_config.NumberColumn("Expense ratio", format="percent"),
    })
    st.download_button("Download table (CSV)", show.to_csv(index=False).encode("utf-8"),
                       file_name=f"us_etf_flows_{date.today():%Y%m%d}.csv", mime="text/csv")

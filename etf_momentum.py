import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="ETF Momentum Backtest", layout="wide")
st.title("Quantitative ETF Momentum Backtester")

# --- 1. Data Ingestion (Cached) ---
@st.cache_data(ttl=3600, show_spinner="Fetching EODHD Market Data...")
def fetch_etf_data(tickers, api_key, start_date, end_date):
    data = {}
    for ticker in tickers:
        symbol = f"{ticker}.US"
        url = f"https://eodhd.com/api/eod/{symbol}?api_token={api_key}&fmt=json&from={start_date}&to={end_date}"
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                data[ticker] = df['adjusted_close']
    
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).dropna(how='all').ffill()

# --- 2. Signal Generation ---
def generate_signals(daily_prices):
    daily_rets = daily_prices.pct_change()
    
    # 231-day rolling volatility for IR signal (approx 11 months)
    rolling_vol_231 = daily_rets.rolling(window=231).std() * np.sqrt(252)
    
    # 63-day rolling volatility for Position Sizing (approx 3 months)
    rolling_vol_63 = daily_rets.rolling(window=63).std() * np.sqrt(252)
    # Prevent division by zero
    rolling_vol_63 = rolling_vol_63.replace(0, np.nan)

    # 12-1 Month Momentum translated to daily (252 days = 1 year, 21 days = 1 month)
    mom_12_1 = daily_prices.shift(21) / daily_prices.shift(252) - 1
    
    # Lag 231d volatility by 21 days to align with the 12-1 momentum gap
    vol_12_1 = rolling_vol_231.shift(21)
    ir_12_1 = mom_12_1 / vol_12_1
    
    return daily_rets, mom_12_1, ir_12_1, rolling_vol_63

# --- 3. Backtest Execution (Vectorized) ---
def run_backtest(daily_rets, signal_df, vol_df, top_n=2, mode="Long Only", target_vol=0.12, use_vol_target=True):
    positions = pd.DataFrame(0.0, index=signal_df.index, columns=signal_df.columns)
    
    required_assets = top_n if mode in ["Long Only", "Short Only"] else top_n * 2
    
    # Check if universe is large enough and has valid data
    if use_vol_target:
        valid_mask = (signal_df.notna() & vol_df.notna()).sum(axis=1) >= required_assets
        # Calculate target weights (Target Vol / Asset Vol) / N
        target_weights = (target_vol / vol_df) / top_n
    else:
        valid_mask = signal_df.notna().sum(axis=1) >= required_assets
        # Equal weights: 1 / N
        target_weights = pd.DataFrame(1.0 / top_n, index=signal_df.index, columns=signal_df.columns)
    
    if mode in ["Long Only", "Long / Short"]:
        # Rank descending: 1 is the highest momentum
        ranks_desc = signal_df.rank(axis=1, ascending=False)
        long_signals = (ranks_desc <= top_n) & valid_mask.values[:, None]
        
        # Apply weighting
        positions[long_signals] = target_weights[long_signals]
        
    if mode in ["Short Only", "Long / Short"]:
        # Rank ascending: 1 is the lowest momentum
        ranks_asc = signal_df.rank(axis=1, ascending=True)
        short_signals = (ranks_asc <= top_n) & valid_mask.values[:, None]
        
        # Apply negative weighting
        positions[short_signals] = -target_weights[short_signals]
            
    # Shift positions by 1 day to prevent look-ahead bias (trade at next day's close)
    strategy_returns = (positions.shift(1) * daily_rets).sum(axis=1)
    
    valid_returns = strategy_returns.loc[strategy_returns != 0]
    valid_positions = positions.loc[valid_returns.index]
    
    return valid_returns, valid_positions

# --- 4. Performance Analytics ---
def calculate_metrics(returns):
    cum_ret = (1 + returns).cumprod()
    years = len(returns) / 252 
    
    cagr = (cum_ret.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol != 0 else 0
    
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    return cagr, ann_vol, sharpe, max_dd

def get_annual_returns(s1, s2, spy, spy_label):
    """Calculates compounded returns grouped by calendar year."""
    df = pd.DataFrame({
        'Strategy 1: 12-1 Momentum': s1.groupby(s1.index.year).apply(lambda x: (1 + x).prod() - 1),
        'Strategy 2: Vol-Adjusted Momentum': s2.groupby(s2.index.year).apply(lambda x: (1 + x).prod() - 1),
        spy_label: spy.groupby(spy.index.year).apply(lambda x: (1 + x).prod() - 1)
    })
    df.index.name = 'Year'
    df = df.sort_index(ascending=False) # Most recent year first
    
    # Format as percentages
    for col in df.columns:
        df[col] = df[col].apply(lambda x: f"{x:.2%}")
        
    return df

# --- Helper: Format Holdings ---
def format_holdings(positions_df, mode):
    """Converts numerical positions into readable tickers with allocated percentages."""
    
    def extract_weights(row, is_long):
        assets = row[row > 0] if is_long else row[row < 0]
        if assets.empty:
            return ""
        # Format as "SPY (15.2%)" or "-SPY (15.2%)" depending on long/short
        return ", ".join([f"{idx} ({abs(val):.1%})" for idx, val in assets.items()])

    df = pd.DataFrame({'Rebalance Date': positions_df.index})
    
    if mode in ["Long Only", "Long / Short"]:
        df['Long ETFs (Weight)'] = positions_df.apply(lambda row: extract_weights(row, True), axis=1).values
        
    if mode in ["Short Only", "Long / Short"]:
        df['Short ETFs (Weight)'] = positions_df.apply(lambda row: extract_weights(row, False), axis=1).values

    df['Rebalance Date'] = df['Rebalance Date'].dt.strftime('%Y-%m-%d')
    return df.sort_values(by='Rebalance Date', ascending=False).reset_index(drop=True)

# --- UI Sidebar ---
with st.sidebar:
    st.header("Strategy Parameters")
    api_key = st.text_input("EODHD API Key", type="password")
    
    default_universe = ['SPY', 'QQQ', 'GLD', 'TLT', 'EWY', 'EEM', 'EWZ', 'XLU', 'XLF', 'VGK', 'EUFN', 'INDA', 'FXI', 'CQQQ',
                        'XLE','XLY','XLK','VCLT','FALN','VNQ']
    universe = st.multiselect("ETF Universe", options=default_universe, default=default_universe)
    
    strategy_mode = st.radio("Strategy Mode", options=["Long Only", "Short Only", "Long / Short"])
    backtest_years = st.slider("Backtest Duration (Years)", min_value=1, max_value=20, value=10)
    top_n = st.number_input("Top N / Bottom N to Hold", min_value=1, max_value=10, value=2)
    
    st.markdown("---")
    use_vol_target = st.checkbox("Apply Volatility Target Weighting", value=True)
    target_vol_pct = st.number_input("Target Portfolio Volatility (%)", min_value=5.0, max_value=30.0, value=12.0, step=1.0, disabled=not use_vol_target)
    
    run_button = st.button("Run Backtest", type="primary", use_container_width=True)

# --- Main Execution ---
if run_button:
    if not api_key:
        st.error("Please enter your EODHD API key to proceed.")
        st.stop()
        
    required_len = top_n if strategy_mode in ["Long Only", "Short Only"] else top_n * 2
    if len(universe) < required_len:
        st.error(f"Universe must contain at least {required_len} ETFs for the '{strategy_mode}' strategy with Top N = {top_n}.")
        st.stop()

    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=(backtest_years + 1) * 365)
    
    # Ensure SPY is fetched for the benchmark even if removed from the tradable universe
    fetch_list = list(set(universe + ['SPY']))
    daily_prices = fetch_etf_data(fetch_list, api_key, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))
    
    if daily_prices.empty:
        st.error("No data returned. Please check your API key and ETF symbols.")
        st.stop()

    # Generate signals ONLY on the selected tradable universe
    daily_rets, mom_12_1, ir_12_1, vol_63d = generate_signals(daily_prices[universe])
    
    # --- Benchmark Calculation ---
    spy_daily_rets = daily_prices['SPY'].pct_change()
    
    if use_vol_target:
        spy_vol_63d = spy_daily_rets.rolling(window=63).std() * np.sqrt(252)
        spy_vol_63d = spy_vol_63d.replace(0, np.nan)
        spy_weights = (target_vol_pct / 100) / spy_vol_63d
        spy_benchmark_rets = (spy_weights.shift(1) * spy_daily_rets).dropna()
        spy_label = f"Benchmark: SPY (Target Vol {target_vol_pct}%)"
    else:
        spy_benchmark_rets = spy_daily_rets.dropna()
        spy_label = "Benchmark: SPY (100% Long)"
        
    # -------------------------------------------
    
    # Unpack strategy returns AND positions, passing Target Volatility params
    strat1_rets, strat1_pos = run_backtest(daily_rets, mom_12_1, vol_63d, top_n, strategy_mode, target_vol_pct/100, use_vol_target)
    strat2_rets, strat2_pos = run_backtest(daily_rets, ir_12_1, vol_63d, top_n, strategy_mode, target_vol_pct/100, use_vol_target)
    
    # Align all dates
    common_idx = strat1_rets.index.intersection(strat2_rets.index).intersection(spy_benchmark_rets.index)
    s1_aligned = strat1_rets.loc[common_idx]
    s2_aligned = strat2_rets.loc[common_idx]
    spy_aligned = spy_benchmark_rets.loc[common_idx]
    
    # Calculate Metrics
    s1_metrics = calculate_metrics(s1_aligned)
    s2_metrics = calculate_metrics(s2_aligned)
    spy_metrics = calculate_metrics(spy_aligned)
    
    # --- UI Layout: Metrics Table ---
    st.subheader(f"Performance Summary ({strategy_mode})")
    
    metrics_df = pd.DataFrame({
        'Strategy': ['Strategy 1: 12-1 Momentum', 'Strategy 2: Vol-Adjusted Momentum', spy_label],
        'CAGR': [f"{s1_metrics[0]:.2%}", f"{s2_metrics[0]:.2%}", f"{spy_metrics[0]:.2%}"],
        'Volatility': [f"{s1_metrics[1]:.2%}", f"{s2_metrics[1]:.2%}", f"{spy_metrics[1]:.2%}"],
        'Sharpe Ratio': [f"{s1_metrics[2]:.2f}", f"{s2_metrics[2]:.2f}", f"{spy_metrics[2]:.2f}"],
        'Max Drawdown': [f"{s1_metrics[3]:.2%}", f"{s2_metrics[3]:.2%}", f"{spy_metrics[3]:.2%}"]
    }).set_index('Strategy')
    
    st.table(metrics_df)

    # --- UI Layout: Plotly Chart ---
    st.subheader("Cumulative Equity Growth")
    
    cum_s1 = (1 + s1_aligned).cumprod()
    cum_s2 = (1 + s2_aligned).cumprod()
    cum_spy = (1 + spy_aligned).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_s1.index, y=cum_s1, mode='lines', name='12-1 Pure Momentum', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=cum_s2.index, y=cum_s2, mode='lines', name='Vol-Adjusted (IR)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy, mode='lines', name=spy_label, line=dict(color='gray', dash='dash')))
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Cumulative Growth ($1)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- UI Layout: Annual Returns Table ---
    st.markdown("---")
    st.subheader("Returns by Year")
    annual_returns_df = get_annual_returns(s1_aligned, s2_aligned, spy_aligned, spy_label)
    st.dataframe(annual_returns_df, use_container_width=True)
    
    # --- UI Layout: Ownership Distribution Chart ---
    st.markdown("---")
    st.subheader("Asset Selection Frequency")
    st.markdown("Displays the percentage of total trading days each asset was held by the strategy.")
    
    # Calculate % of days held for each strategy
    total_days = len(common_idx)
    s1_pos_aligned = strat1_pos.loc[common_idx]
    s2_pos_aligned = strat2_pos.loc[common_idx]
    
    s1_pct = (s1_pos_aligned != 0).sum() / total_days * 100
    s2_pct = (s2_pos_aligned != 0).sum() / total_days * 100
    
    # Sort by Strategy 1 frequency for cleaner horizontal presentation
    s1_pct = s1_pct.sort_values(ascending=True)
    s2_pct = s2_pct.loc[s1_pct.index]
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        y=s1_pct.index, x=s1_pct.values, name='12-1 Pure Momentum', orientation='h', marker_color='blue'
    ))
    fig_dist.add_trace(go.Bar(
        y=s2_pct.index, x=s2_pct.values, name='Vol-Adjusted (IR)', orientation='h', marker_color='orange'
    ))
    
    fig_dist.update_layout(
        barmode='group',
        xaxis_title="% of Trading Days Held",
        yaxis_title="ETF Ticker",
        hovermode="y unified",
        template='plotly_white',
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
        height=500
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # --- UI Layout: Historical Holdings ---
    st.markdown("---")
    st.subheader("Daily Allocated Portfolios")
    
    if use_vol_target:
        st.markdown(f"Assets are selected daily. Weights are sized dynamically to target a **{target_vol_pct}% portfolio volatility** using individual 3-Month rolling realized vol.")
    else:
        st.markdown(f"Assets are selected daily. Weights are allocated **equally** (1/{top_n} per asset class).")
    
    h_col1, h_col2 = st.columns(2)
    
    with h_col1:
        st.markdown("**Strategy 1: 12-1 Momentum**")
        df_s1 = format_holdings(strat1_pos.loc[common_idx], strategy_mode)
        st.dataframe(df_s1, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Pure Mom Allocations (CSV)",
            data=df_s1.to_csv(index=False).encode('utf-8'),
            file_name='strat1_allocations.csv',
            mime='text/csv',
            key='dl_s1'
        )
        
    with h_col2:
        st.markdown("**Strategy 2: Vol-Adjusted Momentum**")
        df_s2 = format_holdings(strat2_pos.loc[common_idx], strategy_mode)
        st.dataframe(df_s2, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Vol-Adjusted Allocations (CSV)",
            data=df_s2.to_csv(index=False).encode('utf-8'),
            file_name='strat2_allocations.csv',
            mime='text/csv',
            key='dl_s2'
        )

else:
    st.info("Configure your parameters in the sidebar and click 'Run Backtest'.")
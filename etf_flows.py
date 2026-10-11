"""
ETF flow estimation and classification on top of EODHD data.

EODHD has no fund-flow, historical AUM or ETF shares-outstanding endpoint: the
fundamentals API only reports the *current* TotalAssets. Flows are therefore
estimated from a history of AUM snapshots that this module builds up over time
(see scripts/snapshot_aum.py and the GitHub Action that runs it daily):

    flow_t = AUM_t - AUM_{t-1} * (P_t / P_{t-1})

i.e. the change in AUM not explained by the change in price. Flows over a
window are the sum of these per-snapshot flows. Historical AUM from another
source (issuer sites, Bloomberg, etc.) can be imported to seed the history.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://eodhd.com/api"
SNAPSHOT_COLUMNS = ["date", "ticker", "aum", "close", "adj_close"]
DEFAULT_SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "aum_snapshots.csv")

FUNDAMENTAL_FIELDS = [
    "General::Name",
    "ETF_Data::Company_Name",
    "ETF_Data::TotalAssets",
    "ETF_Data::NetExpenseRatio",
    "ETF_Data::Inception_Date",
    "ETF_Data::Asset_Allocation",
    "ETF_Data::World_Regions",
    "ETF_Data::Sector_Weights",
    "ETF_Data::Top_10_Holdings",
]

# Large, liquid US-listed ETFs spanning every bucket of the classification.
CURATED_UNIVERSE = [
    # US equity - broad / size / style
    "SPY", "IVV", "VOO", "VTI", "QQQ", "QQQM", "RSP", "DIA", "IWM", "IJH", "IJR", "MDY", "VB", "VO",
    "VUG", "VTV", "IWF", "IWD", "SCHD", "VIG", "DGRO", "SCHG", "MGK", "SPLG", "USMV", "MTUM", "QUAL",
    "JEPI", "JEPQ", "COWZ", "AVUV", "SPMO",
    # US equity - sectors / industries / themes
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "VGT", "SMH", "SOXX",
    "IGV", "XBI", "IBB", "KRE", "KBE", "ITA", "XAR", "PAVE", "GRID", "ICLN", "TAN", "URA", "XME", "GDX",
    "GDXJ", "COPX", "VNQ", "XOP", "OIH", "ARKK", "CIBR", "BOTZ",
    # International developed - regions / countries
    "VEA", "IEFA", "EFA", "VXUS", "VGK", "EZU", "IEUR", "HEDJ", "EWG", "EWU", "EWQ", "EWI", "EWP", "EWL",
    "EWN", "EWD", "EWJ", "DXJ", "EWA", "EWC", "EWY", "EWT", "EWS", "EWH", "EUFN",
    # Emerging markets - regions / countries
    "VWO", "IEMG", "EEM", "INDA", "EPI", "FXI", "MCHI", "KWEB", "CQQQ", "ASHR", "EWZ", "EWW", "ARGT",
    "EIDO", "EPOL", "TUR", "KSA", "EZA", "VNM", "THD",
    # Fixed income
    "AGG", "BND", "BNDX", "TLT", "IEF", "SHY", "GOVT", "VGIT", "VGLT", "SGOV", "BIL", "SHV", "USFR",
    "TFLO", "LQD", "VCIT", "VCSH", "VCLT", "IGSB", "HYG", "JNK", "USHY", "FALN", "SJNK", "BKLN", "SRLN",
    "MUB", "VTEB", "TIP", "SCHP", "STIP", "MBB", "VMBS", "EMB", "EMLC", "JPST", "MINT", "JAAA", "CWB",
    # Commodities
    "GLD", "IAU", "GLDM", "SLV", "PPLT", "USO", "BNO", "UNG", "DBC", "PDBC", "DBA", "CPER", "CMDY",
    # Crypto
    "IBIT", "FBTC", "GBTC", "BITB", "ARKB", "ETHA", "ETHE", "BITO",
    # Multi-asset / alternatives / currency
    "AOR", "AOA", "AOM", "DBMF", "KMLM", "TAIL", "UUP", "FXE", "FXY",
    # Leveraged / inverse
    "TQQQ", "SQQQ", "SOXL", "SOXS", "SPXL", "UPRO", "TNA", "NVDL", "TSLL", "TMF", "SH", "PSQ",
]

# --- Keyword maps used by the classifier (matched against the fund name) ---
COUNTRY_KEYWORDS = {
    "South Korea": ["korea"], "Japan": ["japan"], "China": ["china", "chinese", "csi 300", "csi300"],
    "Hong Kong": ["hong kong"], "Taiwan": ["taiwan"], "India": ["india"], "Brazil": ["brazil"],
    "Mexico": ["mexico"], "Germany": ["germany"], "United Kingdom": ["united kingdom", " uk "],
    "France": ["france"], "Italy": ["italy"], "Spain": ["spain"], "Switzerland": ["switzerland"],
    "Netherlands": ["netherlands"], "Sweden": ["sweden"], "Canada": ["canada"], "Australia": ["australia"],
    "Singapore": ["singapore"], "Indonesia": ["indonesia"], "Poland": ["poland"], "Turkey": ["turkey"],
    "Saudi Arabia": ["saudi"], "South Africa": ["south africa"], "Vietnam": ["vietnam"],
    "Thailand": ["thailand"], "Argentina": ["argentina"], "Israel": ["israel"], "Chile": ["chile"],
    "Malaysia": ["malaysia"], "Philippines": ["philippines"], "Greece": ["greece"],
}
REGION_KEYWORDS = [
    ("Europe", ["europe", "eurozone", "euro stoxx", "emu"]),
    ("Latin America", ["latin america"]),
    ("Asia Pacific", ["pacific", "asia"]),
    ("Emerging Markets", ["emerging", " em "]),
    ("Developed ex-US", ["eafe", "developed", "international", "ex-us", "ex us", "world ex"]),
    ("Global", ["global", "world", "all-world", "acwi", "all country"]),
]
EQUITY_SECTOR_KEYWORDS = [
    ("Semiconductors", ["semiconductor"]),
    ("Technology", ["technology", "tech ", "software", "cybersecurity", "cloud", "internet", "robotics",
                    "artificial intelligence", " ai ", "innovation", "fintech"]),
    ("Financials", ["financial", "bank", "insurance", "broker"]),
    ("Clean Energy", ["clean energy", "solar", "renewable", "uranium", "nuclear"]),
    ("Energy", ["energy", "oil & gas", "oil and gas", "exploration & production", "oil services", "mlp"]),
    ("Healthcare", ["health", "biotech", "pharma", "medical"]),
    ("Industrials", ["industrial", "aerospace", "defense", "infrastructure", "transport"]),
    ("Materials & Mining", ["materials", "metals & mining", "mining", "gold miners", "copper miners", "steel"]),
    ("Consumer Discretionary", ["consumer discretionary", "retail", "homebuilders"]),
    ("Consumer Staples", ["consumer staples"]),
    ("Utilities", ["utilities", "utility"]),
    ("Real Estate", ["real estate", "reit"]),
    ("Communication Services", ["communication"]),
]
# Sector_Weights keys from EODHD -> our sector names
EODHD_SECTOR_MAP = {
    "Technology": "Technology", "Financial Services": "Financials", "Energy": "Energy",
    "Healthcare": "Healthcare", "Industrials": "Industrials", "Basic Materials": "Materials & Mining",
    "Consumer Cyclicals": "Consumer Discretionary", "Consumer Defensive": "Consumer Staples",
    "Utilities": "Utilities", "Real Estate": "Real Estate", "Communication Services": "Communication Services",
}
BOND_SEGMENT_KEYWORDS = [
    ("T-Bills / Ultra-Short", ["0-3 month", "0-1 year", "t-bill", "treasury bill", "floating rate treasury",
                               "treasury floating", "ultra short", "ultra-short", "short maturity", "short treasury",
                               "money market"]),
    ("Inflation-Linked", ["tips", "inflation"]),
    ("Municipal", [" muni", "municipal", "tax-exempt", "tax exempt"]),
    ("High Yield", ["high yield", "junk", "fallen angel", "high-yield"]),
    ("Loans / CLO", ["loan", "clo", "senior"]),
    ("Convertible", ["convertible"]),
    ("Mortgage-Backed", ["mbs", "mortgage"]),
    ("EM Debt", ["emerging"]),
    ("Treasury", ["treasury", "government", "govt"]),
    ("Corporate IG", ["corporate", "investment grade", "credit"]),
    ("Aggregate / Core", ["aggregate", "total bond", "core", "universal", "total international bond"]),
]
COMMODITY_SEGMENT_KEYWORDS = [
    ("Gold", ["gold"]), ("Silver", ["silver"]), ("Platinum / Palladium", ["platinum", "palladium"]),
    ("Oil", ["oil", "brent", "crude"]), ("Natural Gas", ["natural gas"]), ("Copper", ["copper"]),
    ("Agriculture", ["agricultur", "grain", "corn", "wheat"]), ("Broad Commodities", ["commodit"]),
]
CRYPTO_SEGMENT_KEYWORDS = [("Bitcoin", ["bitcoin", "btc"]), ("Ether", ["ether", "eth "]), ("Solana", ["solana"]),
                           ("XRP", ["xrp"])]
# "Ultra-Short" bond funds are not leveraged, hence the lookahead
LEVERAGED_PATTERN = re.compile(r"\b\d(\.\d+)?x\b|ultrapro|\bultra\b(?![- ]short)|\bbull\b|\bbear\b|inverse|leveraged"
                               r"|direxion daily|proshares short", re.I)
ALT_KEYWORDS = ["managed futures", "tail risk", "merger arbitrage", "long/short", "hedge fund"]
CURRENCY_KEYWORDS = ["dollar index", "currencyshares", "currency trust"]
MULTI_ASSET_KEYWORDS = ["allocation", "balanced", "multi-asset", "moderate", "conservative", "aggressive"]


def _num(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _match(name, keywords):
    padded = f" {name.lower()} "
    return any(k in padded for k in keywords)


def _first_match(name, keyword_table):
    for label, keywords in keyword_table:
        if _match(name, keywords):
            return label
    return None


# --- 1. Data access ---
def fetch_us_etf_list(api_key):
    """All ETFs currently listed on US exchanges."""
    r = requests.get(f"{BASE_URL}/exchange-symbol-list/US", params={"api_token": api_key, "fmt": "json", "type": "etf"}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    return sorted(df.loc[df["Type"].str.upper() == "ETF", "Code"].unique().tolist()) if not df.empty else []


def fetch_etf_fundamentals(ticker, api_key):
    """Slimmed-down ETF fundamentals (10 API calls). Returns {} on failure."""
    params = {"api_token": api_key, "fmt": "json", "filter": ",".join(FUNDAMENTAL_FIELDS)}
    try:
        r = requests.get(f"{BASE_URL}/fundamentals/{ticker}.US", params=params, timeout=30)
        if r.status_code != 200:
            return {}
        data = r.json()
    except (requests.RequestException, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Multi-field filters come back keyed by the filter path; flatten to the last path component
    return {k.split("::")[-1]: v for k, v in data.items()}


def fetch_many_fundamentals(tickers, api_key, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(lambda t: (t, fetch_etf_fundamentals(t, api_key)), tickers)
    return {t: f for t, f in results if f}


def fetch_bulk_prices(api_key, on_date=None):
    """Close / adjusted close for every US symbol on one date (bulk endpoint, 100 API calls)."""
    params = {"api_token": api_key, "fmt": "json"}
    if on_date:
        params["date"] = pd.Timestamp(on_date).strftime("%Y-%m-%d")
    r = requests.get(f"{BASE_URL}/eod-bulk-last-day/US", params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return pd.DataFrame(columns=["ticker", "date", "close", "adj_close"])
    df = df.rename(columns={"code": "ticker", "adjusted_close": "adj_close"})
    return df[["ticker", "date", "close", "adj_close"]]


def fetch_price_history(ticker, api_key, start, end):
    params = {"api_token": api_key, "fmt": "json", "from": start, "to": end}
    r = requests.get(f"{BASE_URL}/eod/{ticker}.US", params=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["close", "adjusted_close", "volume"]].rename(columns={"adjusted_close": "adj_close"})


def fetch_many_price_histories(tickers, api_key, start, end=None, max_workers=8):
    """Daily close / adj_close / volume for many tickers in long format (1 API call per ticker)."""
    end = end or date.today().strftime("%Y-%m-%d")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(lambda t: fetch_price_history(t, api_key, start, end).assign(ticker=t), tickers)
    frames = [df.reset_index() for df in results if not df.empty]
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close", "volume"])
    return pd.concat(frames, ignore_index=True)


# --- 2. Classification: asset class -> country / region -> sector / segment ---
def _top_holding_country(fund):
    """Dominant country among the top-10 holdings, if it holds >= 90% of their weight."""
    holdings = fund.get("Top_10_Holdings") or {}
    if not isinstance(holdings, dict):
        return None
    by_country = {}
    for h in holdings.values():
        c = (h or {}).get("Country")
        if c:
            by_country[c] = by_country.get(c, 0.0) + _num(h.get("Assets_%"))
    total = sum(by_country.values())
    if total <= 0:
        return None
    country, weight = max(by_country.items(), key=lambda kv: kv[1])
    if weight / total < 0.9:
        return None
    return {"Korea": "South Korea", "USA": "United States", "United States of America": "United States"}.get(country, country)


def _equity_geography(name, fund, us_pct, non_us_pct):
    for country, keywords in COUNTRY_KEYWORDS.items():
        if _match(name, keywords):
            return country
    region = _first_match(name, REGION_KEYWORDS)
    if region:
        return region
    if us_pct >= 80:
        return "United States"
    holding_country = _top_holding_country(fund)
    if holding_country:
        return holding_country
    regions = {k: _num((v or {}).get("Equity_%")) for k, v in (fund.get("World_Regions") or {}).items()}
    if regions and max(regions.values()) >= 80:
        return max(regions, key=regions.get)
    return "Global" if us_pct > 20 else "Developed ex-US"


def _equity_sector(name, fund, geography):
    sector = _first_match(name, EQUITY_SECTOR_KEYWORDS)
    if sector:
        return sector
    # Weight-based fallback only for US/broad funds; country funds are often concentrated (e.g. Korea ~60% tech)
    if geography not in ("United States", "Global"):
        return "Broad Market"
    weights = {EODHD_SECTOR_MAP.get(k, k): _num((v or {}).get("Equity_%")) for k, v in (fund.get("Sector_Weights") or {}).items()}
    if weights and max(weights.values()) >= 75:
        return max(weights, key=weights.get)
    return "Broad Market"


def classify_etf(fund):
    """Returns (asset_class, geography, sector) for an ETF fundamentals dict."""
    name = str(fund.get("Name") or "")
    alloc = fund.get("Asset_Allocation") or {}
    pct = lambda k: _num((alloc.get(k) or {}).get("Net_Assets_%"))
    us, non_us, bond, other, cash = pct("Stock US"), pct("Stock non-US"), pct("Bond"), pct("Other"), pct("Cash")
    geo_from_name = (next((c for c, kw in COUNTRY_KEYWORDS.items() if _match(name, kw)), None)
                     or _first_match(name, REGION_KEYWORDS))

    if LEVERAGED_PATTERN.search(name):
        underlying = _first_match(name, EQUITY_SECTOR_KEYWORDS) or ("Treasury" if _match(name, ["treasury", "bond"]) else None)
        return "Leveraged / Inverse", geo_from_name or "United States", underlying or "Index / Single Stock"
    crypto = _first_match(name, CRYPTO_SEGMENT_KEYWORDS)
    if (crypto or _match(name, ["crypto", "digital asset"])) and us + non_us < 50:
        return "Crypto", "Global", crypto or "Multi-Coin"
    # Miners are equities; check commodity keywords only when the fund is not mostly stocks
    commodity = _first_match(name, COMMODITY_SEGMENT_KEYWORDS)
    if commodity and us + non_us < 50:
        return "Commodity", "Global", commodity
    if _match(name, CURRENCY_KEYWORDS):
        return "Currency", geo_from_name or "Global", "Currency"
    if _match(name, ALT_KEYWORDS):
        return "Alternatives", "Global", "Alternatives"
    if _match(name, MULTI_ASSET_KEYWORDS) or (bond >= 20 and us + non_us >= 20):
        return "Multi-Asset", geo_from_name or "Global", "Allocation"
    is_bond = bond >= 50 or (cash >= 50 and us + non_us < 10) or _match(name, ["bond", "treasury", "t-bill", " muni", "convertible"])
    if is_bond and us + non_us < 50:
        segment = _first_match(name, BOND_SEGMENT_KEYWORDS) or "Aggregate / Core"
        if segment == "EM Debt":
            geography = "Emerging Markets"
        else:
            geography = geo_from_name or ("Developed ex-US" if _match(name, ["international", "ex-us", "ex us"]) else "United States")
        return "Fixed Income", geography, segment
    if us + non_us >= 50 or other < 50:
        geography = _equity_geography(name, fund, us, non_us)
        return "Equity", geography, _equity_sector(name, fund, geography)
    return "Other", geo_from_name or "Global", "Other"


def build_universe_table(fundamentals):
    rows = []
    for ticker, fund in fundamentals.items():
        asset_class, geography, sector = classify_etf(fund)
        rows.append({
            "ticker": ticker,
            "name": fund.get("Name"),
            "issuer": fund.get("Company_Name"),
            "asset_class": asset_class,
            "geography": geography,
            "sector": sector,
            "aum": _num(fund.get("TotalAssets"), np.nan),
            "expense_ratio": _num(fund.get("NetExpenseRatio"), np.nan),
            "inception": fund.get("Inception_Date"),
        })
    return pd.DataFrame(rows)


# --- 3. Snapshot store ---
def load_snapshots(path=DEFAULT_SNAPSHOT_PATH):
    if not os.path.exists(path):
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def merge_snapshots(existing, new):
    """Union of snapshot rows; on (date, ticker) collisions the new row wins."""
    new = new.copy()
    new["date"] = pd.to_datetime(new["date"])
    for col in SNAPSHOT_COLUMNS:
        if col not in new:
            new[col] = np.nan
    combined = pd.concat([existing[SNAPSHOT_COLUMNS], new[SNAPSHOT_COLUMNS]], ignore_index=True)
    combined = combined.dropna(subset=["aum"])
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)


def save_snapshots(df, path=DEFAULT_SNAPSHOT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def make_snapshot(universe, prices):
    """Today's snapshot rows: EODHD TotalAssets joined to the latest close."""
    snap = universe[["ticker", "aum"]].merge(prices, on="ticker", how="inner")
    return snap.dropna(subset=["aum", "close"])[SNAPSHOT_COLUMNS]


def fill_missing_prices(snapshots, api_key):
    """Look up close / adj_close for imported rows that only carry AUM (one bulk call per date)."""
    missing = snapshots["close"].isna()
    if not missing.any():
        return snapshots
    snapshots = snapshots.copy()
    for d in snapshots.loc[missing, "date"].unique():
        prices = fetch_bulk_prices(api_key, d).set_index("ticker")
        rows = missing & (snapshots["date"] == d)
        snapshots.loc[rows, "close"] = snapshots.loc[rows, "ticker"].map(prices["close"])
        snapshots.loc[rows, "adj_close"] = snapshots.loc[rows, "ticker"].map(prices["adj_close"])
    return snapshots


# --- 4. Volume spikes ---
def volume_spikes(history, window=60, threshold=2.0, sessions=5):
    """
    Sessions where volume exceeded `threshold` x its trailing `window`-session average.

    The average covers the `window` sessions *before* each day, so a spike does not
    dilute its own baseline. Only the last `sessions` sessions of each ticker are
    scanned. Returns one row per spike with the volume ratio, the day's total return
    and the dollar volume traded.
    """
    cols = ["ticker", "date", "volume", "avg_volume", "volume_ratio", "day_return", "dollar_volume"]
    if history.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for ticker, g in history.sort_values("date").groupby("ticker"):
        g = g.set_index("date")
        avg = g["volume"].shift(1).rolling(window, min_periods=window).mean()
        recent = pd.DataFrame({
            "volume": g["volume"], "avg_volume": avg, "volume_ratio": g["volume"] / avg.replace(0, np.nan),
            "day_return": g["adj_close"].pct_change(), "dollar_volume": g["close"] * g["volume"],
        }).tail(sessions)
        hits = recent[recent["volume_ratio"] >= threshold]
        if not hits.empty:
            rows.append(hits.reset_index().assign(ticker=ticker))
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(rows, ignore_index=True)[cols].sort_values("volume_ratio", ascending=False).reset_index(drop=True)


def latest_volume_ratio(history, window=60):
    """Latest session's volume / trailing `window`-session average, per ticker."""
    if history.empty:
        return pd.Series(dtype=float)
    def ratio(g):
        v = g.sort_values("date")["volume"]
        avg = v.shift(1).rolling(window, min_periods=window).mean()
        return v.iloc[-1] / avg.iloc[-1] if avg.iloc[-1] else np.nan
    return history.groupby("ticker").apply(ratio, include_groups=False)


# --- 5. Flow estimation ---
def _price_factor(prev, cur):
    """
    Growth of one share's NAV between snapshots, excluding distributions.

    Distributions leave the fund as cash and are not outflows, so the raw close
    ratio is right. A split makes the raw ratio jump while adjusted close does
    not; when the two disagree by more than 25% we assume a split and fall back
    to the adjusted ratio.
    """
    raw = cur["close"] / prev["close"]
    adj = cur["adj_close"] / prev["adj_close"]
    if pd.isna(adj):
        return raw
    if pd.isna(raw) or abs(np.log(raw / adj)) > np.log(1.25):
        return adj
    return raw


def compute_flow_series(snapshots):
    """Per-snapshot estimated flow for every ticker (first snapshot of each ticker has no flow)."""
    rows = []
    for ticker, g in snapshots.sort_values("date").groupby("ticker"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            prev, cur = g.loc[i - 1], g.loc[i]
            factor = _price_factor(prev, cur)
            if pd.isna(factor) or prev["aum"] <= 0:
                continue
            rows.append({"ticker": ticker, "date": cur["date"], "prev_date": prev["date"],
                         "flow": cur["aum"] - prev["aum"] * factor})
    return pd.DataFrame(rows, columns=["ticker", "date", "prev_date", "flow"])


def window_flows(flow_series, snapshots, start, end=None):
    """
    Sum of estimated flows per ticker over (start, end].

    `coverage_start` is the first snapshot date used: if it is after `start`,
    the history does not reach back to the window start and the figure only
    covers part of the window.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.max
    # The snapshot on/just before `start` anchors the window; flows after it count.
    anchors = (snapshots[snapshots["date"] <= start].groupby("ticker")["date"].max()
               .reindex(snapshots["ticker"].unique()))
    first_seen = snapshots.groupby("ticker")["date"].min()
    anchors = anchors.fillna(first_seen)
    fs = flow_series.copy()
    fs = fs[(fs["date"] <= end) & (fs["prev_date"] >= fs["ticker"].map(anchors))]
    out = fs.groupby("ticker")["flow"].sum().reindex(anchors.index).fillna(0.0).to_frame("flow")
    out["coverage_start"] = anchors
    out["full_coverage"] = anchors <= start
    has_flows = fs.groupby("ticker").size().reindex(anchors.index).fillna(0) > 0
    out.loc[~has_flows, "flow"] = np.nan
    return out


def window_returns(api_key, tickers, start):
    """Price return since `start` from two bulk calls."""
    start_prices = fetch_bulk_prices(api_key, last_business_day(start)).set_index("ticker")["adj_close"]
    latest = fetch_bulk_prices(api_key).set_index("ticker")["adj_close"]
    return (latest / start_prices - 1).reindex(tickers)


def last_business_day(d):
    d = pd.Timestamp(d)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def window_starts(today=None):
    today = pd.Timestamp(today or date.today()).normalize()
    return {"YTD": pd.Timestamp(year=today.year, month=1, day=1) - timedelta(days=1),
            "3M": today - pd.DateOffset(months=3)}


def aggregate_flows(table, levels, value_cols=("aum", "flow_ytd", "flow_3m")):
    """Sum AUM and flows up a classification hierarchy, with flows as % of starting AUM."""
    agg = table.groupby(list(levels), dropna=False)[list(value_cols)].sum(min_count=1)
    agg["n_etfs"] = table.groupby(list(levels), dropna=False).size()
    for col in value_cols:
        if col.startswith("flow_"):
            agg[f"{col}_pct"] = agg[col] / (agg["aum"] - agg[col])
    return agg.reset_index()

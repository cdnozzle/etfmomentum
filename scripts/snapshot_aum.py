"""
Append today's AUM snapshot for US ETFs to data/aum_snapshots.csv.

Run daily (see .github/workflows/aum_snapshot.yml) so the ETF Flows page has
an AUM history to estimate flows from.

    EODHD_API_KEY=... python scripts/snapshot_aum.py [--all] [--min-aum 100e6]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import etf_flows as ef  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Snapshot every US-listed ETF (~10 API calls each) instead of the curated list")
    parser.add_argument("--min-aum", type=float, default=0.0, help="Skip ETFs with AUM below this (USD)")
    parser.add_argument("--path", default=ef.DEFAULT_SNAPSHOT_PATH)
    args = parser.parse_args()

    api_key = os.environ.get("EODHD_API_KEY")
    if not api_key:
        sys.exit("EODHD_API_KEY is not set")

    tickers = ef.fetch_us_etf_list(api_key) if args.all else ef.CURATED_UNIVERSE
    universe = ef.build_universe_table(ef.fetch_many_fundamentals(tickers, api_key))
    universe = universe[universe["aum"] >= args.min_aum]
    snap = ef.make_snapshot(universe, ef.fetch_bulk_prices(api_key))

    snapshots = ef.merge_snapshots(ef.load_snapshots(args.path), snap)
    ef.save_snapshots(snapshots, args.path)
    print(f"Saved {len(snap)} ETF snapshots; history now has {snapshots['date'].nunique()} dates.")


if __name__ == "__main__":
    main()

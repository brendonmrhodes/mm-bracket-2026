"""
Fetch historical NET rankings from WarrenNolan.com.
The NCAA does not provide bulk historical NET downloads; WarrenNolan
is the best free source for end-of-season NET by year.

Saves: data/raw/net_YYYY.csv
       data/raw/net_all.parquet

Note: NET only exists from 2019 onward (replaced RPI).
      This also grabs RPI for earlier years from the same site.
"""

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# NET: 2019-present
NET_YEARS = list(range(2019, 2027))
# RPI: earlier seasons (WarrenNolan also has this)
RPI_YEARS = list(range(2003, 2019))


def fetch_net_year(year: int) -> pd.DataFrame | None:
    url = f"https://www.warrennolan.com/basketball/{year}/net"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  [{year}] NET HTTP error: {e}")
        return None

    try:
        dfs = pd.read_html(StringIO(r.text))
        if not dfs:
            print(f"  [{year}] No tables found")
            return None
        df = dfs[0]
        df.insert(0, "season", year)
        df.insert(1, "ranking_system", "NET")
        return df
    except Exception as e:
        print(f"  [{year}] Parse error: {e}")
        return None


def fetch_rpi_year(year: int) -> pd.DataFrame | None:
    url = f"https://www.warrennolan.com/basketball/{year}/rpi"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  [{year}] RPI HTTP error: {e}")
        return None

    try:
        dfs = pd.read_html(StringIO(r.text))
        if not dfs:
            return None
        df = dfs[0]
        df.insert(0, "season", year)
        df.insert(1, "ranking_system", "RPI")
        return df
    except Exception as e:
        print(f"  [{year}] RPI parse error: {e}")
        return None


def main():
    all_frames = []

    print("Fetching NET rankings (2019-2026)...")
    for year in NET_YEARS:
        df = fetch_net_year(year)
        if df is not None:
            out = RAW_DIR / f"net_{year}.csv"
            df.to_csv(out, index=False)
            all_frames.append(df)
            print(f"  [{year}] {len(df)} teams  cols: {list(df.columns[:6])}")
        time.sleep(2)

    print("\nFetching RPI rankings (2003-2018)...")
    for year in RPI_YEARS:
        df = fetch_rpi_year(year)
        if df is not None:
            out = RAW_DIR / f"rpi_{year}.csv"
            df.to_csv(out, index=False)
            all_frames.append(df)
            print(f"  [{year}] {len(df)} teams")
        time.sleep(2)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_parquet(RAW_DIR / "net_rpi_all.parquet", index=False)
        print(f"\nCombined NET/RPI: {len(combined):,} rows → data/raw/net_rpi_all.parquet")


if __name__ == "__main__":
    main()

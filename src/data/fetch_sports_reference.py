"""
Scrape Sports-Reference CBB for supplemental data not in the Kaggle dataset:
  - SRS (Simple Rating System) per team per season — proxy efficiency for pre-2003
  - Conference standings and records
  - School pages for coach tenure data

Uses direct URL scraping with pandas read_html (no Stathead required).
Rate-limited to ~1 req/3s to be polite.

Saves: data/raw/sportsref_srs_YYYY.csv
       data/raw/sportsref_srs_all.parquet
"""

import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.sports-reference.com/cbb/seasons/men"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

START_YEAR = 1985
END_YEAR   = 2026


def fetch_season_schools(year: int) -> pd.DataFrame | None:
    """
    Fetch the per-school summary table for a given season.
    URL: https://www.sports-reference.com/cbb/seasons/men/YYYY-school-stats.html
    Includes: W, L, SRS, SOS, Pace, ORtg, FTr, 3PAr, TS%, TRB%, AST%, STL%, BLK%, eFG%
    """
    url = f"{BASE_URL}/{year}-school-stats.html"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  [{year}] HTTP error: {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # Sports-Reference embeds tables in comments for some pages
    # Find the stats table — try direct first, then in comments
    table = soup.find("table", {"id": "basic_school_stats"})
    if table is None:
        # Try to find it in HTML comments
        import re
        comments = soup.find_all(string=lambda text: isinstance(text, str) and "basic_school_stats" in text)
        for comment in comments:
            inner_soup = BeautifulSoup(comment, "html.parser")
            table = inner_soup.find("table", {"id": "basic_school_stats"})
            if table:
                break

    if table is None:
        # Fall back to any table on the page
        tables = soup.find_all("table")
        if not tables:
            print(f"  [{year}] No tables found")
            return None
        table = tables[0]

    try:
        html_str = str(table)
        dfs = pd.read_html(StringIO(html_str), header=[0, 1])
        df = dfs[0]
        # Flatten multi-level headers
        df.columns = [
            "_".join(str(c).strip() for c in col if str(c) != "nan").strip("_")
            if isinstance(col, tuple) else str(col)
            for col in df.columns
        ]
    except Exception as e:
        print(f"  [{year}] Parse error: {e}")
        return None

    df.insert(0, "season", year)

    # Drop separator rows (rows where School is NaN or "School")
    school_col = next((c for c in df.columns if "school" in c.lower() or "team" in c.lower()), None)
    if school_col:
        df = df[df[school_col].notna()]
        df = df[~df[school_col].isin(["School", "Team", ""])]
        df = df.rename(columns={school_col: "team"})

    return df


def fetch_season_advanced(year: int) -> pd.DataFrame | None:
    """
    Fetch advanced per-school stats (ORtg, DRtg, Pace, eFG%, etc.)
    URL: https://www.sports-reference.com/cbb/seasons/men/YYYY-advanced-school-stats.html
    """
    url = f"{BASE_URL}/{year}-advanced-school-stats.html"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  [{year}] Advanced HTTP error: {e}")
        return None

    try:
        dfs = pd.read_html(StringIO(r.text), attrs={"id": "adv_school_stats"})
    except Exception:
        try:
            dfs = pd.read_html(StringIO(r.text))
        except Exception as e:
            print(f"  [{year}] Advanced parse error: {e}")
            return None

    if not dfs:
        return None

    df = dfs[0]
    # Flatten multi-level headers
    df.columns = [
        "_".join(str(c).strip() for c in col if "unnamed" not in str(c).lower() and str(c) != "nan").strip("_")
        if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    df.insert(0, "season", year)

    school_col = next((c for c in df.columns if "school" in c.lower() or "team" in c.lower()), None)
    if school_col:
        df = df[df[school_col].notna()]
        df = df[~df[school_col].isin(["School", "Team", ""])]
        df = df.rename(columns={school_col: "team"})

    return df


def main():
    print(f"Fetching Sports-Reference CBB school stats {START_YEAR}–{END_YEAR}...")
    print("(Rate-limited to ~1 req/3s — this will take a few minutes)\n")

    basic_frames = []
    adv_frames = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"  [{year}]", end=" ", flush=True)

        df_basic = fetch_season_schools(year)
        if df_basic is not None:
            out = RAW_DIR / f"sportsref_basic_{year}.csv"
            df_basic.to_csv(out, index=False)
            basic_frames.append(df_basic)
            print(f"basic:{len(df_basic)}", end=" ")
        time.sleep(3)

        df_adv = fetch_season_advanced(year)
        if df_adv is not None:
            out = RAW_DIR / f"sportsref_adv_{year}.csv"
            df_adv.to_csv(out, index=False)
            adv_frames.append(df_adv)
            print(f"adv:{len(df_adv)}", end=" ")
        time.sleep(3)

        print()

    if basic_frames:
        combined = pd.concat(basic_frames, ignore_index=True)
        combined.to_parquet(RAW_DIR / "sportsref_basic_all.parquet", index=False)
        print(f"\nBasic stats: {len(combined):,} rows → data/raw/sportsref_basic_all.parquet")

    if adv_frames:
        combined_adv = pd.concat(adv_frames, ignore_index=True)
        combined_adv.to_parquet(RAW_DIR / "sportsref_adv_all.parquet", index=False)
        print(f"Advanced stats: {len(combined_adv):,} rows → data/raw/sportsref_adv_all.parquet")


if __name__ == "__main__":
    main()

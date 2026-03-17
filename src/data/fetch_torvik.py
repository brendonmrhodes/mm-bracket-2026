"""
Fetch Bart Torvik / T-Rank team ratings for all available seasons.
Free data, no login required.
Saves: data/raw/torvik_YYYY.csv for each year
       data/raw/torvik_all.parquet (combined)

Field mapping documented at: https://adamcwisports.blogspot.com/p/data.html
"""

import requests
import json
import pandas as pd
import time
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TORVIK_COLUMNS = [
    "rank", "team", "conf", "record", "adjoe", "adjoe_rk",
    "adjde", "adjde_rk", "barthag", "barthag_rk",
    "proj_w", "proj_l", "proj_conf_w", "proj_conf_l", "conf_record",
    "sos", "ncsos", "conf_sos", "proj_sos", "proj_nc_sos", "proj_conf_sos",
    "elite_sos", "elite_nc_sos",
    "opp_oe", "opp_de", "opp_proj_oe", "opp_proj_de",
    "conf_adjoe", "conf_adjde",
    "qual_o", "qual_d", "qual_barthag", "qual_games",
    "fun", "conf_pf", "conf_pa", "conf_poss",
    "conf_oe_ratio", "conf_de_ratio", "conf_sos_remain",
    "conf_win_pct", "wab", "wab_rk", "fun_rk", "adjt",
]

# Core metrics we care most about
KEEP_COLS = [
    "season", "team", "conf", "record",
    "adjoe", "adjde", "barthag", "adjt",
    "sos", "ncsos", "elite_sos",
    "qual_o", "qual_d", "qual_barthag", "qual_games",
    "wab", "fun",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

START_YEAR = 2008  # Torvik data is reliable from 2008
END_YEAR   = 2026


def fetch_year(year: int) -> pd.DataFrame | None:
    url = f"https://barttorvik.com/{year}_team_results.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        raw = json.loads(r.text)
    except Exception as e:
        print(f"  [{year}] ERROR: {e}")
        return None

    if not raw or not isinstance(raw[0], list):
        print(f"  [{year}] Unexpected format")
        return None

    # Pad/trim rows to match column count
    n_cols = len(TORVIK_COLUMNS)
    padded = [row[:n_cols] + [None] * max(0, n_cols - len(row)) for row in raw]
    df = pd.DataFrame(padded, columns=TORVIK_COLUMNS)
    df.insert(0, "season", year)
    return df


def main():
    all_frames = []
    print(f"Fetching Torvik data {START_YEAR}–{END_YEAR}...")

    for year in range(START_YEAR, END_YEAR + 1):
        df = fetch_year(year)
        if df is not None:
            out = RAW_DIR / f"torvik_{year}.csv"
            df.to_csv(out, index=False)
            all_frames.append(df)
            print(f"  [{year}] {len(df)} teams → {out.name}")
        time.sleep(0.5)  # be polite

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        # Coerce numeric columns
        for col in combined.columns:
            if col not in ["team", "conf", "record", "conf_record"]:
                combined[col] = pd.to_numeric(combined[col], errors="coerce")

        combined.to_parquet(RAW_DIR / "torvik_all.parquet", index=False)
        print(f"\nCombined: {len(combined):,} rows across {combined['season'].nunique()} seasons")
        print(f"Saved → data/raw/torvik_all.parquet")

        # Also save a slim version with just the key metrics
        slim = combined[KEEP_COLS].dropna(subset=["adjoe", "adjde", "barthag"])
        slim.to_parquet(RAW_DIR / "torvik_slim.parquet", index=False)
        print(f"Slim version: {len(slim):,} rows → data/raw/torvik_slim.parquet")


if __name__ == "__main__":
    main()

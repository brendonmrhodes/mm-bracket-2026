"""
Download FiveThirtyEight's archived NCAA tournament forecast data.
FiveThirtyEight shut down in June 2023; data is frozen but free (CC BY 4.0).

Sources:
  - Historical NCAA tournament forecasts (per-round win probabilities, 2016-2023)
  - Community NCAAB Elo ratings (grdavis/college-basketball-elo, 2000-2023)

Saves: data/raw/538_ncaa_forecasts.csv
       data/raw/538_ncaab_elo.csv
"""

import time
import requests
import pandas as pd
from pathlib import Path
from io import StringIO

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# FiveThirtyEight historical NCAA tournament model results (2016-2023)
FTE_FORECAST_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "historical-ncaa-forecasts/"
    "historical-538-ncaa-tournament-model-results.csv"
)

# FTE 2014 tournament bracket win probability files (62 snapshots of the bracket)
FTE_MM_BASE = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "march-madness-predictions/"
)
# 62 bracket snapshot files: bracket-00.csv through bracket-61.csv
FTE_MM_FILES = [f"bracket-{str(i).zfill(2)}.csv" for i in range(62)]


def download_csv(url: str, label: str) -> pd.DataFrame | None:
    print(f"  Fetching {label}...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        print(f"    {len(df):,} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    print("=== FiveThirtyEight / Elo Data ===\n")

    # 1. FTE historical tournament forecasts
    df_fte = download_csv(FTE_FORECAST_URL, "538 historical NCAA tournament forecasts")
    if df_fte is not None:
        out = RAW_DIR / "538_ncaa_forecasts.csv"
        df_fte.to_csv(out, index=False)
        print(f"    Saved → {out.name}")
        print(f"    Columns: {list(df_fte.columns)}")
        if "year" in df_fte.columns:
            print(f"    Years: {sorted(df_fte['year'].unique())}")
        elif "Season" in df_fte.columns:
            print(f"    Seasons: {sorted(df_fte['Season'].unique())}")

    print()
    # Note: Elo ratings are computed from scratch using compute_elo.py
    # which builds them from the Kaggle game-by-game data (more reliable).
    print()

    # 3. FTE 2014 bracket win probability snapshots (62 files, one per model update)
    print("  Fetching 538 2014 tournament bracket snapshots (62 files)...")
    mm_frames = []
    for fname in FTE_MM_FILES:
        df = download_csv(FTE_MM_BASE + fname, fname)
        if df is not None:
            df["snapshot"] = fname
            mm_frames.append(df)
        time.sleep(0.2)

    if mm_frames:
        combined_mm = pd.concat(mm_frames, ignore_index=True)
        # Use only the final snapshot (bracket-61.csv) as the pre-tournament prediction
        final_snap = combined_mm[combined_mm["snapshot"] == "bracket-61.csv"].copy()
        final_snap["season"] = 2014
        final_snap.to_csv(RAW_DIR / "538_2014_final_bracket.csv", index=False)
        # Coerce mixed-type columns before saving
        for col in combined_mm.columns:
            if col not in ("team_name", "team_region", "snapshot"):
                combined_mm[col] = pd.to_numeric(combined_mm[col], errors="coerce")
        combined_mm.to_parquet(RAW_DIR / "538_2014_all_snapshots.parquet", index=False)
        print(f"\n  2014 final bracket: {len(final_snap)} teams")
        print(f"  All snapshots: {len(combined_mm):,} rows → data/raw/538_2014_all_snapshots.parquet")
        print(f"  Columns: {list(final_snap.columns)}")


if __name__ == "__main__":
    main()

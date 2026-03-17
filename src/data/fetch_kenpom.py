"""
Scrape KenPom main ratings (AdjEM, AdjO, AdjD, AdjT, Luck, SOS) 2002-2026.
Uses Playwright Firefox — not detected by Cloudflare.
Reads existing Firefox session cookies, no manual login needed.

Usage:
    python src/data/fetch_kenpom.py

Saves: data/raw/kenpom_YYYY.csv  +  data/raw/kenpom_all.parquet
"""

import time
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kenpom_session import make_firefox_session, verify_session, fetch_page, parse_table

RAW_DIR    = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2002
END_YEAR   = 2026
DELAY      = 1.5


def parse_ratings(html: str, year: int) -> pd.DataFrame:
    df = parse_table(html, table_id="ratings-table")
    if df.empty:
        return df

    # Rename to canonical KenPom column names
    kenpom_cols = [
        "rank", "team", "conf", "record", "adjEM",
        "adjO", "adjO_rk", "adjD", "adjD_rk",
        "adjT", "adjT_rk", "luck", "luck_rk",
        "sos_adjEM", "sos_adjEM_rk",
        "opp_O", "opp_O_rk", "opp_D", "opp_D_rk",
        "nc_sos_adjEM", "nc_sos_adjEM_rk",
    ]
    df.columns = kenpom_cols[:len(df.columns)]
    df.insert(0, "season", year)

    # Strip seed suffixes: "Duke 1" → "Duke"
    if "team" in df.columns:
        df["team"] = df["team"].str.replace(r"\s+\d+$", "", regex=True).str.strip()

    return df


def main():
    with sync_playwright() as pw:
        browser, context, page = make_firefox_session(pw)

        print("Verifying KenPom session...")
        if not verify_session(page):
            print("ERROR: Session invalid. Log into kenpom.com in Firefox and try again.")
            browser.close()
            return
        print("  Session OK.\n")

        all_frames = []
        print(f"Fetching ratings {START_YEAR}–{END_YEAR}...")

        for year in range(START_YEAR, END_YEAR + 1):
            html = fetch_page(page, f"https://kenpom.com/?y={year}", "#ratings-table")
            if not html:
                print(f"  [{year}] SKIP")
                continue

            df = parse_ratings(html, year)
            if df.empty:
                print(f"  [{year}] empty parse")
                continue

            df.to_csv(RAW_DIR / f"kenpom_{year}.csv", index=False)
            all_frames.append(df)
            em_col = "adjEM" if "adjEM" in df.columns else df.columns[4]
            print(f"  [{year}] {len(df)} teams  "
                  f"adjEM: {df[em_col].min():.1f} – {df[em_col].max():.1f}")
            time.sleep(DELAY)

        browser.close()

    if not all_frames:
        print("No data collected.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_parquet(RAW_DIR / "kenpom_all.parquet", index=False)
    print(f"\nSaved {len(combined):,} rows → data/raw/kenpom_all.parquet")
    latest = combined[combined["season"] == combined["season"].max()]
    print("\nTop 10 (most recent season):")
    print(latest.nlargest(10, "adjEM")[["season","team","conf","adjEM","adjO","adjD","adjT"]].to_string(index=False))


if __name__ == "__main__":
    main()

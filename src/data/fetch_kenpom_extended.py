"""
Scrape all additional KenPom data pages using Playwright Firefox.
Not detected by Cloudflare. Reads Firefox session cookies automatically.

Pages scraped (all years 2002-2026 unless noted):
  stats.php        — Four Factors: eFG%, TO%, OR%, FTR (off + def)
  height.php       — Height & Experience (avg height, experience in min-weighted years)
  teamstats.php    — Misc: 3PA%, A/TO ratio, block%, steal%, FT%
  pointdist.php    — Point distribution: % pts from 2P, 3P, FT
  officials.php    — Referee ratings
  playerstats.php  — Player stats (for roster quality index)
  hca.php          — Home court advantage (single page, no year filter)

Usage:
    python src/data/fetch_kenpom_extended.py
    python src/data/fetch_kenpom_extended.py --pages four_factors height_exp
    python src/data/fetch_kenpom_extended.py --start 2010 --end 2026

Saves: data/raw/kenpom_<page>.parquet for each page type
"""

import argparse
import time
import random
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kenpom_session import make_firefox_session, verify_session, fetch_page, parse_table, _human_pause

RAW_DIR    = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DELAY       = 6.0   # base seconds between pages — do not lower
JITTER      = 3.0   # ± random seconds added to each delay
SECTION_GAP = 20.0  # extra pause between different page types

# ─── Page definitions ─────────────────────────────────────────────────────────
# Each entry: (key, url_template, table_selector, output_file)
PAGES = {
    "four_factors": (
        "stats.php?y={year}",
        "table",
        "kenpom_four_factors.parquet",
        "Four Factors (eFG%, TO%, OR%, FTR off+def)",
    ),
    "height_exp": (
        "height.php?y={year}",
        "table",
        "kenpom_height_exp.parquet",
        "Height & Experience",
    ),
    "misc_stats": (
        "teamstats.php?y={year}",
        "table",
        "kenpom_misc_stats.parquet",
        "Misc stats (3PA%, A/TO, BLK%, STL%, FT%)",
    ),
    "point_dist": (
        "pointdist.php?y={year}",
        "table",
        "kenpom_point_dist.parquet",
        "Point distribution (2P%, 3P%, FT% of scoring)",
    ),
    "officials": (
        "officials.php?y={year}",
        "table",
        "kenpom_officials.parquet",
        "Referee ratings",
    ),
    "players": (
        "playerstats.php?y={year}",
        "table",
        "kenpom_players.parquet",
        "Player stats (roster quality inputs)",
    ),
}

# HCA is a single page with no year parameter
HCA_PAGE = ("hca.php", "table", "kenpom_hca.parquet", "Home court advantage ratings")


def scrape_page_by_year(page, url_template: str, selector: str,
                        start: int, end: int, label: str) -> list[pd.DataFrame]:
    frames = []
    for year in range(start, end + 1):
        url = f"https://kenpom.com/{url_template.format(year=year)}"
        html = fetch_page(page, url, selector)
        if not html:
            print(f"    [{year}] failed")
            time.sleep(DELAY)
            continue

        df = parse_table(html)
        if df.empty:
            print(f"    [{year}] empty")
            time.sleep(DELAY)
            continue

        df.insert(0, "season", year)

        # Strip seed suffixes from team names
        team_col = next((c for c in df.columns if c.lower() in ("team", "name")), None)
        if team_col:
            df[team_col] = df[team_col].str.replace(r"\s+\d+$", "", regex=True).str.strip()

        frames.append(df)
        print(f"    [{year}] {len(df)} rows, {len(df.columns)} cols")
        _human_pause(DELAY, JITTER)

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", nargs="*", default=list(PAGES.keys()) + ["hca"],
                        help="Which pages to scrape (default: all)")
    parser.add_argument("--start", type=int, default=2002)
    parser.add_argument("--end",   type=int, default=2026)
    args = parser.parse_args()

    with sync_playwright() as pw:
        browser, context, page = make_firefox_session(pw)

        print("Verifying KenPom session...")
        if not verify_session(page):
            print("ERROR: Session invalid. Log into kenpom.com in Firefox first.")
            browser.close()
            return
        print("  Session OK.\n")

        first_section = True
        for key, (url_tmpl, selector, outfile, description) in PAGES.items():
            if key not in args.pages:
                continue

            if not first_section:
                print(f"  [inter-section pause {SECTION_GAP:.0f}s...]")
                time.sleep(SECTION_GAP)
            first_section = False

            print(f"{'='*55}")
            print(f"{description}  ({args.start}–{args.end})")
            print(f"{'='*55}")

            frames = scrape_page_by_year(
                page, url_tmpl, selector, args.start, args.end, key
            )

            if frames:
                combined = pd.concat(frames, ignore_index=True)
                out = RAW_DIR / outfile
                combined.to_parquet(out, index=False)
                print(f"  → Saved {len(combined):,} rows to data/raw/{outfile}")
                print(f"    Columns: {list(combined.columns[:18])}\n")
            else:
                print(f"  No data for {key}\n")

        # HCA — single page, no year loop
        if "hca" in args.pages:
            print(f"{'='*55}")
            print("Home court advantage ratings (single page)")
            print(f"{'='*55}")
            html = fetch_page(page, "https://kenpom.com/hca.php", "table")
            if html:
                df_hca = parse_table(html)
                if not df_hca.empty:
                    df_hca.to_parquet(RAW_DIR / "kenpom_hca.parquet", index=False)
                    print(f"  → Saved {len(df_hca)} rows to data/raw/kenpom_hca.parquet")
                    print(f"    Columns: {list(df_hca.columns)}\n")

        browser.close()

    print("Extended KenPom scrape complete.")


if __name__ == "__main__":
    main()

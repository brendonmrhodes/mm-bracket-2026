"""
Scrape coaching history from Sports-Reference CBB.

For each D1 school, fetches per-season records including coach name,
wins, losses, and tournament results. Used to compute:
  - coach_tenure: years as head coach at this school
  - coach_tourney_wins: career NCAA tournament wins (at any school)
  - coach_tourney_appearances: career tournament appearances
  - coach_win_pct: career regular-season win rate
  - coach_final4s: career Final Four appearances
  - coach_championships: career national championships

Output: data/raw/coach_features.parquet

Usage:
    python src/data/fetch_coach_data.py
    python src/data/fetch_coach_data.py --start 2003 --end 2026
"""

import argparse
import time
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from team_name_map import build_name_map, lookup_team_id

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

BASE_URL = "https://www.sports-reference.com/cbb"
DELAY    = 4.0   # seconds between requests


def get_all_schools() -> list[dict]:
    """
    Fetch the list of all active D1 schools from Sports-Reference.
    Returns list of {name, slug} dicts.
    """
    url = f"{BASE_URL}/schools/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  Cannot fetch school list: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": "schools"})
    if table is None:
        print("  Schools table not found — trying fallback")
        table = soup.find("table")

    if table is None:
        return []

    rows = []
    for tr in table.find_all("tr"):
        td = tr.find("td", {"data-stat": "school_name"})
        if td and td.find("a"):
            href = td.find("a")["href"]
            slug = href.rstrip("/").split("/")[-1]
            name = td.get_text(strip=True)
            rows.append({"name": name, "slug": slug})

    print(f"  Found {len(rows)} schools")
    return rows


def fetch_school_seasons(slug: str) -> pd.DataFrame | None:
    """
    Fetch season-by-season records for a school.
    URL: https://www.sports-reference.com/cbb/schools/{slug}/men/
    Returns DataFrame with columns: season, school, coach, wins, losses, tournament_result
    """
    url = f"{BASE_URL}/schools/{slug}/men/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        time.sleep(DELAY)
    except Exception as e:
        print(f"    [{slug}] HTTP error: {e}")
        time.sleep(DELAY)
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # Find the seasons history table
    table = soup.find("table", {"id": "seasons"})
    if table is None:
        # Try to find in comments
        for comment in soup.find_all(string=lambda t: isinstance(t, str) and "seasons" in t):
            inner = BeautifulSoup(comment, "html.parser")
            table = inner.find("table", {"id": "seasons"})
            if table:
                break

    if table is None:
        return None

    try:
        df = pd.read_html(StringIO(str(table)))[0]
    except Exception:
        return None

    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col if "Unnamed" not in str(c)).strip("_")
                      for col in df.columns]

    df.columns = [str(c).lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    df.insert(0, "school_slug", slug)

    return df


def parse_tournament_round(result_str: str) -> int:
    """
    Convert tournament result string to max round reached (wins in tournament).
    R1=0 wins, R2=1 win, S16=2, E8=3, F4=4, 2nd=5, 1st=6
    Returns 0 if no appearance.
    """
    if not isinstance(result_str, str) or result_str.strip() == "" or result_str == "—":
        return 0
    result = result_str.upper().strip()
    mapping = {
        "1ST": 6, "CHAMPION": 6, "NATL CHAMP": 6,
        "2ND": 5, "FINALIST": 5, "RUNNER-UP": 5,
        "F4": 4, "FINAL FOUR": 4, "SEMIFINAL": 4,
        "E8": 3, "ELITE EIGHT": 3, "ELITE 8": 3, "REGIONAL": 3,
        "S16": 2, "SWEET 16": 2, "SWEET SIXTEEN": 2,
        "R2": 1, "2ND ROUND": 1, "SECOND ROUND": 1,
        "R1": 0, "1ST ROUND": 0, "FIRST ROUND": 0,
    }
    for key, val in mapping.items():
        if key in result:
            return val
    # If any tourney result present, at least 1 appearance
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2003)
    parser.add_argument("--end",   type=int, default=2026)
    args = parser.parse_args()

    print(f"Fetching coaching data for seasons {args.start}–{args.end}...")

    # Step 1: get school list
    time.sleep(DELAY)
    schools = get_all_schools()
    if not schools:
        print("ERROR: Could not fetch school list")
        return

    time.sleep(DELAY * 2)

    # Step 2: fetch per-school season records
    all_rows = []
    for i, school in enumerate(schools, 1):
        slug = school["slug"]
        name = school["name"]
        print(f"  [{i:3d}/{len(schools)}] {name:30s}", end=" ", flush=True)

        df = fetch_school_seasons(slug)
        if df is None or len(df) == 0:
            print("skip")
            continue

        # Try to find year and coach columns
        year_col  = next((c for c in df.columns if "year" in c or "season" in c), None)
        coach_col = next((c for c in df.columns if "coach" in c), None)
        wins_col  = next((c for c in df.columns if c in ("w", "wins", "overall_w")), None)
        loss_col  = next((c for c in df.columns if c in ("l", "losses", "overall_l")), None)
        tourn_col = next((c for c in df.columns
                          if any(t in c for t in ["tournament", "ncaa", "postseason", "conf_tourn"])), None)

        if year_col is None:
            print("no year col")
            continue

        df = df.rename(columns={
            year_col:  "year_str",
            **({"coach_name" if coach_col is None else coach_col: "coach_name"} if coach_col else {}),
        })

        # Parse year to int
        def _parse_year(y):
            m = re.search(r"\d{4}", str(y))
            return int(m.group()) if m else None

        df["season"] = df["year_str"].apply(_parse_year)
        df = df[df["season"].notna()]
        df["season"] = df["season"].astype(int)
        df = df[(df["season"] >= args.start) & (df["season"] <= args.end)]

        if len(df) == 0:
            print("no data in range")
            continue

        # Extract numeric W/L if available
        if wins_col and wins_col in df.columns:
            df["wins"] = pd.to_numeric(df[wins_col], errors="coerce")
        else:
            df["wins"] = np.nan
        if loss_col and loss_col in df.columns:
            df["losses"] = pd.to_numeric(df[loss_col], errors="coerce")
        else:
            df["losses"] = np.nan

        # Tournament result
        if tourn_col and tourn_col in df.columns:
            df["tourn_wins"] = df[tourn_col].apply(parse_tournament_round)
        else:
            df["tourn_wins"] = 0

        # Coach name
        if coach_col and coach_col in df.columns:
            df["coach_name"] = df[coach_col].astype(str).str.strip()
        else:
            df["coach_name"] = "Unknown"

        df["school_name"] = name
        keep = ["season", "school_slug", "school_name", "coach_name", "wins", "losses", "tourn_wins"]
        all_rows.append(df[[c for c in keep if c in df.columns]])
        print(f"{len(df)} rows")

    if not all_rows:
        print("No data collected!")
        return

    import numpy as np
    raw = pd.concat(all_rows, ignore_index=True)
    print(f"\nTotal raw rows: {len(raw):,}")

    # Step 3: Build career coach statistics
    # For each coach, compute career stats UP TO but not including each season
    # (to avoid leakage)
    print("\nBuilding career coach features...")

    raw = raw.sort_values(["coach_name", "season"])
    raw["games"] = raw["wins"].fillna(0) + raw["losses"].fillna(0)
    raw["tourn_appearance"] = (raw["tourn_wins"] > 0).astype(int)

    coach_career = {}
    for coach, grp in raw.groupby("coach_name"):
        grp = grp.sort_values("season")
        cum_wins = 0
        cum_losses = 0
        cum_tourn_apps = 0
        cum_tourn_wins = 0
        cum_f4s = 0
        cum_champs = 0
        for _, row in grp.iterrows():
            key = (coach, row["season"])
            coach_career[key] = {
                "career_wins":         cum_wins,
                "career_losses":       cum_losses,
                "career_tourn_apps":   cum_tourn_apps,
                "career_tourn_wins":   cum_tourn_wins,
                "career_f4s":          cum_f4s,
                "career_champs":       cum_champs,
            }
            # Accumulate after storing (cumulative before this season)
            cum_wins        += row.get("wins", 0) or 0
            cum_losses      += row.get("losses", 0) or 0
            cum_tourn_apps  += row.get("tourn_appearance", 0) or 0
            tw               = row.get("tourn_wins", 0) or 0
            cum_tourn_wins  += tw
            cum_f4s         += 1 if tw >= 4 else 0
            cum_champs      += 1 if tw >= 6 else 0

    # Tenure at current school (seasons with same coach at same school)
    raw["school_tenure"] = 0
    for (school, coach), grp in raw.groupby(["school_slug", "coach_name"]):
        seasons = sorted(grp["season"].tolist())
        for i, s in enumerate(seasons):
            raw.loc[(raw["school_slug"] == school) & (raw["coach_name"] == coach) &
                    (raw["season"] == s), "school_tenure"] = i

    # Join career stats back
    career_rows = []
    for _, row in raw.iterrows():
        key = (row["coach_name"], row["season"])
        career = coach_career.get(key, {})
        career_wins = career.get("career_wins", 0)
        career_losses = career.get("career_losses", 0)
        games = career_wins + career_losses
        coach_win_pct = career_wins / games if games > 0 else 0.5
        career_rows.append({
            "season":             row["season"],
            "school_slug":        row["school_slug"],
            "school_name":        row["school_name"],
            "coach_name":         row["coach_name"],
            "coach_tenure":       row.get("school_tenure", 0),
            "coach_win_pct":      round(coach_win_pct, 4),
            "coach_tourn_apps":   career.get("career_tourn_apps", 0),
            "coach_tourn_wins":   career.get("career_tourn_wins", 0),
            "coach_f4s":          career.get("career_f4s", 0),
            "coach_champs":       career.get("career_champs", 0),
        })

    coach_features = pd.DataFrame(career_rows)

    # Step 4: Map school names → Kaggle TeamIDs
    print("Mapping school names to Kaggle TeamIDs...")
    name_map = build_name_map()
    coach_features["TeamID"] = coach_features["school_name"].apply(
        lambda n: lookup_team_id(n, name_map)
    )
    # Also try slug-based name
    coach_features.loc[coach_features["TeamID"].isna(), "TeamID"] = (
        coach_features.loc[coach_features["TeamID"].isna(), "school_slug"]
        .str.replace("-", " ").str.title()
        .apply(lambda n: lookup_team_id(n, name_map))
    )

    matched = coach_features["TeamID"].notna().sum()
    print(f"  Matched {matched}/{len(coach_features)} rows ({matched/len(coach_features):.1%})")

    coach_features = coach_features.dropna(subset=["TeamID"])
    coach_features["TeamID"] = coach_features["TeamID"].astype(int)
    coach_features["Season"] = coach_features["season"]

    out = RAW_DIR / "coach_features.parquet"
    coach_features.to_parquet(out, index=False)
    print(f"\nCoach features saved → {out}")
    print(f"  {len(coach_features):,} rows, {coach_features['season'].nunique()} seasons")
    print(f"  Columns: {list(coach_features.columns)}")

    # Quick preview
    print("\nSample:")
    print(coach_features[["Season", "school_name", "coach_name",
                           "coach_tenure", "coach_tourn_apps", "coach_f4s"]]
          .sort_values(["Season", "school_name"]).tail(20).to_string(index=False))


if __name__ == "__main__":
    import numpy as np
    main()

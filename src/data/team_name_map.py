"""
Team name normalization and cross-source ID mapping.

The Kaggle dataset uses numeric TeamIDs. External sources (KenPom, Torvik,
Sports-Reference) use team name strings — all with slightly different spellings.

This module builds a lookup: normalized_name → Kaggle TeamID

Strategy:
  1. Load MTeamSpellings.csv (Kaggle's own name-matching list)
  2. Fuzzy-match any remaining unmatched names using rapidfuzz
"""

import re
import pandas as pd
from pathlib import Path

try:
    from rapidfuzz import process, fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

KAGGLE_DIR = Path(__file__).resolve().parent.parent.parent / "march-machine-learning-mania-2026"
DATA_DIR   = Path(__file__).resolve().parent.parent.parent / "data"


def _normalize(name: str) -> str:
    """Lowercase, remove punctuation, collapse spaces."""
    name = str(name).lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Common abbreviation expansions
    replacements = {
        "st ": "state ",
        "st.": "state",
        "univ ": "",
        "university": "",
        "college": "",
        "the ": "",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.strip()


def build_name_map() -> dict:
    """
    Returns dict: raw_name_string → TeamID (int)
    """
    spellings = pd.read_csv(KAGGLE_DIR / "MTeamSpellings.csv", encoding="latin-1")
    teams     = pd.read_csv(KAGGLE_DIR / "MTeams.csv")

    # spellings has columns: TeamNameSpelling, TeamID
    # Normalize all spellings
    name_to_id = {}
    for _, row in spellings.iterrows():
        raw = str(row["TeamNameSpelling"]).strip()
        tid = int(row["TeamID"])
        name_to_id[raw.lower()] = tid
        name_to_id[_normalize(raw)] = tid

    # Also add canonical team names from MTeams
    for _, row in teams.iterrows():
        raw = str(row["TeamName"]).strip()
        tid = int(row["TeamID"])
        name_to_id[raw.lower()] = tid
        name_to_id[_normalize(raw)] = tid

    return name_to_id


def lookup_team_id(name: str, name_map: dict, fuzzy_threshold: int = 85) -> int | None:
    """
    Look up a team name string and return its Kaggle TeamID.
    Falls back to fuzzy matching if exact match fails.
    """
    key = name.strip().lower()
    if key in name_map:
        return name_map[key]

    norm = _normalize(name)
    if norm in name_map:
        return name_map[norm]

    if HAS_FUZZY:
        match, score, _ = process.extractOne(norm, list(name_map.keys()), scorer=fuzz.token_sort_ratio)
        if score >= fuzzy_threshold:
            return name_map[match]

    return None


def add_team_ids(df: pd.DataFrame, team_col: str, name_map: dict,
                 out_col: str = "TeamID") -> pd.DataFrame:
    """Add a TeamID column to a DataFrame using the name map."""
    df = df.copy()
    df[out_col] = df[team_col].apply(lambda n: lookup_team_id(n, name_map))
    n_unmatched = df[out_col].isna().sum()
    if n_unmatched > 0:
        unmatched = df[df[out_col].isna()][team_col].unique()
        print(f"  WARNING: {n_unmatched} unmatched teams: {list(unmatched[:10])}")
    return df


if __name__ == "__main__":
    name_map = build_name_map()
    print(f"Name map built: {len(name_map):,} entries")
    # Test a few
    tests = ["Duke", "UConn", "Connecticut", "Michigan St", "Michigan State",
             "St. John's", "Saint John's", "Texas A&M", "UC Irvine"]
    for t in tests:
        tid = lookup_team_id(t, name_map)
        print(f"  {t!r:30s} → {tid}")

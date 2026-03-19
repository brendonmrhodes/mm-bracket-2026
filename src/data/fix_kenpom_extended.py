"""
Fix column alignment in KenPom extended parquet files.

KenPom's tables store (rank, value) pairs for each stat.
The parse_table function combined multi-level headers incorrectly,
causing column names to be off by one. This script remaps
the raw parquet columns to their true semantic meaning.

Verified mapping using Duke 2026 as ground truth:
  - adjO=128.0 → found in 'Offense_eFG%' column
  - adjD=89.1  → found in 'col_14' column
  - AdjTempo=65.3 → found in 'AdjTempo' column ✓

Run once to fix the parquets; re-run feature engineering afterward.

Usage:
    python src/data/fix_kenpom_extended.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# ── Four Factors (stats.php) ────────────────────────────────────────────────
# Column-by-column mapping: raw name → correct name (None = drop, it's a rank)
FOUR_FACTORS_MAP = {
    "season":            "season",
    "Team":              "Team",
    "Conf":              "Conf",
    "AdjTempo":          "AdjTempo",       # value ✓ (65.3 for Duke)
    "AdjOE":             None,             # AdjTempo rank → drop
    "Offense_eFG%":      "AdjOE",          # AdjOE value (128.0 for Duke)
    "Offense_TO%":       None,             # AdjOE rank → drop
    "Offense_OR%":       "O_eFG_pct",      # Offense eFG% value (56.8 for Duke)
    "Offense_FTRate":    None,             # eFG% rank → drop
    "Offense_AdjDE":     "O_TO_pct",       # Offense TO% value (15.7 for Duke)
    "Offense_eFG%_1":    None,             # concatenated mess → drop
    "Offense_TO%_1":     "O_OR_pct",       # Offense OR% value (38.1 for Duke)
    "Offense_OR%_1":     None,             # rank → drop
    "Offense_FTRate_1":  "O_FTRate",       # Offense FTRate value (37.8 for Duke)
    "col_13":            None,             # rank → drop
    "col_14":            "AdjDE",          # AdjDE value (89.1 for Duke) ✓
    "col_15":            None,             # rank → drop
    "col_16":            "D_eFG_pct",      # Defense eFG% allowed (46.2 for Duke)
    "col_17":            None,             # rank → drop
    "col_18":            "D_TO_pct",       # Defense TO% forced (18.1 for Duke)
    "col_19":            None,             # rank → drop
    "col_20":            "D_OR_pct",       # Defense OR% allowed (24.8 for Duke)
    "col_21":            None,             # rank → drop
    "col_22":            "D_FTRate",       # Defense FTRate allowed (23.7 for Duke)
    "col_23":            None,             # rank → drop
}

# ── Height/Experience (height.php) ──────────────────────────────────────────
# The height page was parsed inconsistently across years.
# Key valid columns identified by value range analysis:
#   'Avg Hgt'   : 73–80 (actual avg height in inches)
#   'C Hgt'     : -7 to +9 (center height delta from national avg)
#   'SF Hgt'    : delta (some years have -99 as sentinel for missing)
#   'PG Hgt'    : delta
#   'Bench'     : bench player height delta
#   col_14      : height delta (~+/-5) for some position
#   col_16      : experience in years (0–4.5)
#   col_18      : continuity % (0–100)
#
# The named columns Eff Hgt, PF Hgt, SG Hgt, Experience, Continuity
# are storing RANKS (1–365), not actual values.

HEIGHT_MAP = {
    "season":       "season",
    "Team":         "Team",
    "Conf":         "Conf",
    # Main ratings accidentally included (already in main kenpom) → drop
    "NetRtg":       None,
    "ORtg":         None,
    "DRtg":         None,
    "AdjT":         None,
    "Luck":         None,
    "NetRtg_1":     None,
    "ORtg_1":       None,
    "DRtg_1":       None,
    "NetRtg_2":     None,
    "W-L":          None,
    "Rk":           None,
    # Verified for Duke 2026 (avg_hgt=79.3, experience=0.87yrs, continuity=34.6%)
    "col_13":       None,            # rank → drop
    "col_14":       "eff_hgt_delta", # effective height delta vs national avg (+3.2 Duke) ✓
    "col_15":       None,            # rank → drop
    "col_16":       "experience",    # experience in min-weighted years (0.87 Duke = freshmen) ✓
    "col_17":       None,            # rank → drop
    "col_18":       "continuity",    # % of minutes from returning players (34.6 Duke) ✓
    "col_19":       None,            # rank → drop
    "col_20":       None,            # unclear stat → drop
    # Named height columns (some are ranks due to column shift)
    "Avg Hgt":      "avg_hgt",       # average height in inches (79.3 Duke) ✓
    "Eff Hgt":      None,            # rank → drop
    "C Hgt":        "c_hgt_delta",   # center position height delta (+1.0 Duke) ✓
    "PF Hgt":       None,            # rank → drop
    "SF Hgt":       "pf_hgt_delta",  # PF position height delta (+0.4 Duke) ✓
    "SG Hgt":       None,            # rank → drop
    "PG Hgt":       "sf_hgt_delta",  # SF position height delta (+1.6 Duke) ✓
    "Experience":   None,            # rank → drop (use col_16 instead)
    "Bench":        "sg_hgt_delta",  # SG position height delta (+2.5 Duke) ✓
    "Continuity":   None,            # rank → drop (use col_18 instead)
    "col_12":       "pg_hgt_delta",  # PG position height delta (+2.2 Duke) ✓
    "col_21":       None,            # rank → drop
}


def fix_parquet(in_path: Path, col_map: dict, out_path: Path = None, label: str = "",
                already_fixed_marker: str = None):
    if not in_path.exists():
        print(f"  {label}: file not found — skipping")
        return

    df = pd.read_parquet(in_path)
    print(f"  {label}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Safety: save a backup before modifying
    backup_path = in_path.with_suffix(".bak.parquet")
    if not backup_path.exists():
        df.to_parquet(backup_path, index=False)
        print(f"    Backup saved → {backup_path.name}")
    else:
        print(f"    Backup already exists → {backup_path.name}")

    # Detect if already fixed (look for a key renamed column)
    if already_fixed_marker and already_fixed_marker in df.columns:
        print(f"    Already fixed (found '{already_fixed_marker}') — reloading backup")
        df = pd.read_parquet(backup_path)

    # Apply mapping: keep and rename columns that have non-None mapping
    keep = {}
    for raw_col, new_name in col_map.items():
        if raw_col in df.columns and new_name is not None:
            keep[raw_col] = new_name

    df_fixed = df[list(keep.keys())].rename(columns=keep)

    # Replace sentinel values (-99) with NaN for position height deltas
    # (delta columns: typical range ±10 inches; avg_hgt is absolute ~73–80 inches)
    for col in df_fixed.columns:
        if "delta" in col:
            vals = pd.to_numeric(df_fixed[col], errors="coerce")
            vals = vals.where(vals.abs() <= 15, np.nan)  # deltas > ±15 are sentinels
            df_fixed[col] = vals
    # avg_hgt: absolute height, valid range 70–85 inches
    if "avg_hgt" in df_fixed.columns:
        vals = pd.to_numeric(df_fixed["avg_hgt"], errors="coerce")
        vals = vals.where((vals >= 70) & (vals <= 85), np.nan)
        df_fixed["avg_hgt"] = vals
    # ht_delta_1: typically ±5 inches
    if "ht_delta_1" in df_fixed.columns:
        vals = pd.to_numeric(df_fixed["ht_delta_1"], errors="coerce")
        vals = vals.where(vals.abs() <= 10, np.nan)
        df_fixed["ht_delta_1"] = vals

    # Replace -99 sentinels in experience/continuity
    for col in ["experience", "continuity"]:
        if col in df_fixed.columns:
            vals = pd.to_numeric(df_fixed[col], errors="coerce")
            # experience: valid range 0–6, continuity: valid range 0–100
            upper = 6 if col == "experience" else 100
            vals = vals.where((vals >= 0) & (vals <= upper), np.nan)
            df_fixed[col] = vals

    # Coerce all non-ID columns to numeric
    skip_ids = {"season", "Team", "Conf"}
    for col in df_fixed.columns:
        if col not in skip_ids:
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors="coerce")

    out = out_path or in_path
    df_fixed.to_parquet(out, index=False)
    print(f"    Fixed: {df_fixed.shape[0]:,} rows × {df_fixed.shape[1]} cols")
    print(f"    Columns: {list(df_fixed.columns)}")

    # Quick sanity check on key columns
    if label == "Four Factors":
        for col, expected_range in [
            ("AdjTempo", (55, 90)),
            ("AdjOE",    (80, 140)),
            ("AdjDE",    (60, 125)),
            ("O_eFG_pct",(40, 70)),
        ]:
            if col in df_fixed.columns:
                v = df_fixed[col].dropna()
                pct_valid = ((v >= expected_range[0]) & (v <= expected_range[1])).mean()
                print(f"    {col:15s}: [{v.min():.1f}, {v.max():.1f}]  "
                      f"({pct_valid:.0%} in expected range {expected_range})")

    if label == "Height":
        for col in ["avg_hgt", "experience", "continuity"]:
            if col in df_fixed.columns:
                v = df_fixed[col].dropna()
                if len(v) > 0:
                    print(f"    {col:15s}: [{v.min():.2f}, {v.max():.2f}]  "
                          f"mean={v.mean():.2f}  coverage={df_fixed[col].notna().mean():.0%}")

    print()
    return df_fixed


MISC_STATS_MAP = {
    "season":   "season",
    "Team":     "Team",
    "Conf":     "Conf",
    # Verified mapping for Duke 2026 (adjO=128.0, known 3P%~35%, FT%~72%)
    "3P%":      "ms_3p_pct",      # 3-point shooting % (35.1 for Duke) ✓
    "2P%":      None,             # 3P% rank → drop
    "FT%":      "ms_2p_pct",      # 2-point shooting % (60.1 for Duke) ✓
    "Blk%":     None,             # 2P% rank → drop
    "Stl%":     "ms_ft_pct",      # FT% (72.4 for Duke) ✓
    "NST%":     None,             # FT% rank → drop
    "A%":       None,             # rank → drop (ambiguous)
    "3PA%":     None,             # position ambiguous → drop
    "AdjOE":    None,             # rank → drop
    "col_11":   None,             # rank → drop
    "col_12":   "ms_blk_pct",    # block % (~6.3 for Duke) ✓
    "col_13":   None,             # rank → drop
    "col_14":   "ms_stl_pct",    # steal % (~4.8 for Duke) ✓
    "col_15":   None,             # rank → drop
    "col_16":   "ms_a_pct",      # assist % (~59.2 for Duke) ✓
    "col_17":   None,             # rank → drop
    "col_18":   "ms_3pa_pct",    # 3-point attempt rate (~44.4 for Duke) ✓
    "col_19":   None,             # rank → drop
    "2P Dist":  "ms_2p_dist",    # avg 2P shot distance ✓
    "col_20":   None,             # AdjOE value — already in main kenpom → drop
    "col_21":   None,             # rank → drop
}


def main():
    print("Fixing KenPom extended parquet column alignment...\n")

    fix_parquet(
        RAW_DIR / "kenpom_four_factors.parquet",
        FOUR_FACTORS_MAP,
        label="Four Factors",
        already_fixed_marker="AdjOE",   # if AdjOE exists, data is already fixed
    )

    fix_parquet(
        RAW_DIR / "kenpom_height_exp.parquet",
        HEIGHT_MAP,
        label="Height",
        already_fixed_marker="avg_hgt", # if avg_hgt exists, data is already fixed
    )

    fix_parquet(
        RAW_DIR / "kenpom_misc_stats.parquet",
        MISC_STATS_MAP,
        label="Misc Stats",
        already_fixed_marker="ms_3p_pct",
    )

    print("Done. Re-run src/01_feature_engineering.py to rebuild features.")


if __name__ == "__main__":
    main()

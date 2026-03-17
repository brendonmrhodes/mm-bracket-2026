"""
Compute Championship DNA data for the app.

Generates two output files:
  outputs/champion_history.csv   — per-season champion stats (2003–2025)
  outputs/upset_rates.csv        — historical win rates for each seed matchup

The app uses these to render the Championship DNA tab.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
KAGGLE = ROOT / "march-machine-learning-mania-2026"
OUT    = ROOT / "outputs"

# ── Load source data ───────────────────────────────────────────────────────────
features = pd.read_parquet(ROOT / "data" / "team_season_features.parquet")
results  = pd.read_csv(KAGGLE / "MNCAATourneyCompactResults.csv")
seeds    = pd.read_csv(KAGGLE / "MNCAATourneySeeds.csv")
teams    = pd.read_csv(KAGGLE / "MTeams.csv")

seeds["SeedNum"] = seeds["Seed"].apply(lambda s: int("".join(filter(str.isdigit, s))))

# ── Find champions (winner of NCG each year) ───────────────────────────────────
ncg = results[results["DayNum"] >= 154].copy()
champions = (ncg.sort_values(["Season", "DayNum"], ascending=[True, False])
               .groupby("Season").first()[["WTeamID"]].reset_index()
               .rename(columns={"WTeamID": "TeamID"}))

# Merge with features and team names
champs = (champions
    .merge(features, on=["Season", "TeamID"], how="left")
    .merge(teams[["TeamID", "TeamName"]], on="TeamID", how="left")
    .merge(seeds[["Season", "TeamID", "SeedNum"]], on=["Season", "TeamID"], how="left"))

KEY_STATS = ["adjEM", "adjO", "adjD", "barthag", "wab", "elo_pre_tourney",
             "AvgScoreDiff", "WinPct", "sos_adjEM"]

# Keep only seasons where we have the key stats
champs_clean = champs[champs["Season"] >= 2003].copy()
# SeedNum may appear with suffix from merge; resolve it
if "SeedNum_y" in champs_clean.columns:
    champs_clean["SeedNum"] = champs_clean["SeedNum_y"]
elif "SeedNum_x" in champs_clean.columns:
    champs_clean["SeedNum"] = champs_clean["SeedNum_x"]
keep_cols = ["Season", "TeamName", "SeedNum"] + [c for c in KEY_STATS if c in champs_clean.columns]
champs_clean = champs_clean[keep_cols].copy()

# adjD: lower is better — negate so that higher always = better for percentile calcs
champs_clean["adjD_inv"] = -champs_clean["adjD"]

champs_clean.to_csv(OUT / "champion_history.csv", index=False)
print(f"champion_history.csv: {len(champs_clean)} rows")
print(champs_clean[["Season", "TeamName", "SeedNum", "adjEM", "adjO", "adjD",
                     "barthag", "elo_pre_tourney"]].tail(10).to_string())

# ── Champion profile: percentile thresholds vs all tourney teams ───────────────
# For each key stat, what fraction of champions exceeded each threshold?
all_tourney = (features.merge(seeds[["Season","TeamID","SeedNum"]], on=["Season","TeamID"])
                        .merge(teams[["TeamID","TeamName"]], on="TeamID"))
all_tourney = all_tourney[all_tourney["Season"] >= 2003]

# Compute champion means, medians, and fraction of tournament teams below champion median
profile = {}
for stat in KEY_STATS:
    vals = champs_clean[stat].dropna()
    all_vals = all_tourney[stat].dropna()
    if len(vals) == 0:
        continue
    med = vals.median()
    mn  = vals.mean()
    # Fraction of champions above/below threshold
    if stat == "adjD":
        # Lower is better for defense
        pct_pass = (vals < med).mean()  # fraction of champs with defense < median champ
        pct_tourney_worse = (all_vals > med).mean()  # fraction of tourney teams with worse defense
    else:
        pct_pass = (vals >= med).mean()
        pct_tourney_worse = (all_vals <= med).mean()  # fraction of all tourney teams at or below champ median
    profile[stat] = {
        "mean": round(mn, 2),
        "median": round(med, 2),
        "min": round(vals.min(), 2),
        "max": round(vals.max(), 2),
        "pct_tourney_below_champ_median": round(pct_tourney_worse * 100, 1),
    }

print("\nChampion profile:")
for stat, p in profile.items():
    print(f"  {stat:20s}  median={p['median']:8.2f}  "
          f"(top {100-p['pct_tourney_below_champ_median']:.0f}% of tourney teams)")

# ── Upset history: win rates by seed matchup ───────────────────────────────────
# Standard first-round matchups: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
seed_map = {row["Seed"]: row["SeedNum"]
            for _, row in seeds[seeds["Season"] >= 2003].iterrows()}

# Build a per-game seed lookup
results_seeded = results[results["Season"] >= 2003].copy()

def get_seed(season, team_id):
    row = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team_id)]
    return int(row["SeedNum"].iloc[0]) if len(row) else None

# Only R64 (DayNum 134-136) and R32 (137-139) for upset analysis
early = results_seeded[results_seeded["DayNum"] <= 139].copy()

rows = []
for _, g in early.iterrows():
    ws = get_seed(g["Season"], g["WTeamID"])
    ls = get_seed(g["Season"], g["LTeamID"])
    if ws is None or ls is None:
        continue
    lo, hi = min(ws, ls), max(ws, ls)
    winner_was_lower = (ws < ls)  # lower seed number = better seed = "favourite"
    rows.append({"Season": g["Season"], "Round": "R64" if g["DayNum"] <= 136 else "R32",
                 "FavSeed": lo, "DogSeed": hi, "FavWon": winner_was_lower})

matchup_df = pd.DataFrame(rows)

# Aggregate by seed matchup
upset_records = []
for (fav, dog), grp in matchup_df.groupby(["FavSeed", "DogSeed"]):
    total = len(grp)
    fav_wins = grp["FavWon"].sum()
    upset_records.append({
        "matchup": f"{fav} vs {dog}",
        "FavSeed": fav,
        "DogSeed": dog,
        "Games": total,
        "FavWinPct": round(fav_wins / total * 100, 1),
        "UpsetPct": round((total - fav_wins) / total * 100, 1),
    })

upset_df = pd.DataFrame(upset_records).sort_values(["FavSeed", "DogSeed"])

# Keep standard first/second round matchups
standard = upset_df[
    ((upset_df["FavSeed"] + upset_df["DogSeed"] == 17) |
     ((upset_df["FavSeed"] == 8) & (upset_df["DogSeed"] == 9)))
].copy()

standard.to_csv(OUT / "upset_rates.csv", index=False)
print(f"\nupset_rates.csv: {len(standard)} rows")
print(standard[["matchup", "Games", "FavWinPct", "UpsetPct"]].to_string(index=False))

# ── Save champion profile summary ─────────────────────────────────────────────
profile_rows = [{"stat": k, **v} for k, v in profile.items()]
pd.DataFrame(profile_rows).to_csv(OUT / "champion_profile.csv", index=False)
print("\nchampion_profile.csv saved.")

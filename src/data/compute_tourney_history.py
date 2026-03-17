"""
Build historical tournament performance features per team per season.

For each team in season S, computes metrics from their performance in
past tournaments (seasons < S):
  tourney_appearances     — # of tournament appearances ever
  tourney_recent_appear   — # appearances in last 5 seasons
  tourney_avg_seed        — average seed in past appearances
  tourney_win_rate        — win% in tournament games (historical)
  tourney_best_round      — best round ever reached (R64=1 .. Champion=7)
  tourney_recent_best     — best round in last 5 seasons
  tourney_r32_rate        — fraction of appearances reaching R32+
  tourney_s16_rate        — fraction of appearances reaching S16+
  tourney_e8_rate         — fraction of appearances reaching E8+
  tourney_f4_rate         — fraction of appearances reaching F4+
  tourney_ncg_rate        — fraction of appearances reaching NCG+
  tourney_champ_rate      — fraction of appearances winning it all

Saves: data/raw/tourney_history_features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent.parent
KAGGLE = ROOT / "march-machine-learning-mania-2026"
RAW    = ROOT / "data" / "raw"

# Load tournament results and seeds
results = pd.read_csv(KAGGLE / "MNCAATourneyCompactResults.csv")
seeds   = pd.read_csv(KAGGLE / "MNCAATourneySeeds.csv")

seeds["SeedNum"] = seeds["Seed"].apply(
    lambda s: int("".join(filter(str.isdigit, s)))
)

ROUND_SCORE = {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "NCG": 6, "Champion": 7}

# Build per-season per-team deepest round reached
# Use DayNum brackets to determine round:
#   R64: DayNum 136–137  R32: 138–139  S16: 143–145  E8: 146–148
#   F4: 152–154  NCG: 155–157
def daynum_to_round(day):
    if   day <= 137: return "R64"
    elif day <= 139: return "R32"
    elif day <= 145: return "S16"
    elif day <= 148: return "E8"
    elif day <= 154: return "F4"
    elif day <= 157: return "NCG"
    else:            return "Champion"

results["Round"] = results["DayNum"].apply(daynum_to_round)
results["RoundScore"] = results["Round"].map(ROUND_SCORE)

# Winner reached the round they won in; loser went out at the same round
win_rows = results[["Season", "WTeamID", "Round", "RoundScore"]].copy()
win_rows.columns = ["Season", "TeamID", "Round", "RoundScore"]

los_rows = results[["Season", "LTeamID", "Round", "RoundScore"]].copy()
los_rows.columns = ["Season", "TeamID", "Round", "RoundScore"]

# Loser's deepest round = the round they lost in (they PLAYED in it)
# Winner's contribution is handled by taking the MAX per team per season
all_rows = pd.concat([win_rows, los_rows], ignore_index=True)

# Deepest round per team per season
deepest = (all_rows.groupby(["Season", "TeamID"])["RoundScore"]
           .max().reset_index()
           .rename(columns={"RoundScore": "DeepestRound"}))

# Win counts per team per season
win_counts = (win_rows.groupby(["Season", "TeamID"])["RoundScore"]
              .count().reset_index()
              .rename(columns={"RoundScore": "TourneyWins"}))
game_counts = (all_rows.groupby(["Season", "TeamID"])["RoundScore"]
               .count().reset_index()
               .rename(columns={"RoundScore": "TourneyGames"}))

tourney_perf = (deepest
    .merge(win_counts,  on=["Season", "TeamID"], how="left")
    .merge(game_counts, on=["Season", "TeamID"], how="left")
    .merge(seeds[["Season", "TeamID", "SeedNum"]], on=["Season", "TeamID"], how="left")
)
tourney_perf["TourneyWins"] = tourney_perf["TourneyWins"].fillna(0)

# Get all team-seasons that ever appeared in the tournament
all_teams  = sorted(tourney_perf["TeamID"].unique())
# Include 2026 even though results don't exist yet — features are based on PAST seasons
all_seasons = sorted(set(results["Season"].unique()) | {2026})

records = []
for season in all_seasons:
    for team in all_teams:
        # Only generate rows for teams seeded in this tournament
        if not ((seeds["Season"] == season) & (seeds["TeamID"] == team)).any():
            continue

        # Past tournament data (seasons strictly before this one)
        past = tourney_perf[
            (tourney_perf["TeamID"] == team) &
            (tourney_perf["Season"] < season)
        ].sort_values("Season")

        recent = past[past["Season"] >= season - 5]

        n_appear = len(past)
        n_recent = len(recent)

        rec = {
            "Season":             season,
            "TeamID":             team,
            "tourney_appearances":    n_appear,
            "tourney_recent_appear":  n_recent,
            "tourney_avg_seed":       past["SeedNum"].mean() if n_appear > 0 else np.nan,
            "tourney_best_round":     past["DeepestRound"].max() if n_appear > 0 else 0,
            "tourney_recent_best":    recent["DeepestRound"].max() if n_recent > 0 else 0,
            "tourney_win_rate":       past["TourneyWins"].sum() / past["TourneyGames"].sum()
                                      if n_appear > 0 and past["TourneyGames"].sum() > 0 else 0,
        }

        # Round-reach rates
        for round_name, threshold in [("r32", 2), ("s16", 3), ("e8", 4),
                                       ("f4", 5), ("ncg", 6), ("champ", 7)]:
            if n_appear > 0:
                rec[f"tourney_{round_name}_rate"] = (past["DeepestRound"] >= threshold).mean()
            else:
                rec[f"tourney_{round_name}_rate"] = 0.0

        records.append(rec)

feat = pd.DataFrame(records)

# Fill avg_seed NaN with 17 (never been seeded = treat as worst seed)
feat["tourney_avg_seed"] = feat["tourney_avg_seed"].fillna(17)

out = RAW / "tourney_history_features.parquet"
feat.to_parquet(out, index=False)

print(f"Tournament history features: {len(feat):,} rows")
print(f"Seasons: {feat['Season'].min()} – {feat['Season'].max()}")
print(f"Saved → {out}")
print("\nSample — Florida 2026:")
sample = feat[(feat["TeamID"] == 1314) & (feat["Season"] == 2026)]
if not sample.empty:
    print(sample.iloc[0].to_dict())
print("\nFeature means:")
print(feat[[c for c in feat.columns if c.startswith("tourney_")]].mean().round(3))

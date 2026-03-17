"""
Build conference tournament features per team per season from MConferenceTourneyGames.csv.

Features:
  conf_tourney_wins      — games won in conference tournament
  conf_tourney_losses    — games lost (0 or 1, only lose once)
  conf_tourney_champion  — 1 if won the conf tournament (most wins in their conf)
  conf_tourney_finalist  — 1 if reached the final (top-2 most wins or ≥2 wins)
  conf_tourney_games     — total games played
  conf_tourney_winpct    — win% in conf tournament

Saves: data/raw/conf_tourney_features.parquet
"""
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "raw"
KAGGLE  = ROOT / "march-machine-learning-mania-2026"

df = pd.read_csv(KAGGLE / "MConferenceTourneyGames.csv")

records = []
for (season, conf), grp in df.groupby(["Season", "ConfAbbrev"]):
    # Count wins per team
    wins = grp["WTeamID"].value_counts().to_dict()
    # Every team that appears as a loser gets a loss
    losers = set(grp["LTeamID"].unique())
    winners_of_final = set()  # teams who won the most games in this conf
    all_teams = set(grp["WTeamID"]) | set(grp["LTeamID"])

    # Champion = team with most wins (and not in losers = won every game they played)
    # In a single-elimination bracket, champion lost 0 games
    max_wins = max(wins.get(t, 0) for t in all_teams)
    champions = {t for t in all_teams if wins.get(t, 0) == max_wins and t not in losers}

    # Finalist = played in final game (won or lost the last game of the tournament)
    # Last game = highest DayNum in this conf/season
    last_day = grp["DayNum"].max()
    final_game = grp[grp["DayNum"] == last_day]
    finalists = set(final_game["WTeamID"]) | set(final_game["LTeamID"])

    for team in all_teams:
        w = wins.get(team, 0)
        l = 1 if team in losers else 0
        records.append({
            "Season":                  season,
            "TeamID":                  team,
            "conf_tourney_wins":       w,
            "conf_tourney_losses":     l,
            "conf_tourney_games":      w + l,
            "conf_tourney_winpct":     w / (w + l) if (w + l) > 0 else 0.0,
            "conf_tourney_champion":   int(team in champions),
            "conf_tourney_finalist":   int(team in finalists),
        })

feat = pd.DataFrame(records)
out = DATA_DIR / "conf_tourney_features.parquet"
feat.to_parquet(out, index=False)

print(f"Conf tourney features: {len(feat):,} rows")
print(f"Seasons: {feat.Season.min()} – {feat.Season.max()}")
print(f"Saved → {out}")
print(feat.describe().round(3))

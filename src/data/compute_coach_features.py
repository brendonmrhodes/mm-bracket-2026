"""
Compute coaching features from Kaggle MTeamCoaches.csv.

Features computed (all leak-free — using only data BEFORE the current season):
  coach_tenure        : consecutive seasons this coach has been at this school
  coach_career_winpct : career regular-season win % (all schools, all prior seasons)
  coach_tourn_apps    : career tournament appearances (prior seasons)
  coach_tourn_wins    : career NCAA tournament wins (prior seasons)
  coach_f4s           : career Final Four appearances (prior seasons)
  coach_champs        : career national championships (prior seasons)
  coach_is_new        : 1 if first year at this school

Output: data/raw/coach_features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
KAGGLE  = ROOT / "march-machine-learning-mania-2026"
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    coaches  = pd.read_csv(KAGGLE / "MTeamCoaches.csv")
    tourn    = pd.read_csv(KAGGLE / "MNCAATourneyCompactResults.csv")
    regular  = pd.read_csv(KAGGLE / "MRegularSeasonCompactResults.csv")
    return coaches, tourn, regular


def get_season_coach(coaches: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Season, TeamID), identify the head coach who led the team
    through the end of the regular season / into the tournament.
    If multiple coaches exist (mid-season change), pick the one with the
    highest LastDayNum (coached latest into the year).
    """
    idx = coaches.groupby(["Season", "TeamID"])["LastDayNum"].idxmax()
    return coaches.loc[idx, ["Season", "TeamID", "CoachName"]].reset_index(drop=True)


def build_career_stats(season_coach: pd.DataFrame,
                       tourn: pd.DataFrame,
                       regular: pd.DataFrame) -> pd.DataFrame:
    """
    For each coach-school-season, compute career stats up to (not including)
    the current season to avoid data leakage.
    """
    # ── Regular season wins/losses per coach-season ────────────────────────────
    # Merge coach name onto regular season results
    reg_w = (regular.merge(season_coach.rename(columns={"TeamID":"WTeamID"}),
                           on=["Season","WTeamID"])
             .rename(columns={"CoachName":"coach", "WTeamID":"TeamID"}))
    reg_l = (regular.merge(season_coach.rename(columns={"TeamID":"LTeamID"}),
                           on=["Season","LTeamID"])
             .rename(columns={"CoachName":"coach", "LTeamID":"TeamID"}))

    reg_wins   = reg_w.groupby(["Season","coach"]).size().rename("W")
    reg_losses = reg_l.groupby(["Season","coach"]).size().rename("L")
    reg_record = pd.concat([reg_wins, reg_losses], axis=1).fillna(0).reset_index()

    # ── Tournament results per coach-season ───────────────────────────────────
    # Was the team in the tournament this season? (any appearance = 1 app)
    tourn_teams = pd.concat([
        tourn[["Season","WTeamID"]].rename(columns={"WTeamID":"TeamID"}),
        tourn[["Season","LTeamID"]].rename(columns={"LTeamID":"TeamID"}),
    ]).drop_duplicates()

    tourn_w = (tourn.merge(season_coach.rename(columns={"TeamID":"WTeamID"}),
                           on=["Season","WTeamID"])
               .rename(columns={"CoachName":"coach"}))
    tourn_l = (tourn.merge(season_coach.rename(columns={"TeamID":"LTeamID"}),
                           on=["Season","LTeamID"])
               .rename(columns={"CoachName":"coach"}))

    # Wins = wins in tournament
    tourn_wins = tourn_w.groupby(["Season","coach"]).size().rename("tourn_wins")

    # Appearances = was the team in the tournament
    coach_in_tourn = (tourn_teams
                      .merge(season_coach, on=["Season","TeamID"])
                      .groupby(["Season","CoachName"])
                      .size().rename("tourn_app")
                      .reset_index()
                      .rename(columns={"CoachName":"coach"}))

    # Final Four: DayNum >= 152 in tournament (R64=136, R32=138, S16=143/144,
    # E8=145/146, F4=152/154, NCG=155/156) — use DayNum >= 152
    f4_games = tourn[tourn["DayNum"] >= 152]
    f4_w = (f4_games.merge(season_coach.rename(columns={"TeamID":"WTeamID"}),
                           on=["Season","WTeamID"])
            .rename(columns={"CoachName":"coach"}))
    f4_l = (f4_games.merge(season_coach.rename(columns={"TeamID":"LTeamID"}),
                           on=["Season","LTeamID"])
            .rename(columns={"CoachName":"coach"}))
    f4_coach = pd.concat([
        f4_w[["Season","coach"]],
        f4_l[["Season","coach"]],
    ]).drop_duplicates()
    f4_count = f4_coach.groupby(["Season","coach"]).size().rename("f4").reset_index()

    # Championships: winner of DayNum >= 154 (NCG day)
    ncg = tourn[tourn["DayNum"] >= 154]
    champ_w = (ncg.merge(season_coach.rename(columns={"TeamID":"WTeamID"}),
                         on=["Season","WTeamID"])
               .rename(columns={"CoachName":"coach"}))
    champ_count = champ_w.groupby(["Season","coach"]).size().rename("champ").reset_index()

    # ── Combine into one per-coach-season frame ───────────────────────────────
    coach_seasons = season_coach.copy()
    coach_seasons = coach_seasons.rename(columns={"CoachName":"coach"})

    coach_seasons = (coach_seasons
                     .merge(reg_record, on=["Season","coach"], how="left")
                     .merge(coach_in_tourn, on=["Season","coach"], how="left")
                     .merge(tourn_wins.reset_index(), on=["Season","coach"], how="left")
                     .merge(f4_count, on=["Season","coach"], how="left")
                     .merge(champ_count, on=["Season","coach"], how="left"))

    for col in ["W","L","tourn_app","tourn_wins","f4","champ"]:
        coach_seasons[col] = coach_seasons[col].fillna(0).astype(int)

    return coach_seasons


def compute_features(season_coach: pd.DataFrame,
                     coach_seasons: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Season, TeamID), compute cumulative career stats up to
    but not including the current season (no leakage).
    """
    # Sort so we can cumsum in order
    cs = coach_seasons.sort_values(["coach","Season"]).copy()

    # Career cumulative stats per coach (shift forward one season for leak-free)
    cs["career_W"]           = cs.groupby("coach")["W"].cumsum().shift(1).fillna(0)
    cs["career_L"]           = cs.groupby("coach")["L"].cumsum().shift(1).fillna(0)
    cs["career_tourn_apps"]  = cs.groupby("coach")["tourn_app"].cumsum().shift(1).fillna(0)
    cs["career_tourn_wins"]  = cs.groupby("coach")["tourn_wins"].cumsum().shift(1).fillna(0)
    cs["career_f4s"]         = cs.groupby("coach")["f4"].cumsum().shift(1).fillna(0)
    cs["career_champs"]      = cs.groupby("coach")["champ"].cumsum().shift(1).fillna(0)
    cs["coach_career_winpct"] = (cs["career_W"] /
                                  (cs["career_W"] + cs["career_L"]).clip(lower=1))

    # Tenure at current school: how many consecutive seasons at this TeamID
    # (reset count when coach changes teams)
    def tenure(group):
        """Count consecutive seasons at this team up to and including current."""
        t = []
        count = 0
        prev_team = None
        for _, row in group.iterrows():
            if row["TeamID"] == prev_team:
                count += 1
            else:
                count = 1
            t.append(count)
            prev_team = row["TeamID"]
        return pd.Series(t, index=group.index)

    cs_by_coach = cs.sort_values(["coach","Season"])
    cs["coach_tenure"] = cs_by_coach.groupby("coach", group_keys=False).apply(tenure)
    cs["coach_is_new"] = (cs["coach_tenure"] == 1).astype(int)

    keep = [
        "Season", "TeamID",
        "coach_tenure", "coach_is_new",
        "coach_career_winpct",
        "career_tourn_apps", "career_tourn_wins",
        "career_f4s", "career_champs",
    ]
    return cs[keep].rename(columns={
        "career_tourn_apps": "coach_tourn_apps",
        "career_tourn_wins": "coach_tourn_wins",
        "career_f4s":        "coach_f4s",
        "career_champs":     "coach_champs",
    })


def main():
    print("Computing coach features from Kaggle data...")
    coaches, tourn, regular = load_data()

    print(f"  Coaches: {len(coaches):,} rows, {coaches['Season'].nunique()} seasons")

    season_coach = get_season_coach(coaches)
    print(f"  Season-coach mapping: {len(season_coach):,} team-seasons")

    coach_seasons = build_career_stats(season_coach, tourn, regular)
    features = compute_features(season_coach, coach_seasons)

    out = OUT_DIR / "coach_features.parquet"
    features.to_parquet(out, index=False)
    print(f"  Saved → {out}")
    print(f"  Shape: {features.shape}")

    # Sanity check on 2026 tourney teams
    seeds = pd.read_csv(KAGGLE / "MNCAATourneySeeds.csv")
    teams = pd.read_csv(KAGGLE / "MTeams.csv")
    seed26 = seeds[seeds["Season"] == 2026].merge(teams, on="TeamID")
    check = (features[features["Season"] == 2026]
             .merge(seed26[["TeamID","TeamName","Seed"]], on="TeamID")
             .sort_values("coach_tourn_apps", ascending=False))
    print("\nTop 10 most experienced coaches in 2026 tournament:")
    cols = ["Seed","TeamName","coach_tenure","coach_tourn_apps",
            "coach_tourn_wins","coach_f4s","coach_champs","coach_career_winpct"]
    print(check[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

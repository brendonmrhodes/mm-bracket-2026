"""
Compute Elo ratings + recency momentum features from Kaggle game history.

Outputs:
  data/raw/elo_game_by_game.parquet   — every game with pre-game Elo
  data/raw/elo_pre_tourney.parquet    — final Elo per team per season (pre-tournament)
  data/raw/elo_momentum.parquet       — recency features per team per season:
      elo_pre_tourney   : Elo entering the tournament
      elo_last10        : avg Elo over the last 10 regular-season games
      elo_momentum      : Elo change over last 10 games (positive = hot)
      elo_peak          : peak Elo reached during the season
      elo_consistency   : std of game-by-game Elo changes (low = steady team)
      elo_late_winpct   : win% in last 10 regular-season games
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

KAGGLE_DIR = Path(__file__).resolve().parent.parent.parent / "march-machine-learning-mania-2026"
RAW_DIR    = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_ELO    = 1500
K_BASE         = 20
HOME_ADVANTAGE = 100
REGRESS_FRAC   = 1 / 3


def expected_prob(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def mov_multiplier(margin: int, elo_diff: float) -> float:
    return np.log(abs(margin) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))


def update_elo(elo_a, elo_b, score_a, score_b, loc):
    adj_a = elo_a + (HOME_ADVANTAGE if loc == "H" else (-HOME_ADVANTAGE if loc == "A" else 0))
    exp_a = expected_prob(adj_a, elo_b)
    won   = 1 if score_a > score_b else 0
    k     = K_BASE * mov_multiplier(abs(score_a - score_b), adj_a - elo_b)
    delta = k * (won - exp_a)
    return elo_a + delta, elo_b - delta


def regress(elo, mean=INITIAL_ELO, frac=REGRESS_FRAC):
    return elo + frac * (mean - elo)


def main():
    print("Loading game data...")
    reg = pd.read_csv(KAGGLE_DIR / "MRegularSeasonCompactResults.csv")
    trn = pd.read_csv(KAGGLE_DIR / "MNCAATourneyCompactResults.csv")

    all_games = pd.concat([reg, trn], ignore_index=True)
    all_games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)
    seasons   = sorted(all_games["Season"].unique())
    all_teams = pd.unique(all_games[["WTeamID", "LTeamID"]].values.ravel())

    print(f"  {len(all_games):,} games | {len(seasons)} seasons | {len(all_teams)} teams")

    elo = {t: INITIAL_ELO for t in all_teams}

    game_records      = []
    momentum_records  = []
    current_season    = None

    # Per-season tracking for recency features
    season_game_log = defaultdict(list)  # team -> [(day, elo_after, won)]

    for _, g in all_games.iterrows():
        season = g.Season
        is_reg = g.DayNum <= 133

        if season != current_season:
            # ── End of previous season: compute momentum features ──────────
            if current_season is not None:
                for team_id in all_teams:
                    log = season_game_log[team_id]
                    # Only regular-season games
                    reg_log = [(d, e, w) for d, e, w in log if True]

                    elo_final = elo.get(team_id, INITIAL_ELO)

                    if len(reg_log) >= 2:
                        last10 = reg_log[-10:]
                        elo_vals   = [e for _, e, _ in last10]
                        elo_first  = reg_log[-min(10, len(reg_log))][1]
                        all_elos   = [e for _, e, _ in reg_log]

                        momentum_records.append({
                            "season":          current_season,
                            "TeamID":          team_id,
                            "elo_pre_tourney": elo_final,
                            "elo_last10":      np.mean(elo_vals),
                            "elo_momentum":    elo_vals[-1] - elo_vals[0] if len(elo_vals) > 1 else 0,
                            "elo_peak":        max(all_elos),
                            "elo_consistency": np.std([e2 - e1 for e1, e2 in zip(all_elos, all_elos[1:])]),
                            "elo_late_winpct": np.mean([w for _, _, w in last10]),
                        })
                    else:
                        momentum_records.append({
                            "season":          current_season,
                            "TeamID":          team_id,
                            "elo_pre_tourney": elo_final,
                            "elo_last10":      elo_final,
                            "elo_momentum":    0.0,
                            "elo_peak":        elo_final,
                            "elo_consistency": 0.0,
                            "elo_late_winpct": 0.5,
                        })

            # Regress toward mean for new season
            elo = {t: regress(r) for t, r in elo.items()}
            season_game_log = defaultdict(list)
            current_season = season

        w, l   = int(g.WTeamID), int(g.LTeamID)
        ws, ls = int(g.WScore),  int(g.LScore)
        loc    = g.WLoc

        elo_w_pre = elo.get(w, INITIAL_ELO)
        elo_l_pre = elo.get(l, INITIAL_ELO)

        # Record pre-game state
        if is_reg:
            game_records.append({
                "season":       season,
                "day":          g.DayNum,
                "WTeamID":      w,
                "LTeamID":      l,
                "elo_w_pre":    elo_w_pre,
                "elo_l_pre":    elo_l_pre,
                "w_win_prob":   expected_prob(
                    elo_w_pre + (HOME_ADVANTAGE if loc=="H" else -HOME_ADVANTAGE if loc=="A" else 0),
                    elo_l_pre
                ),
            })

        elo_w_new, elo_l_new = update_elo(elo_w_pre, elo_l_pre, ws, ls, loc)
        elo[w] = elo_w_new
        elo[l] = elo_l_new

        if is_reg:
            season_game_log[w].append((g.DayNum, elo_w_new, 1))
            season_game_log[l].append((g.DayNum, elo_l_new, 0))

    # Final season
    for team_id in all_teams:
        log = season_game_log[team_id]
        elo_final = elo.get(team_id, INITIAL_ELO)
        if len(log) >= 2:
            last10 = log[-10:]
            elo_vals = [e for _, e, _ in last10]
            all_elos = [e for _, e, _ in log]
            momentum_records.append({
                "season":          current_season,
                "TeamID":          team_id,
                "elo_pre_tourney": elo_final,
                "elo_last10":      np.mean(elo_vals),
                "elo_momentum":    elo_vals[-1] - elo_vals[0] if len(elo_vals) > 1 else 0,
                "elo_peak":        max(all_elos),
                "elo_consistency": np.std([e2 - e1 for e1, e2 in zip(all_elos, all_elos[1:])]),
                "elo_late_winpct": np.mean([w for _, _, w in last10]),
            })
        else:
            momentum_records.append({
                "season": current_season, "TeamID": team_id,
                "elo_pre_tourney": elo_final, "elo_last10": elo_final,
                "elo_momentum": 0.0, "elo_peak": elo_final,
                "elo_consistency": 0.0, "elo_late_winpct": 0.5,
            })

    games_df    = pd.DataFrame(game_records)
    momentum_df = pd.DataFrame(momentum_records).drop_duplicates(
        subset=["season", "TeamID"], keep="last"
    )

    # Legacy pre-tourney parquet (backward compat)
    pre_tourney = momentum_df[["season","TeamID","elo_pre_tourney"]].copy()

    games_df.to_parquet(RAW_DIR / "elo_game_by_game.parquet",  index=False)
    pre_tourney.to_parquet(RAW_DIR / "elo_pre_tourney.parquet", index=False)
    momentum_df.to_parquet(RAW_DIR / "elo_momentum.parquet",    index=False)

    print(f"\nElo outputs:")
    print(f"  Game-by-game : {len(games_df):,} rows")
    print(f"  Pre-tourney  : {len(pre_tourney):,} rows")
    print(f"  Momentum     : {len(momentum_df):,} rows  cols: {list(momentum_df.columns)}")

    # Validation: top teams by Elo momentum entering most recent tournament
    teams = pd.read_csv(KAGGLE_DIR / "MTeams.csv")
    latest_s = momentum_df["season"].max()
    top = (momentum_df[momentum_df["season"] == latest_s]
           .merge(teams[["TeamID","TeamName"]], on="TeamID")
           .sort_values("elo_pre_tourney", ascending=False)
           .head(15))
    print(f"\nTop 15 by Elo entering {latest_s} tournament:")
    print(top[["TeamName","elo_pre_tourney","elo_last10","elo_momentum","elo_late_winpct"]]
          .to_string(index=False))

    hot = (momentum_df[momentum_df["season"] == latest_s]
           .merge(teams[["TeamID","TeamName"]], on="TeamID")
           .sort_values("elo_momentum", ascending=False)
           .head(10))
    print(f"\nHottest teams (biggest Elo gain in last 10 games):")
    print(hot[["TeamName","elo_momentum","elo_late_winpct"]].to_string(index=False))


if __name__ == "__main__":
    main()

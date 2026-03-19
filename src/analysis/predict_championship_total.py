"""
Predict the combined total score for the 2026 NCAA championship game.

Method (principled KenPom-based):
  1. Use the KenPom efficiency formula to compute a theoretical total:
       pts_A = AdjO_A × (AdjD_B / 100) × (avg_tempo / 100)
       pts_B = AdjO_B × (AdjD_A / 100) × (avg_tempo / 100)
       theoretical_total = pts_A + pts_B

  2. Calibrate against all 22 historical championship games (2003–2025)
     with a simple linear regression: actual = a × theoretical + b
     (the raw formula systematically overestimates by ~4–15 pts because
     championship matchups pair elite defenses against each other)

  3. Report prediction ± 1 std dev of residuals (~16 pts) as the range.

Honest caveats:
  - Leave-one-out MAE ≈ 14 pts on 22 games
  - Historical range: 94 (2011 UConn-Butler) to 162 (2019 Virginia-Texas Tech)
  - Overtime games inflate the total (Virginia 2019 went OT)
  - Game pace is the biggest driver; defensive matchup is secondary

Outputs:
  outputs/championship_total_2026.csv  — predicted totals for top matchups
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
DATA    = ROOT / "data"
KAGGLE  = ROOT / "march-machine-learning-mania-2026"
OUT_DIR = ROOT / "outputs"

PREDICT_SEASON = 2026


def build_calibration_model():
    detail = pd.read_csv(KAGGLE / "MNCAATourneyDetailedResults.csv")
    feats  = pd.read_parquet(DATA / "team_season_features.parquet")

    ncg = detail[detail["DayNum"] >= 154].copy()
    cols = ["Season", "TeamID", "adjO", "adjD", "adjT"]
    ncg = (
        ncg.merge(
            feats[cols].rename(columns={"TeamID": "WTeamID",
                                        "adjO": "W_adjO", "adjD": "W_adjD", "adjT": "W_adjT"}),
            on=["Season", "WTeamID"], how="left",
        ).merge(
            feats[cols].rename(columns={"TeamID": "LTeamID",
                                        "adjO": "L_adjO", "adjD": "L_adjD", "adjT": "L_adjT"}),
            on=["Season", "LTeamID"], how="left",
        )
    )
    ncg = ncg.dropna(subset=["W_adjO", "L_adjO", "W_adjD", "L_adjD", "W_adjT", "L_adjT"])
    ncg["Total"] = ncg["WScore"] + ncg["LScore"]
    ncg["avg_T"] = (ncg["W_adjT"] + ncg["L_adjT"]) / 2
    ncg["theo"]  = (
        ncg["W_adjO"] * (ncg["L_adjD"] / 100) +
        ncg["L_adjO"] * (ncg["W_adjD"] / 100)
    ) * (ncg["avg_T"] / 100)

    model = LinearRegression().fit(ncg["theo"].values.reshape(-1, 1), ncg["Total"].values)
    resid = ncg["Total"].values - model.predict(ncg["theo"].values.reshape(-1, 1))
    resid_std = float(np.std(resid))

    return model, resid_std, ncg


def predict_total(t1_feats, t2_feats, model):
    avg_T = (t1_feats["adjT"] + t2_feats["adjT"]) / 2
    theo  = (
        t1_feats["adjO"] * (t2_feats["adjD"] / 100) +
        t2_feats["adjO"] * (t1_feats["adjD"] / 100)
    ) * (avg_T / 100)
    total = float(model.predict([[theo]])[0])
    return total, float(theo), float(avg_T)


def main():
    print("Predicting 2026 championship game total score...\n")

    cal_model, resid_std, hist = build_calibration_model()
    print(f"Calibration model fit on {len(hist)} championship games (2003–2025)")
    print(f"  Calibration: Total = {cal_model.coef_[0]:.3f} × theoretical + {cal_model.intercept_:.1f}")
    print(f"  Residual std: ±{resid_std:.1f} pts  (historical range: {hist['Total'].min()}–{hist['Total'].max()})")

    feats    = pd.read_parquet(DATA / "team_season_features.parquet")
    teams_df = pd.read_csv(KAGGLE / "MTeams.csv")
    seeds_df = pd.read_csv(KAGGLE / "MNCAATourneySeeds.csv")
    round_df = pd.read_csv(OUT_DIR / "round_probs_2026_calibrated.csv")

    feat26   = feats[feats["Season"] == PREDICT_SEASON].set_index("TeamID")
    t_map    = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    seeds26  = seeds_df[seeds_df["Season"] == PREDICT_SEASON].copy()

    # Top 6 teams by probability of reaching NCG
    top6 = round_df.nlargest(6, "prob_NCG")["TeamID"].tolist()
    top6 = [t for t in top6 if t in feat26.index]

    rows = []
    for i, t1 in enumerate(top6):
        for t2 in top6[i + 1:]:
            f1, f2   = feat26.loc[t1], feat26.loc[t2]
            total, theo, avg_T = predict_total(f1, f2, cal_model)
            p_ncg_t1 = float(round_df[round_df["TeamID"] == t1]["prob_NCG"].values[0])
            p_ncg_t2 = float(round_df[round_df["TeamID"] == t2]["prob_NCG"].values[0])
            # Probability this specific final happens (approx)
            p_final  = p_ncg_t1 * p_ncg_t2 * 4   # rough — 4 possible slot combos

            s1 = seeds26[seeds26["TeamID"] == t1]["Seed"].values
            s2 = seeds26[seeds26["TeamID"] == t2]["Seed"].values
            rows.append({
                "Team1":       t_map.get(t1, "?"),
                "Seed1":       s1[0] if len(s1) else "?",
                "Team2":       t_map.get(t2, "?"),
                "Seed2":       s2[0] if len(s2) else "?",
                "pred_total":  round(total),
                "low_total":   round(total - resid_std),
                "high_total":  round(total + resid_std),
                "resid_std":   round(resid_std, 1),
                "theo_total":  round(theo, 1),
                "avg_pace":    round(avg_T, 1),
                "T1_adjO":     round(float(f1["adjO"]), 1),
                "T1_adjD":     round(float(f1["adjD"]), 1),
                "T1_adjT":     round(float(f1["adjT"]), 1),
                "T2_adjO":     round(float(f2["adjO"]), 1),
                "T2_adjD":     round(float(f2["adjD"]), 1),
                "T2_adjT":     round(float(f2["adjT"]), 1),
            })

    df = pd.DataFrame(rows).sort_values("pred_total", ascending=False)
    out = OUT_DIR / "championship_total_2026.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")

    print(f"\n{'Matchup':<40} {'Pred':>5} {'Range':>12} {'Pace':>6}")
    print("─" * 68)
    for _, r in df.iterrows():
        matchup = f"{r['Team1']} ({r['Seed1']}) vs {r['Team2']} ({r['Seed2']})"
        rng     = f"{r['low_total']}–{r['high_total']}"
        print(f"{matchup:<40} {r['pred_total']:>5}  {rng:>12}  {r['avg_pace']:>5.1f}")

    print(f"\nNote: ±1 std dev = ±{resid_std:.0f} pts. Range shown is ±1σ.")
    print(f"Historically, championship totals range 94–162. Two defensive powerhouses")
    print(f"playing slow (Houston-style) skew toward 120s; two fast offensive teams toward 150+.")


if __name__ == "__main__":
    main()

"""
Round-Stratified Model Experiment (src/05_round_model.py)

Hypothesis: models trained specifically on early-round vs late-round tournament
games may be better calibrated for each phase, especially at detecting upsets
in R64 where seed-based priors are strongest.

Approach:
  - Add Round info to matchup_train by joining with tournament DayNum
  - Train three round-group models via walk-forward CV:
      Group R64    : Round of 64 (first-round games)
      Group R32S16 : Round of 32 + Sweet 16
      Group E8plus : Elite Eight + Final Four + Championship
  - Compare per-round accuracy & log-loss vs the unified model (models_tuned.pkl)
  - Produce 2026 R64 predictions using the R64-specific model

Outputs:
  outputs/round_model_cv_results.csv   — per-round CV comparison
  outputs/round_model_r64_preds.csv    — R64-specific predictions for 2026
  outputs/round_models.pkl             — trained round-specific models
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
KAGGLE_DIR = ROOT / "march-machine-learning-mania-2026"
OUT_DIR    = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ROUND_MAP = {
    (134, 135): "FF",
    (136, 137): "R64",
    (138, 140): "R32",
    (141, 145): "S16",
    (146, 148): "E8",
    (149, 153): "F4",
    (154, 156): "NCG",
}


def day_to_round(day: int) -> str:
    for (lo, hi), rnd in ROUND_MAP.items():
        if lo <= day <= hi:
            return rnd
    return "Unknown"


def load_data():
    matchup = pd.read_parquet(DATA_DIR / "matchup_train.parquet")
    # Add round info from compact tournament results
    compact = pd.read_csv(KAGGLE_DIR / "MNCAATourneyCompactResults.csv")
    compact["Round"] = compact["DayNum"].apply(day_to_round)

    # Merge round onto matchup_train
    # matchup_train rows have T1 < T2; compact has WTeamID/LTeamID
    compact["T1"] = compact[["WTeamID", "LTeamID"]].min(axis=1)
    compact["T2"] = compact[["WTeamID", "LTeamID"]].max(axis=1)
    round_info = compact[["Season", "T1", "T2", "Round", "DayNum"]].copy()

    matchup = matchup.merge(round_info, on=["Season", "T1", "T2"], how="left")
    matchup["Round"] = matchup["Round"].fillna("R64")  # default for pre-1985 edge cases

    print(f"Round distribution in training data:")
    print(matchup["Round"].value_counts().to_string())
    return matchup


def get_features(df):
    """Return feature matrix (drop meta columns)."""
    drop = {"Season", "T1", "T2", "Label", "Round", "DayNum"}
    feat_cols = [c for c in df.columns if c not in drop]
    # Drop columns with >50% missing
    miss = df[feat_cols].isnull().mean()
    feat_cols = list(miss[miss < 0.5].index)
    X = df[feat_cols].fillna(df[feat_cols].median())
    return X, feat_cols


def make_xgb(**kwargs):
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        **kwargs,
    )


def make_lgb(**kwargs):
    return LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=10,
        verbose=-1,
        **kwargs,
    )


def make_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=1000)),
    ])


def ensemble_predict(models, X):
    """Weighted average ensemble: XGB 45%, LGB 45%, LR 10%."""
    xgb_p = models["xgb"].predict_proba(X)[:, 1]
    lgb_p = models["lgb"].predict_proba(X)[:, 1]
    lr_p  = models["lr"].predict_proba(X)[:, 1]
    return 0.45 * xgb_p + 0.45 * lgb_p + 0.10 * lr_p


def train_models(X_train, y_train):
    """Train XGB + LGB + LR ensemble with isotonic calibration."""
    xgb_base = make_xgb()
    lgb_base = make_lgb()
    lr       = make_lr()

    xgb = CalibratedClassifierCV(xgb_base, cv=3, method="isotonic")
    lgb = CalibratedClassifierCV(lgb_base, cv=3, method="isotonic")

    xgb.fit(X_train, y_train)
    lgb.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    return {"xgb": xgb, "lgb": lgb, "lr": lr}


# ── Round group definitions ────────────────────────────────────────────────────
ROUND_GROUPS = {
    "R64":    ["R64"],
    "R32S16": ["R32", "S16"],
    "E8plus": ["E8", "F4", "NCG"],
}

# For a given prediction request, which model group to use:
ROUND_TO_GROUP = {
    "FF":  "R64",
    "R64": "R64",
    "R32": "R32S16",
    "S16": "R32S16",
    "E8":  "E8plus",
    "F4":  "E8plus",
    "NCG": "E8plus",
}


def run_cv(matchup: pd.DataFrame):
    """
    Walk-forward CV: for each test season, train on all earlier seasons.
    Compute per-round accuracy and log-loss for:
      (a) unified model (all rounds, one model)
      (b) round-group models (separate model per group)
    """
    seasons = sorted(matchup["Season"].unique())
    # Need at least 5 seasons of training data
    test_seasons = [s for s in seasons if s >= seasons[4]]

    cv_rows = []

    for test_season in test_seasons:
        train_df = matchup[matchup["Season"] < test_season].copy()
        test_df  = matchup[matchup["Season"] == test_season].copy()

        if len(train_df) < 100 or len(test_df) == 0:
            continue

        # --- Unified model ---
        X_train, feat_cols = get_features(train_df)
        y_train = train_df["Label"].values
        unified_models = train_models(X_train, y_train)

        X_test = test_df[feat_cols].fillna(train_df[feat_cols].median())
        y_test = test_df["Label"].values
        unified_preds = ensemble_predict(unified_models, X_test)
        unified_preds = np.clip(unified_preds, 0.01, 0.99)

        # --- Round-group models ---
        group_models = {}
        group_feat_cols = {}
        for group_name, rounds in ROUND_GROUPS.items():
            g_train = train_df[train_df["Round"].isin(rounds)]
            if len(g_train) < 50:
                continue
            Xg, gc = get_features(g_train)
            yg = g_train["Label"].values
            group_models[group_name] = train_models(Xg, yg)
            group_feat_cols[group_name] = gc

        # Round-group predictions for test (row-by-row to use correct model per round)
        test_reset = test_df.reset_index(drop=True)
        train_medians = {g: train_df[group_feat_cols[g]].median()
                         for g in group_feat_cols}
        round_preds = np.zeros(len(test_reset))
        for i in range(len(test_reset)):
            row = test_reset.iloc[i]
            g = ROUND_TO_GROUP.get(row["Round"], "R64")
            if g not in group_models:
                g = "R64"
            m  = group_models[g]
            fc = group_feat_cols[g]
            x_row = test_reset.loc[i, fc].copy()
            x_row = x_row.fillna(train_medians[g])
            round_preds[i] = ensemble_predict(m, x_row.values.reshape(1, -1))[0]
        round_preds = np.clip(round_preds, 0.01, 0.99)

        # Realign mask to test_reset (same order, just reset index)
        test_df = test_reset

        # Per-round metrics
        y_test_reset = test_df["Label"].values
        for rnd in test_df["Round"].unique():
            mask = (test_df["Round"] == rnd).values
            if mask.sum() < 2:
                continue
            yt = y_test_reset[mask]
            up = unified_preds[mask]
            rp = round_preds[mask]
            # Skip rounds where test set has only one class
            if len(np.unique(yt)) < 2:
                continue

            cv_rows.append({
                "Season":        test_season,
                "Round":         rnd,
                "N":             mask.sum(),
                "unified_ll":    log_loss(yt, up, labels=[0, 1]),
                "unified_acc":   accuracy_score(yt, (up >= 0.5).astype(int)),
                "round_ll":      log_loss(yt, rp, labels=[0, 1]),
                "round_acc":     accuracy_score(yt, (rp >= 0.5).astype(int)),
            })

        y_test_all = test_df["Label"].values
        print(f"  Season {test_season}: unified LL={log_loss(y_test_all, unified_preds, labels=[0,1]):.4f}  "
              f"round-stratified LL={log_loss(y_test_all, round_preds, labels=[0,1]):.4f}")

    return pd.DataFrame(cv_rows)


def train_final_models(matchup: pd.DataFrame):
    """Train final round-group models on all available data for 2026 predictions."""
    final_models = {}
    final_feat_cols = {}
    for group_name, rounds in ROUND_GROUPS.items():
        g_data = matchup[matchup["Round"].isin(rounds)]
        if len(g_data) < 50:
            print(f"  {group_name}: too few samples ({len(g_data)}), skipping")
            continue
        X, feat_cols = get_features(g_data)
        y = g_data["Label"].values
        final_models[group_name] = train_models(X, y)
        final_feat_cols[group_name] = feat_cols
        print(f"  {group_name}: trained on {len(g_data):,} games ({len(feat_cols)} features)")
    return final_models, final_feat_cols


def predict_2026_r64(final_models, final_feat_cols, matchup_train: pd.DataFrame):
    """
    Predict all possible 2026 matchups using the R64 model,
    then annotate with seed info for interpretation.
    """
    # Load 2026 team features
    tsf = pd.read_parquet(DATA_DIR / "team_season_features.parquet")
    seeds = tsf[tsf["Season"] == 2026][["TeamID", "SeedNum"]].copy()
    seeds_2026 = set(seeds["TeamID"].tolist())

    # Build all possible 2026 matchups (same as 03_predict.py)
    teams_2026 = sorted(seeds_2026)
    feat_matrix = tsf[tsf["Season"] == 2026].set_index("TeamID")

    EXCLUDE = {"Season", "TeamID", "Region"}
    feat_cols_r64 = final_feat_cols.get("R64", final_feat_cols.get("R64", []))
    if not feat_cols_r64:
        print("  No R64 model available — skipping 2026 predictions")
        return pd.DataFrame()

    # Derive feature columns from team-season features
    # (these map to d_, t1_, t2_ in matchup format)
    all_team_feats = [c for c in feat_matrix.columns if c not in EXCLUDE]
    miss = feat_matrix[all_team_feats].isnull().mean()
    valid_team_feats = list(miss[miss < 0.5].index)

    rows = []
    for i, t1 in enumerate(teams_2026):
        for t2 in teams_2026[i + 1:]:
            if t1 >= t2:
                continue
            if t1 not in feat_matrix.index or t2 not in feat_matrix.index:
                continue
            f1 = feat_matrix.loc[t1]
            f2 = feat_matrix.loc[t2]
            row = {"T1": t1, "T2": t2}
            for col in valid_team_feats:
                v1 = f1.get(col, np.nan) if hasattr(f1, "get") else f1[col] if col in f1.index else np.nan
                v2 = f2.get(col, np.nan) if hasattr(f2, "get") else f2[col] if col in f2.index else np.nan
                row[f"d_{col}"]  = v1 - v2
                row[f"t1_{col}"] = v1
                row[f"t2_{col}"] = v2
            rows.append(row)

    pred_df = pd.DataFrame(rows)

    # Get features available in R64 model
    available = [c for c in feat_cols_r64 if c in pred_df.columns]
    X_pred = pred_df[available].fillna(
        matchup_train[available].median() if all(c in matchup_train.columns for c in available)
        else pred_df[available].median()
    )

    probs = ensemble_predict(final_models["R64"], X_pred)
    probs = np.clip(probs, 0.01, 0.99)

    pred_df["r64_model_prob_t1"] = probs
    pred_df["r64_model_prob_t2"] = 1 - probs

    # Add seed info
    seed_map = dict(zip(seeds["TeamID"], seeds["SeedNum"]))
    pred_df["t1_seed"] = pred_df["T1"].map(seed_map)
    pred_df["t2_seed"] = pred_df["T2"].map(seed_map)

    # Load team names
    teams = pd.read_csv(KAGGLE_DIR / "MTeams.csv")[["TeamID", "TeamName"]]
    name_map = dict(zip(teams["TeamID"], teams["TeamName"]))
    pred_df["t1_name"] = pred_df["T1"].map(name_map)
    pred_df["t2_name"] = pred_df["T2"].map(name_map)

    return pred_df


def main():
    print("=" * 60)
    print("Round-Stratified Model Experiment")
    print("=" * 60)

    print("\nLoading data...")
    matchup = load_data()
    print(f"  Total matchups: {len(matchup):,}")

    # ── Walk-forward CV ──────────────────────────────────────────────────────
    print("\nRunning walk-forward CV (unified vs round-stratified)...")
    print("  (This may take 10–20 minutes)")
    cv_results = run_cv(matchup)

    if len(cv_results) > 0:
        print("\n" + "=" * 60)
        print("CV RESULTS: Unified vs Round-Stratified")
        print("=" * 60)
        summary = (cv_results.groupby("Round")
                   .agg(N=("N", "sum"),
                        unified_ll=("unified_ll", "mean"),
                        round_ll=("round_ll", "mean"),
                        unified_acc=("unified_acc", "mean"),
                        round_acc=("round_acc", "mean"))
                   .reset_index())
        summary["ll_improvement"] = summary["unified_ll"] - summary["round_ll"]
        summary["acc_improvement"] = summary["round_acc"] - summary["unified_acc"]

        round_order = ["R64", "R32", "S16", "E8", "F4", "NCG"]
        summary["_order"] = summary["Round"].apply(
            lambda r: round_order.index(r) if r in round_order else 99
        )
        summary = summary.sort_values("_order").drop(columns="_order")

        print(summary.to_string(index=False, float_format="{:.4f}".format))
        print("\n  ll_improvement > 0 means round model is better (lower log-loss)")

        cv_results.to_csv(OUT_DIR / "round_model_cv_results.csv", index=False)
        print(f"\n  Saved → outputs/round_model_cv_results.csv")

    # ── Train final models on all data ──────────────────────────────────────
    print("\nTraining final round-group models on all data...")
    final_models, final_feat_cols = train_final_models(matchup)

    # ── 2026 R64 predictions ─────────────────────────────────────────────────
    if "R64" in final_models:
        print("\nGenerating 2026 predictions using R64-specific model...")
        preds = predict_2026_r64(final_models, final_feat_cols, matchup)
        if len(preds) > 0:
            preds.to_csv(OUT_DIR / "round_model_r64_preds.csv", index=False)
            print(f"  Saved → outputs/round_model_r64_preds.csv ({len(preds):,} matchups)")

    # ── Save models ──────────────────────────────────────────────────────────
    with open(OUT_DIR / "round_models.pkl", "wb") as f:
        pickle.dump({"models": final_models, "feat_cols": final_feat_cols}, f)
    print(f"  Saved → outputs/round_models.pkl")


if __name__ == "__main__":
    main()

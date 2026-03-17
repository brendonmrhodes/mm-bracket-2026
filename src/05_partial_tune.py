"""
Build final tuned model using best params from best_params.json (all 3 phases).
Run after 04_tune.py has completed phases 1–3 (even if phase 4 crashed).

Saves to outputs/models_tuned.pkl so 03_predict.py picks it up immediately.
"""
import warnings; warnings.filterwarnings("ignore")

import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"
SEED     = 42

print("Loading training data...")
matchup = pd.read_parquet(DATA_DIR / "matchup_train.parquet")
EXCLUDE = ["Season", "T1", "T2", "Label"]
feature_cols = [c for c in matchup.columns if c not in EXCLUDE]
missing_rate = matchup[feature_cols].isnull().mean()
feature_cols = list(missing_rate[missing_rate < 0.4].index)
X = matchup[feature_cols].fillna(matchup[feature_cols].median())
y = matchup["Label"]
print(f"  {len(X):,} matchups, {len(feature_cols)} features")

# Load best params from Optuna
params_path = OUT_DIR / "best_params.json"
with open(params_path) as f:
    best = json.load(f)

best_xgb_params = best["xgb"]
best_lgb_params = best["lgb"]
w = best["ensemble_weights"]
ensemble_weights = (w["xgb"], w["lgb"], w["lr"])

print(f"\nLoaded from {params_path}")
print(f"  CV log-loss — XGB: {best['cv_log_loss']['xgb_only']:.5f}  "
      f"LGB: {best['cv_log_loss']['lgb_only']:.5f}  "
      f"Ensemble: {best['cv_log_loss']['ensemble']:.5f}")
print(f"  Ensemble weights — XGB: {ensemble_weights[0]:.3f}  "
      f"LGB: {ensemble_weights[1]:.3f}  LR: {ensemble_weights[2]:.3f}")

print("\nTraining XGBoost with tuned params (cv=5 isotonic calibration)...")
xgb_model = CalibratedClassifierCV(
    xgb.XGBClassifier(
        **best_xgb_params,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=SEED,
        n_jobs=-1,
    ),
    method="isotonic", cv=5,
)
xgb_model.fit(X, y)
print("  Done.")

print("Training LightGBM with tuned params (cv=5 isotonic calibration)...")
lgb_model = CalibratedClassifierCV(
    lgb.LGBMClassifier(
        **best_lgb_params,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    ),
    method="isotonic", cv=5,
)
lgb_model.fit(X, y)
print("  Done.")

print("Training Logistic Regression...")
lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
])
lr_model.fit(X, y)
print("  Done.")

bundle = {
    "xgb":              xgb_model,
    "lgb":              lgb_model,
    "lr":               lr_model,
    "feature_cols":     feature_cols,
    "ensemble_weights": ensemble_weights,
    "best_params":      best,
}
out = OUT_DIR / "models_tuned.pkl"
joblib.dump(bundle, out)
print(f"\nSaved → {out}")
print(f"Ensemble weights — XGB: {ensemble_weights[0]:.3f}  "
      f"LGB: {ensemble_weights[1]:.3f}  LR: {ensemble_weights[2]:.3f}")

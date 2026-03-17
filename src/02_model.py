"""
Model Training for March Madness Prediction
- Walk-forward cross-validation (train on years 1..N-1, predict year N)
- XGBoost + LightGBM ensemble with calibration
- Outputs per-matchup win probabilities for submission
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading features...")
matchup = pd.read_parquet(DATA_DIR / "matchup_train.parquet")
print(f"  {len(matchup):,} matchups | {matchup['Season'].nunique()} seasons")

# Feature columns: use diff features + seed features as primary signals
# Diff features capture relative strength; keep raw values too for tree models
EXCLUDE = ["Season", "T1", "T2", "Label"]
feature_cols = [c for c in matchup.columns if c not in EXCLUDE]

# Drop columns with >40% missing
missing_rate = matchup[feature_cols].isnull().mean()
feature_cols = list(missing_rate[missing_rate < 0.4].index)
print(f"  Feature columns after missingness filter: {len(feature_cols)}")

X = matchup[feature_cols].fillna(matchup[feature_cols].median())
y = matchup["Label"]
seasons = matchup["Season"]


# ── Walk-forward cross-validation ─────────────────────────────────────────────

def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

def make_lgb():
    return lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

def make_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ])


print("\nRunning walk-forward cross-validation...")
print(f"{'Season':>8}  {'N':>5}  {'Acc':>6}  {'LogLoss':>8}  {'Brier':>7}")
print("-" * 44)

all_preds = []
cv_years = sorted(matchup["Season"].unique())
# Start training after we have at least 5 seasons of data
start_year = cv_years[5]

for season in cv_years:
    if season < start_year:
        continue

    train_mask = seasons < season
    test_mask  = seasons == season

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        continue

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]

    # XGBoost with early stopping on a validation slice
    val_size = max(1, int(len(X_tr) * 0.15))
    X_val, y_val = X_tr.iloc[-val_size:], y_tr.iloc[-val_size:]
    X_tr2, y_tr2 = X_tr.iloc[:-val_size], y_tr.iloc[:-val_size]

    xgb_model = make_xgb()
    xgb_model.fit(
        X_tr2, y_tr2,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    lgb_model = make_lgb()
    lgb_model.fit(
        X_tr2, y_tr2,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )

    lr_model = make_lr()
    lr_model.fit(X_tr, y_tr)

    # Ensemble: weighted average of probabilities
    p_xgb = xgb_model.predict_proba(X_te)[:, 1]
    p_lgb = lgb_model.predict_proba(X_te)[:, 1]
    p_lr  = lr_model.predict_proba(X_te)[:, 1]
    p_ens = 0.45 * p_xgb + 0.45 * p_lgb + 0.10 * p_lr

    # Clip to avoid log(0)
    p_ens = np.clip(p_ens, 0.01, 0.99)

    acc  = accuracy_score(y_te, (p_ens >= 0.5).astype(int))
    ll   = log_loss(y_te, p_ens)
    brier = brier_score_loss(y_te, p_ens)

    print(f"{season:>8}  {len(y_te):>5}  {acc:>6.3f}  {ll:>8.4f}  {brier:>7.4f}")

    subset = matchup[test_mask][["Season", "T1", "T2", "Label"]].copy()
    subset["pred_prob"] = p_ens
    all_preds.append(subset)

cv_df = pd.concat(all_preds, ignore_index=True)
overall_acc  = accuracy_score(cv_df["Label"], (cv_df["pred_prob"] >= 0.5).astype(int))
overall_ll   = log_loss(cv_df["Label"], cv_df["pred_prob"])
overall_brier = brier_score_loss(cv_df["Label"], cv_df["pred_prob"])

print("-" * 44)
print(f"{'OVERALL':>8}  {len(cv_df):>5}  {overall_acc:>6.3f}  {overall_ll:>8.4f}  {overall_brier:>7.4f}")
cv_df.to_parquet(OUT_DIR / "cv_predictions.parquet", index=False)


# ── Train final model on ALL data ─────────────────────────────────────────────

print("\nTraining final model on all data...")
val_size = max(1, int(len(X) * 0.10))
X_final_val, y_final_val = X.iloc[-val_size:], y.iloc[-val_size:]
X_final_tr, y_final_tr   = X.iloc[:-val_size], y.iloc[:-val_size]

final_xgb = make_xgb()
final_xgb.fit(
    X_final_tr, y_final_tr,
    eval_set=[(X_final_val, y_final_val)],
    verbose=False,
)

final_lgb = make_lgb()
final_lgb.fit(
    X_final_tr, y_final_tr,
    eval_set=[(X_final_val, y_final_val)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
)

final_lr = make_lr()
final_lr.fit(X, y)

# Save feature importances
importance_xgb = pd.Series(
    final_xgb.feature_importances_, index=feature_cols
).sort_values(ascending=False)
importance_xgb.to_csv(OUT_DIR / "feature_importance_xgb.csv")

importance_lgb = pd.Series(
    final_lgb.feature_importances_, index=feature_cols
).sort_values(ascending=False)
importance_lgb.to_csv(OUT_DIR / "feature_importance_lgb.csv")

print("\nTop 20 features (XGBoost):")
print(importance_xgb.head(20).to_string())

# Persist models
import joblib
joblib.dump({"xgb": final_xgb, "lgb": final_lgb, "lr": final_lr,
             "feature_cols": feature_cols}, OUT_DIR / "models.pkl")
print(f"\nModels saved → outputs/models.pkl")
print(f"CV log-loss: {overall_ll:.4f} | Accuracy: {overall_acc:.3f}")

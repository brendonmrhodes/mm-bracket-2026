"""
Hyperparameter Tuning for March Madness Prediction Model
=========================================================
Uses Optuna to optimize XGBoost, LightGBM, and ensemble weights
via walk-forward cross-validation (train on years < N, test on year N).

Phases:
  1. 200 Optuna trials for XGBoost hyperparameters
  2. 200 Optuna trials for LightGBM hyperparameters
  3.  50 Optuna trials for ensemble blend weights
  4. Retrain final calibrated models on all data
  5. Print comparison table: default vs tuned

Output:
  outputs/best_params.json   — best hyperparameters found
  outputs/models_tuned.pkl   — final calibrated models + metadata
"""

import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EARLY_STOPPING_ROUNDS = 30   # early-stopping patience for tree models
# Walk-forward CV starts at this season (need enough historical training data)
CV_START_SEASON = 2010
# Random seed for reproducibility
SEED = 42

# Default ensemble weights from 02_model.py (used for comparison baseline)
DEFAULT_WEIGHTS = (0.45, 0.45, 0.10)  # xgb, lgb, lr


# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading matchup training data...")
matchup = pd.read_parquet(DATA_DIR / "matchup_train.parquet")
print(f"  {len(matchup):,} matchups across {matchup['Season'].nunique()} seasons")

EXCLUDE      = ["Season", "T1", "T2", "Label"]
feature_cols = [c for c in matchup.columns if c not in EXCLUDE]

# Drop columns with >40% missing values (mirrors 02_model.py)
missing_rate = matchup[feature_cols].isnull().mean()
feature_cols = list(missing_rate[missing_rate < 0.4].index)
print(f"  Feature columns after missingness filter: {len(feature_cols)}")

X       = matchup[feature_cols].fillna(matchup[feature_cols].median())
y       = matchup["Label"]
seasons = matchup["Season"]

cv_years     = sorted(matchup["Season"].unique())
eval_seasons = [s for s in cv_years if s >= CV_START_SEASON]
print(f"  Walk-forward CV seasons: {eval_seasons[0]} – {eval_seasons[-1]} "
      f"({len(eval_seasons)} folds)")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_val(X_tr, y_tr, val_frac=0.15):
    """Reserve the last val_frac rows as an early-stopping validation set."""
    val_size = max(1, int(len(X_tr) * val_frac))
    X_v, y_v = X_tr.iloc[-val_size:], y_tr.iloc[-val_size:]
    X_t, y_t = X_tr.iloc[:-val_size], y_tr.iloc[:-val_size]
    return X_t, y_t, X_v, y_v


def walk_forward_logloss_xgb(params: dict) -> float:
    """
    Evaluate an XGBoost parameter set using walk-forward CV.
    Returns mean log-loss across all evaluation folds.
    """
    losses = []
    for season in eval_seasons:
        train_mask = seasons < season
        test_mask  = seasons == season
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]
        X_t, y_t, X_v, y_v = _split_val(X_tr, y_tr)

        model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=SEED,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

        proba = np.clip(model.predict_proba(X_te)[:, 1], 1e-7, 1 - 1e-7)
        losses.append(log_loss(y_te, proba))

    return float(np.mean(losses))


def walk_forward_logloss_lgb(params: dict) -> float:
    """
    Evaluate a LightGBM parameter set using walk-forward CV.
    Returns mean log-loss across all evaluation folds.
    """
    losses = []
    for season in eval_seasons:
        train_mask = seasons < season
        test_mask  = seasons == season
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]
        X_t, y_t, X_v, y_v = _split_val(X_tr, y_tr)

        model = lgb.LGBMClassifier(
            **params,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        proba = np.clip(model.predict_proba(X_te)[:, 1], 1e-7, 1 - 1e-7)
        losses.append(log_loss(y_te, proba))

    return float(np.mean(losses))


def walk_forward_ensemble(w_xgb: float, w_lgb: float, w_lr: float,
                          xgb_params: dict, lgb_params: dict) -> float:
    """
    Evaluate an ensemble blend (w_xgb, w_lgb, w_lr) using walk-forward CV.
    Weights are assumed to already sum to 1.
    Returns mean log-loss across all evaluation folds.
    """
    losses = []
    for season in eval_seasons:
        train_mask = seasons < season
        test_mask  = seasons == season
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]
        X_t, y_t, X_v, y_v = _split_val(X_tr, y_tr)

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            **xgb_params,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=SEED,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        xgb_model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            **lgb_params,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        # Logistic Regression (trained on full training set, no early stopping)
        lr_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
        ])
        lr_model.fit(X_tr, y_tr)

        p_xgb = xgb_model.predict_proba(X_te)[:, 1]
        p_lgb = lgb_model.predict_proba(X_te)[:, 1]
        p_lr  = lr_model.predict_proba(X_te)[:, 1]

        p_ens = w_xgb * p_xgb + w_lgb * p_lgb + w_lr * p_lr
        p_ens = np.clip(p_ens, 1e-7, 1 - 1e-7)
        losses.append(log_loss(y_te, p_ens))

    return float(np.mean(losses))


# ── Phase 1: Tune XGBoost ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 1: Tuning XGBoost (200 trials)")
print("=" * 60)

def xgb_objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 2000),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 5.0),
    }
    return walk_forward_logloss_xgb(params)


xgb_study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="xgb_tuning",
)
xgb_study.optimize(xgb_objective, n_trials=200, show_progress_bar=True)

best_xgb_params = xgb_study.best_params
best_xgb_loss   = xgb_study.best_value
print(f"\nBest XGBoost log-loss: {best_xgb_loss:.5f}")
print(f"Best XGBoost params:  {best_xgb_params}")


# ── Phase 2: Tune LightGBM ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2: Tuning LightGBM (200 trials)")
print("=" * 60)

def lgb_objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 2000),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 5.0),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
    }
    return walk_forward_logloss_lgb(params)


lgb_study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="lgb_tuning",
)
lgb_study.optimize(lgb_objective, n_trials=200, show_progress_bar=True)

best_lgb_params = lgb_study.best_params
best_lgb_loss   = lgb_study.best_value
print(f"\nBest LightGBM log-loss: {best_lgb_loss:.5f}")
print(f"Best LightGBM params:   {best_lgb_params}")


# ── Phase 3: Tune ensemble weights ────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: Tuning ensemble weights (50 trials)")
print("=" * 60)

def ensemble_objective(trial: optuna.Trial) -> float:
    # Sample two weights; derive the third so they sum to 1.
    # w_xgb and w_lgb each in [0.1, 0.8]; remainder goes to LR (clamped >= 0.05).
    w_xgb_raw = trial.suggest_float("w_xgb", 0.1, 0.8)
    w_lgb_raw = trial.suggest_float("w_lgb", 0.1, 0.8)

    total = w_xgb_raw + w_lgb_raw
    if total > 0.95:
        # Rescale so LR gets at least 0.05
        scale  = 0.95 / total
        w_xgb  = w_xgb_raw * scale
        w_lgb  = w_lgb_raw * scale
    else:
        w_xgb = w_xgb_raw
        w_lgb = w_lgb_raw

    w_lr = 1.0 - w_xgb - w_lgb

    return walk_forward_ensemble(
        w_xgb, w_lgb, w_lr,
        best_xgb_params, best_lgb_params,
    )


ens_study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="ensemble_tuning",
)
ens_study.optimize(ensemble_objective, n_trials=50, show_progress_bar=True)

# Reconstruct the final normalised weights from the best trial
_w_xgb_raw = ens_study.best_params["w_xgb"]
_w_lgb_raw = ens_study.best_params["w_lgb"]
_total     = _w_xgb_raw + _w_lgb_raw
if _total > 0.95:
    _scale     = 0.95 / _total
    best_w_xgb = _w_xgb_raw * _scale
    best_w_lgb = _w_lgb_raw * _scale
else:
    best_w_xgb = _w_xgb_raw
    best_w_lgb = _w_lgb_raw
best_w_lr = 1.0 - best_w_xgb - best_w_lgb

best_ens_loss = ens_study.best_value
print(f"\nBest ensemble log-loss: {best_ens_loss:.5f}")
print(f"Best weights -> XGB: {best_w_xgb:.3f}  LGB: {best_w_lgb:.3f}  LR: {best_w_lr:.3f}")


# ── Save best params ──────────────────────────────────────────────────────────
best_params_all = {
    "xgb":             best_xgb_params,
    "lgb":             best_lgb_params,
    "ensemble_weights": {
        "xgb": best_w_xgb,
        "lgb": best_w_lgb,
        "lr":  best_w_lr,
    },
    "cv_log_loss": {
        "xgb_only":  best_xgb_loss,
        "lgb_only":  best_lgb_loss,
        "ensemble":  best_ens_loss,
    },
}
params_path = OUT_DIR / "best_params.json"
with open(params_path, "w") as fh:
    json.dump(best_params_all, fh, indent=2)
print(f"\nBest params saved -> {params_path}")


# ── Phase 4: Retrain final calibrated models on all data ──────────────────────
print("\n" + "=" * 60)
print("PHASE 4: Retraining final models on all data (with calibration)")
print("=" * 60)

# Reserve a small hold-out for early stopping (tree models) and isotonic calibration.
# We use the last 10% of rows for early stopping and a separate 15% for calibration.
cal_size  = max(1, int(len(X) * 0.15))
val_size  = max(1, int(len(X) * 0.10))

# Layout (chronological order):  [train | val (ES) | cal | ...]
# We keep calibration data as the last cal_size rows; early-stopping val
# immediately before that.
X_cal,   y_cal   = X.iloc[-cal_size:],              y.iloc[-cal_size:]
X_es_val, y_es_val = X.iloc[-(cal_size + val_size):-cal_size], \
                     y.iloc[-(cal_size + val_size):-cal_size]
X_tr_final, y_tr_final = X.iloc[:-(cal_size + val_size)], \
                          y.iloc[:-(cal_size + val_size)]

print(f"  Train: {len(X_tr_final):,}  |  ES-val: {len(X_es_val):,}  "
      f"|  Calibration: {len(X_cal):,}")

# --- XGBoost (tuned) ---
print("  Fitting tuned XGBoost...")
raw_xgb = xgb.XGBClassifier(
    **best_xgb_params,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=SEED,
    n_jobs=-1,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
)
raw_xgb.fit(
    X_tr_final, y_tr_final,
    eval_set=[(X_es_val, y_es_val)],
    verbose=False,
)
# Isotonic calibration on held-out set — sklearn 1.4+ removed cv='prefit',
# so we re-wrap with cv=5 on the calibration split.
final_xgb = CalibratedClassifierCV(
    xgb.XGBClassifier(**best_xgb_params, eval_metric="logloss",
                      use_label_encoder=False, random_state=SEED, n_jobs=-1),
    method="isotonic", cv=5,
)
final_xgb.fit(X_cal, y_cal)

# --- LightGBM (tuned) ---
print("  Fitting tuned LightGBM...")
raw_lgb = lgb.LGBMClassifier(
    **best_lgb_params,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
raw_lgb.fit(
    X_tr_final, y_tr_final,
    eval_set=[(X_es_val, y_es_val)],
    callbacks=[
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(-1),
    ],
)
final_lgb = CalibratedClassifierCV(
    lgb.LGBMClassifier(**best_lgb_params, random_state=SEED, n_jobs=-1, verbose=-1),
    method="isotonic", cv=5,
)
final_lgb.fit(X_cal, y_cal)

# --- Logistic Regression ---
print("  Fitting Logistic Regression (full data)...")
final_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
])
final_lr.fit(X, y)

# Save
model_bundle = {
    "xgb":             final_xgb,
    "lgb":             final_lgb,
    "lr":              final_lr,
    "feature_cols":    feature_cols,
    "ensemble_weights": {
        "xgb": best_w_xgb,
        "lgb": best_w_lgb,
        "lr":  best_w_lr,
    },
    "best_params":     best_params_all,
}
model_path = OUT_DIR / "models_tuned.pkl"
joblib.dump(model_bundle, model_path)
print(f"  Tuned models saved -> {model_path}")


# ── Phase 5: Comparison table — default vs tuned ──────────────────────────────
print("\n" + "=" * 60)
print("PHASE 5: Default params vs Tuned params (walk-forward CV)")
print("=" * 60)


def _default_xgb_params():
    return dict(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    )


def _default_lgb_params():
    return dict(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=1.0, num_leaves=31,
    )


def _collect_cv_preds(xgb_params, lgb_params, w_xgb, w_lgb, w_lr):
    """
    Run the full walk-forward CV with given params and blend weights.
    Returns a DataFrame with columns [Label, pred_prob].
    """
    records = []
    for season in eval_seasons:
        train_mask = seasons < season
        test_mask  = seasons == season
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]
        X_t, y_t, X_v, y_v = _split_val(X_tr, y_tr)

        xgb_m = xgb.XGBClassifier(
            **xgb_params,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=SEED,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        xgb_m.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

        lgb_m = lgb.LGBMClassifier(
            **lgb_params,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_m.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        lr_m = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
        ])
        lr_m.fit(X_tr, y_tr)

        p_xgb = xgb_m.predict_proba(X_te)[:, 1]
        p_lgb = lgb_m.predict_proba(X_te)[:, 1]
        p_lr  = lr_m.predict_proba(X_te)[:, 1]
        p_ens = np.clip(w_xgb * p_xgb + w_lgb * p_lgb + w_lr * p_lr, 1e-7, 1 - 1e-7)

        for label, prob in zip(y_te, p_ens):
            records.append({"Label": int(label), "pred_prob": float(prob)})

    return pd.DataFrame(records)


print("\nEvaluating DEFAULT params on walk-forward CV...")
df_default = _collect_cv_preds(
    _default_xgb_params(), _default_lgb_params(),
    DEFAULT_WEIGHTS[0], DEFAULT_WEIGHTS[1], DEFAULT_WEIGHTS[2],
)

print("Evaluating TUNED params on walk-forward CV...")
df_tuned = _collect_cv_preds(
    best_xgb_params, best_lgb_params,
    best_w_xgb, best_w_lgb, best_w_lr,
)


def _metrics(df: pd.DataFrame) -> dict:
    labels = df["Label"].values
    probs  = df["pred_prob"].values
    return {
        "log_loss": log_loss(labels, probs),
        "accuracy": accuracy_score(labels, (probs >= 0.5).astype(int)),
        "brier":    brier_score_loss(labels, probs),
        "n":        len(df),
    }


m_def   = _metrics(df_default)
m_tuned = _metrics(df_tuned)

# Improvement
delta_ll  = m_def["log_loss"] - m_tuned["log_loss"]
delta_acc = m_tuned["accuracy"] - m_def["accuracy"]
delta_b   = m_def["brier"] - m_tuned["brier"]

print("\n" + "=" * 60)
print(f"{'Metric':<18} {'Default':>10} {'Tuned':>10} {'Delta':>10}")
print("-" * 52)
print(f"{'Log-Loss':<18} {m_def['log_loss']:>10.5f} {m_tuned['log_loss']:>10.5f} "
      f"{delta_ll:>+10.5f}")
print(f"{'Accuracy':<18} {m_def['accuracy']:>10.4f} {m_tuned['accuracy']:>10.4f} "
      f"{delta_acc:>+10.4f}")
print(f"{'Brier Score':<18} {m_def['brier']:>10.5f} {m_tuned['brier']:>10.5f} "
      f"{delta_b:>+10.5f}")
print(f"{'N Matchups':<18} {m_def['n']:>10,} {m_tuned['n']:>10,}")
print("=" * 60)
print("\nBest ensemble weights (tuned):")
print(f"  XGB: {best_w_xgb:.4f}  |  LGB: {best_w_lgb:.4f}  |  LR: {best_w_lr:.4f}")
print(f"\nOutputs written:")
print(f"  {params_path}")
print(f"  {model_path}")

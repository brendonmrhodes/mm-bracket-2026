"""
Generate 2026 tournament predictions.

Automatically uses tuned model (outputs/models_tuned.pkl) if available,
otherwise falls back to default model (outputs/models.pkl).

Outputs:
  outputs/submission_2026.csv          — Kaggle submission format
  outputs/championship_probs_2026.csv  — championship probability per team
  outputs/round_probs_2026.csv         — probability of reaching each round
  outputs/pool_strategy_2026.csv       — EV analysis for bracket pool picks
"""

import numpy as np
import pandas as pd
import itertools
import joblib
from pathlib import Path

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
OUT_DIR    = Path(__file__).resolve().parent.parent / "outputs"
KAGGLE_DIR = Path(__file__).resolve().parent.parent / "march-machine-learning-mania-2026"
OUT_DIR.mkdir(exist_ok=True)

PREDICT_SEASON = 2026
N_SIM          = 100_000

# ── Load model ─────────────────────────────────────────────────────────────────
tuned_path   = OUT_DIR / "models_tuned.pkl"
default_path = OUT_DIR / "models.pkl"

if tuned_path.exists():
    bundle = joblib.load(tuned_path)
    print("Using TUNED model (outputs/models_tuned.pkl)")
else:
    bundle = joblib.load(default_path)
    print("Using default model (outputs/models.pkl)")

xgb_model    = bundle["xgb"]
lgb_model    = bundle["lgb"]
lr_model     = bundle["lr"]
feature_cols = bundle["feature_cols"]
weights = bundle.get("ensemble_weights", (0.45, 0.45, 0.10))
if isinstance(weights, dict):
    w_xgb, w_lgb, w_lr = weights["xgb"], weights["lgb"], weights["lr"]
else:
    w_xgb, w_lgb, w_lr = weights
print(f"Ensemble weights — XGB: {w_xgb:.2f}  LGB: {w_lgb:.2f}  LR: {w_lr:.2f}\n")

# ── Load features & seeds ──────────────────────────────────────────────────────
features = pd.read_parquet(DATA_DIR / "team_season_features.parquet")
seeds_df  = pd.read_csv(KAGGLE_DIR / "MNCAATourneySeeds.csv")
teams_df  = pd.read_csv(KAGGLE_DIR / "MTeams.csv")

seeds_df["SeedNum"] = seeds_df["Seed"].apply(
    lambda s: int("".join(filter(str.isdigit, s)))
)

seed_2026 = seeds_df[seeds_df["Season"] == PREDICT_SEASON].copy()
feat_2026  = features[features["Season"] == PREDICT_SEASON].set_index("TeamID")
team_list  = sorted(seed_2026["TeamID"].unique())

print(f"Tournament teams: {len(team_list)}")


# ── Win probability function ───────────────────────────────────────────────────
def build_prob_lookup(team_list, feat_map, feature_cols, xgb_m, lgb_m, lr_m, weights):
    """Pre-compute win probabilities for all possible matchups."""
    w_xgb, w_lgb, w_lr = weights
    prob = {}
    pairs = list(itertools.combinations(sorted(team_list), 2))

    rows = []
    meta = []
    for t1, t2 in pairs:
        if t1 not in feat_map.index or t2 not in feat_map.index:
            continue
        f1, f2 = feat_map.loc[t1], feat_map.loc[t2]
        row = {}
        for col in feature_cols:
            if col.startswith("d_"):
                base = col[2:]
                row[col] = f1.get(base, np.nan) - f2.get(base, np.nan)
            elif col.startswith("t1_"):
                row[col] = f1.get(col[3:], np.nan)
            elif col.startswith("t2_"):
                row[col] = f2.get(col[3:], np.nan)
            else:
                row[col] = np.nan
        rows.append(row)
        meta.append((t1, t2))

    if not rows:
        return prob

    X = pd.DataFrame(rows, columns=feature_cols).fillna(0)
    p_xgb = xgb_m.predict_proba(X)[:, 1]
    p_lgb = lgb_m.predict_proba(X)[:, 1]
    p_lr  = lr_m.predict_proba(X)[:, 1]
    p_ens = np.clip(w_xgb * p_xgb + w_lgb * p_lgb + w_lr * p_lr, 0.01, 0.99)

    for (t1, t2), p in zip(meta, p_ens):
        prob[(t1, t2)] = p        # P(t1 wins) where t1 < t2
        prob[(t2, t1)] = 1 - p

    return prob


print("Computing matchup probabilities...")
prob_lookup = build_prob_lookup(
    team_list, feat_2026, feature_cols,
    xgb_model, lgb_model, lr_model, (w_xgb, w_lgb, w_lr)
)

def win_prob(t1, t2):
    key = (min(t1, t2), max(t1, t2))
    p = prob_lookup.get(key, 0.5)
    return p if t1 < t2 else 1 - p


# ── Kaggle submission ──────────────────────────────────────────────────────────
print("Building Kaggle submission...")
rows = []
for t1, t2 in itertools.combinations(sorted(team_list), 2):
    rows.append({
        "ID":   f"{PREDICT_SEASON}_{t1}_{t2}",
        "Pred": prob_lookup.get((t1, t2), 0.5),
    })
submission = pd.DataFrame(rows)
submission.to_csv(OUT_DIR / "submission_2026.csv", index=False)
print(f"  Saved → outputs/submission_2026.csv ({len(submission):,} matchups)")


# ── Monte Carlo bracket simulation ────────────────────────────────────────────
seeded = dict(zip(seed_2026["Seed"], seed_2026["TeamID"]))


def sim_game(t1, t2):
    p = win_prob(t1, t2)
    return t1 if np.random.random() < p else t2


def simulate_region(region):
    """Simulate one region. Returns (winner, {team_id: deepest_round_reached}).
    Rounds within a region: R64 (all teams), R32, S16, E8; winner goes to F4.
    """
    resolved = {}
    for seed_str, tid in seeded.items():
        if not seed_str.startswith(region):
            continue
        base = seed_str  # e.g. "W01", "Y11a"
        if base.endswith("a") or base.endswith("b"):
            key = base[:-1]
            resolved.setdefault(key, []).append(tid)
        else:
            resolved[seed_str] = tid

    # Play First Four games
    for key, pair in list(resolved.items()):
        if isinstance(pair, list) and len(pair) == 2:
            resolved[key] = sim_game(pair[0], pair[1])

    # Standard bracket order: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
    bracket = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    teams = []
    for s in bracket:
        key = f"{region}{str(s).zfill(2)}"
        t = resolved.get(key)
        if t is not None and not isinstance(t, list):
            teams.append(t)

    # Track deepest round each team reaches within this region.
    # All 16 teams reach R64. Three rounds of play get from 16 → 2 (E8).
    # Then one final game (regional final) determines the F4 representative.
    round_reached = {t: "R64" for t in teams}

    for round_name in ["R32", "S16", "E8"]:
        if len(teams) <= 1:
            break
        next_round = []
        for i in range(0, len(teams) - 1, 2):
            w = sim_game(teams[i], teams[i + 1])
            next_round.append(w)
            round_reached[w] = round_name   # reached this round
        teams = next_round

    # Regional final: both remaining teams have reached E8;
    # play the game to find the actual F4-bound champion.
    if len(teams) == 2:
        winner = sim_game(teams[0], teams[1])
    elif len(teams) == 1:
        winner = teams[0]
    else:
        winner = None

    return winner, round_reached


REGIONS = ["W", "X", "Y", "Z"]
# Standard Final Four pairings: W vs X, Y vs Z
FF_PAIRS = [(0, 1), (2, 3)]

ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "NCG", "Champion"]

print(f"Running {N_SIM:,} bracket simulations...")
reach = {t: {r: 0 for r in ROUND_NAMES} for t in team_list}

# Maps round name -> index for cumulative crediting
REGION_ROUND_ORDER = ["R64", "R32", "S16", "E8"]

for _ in range(N_SIM):
    region_results = [simulate_region(r) for r in REGIONS]
    ff  = [res[0] for res in region_results if res[0] is not None]
    rds = [res[1] for res in region_results]

    # Credit R64/R32/S16/E8 from within-region simulation
    for round_dict in rds:
        for team_id, deepest in round_dict.items():
            if team_id not in reach:
                continue
            # Credit all rounds up to and including deepest reached
            for rname in REGION_ROUND_ORDER:
                reach[team_id][rname] += 1
                if rname == deepest:
                    break

    if len(ff) != 4:
        continue

    # Credit F4
    for t in ff:
        reach[t]["F4"] += 1

    # Semifinal games
    finalists = []
    for i, j in FF_PAIRS:
        if i < len(ff) and j < len(ff):
            w = sim_game(ff[i], ff[j])
            finalists.append(w)
            reach[w]["NCG"] += 1

    if len(finalists) == 2:
        champ = sim_game(finalists[0], finalists[1])
        reach[champ]["Champion"] += 1


# Convert to probabilities
round_probs = []
for tid in team_list:
    row = {"TeamID": tid}
    for r in ROUND_NAMES:
        row[f"prob_{r}"] = reach[tid][r] / N_SIM
    round_probs.append(row)

round_df = (
    pd.DataFrame(round_probs)
    .merge(seed_2026[["TeamID","Seed","SeedNum"]], on="TeamID")
    .merge(teams_df[["TeamID","TeamName"]], on="TeamID")
    .sort_values("prob_Champion", ascending=False)
)

round_df.to_csv(OUT_DIR / "round_probs_2026.csv", index=False)

champ_df = round_df[["Seed","TeamName","prob_Champion","prob_F4","prob_NCG"]].copy()
champ_df.columns = ["Seed","Team","Champ%","F4%","Final%"]
for col in ["Champ%","F4%","Final%"]:
    champ_df[col] = (champ_df[col] * 100).round(1)


# ── Pool strategy analysis ─────────────────────────────────────────────────────
# "Field ownership" — roughly approximate public pick % using seed-based priors
# In large pools, the value of a pick = your probability / field's probability
# We approximate field using 538-style seed-based weights
def seed_prior(seed_num):
    """Approximate public championship pick% by seed (based on historical pool data)."""
    priors = {1: 0.24, 2: 0.12, 3: 0.07, 4: 0.05, 5: 0.03, 6: 0.02,
              7: 0.015, 8: 0.01, 9: 0.008, 10: 0.007, 11: 0.006,
              12: 0.005, 13: 0.003, 14: 0.002, 15: 0.001, 16: 0.0005}
    return priors.get(seed_num, 0.001)

pool_rows = []
for _, row in round_df.iterrows():
    seed_num  = int(row["SeedNum"])
    model_p   = row["prob_Champion"]
    field_p   = seed_prior(seed_num)
    ev_ratio  = model_p / field_p if field_p > 0 else 0
    pool_rows.append({
        "Seed":      row["Seed"],
        "Team":      row["TeamName"],
        "Model%":    round(model_p * 100, 1),
        "Field%":    round(field_p * 100, 1),
        "EV_ratio":  round(ev_ratio, 2),
        "F4%":       round(row["prob_F4"] * 100, 1),
    })

pool_df = pd.DataFrame(pool_rows).sort_values("EV_ratio", ascending=False)
pool_df.to_csv(OUT_DIR / "pool_strategy_2026.csv", index=False)


# ── Print results ──────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print(f"{'2026 CHAMPIONSHIP PROBABILITIES':^58}")
print(f"{'='*58}")
print(f"{'Seed':<6} {'Team':<22} {'Champ%':>7} {'Final%':>7} {'F4%':>6}")
print("-" * 58)
for _, r in champ_df.head(20).iterrows():
    print(f"{r['Seed']:<6} {r['Team']:<22} {r['Champ%']:>6.1f}% {r['Final%']:>6.1f}% {r['F4%']:>5.1f}%")

print(f"\n{'='*58}")
print(f"{'POOL STRATEGY — BEST EV PICKS (large pool)':^58}")
print(f"{'='*58}")
print(f"{'Seed':<6} {'Team':<22} {'Model%':>7} {'Field%':>7} {'EV':>6}")
print("-" * 58)
for _, r in pool_df[pool_df["Model%"] >= 1.0].head(12).iterrows():
    arrow = "◄ VALUE" if r["EV_ratio"] > 1.3 else ""
    print(f"{r['Seed']:<6} {r['Team']:<22} {r['Model%']:>6.1f}% {r['Field%']:>6.1f}% {r['EV_ratio']:>5.2f}x  {arrow}")

print(f"\nAll outputs saved to outputs/")
print(f"  submission_2026.csv       — Kaggle submission")
print(f"  round_probs_2026.csv      — full round-by-round probabilities")
print(f"  pool_strategy_2026.csv    — EV analysis for bracket pools")

"""
Post-hoc calibration for 2026 predictions.

Two problems fixed:

1. SEED-STRATIFIED BLENDING (submission + simulation)
   The model gives a 16-seed ~15% vs a 1-seed; historically it's 1.8%.
   For R64 seed pairs (1v16, 2v15, … 8v9), blend model probability with
   the empirical historical win rate. Shrinkage is proportional to seed
   differential — large mismatches receive more historical prior weight.

   Formula:
     α = exp(−seed_diff / ALPHA_SCALE)     # how much to trust the model
     p_cal = α × p_model + (1−α) × p_hist_fav

2. TEMPERATURE SCALING (Monte Carlo only — not written to Kaggle submission)
   Championship probability compounds over 6 rounds. A team with p=0.80 in
   every game wins 6 straight with probability 0.80⁶ = 26%, and small
   calibration errors balloon. Applying T > 1 to per-game logits before
   each simulated draw pushes individual probs toward 0.5, reducing
   over-concentration.

   Formula:
     p_scaled = σ( logit(p) / T )

   T is auto-calibrated so the 4 1-seeds collectively win ≈ 32% of
   championships (historical average 1985–2025).

Outputs:
  outputs/submission_2026_calibrated.csv        — blended probabilities (Kaggle)
  outputs/round_probs_2026_calibrated.csv       — temperature-scaled round %s
  outputs/championship_comparison.csv           — side-by-side before/after
"""

import numpy as np
import pandas as pd
import itertools
import joblib
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
OUT_DIR    = ROOT / "outputs"
KAGGLE_DIR = ROOT / "march-machine-learning-mania-2026"
OUT_DIR.mkdir(exist_ok=True)

PREDICT_SEASON = 2026
N_SIM          = 100_000

# ── Calibration parameters ─────────────────────────────────────────────────────
ALPHA_SCALE = 10.0   # seed-blending decay constant (higher = more model trust)
TARGET_1SEED_CHAMP = 0.32   # historical 1-seed combined championship rate

# ── Historical first-round upset rates (1985–2025) ────────────────────────────
# P(higher seed wins), i.e., the upset probability
HIST_UPSET = {
    (1, 16): 0.018,
    (2, 15): 0.072,
    (3, 14): 0.155,
    (4, 13): 0.271,
    (5, 12): 0.352,
    (6, 11): 0.368,
    (7, 10): 0.394,
    (8,  9): 0.494,
}
# P(favorite wins) = 1 − P(upset)
HIST_FAV = {k: 1.0 - v for k, v in HIST_UPSET.items()}


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
mlp_model    = bundle.get("mlp", None)
feature_cols = bundle["feature_cols"]
weights      = bundle.get("ensemble_weights", (0.41, 0.41, 0.08, 0.10))

if isinstance(weights, dict):
    w_xgb = weights.get("xgb", 0.38)
    w_lgb = weights.get("lgb", 0.38)
    w_lr  = weights.get("lr",  0.09)
    w_mlp = weights.get("mlp", 0.15) if mlp_model else 0.0
elif len(weights) == 4:
    w_xgb, w_lgb, w_lr, w_mlp = weights
    if mlp_model is None:
        w_mlp = 0.0
else:
    w_xgb, w_lgb, w_lr = weights
    w_mlp = 0.0

if mlp_model is None and (w_xgb + w_lgb + w_lr) < 0.99:
    total = w_xgb + w_lgb + w_lr
    w_xgb /= total; w_lgb /= total; w_lr /= total

print(f"Ensemble weights — XGB:{w_xgb:.2f}  LGB:{w_lgb:.2f}  LR:{w_lr:.2f}  "
      f"MLP:{w_mlp:.2f}" if mlp_model else
      f"Ensemble weights — XGB:{w_xgb:.2f}  LGB:{w_lgb:.2f}  LR:{w_lr:.2f}")

# ── Load features & seeds ──────────────────────────────────────────────────────
features = pd.read_parquet(DATA_DIR / "team_season_features.parquet")
seeds_df  = pd.read_csv(KAGGLE_DIR / "MNCAATourneySeeds.csv")
teams_df  = pd.read_csv(KAGGLE_DIR / "MTeams.csv")

seeds_df["SeedNum"] = seeds_df["Seed"].apply(
    lambda s: int("".join(filter(str.isdigit, s)))
)

seed_2026 = seeds_df[seeds_df["Season"] == PREDICT_SEASON].copy()
feat_2026 = features[features["Season"] == PREDICT_SEASON].set_index("TeamID")
team_list = sorted(seed_2026["TeamID"].unique())

# Map TeamID → seed number (First Four teams use base seed)
seed_map = {}
for _, row in seed_2026.iterrows():
    tid = row["TeamID"]
    snum = row["SeedNum"]
    # keep the lower seed if duplicate (shouldn't be, but just in case)
    if tid not in seed_map or snum < seed_map[tid]:
        seed_map[tid] = snum

print(f"Tournament teams: {len(team_list)}")


# ── Raw model probabilities ────────────────────────────────────────────────────
print("Computing raw matchup probabilities...")

pairs = list(itertools.combinations(sorted(team_list), 2))
rows_feats = []
meta = []
for t1, t2 in pairs:
    if t1 not in feat_2026.index or t2 not in feat_2026.index:
        continue
    f1, f2 = feat_2026.loc[t1], feat_2026.loc[t2]
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
    rows_feats.append(row)
    meta.append((t1, t2))

X = pd.DataFrame(rows_feats, columns=feature_cols).fillna(0)

p_xgb = xgb_model.predict_proba(X)[:, 1]
p_lgb = lgb_model.predict_proba(X)[:, 1]
p_lr  = lr_model.predict_proba(X)[:, 1]
p_raw = w_xgb * p_xgb + w_lgb * p_lgb + w_lr * p_lr
if mlp_model is not None and w_mlp > 0:
    p_mlp = mlp_model.predict_proba(X)[:, 1]
    p_raw = p_raw + w_mlp * p_mlp
p_raw = np.clip(p_raw, 0.01, 0.99)

# Build lookup: (t1, t2) → P(t1 wins) with t1 < t2
raw_lookup = {}
for (t1, t2), p in zip(meta, p_raw):
    raw_lookup[(t1, t2)] = p


# ── Seed-stratified blending ───────────────────────────────────────────────────
def blend_with_seed_prior(p_model, s_fav, s_dog, alpha_scale=ALPHA_SCALE):
    """
    Blend model prob with historical seed-pair win rate.
    p_model  = P(favorite wins)  [favorite = lower seed number]
    s_fav    = seed number of favorite (1–16)
    s_dog    = seed number of underdog (1–16)
    Returns blended P(favorite wins).
    """
    pair = (min(s_fav, s_dog), max(s_fav, s_dog))
    if pair not in HIST_FAV:
        return p_model          # no historical prior for this pair

    hist_p_fav = HIST_FAV[pair]
    seed_diff  = s_dog - s_fav  # always positive (underdog − favorite)
    alpha      = np.exp(-seed_diff / alpha_scale)  # higher diff → less model weight
    return alpha * p_model + (1 - alpha) * hist_p_fav


print(f"\nApplying seed-stratified blending (α_scale={ALPHA_SCALE})...")
blended_lookup = {}
blend_records  = []

for (t1, t2), p_model in raw_lookup.items():
    s1 = seed_map.get(t1, 8)   # default to 8 if seed unknown
    s2 = seed_map.get(t2, 8)

    # Identify favorite (lower seed number = better team)
    if s1 <= s2:
        s_fav, s_dog = s1, s2
        p_model_fav  = p_model          # p_model = P(t1 wins), t1 is favorite
        p_blend_fav  = blend_with_seed_prior(p_model_fav, s_fav, s_dog)
        p_blend_t1   = p_blend_fav
    else:
        s_fav, s_dog = s2, s1
        p_model_fav  = 1.0 - p_model    # flip: P(t2 wins)
        p_blend_fav  = blend_with_seed_prior(p_model_fav, s_fav, s_dog)
        p_blend_t1   = 1.0 - p_blend_fav

    p_blend_t1 = float(np.clip(p_blend_t1, 0.01, 0.99))
    blended_lookup[(t1, t2)] = p_blend_t1

    pair = (min(s_fav, s_dog), max(s_fav, s_dog))
    if pair in HIST_UPSET:
        seed_diff = s_dog - s_fav
        alpha     = np.exp(-seed_diff / ALPHA_SCALE)
        blend_records.append({
            "pair":        f"{s_fav}v{s_dog}",
            "seed_diff":   seed_diff,
            "alpha":       round(alpha, 3),
            "model_fav%":  round(p_model_fav * 100, 1),
            "hist_fav%":   round(HIST_FAV[pair] * 100, 1),
            "blend_fav%":  round(p_blend_fav * 100, 1),
        })

# Show blending summary for R64 pairs in 2026
if blend_records:
    blend_df = (pd.DataFrame(blend_records)
                .sort_values("seed_diff")
                .drop_duplicates("pair"))
    print("\nSeed blending summary (2026, representative matchups):")
    print(blend_df.to_string(index=False))


# ── Write calibrated submission ────────────────────────────────────────────────
print("\nWriting calibrated submission...")
sub_rows = []
for t1, t2 in itertools.combinations(sorted(team_list), 2):
    sub_rows.append({
        "ID":   f"{PREDICT_SEASON}_{t1}_{t2}",
        "Pred": blended_lookup.get((t1, t2), 0.5),
    })
sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv(OUT_DIR / "submission_2026_calibrated.csv", index=False)
print(f"  Saved → outputs/submission_2026_calibrated.csv ({len(sub_df):,} matchups)")


# ── Temperature scaling for Monte Carlo ───────────────────────────────────────
def apply_temperature(p, T):
    """Scale probability toward 0.5 using temperature T > 1."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit_p / T))


def run_monte_carlo(prob_fn, temperature=1.0, n_sim=N_SIM):
    """
    Run bracket simulation using prob_fn(t1, t2) → P(t1 wins).
    If temperature > 1, scale each per-game probability toward 0.5.
    """
    seeded = {}
    for _, row in seed_2026.iterrows():
        seeded[row["Seed"]] = row["TeamID"]

    REGIONS         = ["W", "X", "Y", "Z"]
    FF_PAIRS        = [(0, 1), (2, 3)]
    ROUND_NAMES     = ["R64", "R32", "S16", "E8", "F4", "NCG", "Champion"]
    REGION_ROUNDS   = ["R64", "R32", "S16", "E8"]

    reach = {t: {r: 0 for r in ROUND_NAMES} for t in team_list}

    def sim_game_t(t1, t2):
        p = prob_fn(t1, t2)
        if temperature != 1.0:
            p = apply_temperature(p, temperature)
        return t1 if np.random.random() < p else t2

    def simulate_region(region):
        resolved = {}
        for seed_str, tid in seeded.items():
            if not seed_str.startswith(region):
                continue
            if seed_str.endswith("a") or seed_str.endswith("b"):
                key = seed_str[:-1]
                resolved.setdefault(key, []).append(tid)
            else:
                resolved[seed_str] = tid

        for key, pair in list(resolved.items()):
            if isinstance(pair, list) and len(pair) == 2:
                resolved[key] = sim_game_t(pair[0], pair[1])

        bracket = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        teams = []
        for s in bracket:
            key = f"{region}{str(s).zfill(2)}"
            t = resolved.get(key)
            if t is not None and not isinstance(t, list):
                teams.append(t)

        round_reached = {t: "R64" for t in teams}
        for round_name in ["R32", "S16", "E8"]:
            if len(teams) <= 1:
                break
            next_round = []
            for i in range(0, len(teams) - 1, 2):
                w = sim_game_t(teams[i], teams[i + 1])
                next_round.append(w)
                round_reached[w] = round_name
            teams = next_round

        if len(teams) == 2:
            winner = sim_game_t(teams[0], teams[1])
        elif len(teams) == 1:
            winner = teams[0]
        else:
            winner = None
        return winner, round_reached

    for _ in range(n_sim):
        region_results = [simulate_region(r) for r in REGIONS]
        ff  = [res[0] for res in region_results if res[0] is not None]
        rds = [res[1] for res in region_results]

        for round_dict in rds:
            for team_id, deepest in round_dict.items():
                if team_id not in reach:
                    continue
                for rname in REGION_ROUNDS:
                    reach[team_id][rname] += 1
                    if rname == deepest:
                        break

        if len(ff) != 4:
            continue

        for t in ff:
            reach[t]["F4"] += 1

        finalists = []
        for i, j in FF_PAIRS:
            if i < len(ff) and j < len(ff):
                w = sim_game_t(ff[i], ff[j])
                finalists.append(w)
                reach[w]["NCG"] += 1

        if len(finalists) == 2:
            champ = sim_game_t(finalists[0], finalists[1])
            reach[champ]["Champion"] += 1

    round_probs = []
    for tid in team_list:
        row = {"TeamID": tid}
        for r in ROUND_NAMES:
            row[f"prob_{r}"] = reach[tid][r] / n_sim
        round_probs.append(row)

    return (pd.DataFrame(round_probs)
            .merge(seed_2026[["TeamID", "Seed", "SeedNum"]], on="TeamID")
            .merge(teams_df[["TeamID", "TeamName"]], on="TeamID")
            .sort_values("prob_Champion", ascending=False))


def blended_win_prob(t1, t2):
    key = (min(t1, t2), max(t1, t2))
    p = blended_lookup.get(key, 0.5)
    return p if t1 < t2 else 1 - p


# ── Auto-calibrate temperature ─────────────────────────────────────────────────
# Use a coarse simulation (10k) to find T such that combined 1-seed championship
# probability ≈ TARGET_1SEED_CHAMP (historically ~32%).

one_seed_teams = [tid for tid, s in seed_map.items() if s == 1]
print(f"\n1-seed teams: {len(one_seed_teams)} — target combined champion% = "
      f"{TARGET_1SEED_CHAMP:.0%}")
print("Auto-calibrating temperature (coarse 10k sims)...")

CALIBRATE_N = 10_000
T_CANDIDATES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]
best_T = 1.0
best_delta = 999.0

for T in T_CANDIDATES:
    df_t = run_monte_carlo(blended_win_prob, temperature=T, n_sim=CALIBRATE_N)
    one_seed_total = df_t[df_t["TeamID"].isin(one_seed_teams)]["prob_Champion"].sum()
    delta = abs(one_seed_total - TARGET_1SEED_CHAMP)
    print(f"  T={T:.1f}  → 1-seed combined champ% = {one_seed_total:.1%}  "
          f"(Δ={delta:.3f})")
    if delta < best_delta:
        best_delta = delta
        best_T = T

print(f"\n→ Selected T = {best_T:.1f}  (best match to historical {TARGET_1SEED_CHAMP:.0%})")


# ── Full simulation with calibrated T ─────────────────────────────────────────
print(f"\nRunning {N_SIM:,} simulations with T={best_T:.1f}...")
cal_df = run_monte_carlo(blended_win_prob, temperature=best_T, n_sim=N_SIM)

cal_df.to_csv(OUT_DIR / "round_probs_2026_calibrated.csv", index=False)
print(f"  Saved → outputs/round_probs_2026_calibrated.csv")


# ── Load original (uncalibrated) results for comparison ───────────────────────
orig_path = OUT_DIR / "round_probs_2026.csv"
if orig_path.exists():
    orig_df = pd.read_csv(orig_path)
    compare = (cal_df[["Seed", "TeamName", "prob_Champion", "prob_F4", "prob_NCG"]]
               .rename(columns={
                   "prob_Champion": "cal_Champ%",
                   "prob_F4":       "cal_F4%",
                   "prob_NCG":      "cal_Final%",
               })
               .merge(
                   orig_df[["Seed", "TeamName", "prob_Champion", "prob_F4", "prob_NCG"]]
                   .rename(columns={
                       "prob_Champion": "raw_Champ%",
                       "prob_F4":       "raw_F4%",
                       "prob_NCG":      "raw_Final%",
                   }),
                   on=["Seed", "TeamName"], how="left",
               ))
    for col in ["cal_Champ%", "cal_F4%", "cal_Final%",
                "raw_Champ%", "raw_F4%", "raw_Final%"]:
        compare[col] = (compare[col] * 100).round(1)
    compare = compare.sort_values("cal_Champ%", ascending=False)
    compare.to_csv(OUT_DIR / "championship_comparison.csv", index=False)
    print(f"  Saved → outputs/championship_comparison.csv")
else:
    compare = None


# ── Print results ──────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"{'2026 CHAMPIONSHIP PROBABILITIES — CALIBRATED':^72}")
print(f"{'(seed-blended + temperature-scaled, T=' + str(best_T) + ')':^72}")
print(f"{'='*72}")

if compare is not None:
    print(f"{'Seed':<6} {'Team':<22} {'Raw Champ%':>10} {'Cal Champ%':>10} {'Change':>8}")
    print("-" * 72)
    for _, r in compare.head(20).iterrows():
        raw = r.get("raw_Champ%", float("nan"))
        cal = r["cal_Champ%"]
        delta = cal - raw if not np.isnan(raw) else 0
        arrow = "▲" if delta > 1 else ("▼" if delta < -1 else " ")
        print(f"{r['Seed']:<6} {r['TeamName']:<22} {raw:>9.1f}% {cal:>9.1f}%  "
              f"{arrow}{abs(delta):>5.1f}%")
else:
    print(f"{'Seed':<6} {'Team':<22} {'Champ%':>8} {'F4%':>7} {'Final%':>8}")
    print("-" * 72)
    for _, r in cal_df.head(20).iterrows():
        print(f"{r['Seed']:<6} {r['TeamName']:<22} "
              f"{r['prob_Champion']*100:>7.1f}% "
              f"{r['prob_F4']*100:>6.1f}% "
              f"{r['prob_NCG']*100:>7.1f}%")

# Seed concentration summary
print(f"\n{'─'*72}")
print("Concentration summary (calibrated vs raw):")
for snum in [1, 2, 3, 4]:
    teams_s = [tid for tid, s in seed_map.items() if s == snum]
    cal_total = cal_df[cal_df["TeamID"].isin(teams_s)]["prob_Champion"].sum()
    label = f"All {snum}-seeds combined"
    if orig_path.exists():
        raw_total = orig_df[orig_df["TeamID"].isin(teams_s)]["prob_Champion"].sum()
        print(f"  {label}: raw={raw_total:.1%}  calibrated={cal_total:.1%}")
    else:
        print(f"  {label}: calibrated={cal_total:.1%}")

print(f"\nAll outputs saved to outputs/")
print(f"  submission_2026_calibrated.csv    — seed-blended Kaggle submission")
print(f"  round_probs_2026_calibrated.csv   — calibrated round probabilities")
print(f"  championship_comparison.csv       — before/after comparison table")

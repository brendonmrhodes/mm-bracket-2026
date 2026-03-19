"""
Generate a diverse set of model-based brackets to maximize collective coverage.

Strategy: bracket portfolio optimization
  - Run N Monte Carlo simulations drawing complete brackets from the model distribution
  - Select K brackets using a greedy max-diversity algorithm:
      each new bracket maximizes its minimum weighted distance to all already-selected ones
  - Round weights increase exponentially (getting the champion right >> R64 pick)

This maximizes the probability that at least one of your brackets covers whatever
actually happens, while keeping every bracket individually reasonable.

Usage:
    python src/analysis/generate_brackets.py          # generates 19 brackets
    python src/analysis/generate_brackets.py --n 10   # generates 10 brackets

Outputs:
    outputs/generated_brackets.csv   — all picks in long format
    outputs/generated_brackets_summary.csv — one row per bracket, key picks
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import itertools
from pathlib import Path
from collections import defaultdict

ROOT       = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = ROOT / "data"
OUT_DIR    = ROOT / "outputs"
KAGGLE_DIR = ROOT / "march-machine-learning-mania-2026"

PREDICT_SEASON = 2026
N_CANDIDATES   = 5_000   # simulations to draw from

# Round weights for diversity metric — later rounds count exponentially more
ROUND_WEIGHTS = {
    "R64": 1, "R32": 2, "S16": 4, "E8": 8,
    "F4": 16, "NCG": 64, "Champion": 128,
}


# ── Load model & data ──────────────────────────────────────────────────────────
def load_everything():
    tuned = OUT_DIR / "models_tuned.pkl"
    default = OUT_DIR / "models.pkl"
    bundle = joblib.load(tuned if tuned.exists() else default)

    xgb_m    = bundle["xgb"]
    lgb_m    = bundle["lgb"]
    lr_m     = bundle["lr"]
    mlp_m    = bundle.get("mlp")
    feat_cols = bundle["feature_cols"]
    weights  = bundle.get("ensemble_weights", (0.45, 0.45, 0.10))

    if isinstance(weights, dict):
        w_xgb = weights.get("xgb", 0.45)
        w_lgb = weights.get("lgb", 0.45)
        w_lr  = weights.get("lr",  0.10)
        w_mlp = weights.get("mlp", 0.0) if mlp_m else 0.0
    else:
        w_xgb, w_lgb, w_lr = (weights[:3])
        w_mlp = weights[3] if len(weights) > 3 and mlp_m else 0.0

    features  = pd.read_parquet(DATA_DIR / "team_season_features.parquet")
    seeds_df  = pd.read_csv(KAGGLE_DIR / "MNCAATourneySeeds.csv")
    teams_df  = pd.read_csv(KAGGLE_DIR / "MTeams.csv")

    seeds_df["SeedNum"] = seeds_df["Seed"].str.extract(r"(\d+)").astype(int)
    seed_2026 = seeds_df[seeds_df["Season"] == PREDICT_SEASON].copy()
    feat_2026 = features[features["Season"] == PREDICT_SEASON].set_index("TeamID")
    team_list = sorted(seed_2026["TeamID"].unique())

    return (xgb_m, lgb_m, lr_m, mlp_m, feat_cols,
            w_xgb, w_lgb, w_lr, w_mlp,
            feat_2026, seed_2026, team_list,
            dict(zip(teams_df["TeamID"], teams_df["TeamName"])))


# ── Build seed-blended probability lookup ─────────────────────────────────────
HIST_FAV = {
    (1,16): 0.982, (2,15): 0.928, (3,14): 0.845, (4,13): 0.729,
    (5,12): 0.648, (6,11): 0.632, (7,10): 0.606, (8, 9): 0.506,
}
ALPHA_SCALE = 10.0

def build_blended_lookup(team_list, feat_map, feat_cols,
                         xgb_m, lgb_m, lr_m, mlp_m,
                         w_xgb, w_lgb, w_lr, w_mlp, seed_map):
    pairs = list(itertools.combinations(sorted(team_list), 2))
    rows, meta = [], []
    for t1, t2 in pairs:
        if t1 not in feat_map.index or t2 not in feat_map.index:
            continue
        f1, f2 = feat_map.loc[t1], feat_map.loc[t2]
        row = {}
        for col in feat_cols:
            if col.startswith("d_"):
                row[col] = f1.get(col[2:], np.nan) - f2.get(col[2:], np.nan)
            elif col.startswith("t1_"):
                row[col] = f1.get(col[3:], np.nan)
            elif col.startswith("t2_"):
                row[col] = f2.get(col[3:], np.nan)
            else:
                row[col] = np.nan
        rows.append(row)
        meta.append((t1, t2))

    X = pd.DataFrame(rows, columns=feat_cols).fillna(0)
    p = w_xgb * xgb_m.predict_proba(X)[:, 1]
    p += w_lgb * lgb_m.predict_proba(X)[:, 1]
    p += w_lr  * lr_m.predict_proba(X)[:, 1]
    if mlp_m and w_mlp > 0:
        p += w_mlp * mlp_m.predict_proba(X)[:, 1]
    p = np.clip(p, 0.01, 0.99)

    lookup = {}
    for (t1, t2), p_raw in zip(meta, p):
        s1 = seed_map.get(t1, 8)
        s2 = seed_map.get(t2, 8)
        s_fav, s_dog = (s1, s2) if s1 <= s2 else (s2, s1)
        pair = (s_fav, s_dog)
        if pair in HIST_FAV:
            diff  = s_dog - s_fav
            alpha = np.exp(-diff / ALPHA_SCALE)
            p_fav_model = p_raw if s1 == s_fav else 1 - p_raw
            p_fav_blend = alpha * p_fav_model + (1 - alpha) * HIST_FAV[pair]
            p_t1 = float(np.clip(p_fav_blend if s1 == s_fav else 1 - p_fav_blend, 0.01, 0.99))
        else:
            p_t1 = float(p_raw)
        lookup[(t1, t2)] = p_t1

    def win_prob(a, b):
        key = (min(a, b), max(a, b))
        p = lookup.get(key, 0.5)
        return p if a < b else 1 - p

    return win_prob


# ── Simulate one complete bracket ─────────────────────────────────────────────
def simulate_bracket(win_prob, seeded):
    """Return dict mapping game_key -> winning TeamID for all 63 games."""
    results = {}
    REGIONS = ["W", "X", "Y", "Z"]

    def sim(a, b, key):
        p = win_prob(a, b)
        w = a if np.random.random() < p else b
        results[key] = w
        return w

    ff_teams = []
    for reg in REGIONS:
        resolved = {}
        for seed_str, tid in seeded.items():
            if not seed_str.startswith(reg):
                continue
            if seed_str[-1] in "ab":
                resolved.setdefault(seed_str[:-1], []).append(tid)
            else:
                resolved[seed_str] = tid

        # First Four
        for key, pair in list(resolved.items()):
            if isinstance(pair, list) and len(pair) == 2:
                resolved[key] = sim(pair[0], pair[1], f"FF_{key}")

        bracket_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        teams = []
        for s in bracket_order:
            k = f"{reg}{str(s).zfill(2)}"
            t = resolved.get(k)
            if t is not None and not isinstance(t, list):
                teams.append(t)

        for rnd in ["R32", "S16", "E8"]:
            next_t = []
            for i in range(0, len(teams) - 1, 2):
                w = sim(teams[i], teams[i+1], f"{rnd}_{reg}_{i//2}")
                next_t.append(w)
            teams = next_t

        # Regional final
        if len(teams) == 2:
            reg_winner = sim(teams[0], teams[1], f"RegFinal_{reg}")
        else:
            reg_winner = teams[0] if teams else None
        ff_teams.append(reg_winner)

    # Final Four: W vs X, Y vs Z
    finalists = []
    for (i, j), slot in [((0,1),"F4_WX"), ((2,3),"F4_YZ")]:
        if ff_teams[i] and ff_teams[j]:
            w = sim(ff_teams[i], ff_teams[j], slot)
            finalists.append(w)

    champion = None
    if len(finalists) == 2:
        champion = sim(finalists[0], finalists[1], "NCG")
        results["Champion"] = champion

    return results, champion


# ── Diversity metric ───────────────────────────────────────────────────────────
def bracket_distance(b1_results, b2_results):
    """Weighted Hamming distance between two brackets."""
    all_keys = set(b1_results) | set(b2_results)
    dist = 0
    for key in all_keys:
        if b1_results.get(key) != b2_results.get(key):
            # Map key prefix to round weight
            if key == "Champion":     dist += ROUND_WEIGHTS["Champion"]
            elif key.startswith("NCG"):   dist += ROUND_WEIGHTS["NCG"]
            elif key.startswith("F4"):    dist += ROUND_WEIGHTS["F4"]
            elif key.startswith("RegF"): dist += ROUND_WEIGHTS["E8"]
            elif key.startswith("E8"):   dist += ROUND_WEIGHTS["E8"]
            elif key.startswith("S16"):  dist += ROUND_WEIGHTS["S16"]
            elif key.startswith("R32"):  dist += ROUND_WEIGHTS["R32"]
            else:                         dist += ROUND_WEIGHTS["R64"]
    return dist


# ── Greedy diverse selection ───────────────────────────────────────────────────
def select_diverse(candidates, n, seed_map, t_name):
    """Greedy: pick bracket that maximises min-distance to already-selected set."""
    # Start with the highest-probability champion bracket (most common champion)
    champ_counts = defaultdict(int)
    for _, champ in candidates:
        if champ:
            champ_counts[champ] += 1
    most_common_champ = max(champ_counts, key=champ_counts.get)
    first = next((cand for cand in candidates if cand[1] == most_common_champ), candidates[0])

    selected = [first]
    remaining = [c for c in candidates if c is not first]

    while len(selected) < n and remaining:
        best, best_score = None, -1
        for candidate in remaining:
            min_d = min(bracket_distance(candidate[0], s[0]) for s in selected)
            if min_d > best_score:
                best_score = min_d
                best = candidate
        selected.append(best)
        remaining.remove(best)
        champ = best[1]
        print(f"  Bracket {len(selected):2d}: champion = {t_name.get(champ,'?'):<20}  "
              f"min-distance = {best_score}")

    return selected


# ── Format output ──────────────────────────────────────────────────────────────
def format_brackets(selected, seed_2026, t_name):
    seed_map = {}
    for _, row in seed_2026.iterrows():
        seed_map[row["TeamID"]] = row["Seed"]

    rows = []
    summary_rows = []

    for idx, (results, champion) in enumerate(selected, 1):
        label = f"Bracket_{idx:02d}"

        # Pull key picks
        champ_name = t_name.get(champion, "?")
        champ_seed = seed_map.get(champion, "?")

        # Final Four: teams that won RegFinal
        ff = [results.get(f"RegFinal_{r}") for r in ["W","X","Y","Z"]]
        ff_names = " / ".join(f"{t_name.get(t,'?')} ({seed_map.get(t,'?')})"
                               for t in ff if t)

        # Finalists
        f1 = results.get("F4_WX")
        f2 = results.get("F4_YZ")
        final = (f"{t_name.get(f1,'?')} ({seed_map.get(f1,'?')}) vs "
                 f"{t_name.get(f2,'?')} ({seed_map.get(f2,'?')})" if f1 and f2 else "?")

        summary_rows.append({
            "Bracket": label,
            "Champion": f"{champ_name} ({champ_seed})",
            "Championship_Game": final,
            "Final_Four": ff_names,
        })

        # All picks
        for game_key, winner_id in results.items():
            rows.append({
                "Bracket": label,
                "Game":    game_key,
                "Winner":  t_name.get(winner_id, str(winner_id)),
                "Seed":    seed_map.get(winner_id, "?"),
            })

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


# ── Main ───────────────────────────────────────────────────────────────────────
def main(n_brackets=19):
    print(f"Generating {n_brackets} diverse model brackets from {N_CANDIDATES:,} simulations...\n")

    (xgb_m, lgb_m, lr_m, mlp_m, feat_cols,
     w_xgb, w_lgb, w_lr, w_mlp,
     feat_2026, seed_2026, team_list, t_name) = load_everything()

    seed_map = {}
    for _, row in seed_2026.iterrows():
        seed_map[row["TeamID"]] = int(row["SeedNum"])

    seeded = dict(zip(seed_2026["Seed"], seed_2026["TeamID"]))

    print("Building probability lookup...")
    win_prob = build_blended_lookup(
        team_list, feat_2026, feat_cols,
        xgb_m, lgb_m, lr_m, mlp_m,
        w_xgb, w_lgb, w_lr, w_mlp, seed_map,
    )

    print(f"Running {N_CANDIDATES:,} bracket simulations...")
    candidates = []
    champ_tally = defaultdict(int)
    for _ in range(N_CANDIDATES):
        res, champ = simulate_bracket(win_prob, seeded)
        candidates.append((res, champ))
        if champ:
            champ_tally[champ] += 1

    print("\nChampion distribution across simulations:")
    for tid, cnt in sorted(champ_tally.items(), key=lambda x: -x[1])[:12]:
        pct = cnt / N_CANDIDATES * 100
        bar = "█" * int(pct / 2)
        print(f"  {t_name.get(tid,'?'):<22} {pct:5.1f}%  {bar}")

    print(f"\nSelecting {n_brackets} maximally diverse brackets (greedy algorithm)...")
    selected = select_diverse(candidates, n_brackets, seed_map, t_name)

    detail_df, summary_df = format_brackets(selected, seed_2026, t_name)

    detail_path  = OUT_DIR / "generated_brackets.csv"
    summary_path = OUT_DIR / "generated_brackets_summary.csv"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*65}")
    print(f"{'GENERATED BRACKETS — SUMMARY':^65}")
    print(f"{'='*65}")
    for _, r in summary_df.iterrows():
        print(f"\n{r['Bracket']}")
        print(f"  Champion : {r['Champion']}")
        print(f"  Final    : {r['Championship_Game']}")
        print(f"  Final 4  : {r['Final_Four']}")

    print(f"\nSaved:")
    print(f"  {detail_path}   — all picks")
    print(f"  {summary_path} — champion/F4/final per bracket")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=19, help="Number of brackets to generate")
    args = parser.parse_args()
    main(args.n)

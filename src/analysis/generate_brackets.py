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

# ── Probability-realism controls ───────────────────────────────────────────────
# Pool filtering uses DEEP-round log-probability only (R32 onwards).
# First-round upset brackets are NOT penalised — a bracket that picks a 12-5
# upset but has a great Final Four path will survive the filter.
# 0.20 = keep the top 20% of brackets ranked by their R32-onwards log-prob.
PROB_POOL_FRAC  = 0.20

# In the diversity scoring, blend min-distance (diversity) with bracket
# probability.  score = min_distance * prob_rank_score ^ PROB_BLEND_ALPHA
# 0.0 = pure diversity  |  1.0 = strongly probability-weighted
PROB_BLEND_ALPHA = 0.6

# Round weights for diversity metric — later rounds count exponentially more.
# First-round (R64) weight is deliberately raised so the greedy algorithm
# actively seeks brackets with DIFFERENT upset patterns, not just the same
# 11-6 or 12-5 pick repeated across all 19 brackets.
ROUND_WEIGHTS = {
    "R64": 1, "R32": 6, "S16": 4, "E8": 8,
    "F4": 16, "NCG": 64, "Champion": 128,
}
# NOTE: keys starting with "R32_" are first-round (Round of 64) games — the
# winner advances *to* the Round of 32.  The weight above (6) applies to them.

# ── Historical plausibility constraints ────────────────────────────────────────
# Hard seed caps: no team seeded higher than this has reached the stage (2003–2025)
#   E8: Saint Peter's (15-seed, 2022); F4: George Mason/VCU/Loyola/NC State (11-seed)
#   NCG/Champion: Butler/Kentucky/UNC (8-seed)
ROUND_SEED_CAPS = {
    "E8":       15,   # max seed to reach Elite Eight
    "F4":       11,   # max seed to reach Final Four
    "NCG":       8,   # max seed to reach Championship Game
    "Champion":  8,   # max seed to win the championship
}

# Soft efficiency floors (adjEM): teams below these get their win prob penalized
#   Based on 5th-percentile adjEM of historical teams at each stage (2003–2025)
#   F4 floor=14: NC State 2024 had 15.9, Loyola 2018 had 16.4 as all-time F4 lows
#   NCG floor=16: Butler 2011 had 16.5 as the all-time championship game low
ROUND_EFF_FLOORS = {
    "E8":        5.0,   # Oral Roberts 2021 (4.83) is the only E8 team below 5
    "F4":       14.0,   # filters teams that historically never reach F4
    "NCG":      16.0,   # championship game requires elite efficiency
    "Champion": 18.0,   # no champion has had adjEM below ~20
}
EFF_PENALTY = 0.15   # reduce win prob to 15% of raw value if below efficiency floor


def _get_constraint_round(key):
    """Map a game key to the stage the WINNER will advance into."""
    if key.startswith("E8_"):        return "E8"
    if key.startswith("RegFinal_"):  return "F4"
    if key.startswith("F4_"):        return "NCG"
    if key == "NCG":                 return "Champion"
    return None


def _constrained_prob(a, b, key, raw_prob_fn, seed_map, feat_map):
    """
    Adjust win probability with historical seed caps and efficiency floors.
    - Seed cap violations → near-zero probability (0.02) for the ineligible team
    - Efficiency floor violations → win prob * EFF_PENALTY (soft, not zero)
    """
    p = raw_prob_fn(a, b)
    stage = _get_constraint_round(key)
    if stage is None:
        return p

    s_a = seed_map.get(a, 8)
    s_b = seed_map.get(b, 8)
    cap = ROUND_SEED_CAPS[stage]

    a_ok_seed = s_a <= cap
    b_ok_seed = s_b <= cap

    if not a_ok_seed and not b_ok_seed:
        return 0.5    # both violate — fallback to coin flip (shouldn't happen)
    elif not a_ok_seed:
        return 0.02   # A can't advance; B wins with near-certainty
    elif not b_ok_seed:
        return 0.98   # B can't advance

    # Efficiency floor (soft — penalise, don't block)
    if stage in ROUND_EFF_FLOORS and feat_map is not None:
        floor = ROUND_EFF_FLOORS[stage]
        adj_a = float(feat_map.loc[a, "adjEM"]) if a in feat_map.index else 15.0
        adj_b = float(feat_map.loc[b, "adjEM"]) if b in feat_map.index else 15.0
        if adj_a < floor and adj_b >= floor:
            p = p * EFF_PENALTY
        elif adj_b < floor and adj_a >= floor:
            p = 1.0 - (1.0 - p) * EFF_PENALTY
        # If both below floor: no adjustment (let seed caps / raw prob decide)

    return float(np.clip(p, 0.01, 0.99))


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
def _is_first_round(key):
    """True for Round-of-64 and First Four games (winner advances to R32)."""
    return key.startswith("R32_") or key.startswith("FF_")


def simulate_bracket(win_prob, seeded, seed_map=None, feat_map=None):
    """Return (results, champion, log_prob_all, log_prob_deep) for one bracket.

    log_prob_all  — log-probability of every game outcome (all 63 games)
    log_prob_deep — log-probability of R32-onwards games only; used for pool
                    filtering so that upset-heavy first rounds aren't penalised
    """
    results        = {}
    log_prob_all   = 0.0
    log_prob_deep  = 0.0
    REGIONS        = ["W", "X", "Y", "Z"]

    def sim(a, b, key):
        nonlocal log_prob_all, log_prob_deep
        if seed_map is not None:
            p = _constrained_prob(a, b, key, win_prob, seed_map, feat_map)
        else:
            p = win_prob(a, b)
        if np.random.random() < p:
            w, p_win = a, p
        else:
            w, p_win = b, 1.0 - p
        lp = np.log(max(p_win, 1e-9))
        log_prob_all  += lp
        if not _is_first_round(key):
            log_prob_deep += lp
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

    return results, champion, log_prob_all, log_prob_deep


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


# ── Greedy probability-weighted diverse selection ──────────────────────────────
def select_diverse(candidates, n, seed_map, t_name):
    """
    Greedy selection balancing realism (bracket probability) with diversity.

    Algorithm:
      1. Pre-filter to the top PROB_POOL_FRAC of candidates by log-probability.
         This ensures every selected bracket is drawn from the high-probability
         region of the distribution — no tail outliers just for variety's sake.
      2. Normalise log-probs to [0, 1] within the filtered pool so brackets can
         be ranked by relative plausibility.
      3. Greedy: at each step score = min_distance × prob_rank ^ PROB_BLEND_ALPHA
         This keeps diversity meaningful while weighting toward likelier brackets.
      4. Start from the single highest-probability bracket (not just most common
         champion) so bracket 1 is always the model's best single prediction.
    """
    # Step 1 — pre-filter by DEEP-round log-probability (index 3).
    # First-round upsets don't lower a bracket's eligibility; only the quality
    # of deep-tournament predictions (R32 onwards) gates entry to the pool.
    deep_lps = np.array([c[3] for c in candidates])
    cutoff   = np.percentile(deep_lps, (1.0 - PROB_POOL_FRAC) * 100)
    pool     = [c for c in candidates if c[3] >= cutoff]
    if len(pool) < n:
        pool = candidates  # safety fallback

    # Step 2 — normalise deep log-probs to [0, 1] for blended scoring
    pool_lp        = np.array([c[3] for c in pool])
    lp_min, lp_max = pool_lp.min(), pool_lp.max()
    lp_range       = lp_max - lp_min if lp_max > lp_min else 1.0
    prob_rank      = {id(c): (c[3] - lp_min) / lp_range for c in pool}

    # Step 3 — start with the highest-probability bracket overall
    first     = max(pool, key=lambda c: c[2])
    selected  = [first]
    remaining = [c for c in pool if c is not first]

    while len(selected) < n and remaining:
        best, best_score = None, -1.0
        for candidate in remaining:
            min_d = min(bracket_distance(candidate[0], s[0]) for s in selected)
            score = min_d * (prob_rank[id(candidate)] ** PROB_BLEND_ALPHA)
            if score > best_score:
                best_score = score
                best = candidate
        selected.append(best)
        remaining.remove(best)
        champ = best[1]
        prank = prob_rank[id(best)]
        print(f"  Bracket {len(selected):2d}: champion = {t_name.get(champ,'?'):<22} "
              f"prob-rank = {prank:.2f}  score = {best_score:.1f}")

    return selected


# ── Format output ──────────────────────────────────────────────────────────────
def format_brackets(selected, seed_2026, t_name):
    seed_map = {}
    for _, row in seed_2026.iterrows():
        seed_map[row["TeamID"]] = row["Seed"]

    rows = []
    summary_rows = []

    for idx, (results, champion, _lp_all, _lp_deep) in enumerate(selected, 1):
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

    print(f"Running {N_CANDIDATES:,} bracket simulations (with historical constraints)...")
    candidates = []
    champ_tally = defaultdict(int)
    for _ in range(N_CANDIDATES):
        res, champ, lp_all, lp_deep = simulate_bracket(win_prob, seeded, seed_map, feat_2026)
        candidates.append((res, champ, lp_all, lp_deep))
        if champ:
            champ_tally[champ] += 1

    print("\nChampion distribution across simulations:")
    for tid, cnt in sorted(champ_tally.items(), key=lambda x: -x[1])[:12]:
        pct = cnt / N_CANDIDATES * 100
        bar = "█" * int(pct / 2)
        print(f"  {t_name.get(tid,'?'):<22} {pct:5.1f}%  {bar}")

    pool_size = int(N_CANDIDATES * PROB_POOL_FRAC)
    print(f"\nSelecting {n_brackets} probability-weighted diverse brackets "
          f"(top {pool_size:,} of {N_CANDIDATES:,} by prob, α={PROB_BLEND_ALPHA})...")
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

"""
Round-by-Round EV Analysis & Upset Finder
==========================================
Extends pool strategy beyond just championship to every round.

Outputs:
  outputs/round_ev_2026.csv       — EV ratio vs public for every team × every round
  outputs/upset_finder_2026.csv   — First-round matchups ranked by upset potential
  outputs/historical_seed_rates.csv — Historical advance rates by seed (the 'public' baseline)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent.parent
KAGGLE   = ROOT / "march-machine-learning-mania-2026"
OUT_DIR  = ROOT / "outputs"

ROUND_DAYS = {
    "R64": (136, 137),
    "R32": (138, 140),
    "S16": (143, 145),
    "E8":  (146, 148),
    "F4":  (152, 152),
    "NCG": (154, 156),
}
ROUNDS = ["R64", "R32", "S16", "E8", "F4", "NCG", "Champion"]
PREDICT_SEASON = 2026


def load_data():
    tourn   = pd.read_csv(KAGGLE / "MNCAATourneyCompactResults.csv")
    seeds   = pd.read_csv(KAGGLE / "MNCAATourneySeeds.csv")
    teams   = pd.read_csv(KAGGLE / "MTeams.csv")
    slots   = pd.read_csv(KAGGLE / "MNCAATourneySlots.csv")

    def parse_seed(s): return int("".join(filter(str.isdigit, s)))
    seeds["SeedNum"] = seeds["Seed"].apply(parse_seed)
    return tourn, seeds, teams, slots


def assign_round(daynum):
    for rname, (lo, hi) in ROUND_DAYS.items():
        if lo <= daynum <= hi:
            return rname
    if daynum <= 135:
        return "FirstFour"
    return "Unknown"


def compute_historical_rates(tourn, seeds):
    """
    For each seed 1-16, compute historical advance rates per round.
    These are the 'public expected' rates for bracket pools.
    """
    tourn = tourn[tourn["Season"] >= 1985].copy()
    tourn["Round"] = tourn["DayNum"].apply(assign_round)
    tourn = tourn[tourn["Round"] != "FirstFour"]

    # Teams in tournament each season (excluding First Four play-ins)
    seed_map = seeds[~seeds["Seed"].str[-1].isin(["a","b"])][["Season","TeamID","SeedNum"]].copy()

    records = []
    for season in tourn["Season"].unique():
        season_games = tourn[tourn["Season"] == season]
        season_seeds = seed_map[seed_map["Season"] == season]
        all_teams = set(season_seeds["TeamID"])

        # Teams that won at least one game in each round
        round_winners = {}
        for rname in ROUND_DAYS:
            rnd_games = season_games[season_games["Round"] == rname]
            round_winners[rname] = set(rnd_games["WTeamID"])

        # R64: all 64 teams "entered" (reached R64 = played in it)
        r64_teams = set(season_seeds["TeamID"])
        # Champion = who won the NCG
        ncg_games = season_games[season_games["Round"] == "NCG"]
        champs = set(ncg_games["WTeamID"]) if len(ncg_games) > 0 else set()

        for _, row in season_seeds.iterrows():
            tid, seed = row["TeamID"], row["SeedNum"]
            rec = {"Season": season, "TeamID": tid, "SeedNum": seed}
            rec["reached_R64"]     = int(tid in r64_teams)
            rec["reached_R32"]     = int(tid in round_winners.get("R64",  set()))
            rec["reached_S16"]     = int(tid in round_winners.get("R32",  set()))
            rec["reached_E8"]      = int(tid in round_winners.get("S16",  set()))
            rec["reached_F4"]      = int(tid in round_winners.get("E8",   set()))
            rec["reached_NCG"]     = int(tid in round_winners.get("F4",   set()))
            rec["reached_Champion"]= int(tid in champs)
            records.append(rec)

    df = pd.DataFrame(records)

    # Aggregate by seed
    rate_cols = [c for c in df.columns if c.startswith("reached_")]
    rates = (df.groupby("SeedNum")[rate_cols].mean()
               .rename(columns=lambda c: c.replace("reached_", "hist_"))
               .reset_index())
    # Count seasons per seed
    counts = df.groupby("SeedNum").size().rename("n_appearances").reset_index()
    rates = rates.merge(counts, on="SeedNum")
    return rates, df


def build_round_ev(round_df_path, seed_rates, seeds, teams, season=PREDICT_SEASON):
    """
    For each team in the 2026 tournament, compute:
      model_prob vs hist_prob for each round → EV ratio
    """
    round_df = pd.read_csv(round_df_path)

    seed_2026 = seeds[seeds["Season"] == season].copy()
    seed_2026 = seed_2026[~seed_2026["Seed"].str[-1].isin(["a","b"])]
    seed_2026 = seed_2026.merge(teams[["TeamID","TeamName"]], on="TeamID")
    seed_2026 = seed_2026.merge(seed_rates, on="SeedNum")

    # Merge model round probs
    model_cols = {
        "prob_R64":      "model_R64",
        "prob_R32":      "model_R32",
        "prob_S16":      "model_S16",
        "prob_E8":       "model_E8",
        "prob_F4":       "model_F4",
        "prob_NCG":      "model_NCG",
        "prob_Champion": "model_Champion",
    }
    round_probs = round_df[["TeamID"] + list(model_cols.keys())].rename(columns=model_cols)
    seed_2026 = seed_2026.merge(round_probs, on="TeamID")

    rows = []
    for _, r in seed_2026.iterrows():
        base = {"Seed": r["Seed"], "SeedNum": r["SeedNum"], "TeamName": r["TeamName"]}
        for rnd in ROUNDS:
            model_col = f"model_{rnd}"
            hist_col  = f"hist_{rnd}"
            if model_col not in r or hist_col not in r:
                continue
            mp = r[model_col]
            hp = r[hist_col]
            ev = (mp / hp) if hp > 0.001 else 0.0
            base[f"model_{rnd}"] = round(mp * 100, 1)
            base[f"hist_{rnd}"]  = round(hp * 100, 1)
            base[f"ev_{rnd}"]    = round(ev, 2)
        rows.append(base)

    return pd.DataFrame(rows)


def build_upset_finder(features_path, seeds, teams, season=PREDICT_SEASON):
    """
    For each first-round matchup, identify upset potential:
      - Model win prob for the underdog
      - Historical upset rate for that seed matchup
      - 'Upset edge' = model prob - historical rate
    """
    round_probs_path = OUT_DIR / "round_probs_2026.csv"
    if not round_probs_path.exists():
        print("  round_probs_2026.csv not found — skipping upset finder")
        return None

    round_df = pd.read_csv(round_probs_path)
    feat = pd.read_parquet(features_path)

    seed_2026 = seeds[seeds["Season"] == season].copy()
    # Resolve First Four: use a/b pairs, pick the team that appears without a/b
    base_seeds = seed_2026[~seed_2026["Seed"].str[-1].isin(["a","b"])].copy()
    base_seeds["SeedNum"] = base_seeds["Seed"].apply(
        lambda s: int("".join(filter(str.isdigit, s)))
    )

    # Build standard R64 bracket: 1v16,8v9,5v12,4v13,6v11,3v14,7v10,2v15
    matchup_pairs = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

    feat_2026 = feat[feat["Season"] == season].set_index("TeamID")
    rp = round_df[["TeamID","prob_Champion","prob_F4","prob_S16","prob_R32"]].set_index("TeamID")
    name_map = teams.set_index("TeamID")["TeamName"]

    # Historical upset rates by seed matchup
    hist_upset = {
        (5,12): 0.352, (4,13): 0.271, (6,11): 0.368, (3,14): 0.155,
        (7,10): 0.394, (2,15): 0.072, (1,16): 0.018, (8,9):  0.494,
    }

    rows = []
    regions = base_seeds["Seed"].str[0].unique()

    for region in sorted(regions):
        reg_seeds = base_seeds[base_seeds["Seed"].str.startswith(region)]
        seed_to_tid = dict(zip(reg_seeds["SeedNum"], reg_seeds["TeamID"]))

        for hi_seed, lo_seed in matchup_pairs:  # hi_seed = favorite (lower number)
            fav_tid = seed_to_tid.get(hi_seed)
            dog_tid = seed_to_tid.get(lo_seed)
            if fav_tid is None or dog_tid is None:
                continue

            lo, hi = min(fav_tid, dog_tid), max(fav_tid, dog_tid)
            hist_rate = hist_upset.get((hi_seed, lo_seed), 0.5)

            # Get features for both teams
            if fav_tid not in feat_2026.index or dog_tid not in feat_2026.index:
                continue
            ff = feat_2026.loc[fav_tid]
            fd = feat_2026.loc[dog_tid]

            # Model win prob for underdog (from round_probs, compute ratio)
            # Use prob_R32 as proxy for "winning R64"
            dog_r32 = rp.loc[dog_tid, "prob_R32"] if dog_tid in rp.index else 0.5
            fav_r32 = rp.loc[fav_tid, "prob_R32"] if fav_tid in rp.index else 0.5
            # Normalize to just this matchup
            total = dog_r32 + fav_r32
            if total > 0:
                model_dog_prob = dog_r32 / total
            else:
                model_dog_prob = hist_rate

            # Upset edge = model prob - historical base rate
            upset_edge = model_dog_prob - hist_rate

            # Key upset indicator features
            tempo_diff = fd.get("avg_FGA3_rate", np.nan) - ff.get("avg_FGA3_rate", np.nan)
            elo_diff   = fd.get("elo_pre_tourney", np.nan) - ff.get("elo_pre_tourney", np.nan)
            adjEM_diff = fd.get("adjEM", np.nan) - ff.get("adjEM", np.nan)
            momentum   = fd.get("elo_momentum", np.nan)
            continuity = fd.get("kp_ht_continuity", np.nan)
            ft_rate    = fd.get("avg_FTR", np.nan)  # underdog FT rate

            rows.append({
                "Region":        region,
                "Matchup":       f"{hi_seed} vs {lo_seed}",
                "Favorite":      name_map.get(fav_tid, str(fav_tid)),
                "Underdog":      name_map.get(dog_tid, str(dog_tid)),
                "Fav_Seed":      hi_seed,
                "Dog_Seed":      lo_seed,
                "Model_Dog%":    round(model_dog_prob * 100, 1),
                "Hist_Dog%":     round(hist_rate * 100, 1),
                "Upset_Edge":    round(upset_edge * 100, 1),
                "Elo_Gap":       round(elo_diff, 0) if not np.isnan(elo_diff) else None,
                "AdjEM_Gap":     round(adjEM_diff, 2) if not np.isnan(adjEM_diff) else None,
                "Dog_Momentum":  round(momentum, 1) if not np.isnan(momentum) else None,
                "Dog_3PA_Edge":  round(tempo_diff * 100, 1) if not np.isnan(tempo_diff) else None,
                "Dog_FTRate":    round(ft_rate * 100, 1) if not np.isnan(ft_rate) else None,
                "Dog_Continuity": round(continuity, 1) if continuity and not np.isnan(continuity) else None,
            })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("Upset_Edge", ascending=False)
    return df


def main():
    print("=" * 60)
    print("Round-by-Round EV Analysis & Upset Finder")
    print("=" * 60)

    tourn, seeds, teams, slots = load_data()

    # ── Historical seed rates ─────────────────────────────────────────────────
    print("\nComputing historical advance rates by seed (1985–2025)...")
    seed_rates, seed_detail = compute_historical_rates(tourn, seeds)
    seed_rates.to_csv(OUT_DIR / "historical_seed_rates.csv", index=False)
    print("  Historical rates by seed saved.")
    print(f"\n{'Seed':>4}  {'R64':>6}  {'R32':>6}  {'S16':>6}  {'E8':>6}  {'F4':>6}  {'NCG':>6}  {'Champ':>6}")
    print("-" * 55)
    for _, r in seed_rates.iterrows():
        print(f"{int(r['SeedNum']):>4}  "
              f"{r['hist_R64']:>5.1%}  {r['hist_R32']:>5.1%}  {r['hist_S16']:>5.1%}  "
              f"{r['hist_E8']:>5.1%}  {r['hist_F4']:>5.1%}  {r['hist_NCG']:>5.1%}  "
              f"{r['hist_Champion']:>5.1%}")

    # ── Round EV table ────────────────────────────────────────────────────────
    print("\nBuilding round-by-round EV table for 2026 tournament teams...")
    round_probs_path = OUT_DIR / "round_probs_2026.csv"
    if not round_probs_path.exists():
        print("  round_probs_2026.csv not found — run 03_predict.py first")
        return

    ev_df = build_round_ev(round_probs_path, seed_rates, seeds, teams)
    ev_df.to_csv(OUT_DIR / "round_ev_2026.csv", index=False)

    print(f"\n{'Team':<22}{'Seed':>5}", end="")
    for rnd in ["R32","S16","E8","F4","Champion"]:
        print(f"  {rnd:>8}", end="")
    print()
    print("-" * 75)
    # Show best EV plays at each stage
    for rnd in ["R32","S16","E8","F4","Champion"]:
        ev_col = f"ev_{rnd}"
        if ev_col not in ev_df.columns:
            continue
        top = ev_df.nlargest(3, ev_col)
        for _, r in top.iterrows():
            m = r.get(f"model_{rnd}", 0)
            h = r.get(f"hist_{rnd}", 0)
            ev = r.get(ev_col, 0)
            if m >= 1.0 and ev > 1.0:
                flag = " ◄ VALUE" if ev > 1.5 else ""
                print(f"  {rnd} | {r['TeamName']:<22} ({r['Seed']}) "
                      f"Model: {m:.1f}%  Hist: {h:.1f}%  EV: {ev:.2f}x{flag}")

    # ── Upset finder ─────────────────────────────────────────────────────────
    print("\nBuilding first-round upset finder...")
    feat_path = ROOT / "data" / "team_season_features.parquet"
    upset_df = build_upset_finder(feat_path, seeds, teams)
    if upset_df is not None and len(upset_df):
        upset_df.to_csv(OUT_DIR / "upset_finder_2026.csv", index=False)
        print(f"\n{'Matchup':<10} {'Favorite':<22} {'Underdog':<22} {'Model%':>7} {'Hist%':>6} {'Edge':>6}  Notes")
        print("-" * 90)
        for _, r in upset_df.head(12).iterrows():
            edge_flag = " ◄◄ UPSET ALERT" if r["Upset_Edge"] > 10 else (" ◄ WATCH" if r["Upset_Edge"] > 5 else "")
            elo_note = f"  EloGap:{r['Elo_Gap']:+.0f}" if r["Elo_Gap"] is not None else ""
            mom_note = f"  Mom:{r['Dog_Momentum']:+.0f}" if r["Dog_Momentum"] is not None else ""
            print(f"{r['Region']}{r['Matchup']:<10} {r['Favorite']:<22} {r['Underdog']:<22} "
                  f"{r['Model_Dog%']:>6.1f}% {r['Hist_Dog%']:>5.1f}%  {r['Upset_Edge']:>+5.1f}%"
                  f"{elo_note}{mom_note}{edge_flag}")

    print(f"\nOutputs saved to outputs/")
    print(f"  round_ev_2026.csv        — EV ratios for every team × every round")
    print(f"  upset_finder_2026.csv    — First-round upset candidates ranked by edge")
    print(f"  historical_seed_rates.csv — Historical advance rates by seed 1985–2025")


if __name__ == "__main__":
    main()

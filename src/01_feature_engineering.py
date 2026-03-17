"""
Feature Engineering Pipeline v3
Integrates all data sources into a single team-season feature matrix:

  Source                      Features
  ─────────────────────────── ────────────────────────────────────────────
  Kaggle box scores           Four Factors (eFG%, TO%, OR%, FTR), PPP,
                              recency-weighted season averages
  KenPom (2002-2026)          AdjEM, AdjO, AdjD, AdjT, Luck, SOS metrics
  KenPom extended             Four Factors, Height/Experience, Misc Stats
                              (scraped via kenpom_extended — if available)
  Torvik T-Rank (2008-2026)   BARTHAG, WAB, qual_barthag, elite_sos
  Massey Ordinals             Composite rank from 10 rating systems
  Elo (1985-2026)             Pre-tournament Elo rating
  Conf Tournament             Wins, champion flag, finalist flag
  Seeds                       Seed number, region
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "data"))
from team_name_map import build_name_map, lookup_team_id

DATA_DIR   = Path(__file__).resolve().parent.parent / "march-machine-learning-mania-2026"
RAW_DIR    = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR    = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load raw sources ────────────────────────────────────────────────────────
print("Loading data...")
rsd       = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv")
tourney   = pd.read_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv")
seeds     = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
massey    = pd.read_csv(DATA_DIR / "MMasseyOrdinals.csv")
kenpom    = pd.read_parquet(RAW_DIR / "kenpom_all.parquet")
torvik    = pd.read_parquet(RAW_DIR / "torvik_slim.parquet")
elo       = pd.read_parquet(RAW_DIR / "elo_momentum.parquet")
conf_tourn   = pd.read_parquet(RAW_DIR / "conf_tourney_features.parquet")
tourn_hist   = pd.read_parquet(RAW_DIR / "tourney_history_features.parquet")
name_map  = build_name_map()

print(f"  Box scores    : {len(rsd):,} games ({rsd['Season'].min()}–{rsd['Season'].max()})")
print(f"  Tournament    : {len(tourney):,} games")
print(f"  KenPom        : {len(kenpom):,} rows ({kenpom['season'].min()}–{kenpom['season'].max()})")
print(f"  Torvik        : {len(torvik):,} rows ({torvik['season'].min()}–{torvik['season'].max()})")
print(f"  Elo           : {len(elo):,} rows")
print(f"  Conf tourney  : {len(conf_tourn):,} rows")
print(f"  Tourney hist  : {len(tourn_hist):,} rows")

# KenPom extended — optional, load if available
kp_four_factors = None
kp_height_exp   = None
kp_misc_stats   = None
for fname, label, attr in [
    ("kenpom_four_factors.parquet", "Four Factors",     "kp_four_factors"),
    ("kenpom_height_exp.parquet",   "Height/Exp",       "kp_height_exp"),
    ("kenpom_misc_stats.parquet",   "Misc stats",       "kp_misc_stats"),
]:
    p = RAW_DIR / fname
    if p.exists():
        df = pd.read_parquet(p)
        print(f"  KenPom {label:12s}: {len(df):,} rows — {list(df.columns[:6])}")
        if attr == "kp_four_factors": kp_four_factors = df
        elif attr == "kp_height_exp": kp_height_exp   = df
        elif attr == "kp_misc_stats": kp_misc_stats   = df
    else:
        print(f"  KenPom {label:12s}: not available (run fetch_kenpom_extended.py)")


# ── 2. Box-score Four Factors (from Kaggle game logs) ─────────────────────────
print("\nComputing Four Factors from box scores...")

def compute_game_features(df):
    rows = []
    for _, g in df.iterrows():
        w_poss = g.WFGA - g.WOR + g.WTO + 0.475 * g.WFTA
        l_poss = g.LFGA - g.LOR + g.LTO + 0.475 * g.LFTA
        poss   = (w_poss + l_poss) / 2

        def ff(fgm, fga, fgm3, to, orb, opp_drb, ftm, fta, poss):
            efg  = (fgm + 0.5 * fgm3) / fga  if fga  > 0 else np.nan
            tov  = to / poss                   if poss > 0 else np.nan
            orb_ = orb / (orb + opp_drb)       if (orb + opp_drb) > 0 else np.nan
            ftr  = ftm / fga                   if fga  > 0 else np.nan
            ppp  = (fgm * 2 + fgm3 + ftm)/poss if poss > 0 else np.nan
            return efg, tov, orb_, ftr, ppp

        w_efg, w_tov, w_orb, w_ftr, w_ppp = ff(g.WFGM, g.WFGA, g.WFGM3, g.WTO, g.WOR, g.LDR, g.WFTM, g.WFTA, poss)
        l_efg, l_tov, l_orb, l_ftr, l_ppp = ff(g.LFGM, g.LFGA, g.LFGM3, g.LTO, g.LOR, g.WDR, g.LFTM, g.LFTA, poss)

        base = {"Season": g.Season, "DayNum": g.DayNum, "Poss": poss}
        for team, opp, efg, tov, orb, ftr, ppp, d_efg, d_tov, d_orb, d_ftr, d_ppp, fga, fga3, loc in [
            (g.WTeamID, g.LTeamID, w_efg, w_tov, w_orb, w_ftr, w_ppp,
             l_efg, l_tov, l_orb, l_ftr, l_ppp, g.WFGA, g.WFGA3, g.WLoc),
            (g.LTeamID, g.WTeamID, l_efg, l_tov, l_orb, l_ftr, l_ppp,
             w_efg, w_tov, w_orb, w_ftr, w_ppp, g.LFGA, g.LFGA3,
             "A" if g.WLoc == "H" else ("H" if g.WLoc == "A" else "N")),
        ]:
            rows.append({**base,
                "TeamID": team, "OppID": opp,
                "Win": 1 if team == g.WTeamID else 0,
                "ScoreDiff": (g.WScore - g.LScore) if team == g.WTeamID else (g.LScore - g.WScore),
                "eFG": efg, "TOV": tov, "ORB": orb, "FTR": ftr, "PPP": ppp,
                "Def_eFG": d_efg, "Def_TOV": d_tov, "Def_ORB": d_orb,
                "Def_FTR": d_ftr, "Def_PPP": d_ppp,
                "FGA3_rate": fga3 / fga if fga > 0 else np.nan,
                "Loc": loc})
    return pd.DataFrame(rows)

game_df = compute_game_features(rsd)

def season_aggregates(game_df):
    game_df = game_df.copy()
    # Recency weight: last ~35 days of regular season (day 100+) get 2x
    game_df["weight"] = np.where(game_df["DayNum"] >= 100, 2.0, 1.0)
    feat_cols = ["eFG","TOV","ORB","FTR","PPP",
                 "Def_eFG","Def_TOV","Def_ORB","Def_FTR","Def_PPP",
                 "FGA3_rate","ScoreDiff"]
    records = []
    for (season, team), grp in game_df.groupby(["Season", "TeamID"]):
        w = grp["weight"].values
        rec = {"Season": season, "TeamID": team, "Games": len(grp)}
        rec["WinPct"]       = np.average(grp["Win"], weights=w)
        rec["AvgScoreDiff"] = np.average(grp["ScoreDiff"], weights=w)
        rec["ScoreDiff_std"]= grp["ScoreDiff"].std()
        for col in feat_cols:
            vals = grp[col].dropna()
            if len(vals):
                rec[f"avg_{col}"] = np.average(vals, weights=grp.loc[vals.index, "weight"])
            else:
                rec[f"avg_{col}"] = np.nan
        neutral = grp[grp["Loc"] == "N"]
        rec["Neutral_WinPct"] = neutral["Win"].mean() if len(neutral) > 5 else rec["WinPct"]
        records.append(rec)
    return pd.DataFrame(records)

print("  Aggregating season-team features...")
box_features = season_aggregates(game_df)
print(f"  Box feature rows: {len(box_features):,}")


# ── 3. Massey ordinals composite rank ─────────────────────────────────────────
print("\nExtracting Massey ordinals...")
SYSTEMS = ["POM", "BPI", "NET", "SAG", "MOR", "KPI", "DOK", "WOL", "COL", "RPI"]
massey_pre = massey[massey["RankingDayNum"] <= 133].copy()
massey_latest = (massey_pre.sort_values("RankingDayNum")
                 .groupby(["Season","SystemName","TeamID"]).last().reset_index())
massey_pivot = (massey_latest[massey_latest["SystemName"].isin(SYSTEMS)]
                .pivot_table(index=["Season","TeamID"], columns="SystemName", values="OrdinalRank")
                .reset_index())
massey_pivot.columns.name = None
massey_pivot.columns = ["Season","TeamID"] + [f"rank_{c}" for c in massey_pivot.columns[2:]]
rank_cols = [c for c in massey_pivot.columns if c.startswith("rank_")]
massey_pivot["rank_composite"] = massey_pivot[rank_cols].mean(axis=1)
print(f"  Massey pivot: {massey_pivot.shape}")


# ── 4. KenPom — add TeamID via name matching ──────────────────────────────────
print("\nMatching KenPom team names to Kaggle IDs...")
kenpom["team_clean"] = kenpom["team"].str.replace(r"\s+\d+$", "", regex=True).str.strip()
kenpom["TeamID"] = kenpom["team_clean"].apply(lambda n: lookup_team_id(n, name_map))
kenpom = kenpom.rename(columns={"season": "Season"})

n_matched = kenpom["TeamID"].notna().sum()
print(f"  Matched {n_matched}/{len(kenpom)} rows ({n_matched/len(kenpom):.1%})")

kp_cols = ["Season","TeamID","adjEM","adjO","adjD","adjT",
           "luck","sos_adjEM","opp_O","opp_D","nc_sos_adjEM"]
kenpom_clean = kenpom[kp_cols].dropna(subset=["TeamID"]).copy()
kenpom_clean["TeamID"] = kenpom_clean["TeamID"].astype(int)


# ── 5. Torvik — add TeamID ────────────────────────────────────────────────────
print("\nMatching Torvik team names to Kaggle IDs...")
torvik["TeamID"] = torvik["team"].apply(lambda n: lookup_team_id(n, name_map))
torvik = torvik.rename(columns={"season": "Season"})

n_matched = torvik["TeamID"].notna().sum()
print(f"  Matched {n_matched}/{len(torvik)} rows ({n_matched/len(torvik):.1%})")

tv_cols = ["Season","TeamID","barthag","wab","qual_barthag","qual_games",
           "elite_sos","fun","adjoe","adjde","adjt"]
torvik_clean = torvik[tv_cols].dropna(subset=["TeamID"]).copy()
torvik_clean["TeamID"] = torvik_clean["TeamID"].astype(int)
# Rename to avoid collision with KenPom adjO/adjD
torvik_clean = torvik_clean.rename(columns={
    "adjoe": "tv_adjoe", "adjde": "tv_adjde", "adjt": "tv_adjt"
})


# ── 6. Seeds ──────────────────────────────────────────────────────────────────
def parse_seed(s):
    return int("".join(filter(str.isdigit, s)))

seeds["SeedNum"] = seeds["Seed"].apply(parse_seed)
seeds["Region"]  = seeds["Seed"].str[0]


# ── 7. Merge everything ───────────────────────────────────────────────────────
print("\nMerging all features...")

# Conference tournament features
ct_cols = ["Season","TeamID","conf_tourney_wins","conf_tourney_winpct",
           "conf_tourney_champion","conf_tourney_finalist"]
conf_tourn_clean = conf_tourn[ct_cols].copy()

th_cols = ["Season","TeamID",
           "tourney_appearances","tourney_recent_appear","tourney_avg_seed",
           "tourney_best_round","tourney_recent_best","tourney_win_rate",
           "tourney_r32_rate","tourney_s16_rate","tourney_e8_rate",
           "tourney_f4_rate","tourney_ncg_rate","tourney_champ_rate"]
tourn_hist_clean = tourn_hist[th_cols].astype({"Season": int, "TeamID": int})

features = (
    box_features
    .merge(massey_pivot,      on=["Season","TeamID"], how="left")
    .merge(kenpom_clean,      on=["Season","TeamID"], how="left")
    .merge(torvik_clean,      on=["Season","TeamID"], how="left")
    .merge(elo.rename(columns={"season":"Season"}), on=["Season","TeamID"], how="left")
    .merge(conf_tourn_clean,  on=["Season","TeamID"], how="left")
    .merge(tourn_hist_clean,  on=["Season","TeamID"], how="left")
    .merge(seeds[["Season","TeamID","SeedNum","Region"]], on=["Season","TeamID"], how="left")
)

# Fill conf tourney features for teams that didn't participate (auto-bids that skipped)
features["conf_tourney_wins"]      = features["conf_tourney_wins"].fillna(0)
features["conf_tourney_winpct"]    = features["conf_tourney_winpct"].fillna(0)
features["conf_tourney_champion"]  = features["conf_tourney_champion"].fillna(0)
features["conf_tourney_finalist"]  = features["conf_tourney_finalist"].fillna(0)

# Fill tourney history features for first-time participants
for col in th_cols[2:]:  # skip Season, TeamID
    if col in features.columns:
        fill = 17.0 if col == "tourney_avg_seed" else 0.0
        features[col] = features[col].fillna(fill)

# KenPom extended — merge if available
def _merge_kenpom_ext(features, df, prefix, key_cols):
    """Match by season + team name → TeamID, then merge."""
    df = df.copy()
    # Find team name column (case-insensitive)
    team_col = next((c for c in df.columns if c.lower() == "team"), None)
    if team_col:
        df["TeamID"] = df[team_col].apply(lambda n: lookup_team_id(n, name_map))
    else:
        df["TeamID"] = None
    if "TeamID" not in df.columns or df["TeamID"].isna().all():
        return features
    df = df.rename(columns={"season": "Season"})
    df["TeamID"] = df["TeamID"].dropna().astype(int)
    df = df.dropna(subset=["TeamID"])
    # Drop garbage columns (col_N pattern, Rk, Rank, W-L, Conf)
    import re as _re
    drop_pats = {"Rk", "Rank", "W-L", "Conf", "conf"}
    df = df[[c for c in df.columns
             if c not in drop_pats and not _re.match(r'^col_\d+$', c)]]
    # Coerce all non-identity columns to float (handles StringDtype from parquet)
    skip_ids = {"Season", "TeamID", team_col or "Team"}
    for c in df.columns:
        if c not in skip_ids:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop columns that are entirely NaN after coercion
    df = df.dropna(axis=1, how="all")
    # Rename numeric columns to avoid collisions
    rename_map = {c: f"{prefix}_{c}" for c in df.columns
                  if c not in skip_ids and not c.startswith(prefix)}
    df = df.rename(columns=rename_map)
    keep = ["Season","TeamID"] + [c for c in df.columns if c.startswith(f"{prefix}_")]
    return features.merge(df[keep], on=["Season","TeamID"], how="left")

if kp_four_factors is not None:
    print("  Merging KenPom Four Factors...")
    features = _merge_kenpom_ext(features, kp_four_factors, "kp_ff", [])
if kp_height_exp is not None:
    print("  Merging KenPom Height/Experience...")
    features = _merge_kenpom_ext(features, kp_height_exp, "kp_ht", [])
if kp_misc_stats is not None:
    print("  Merging KenPom Misc Stats...")
    features = _merge_kenpom_ext(features, kp_misc_stats, "kp_ms", [])

features["SeedNum"]  = features["SeedNum"].fillna(17)
features["InTourney"]= (features["SeedNum"] <= 16).astype(int)

# Fill Elo for teams with no history (new D1 programs) with the mean
# Fill Elo features with column means for teams with no history
elo_cols = ["elo_pre_tourney","elo_last10","elo_momentum","elo_peak","elo_consistency","elo_late_winpct"]
for col in elo_cols:
    if col in features.columns:
        features[col] = features[col].fillna(features[col].mean())
elo_mean = features["elo_pre_tourney"].mean()

print(f"  Feature table: {features.shape}")
features.to_parquet(OUT_DIR / "team_season_features.parquet", index=False)
print(f"  Saved → data/team_season_features.parquet")

# Quick coverage report
kp_coverage  = features["adjEM"].notna().mean()
tv_coverage  = features["barthag"].notna().mean()
elo_coverage = (features["elo_pre_tourney"] != elo_mean).mean()
ct_coverage  = (features["conf_tourney_wins"] > 0).mean()
print(f"\n  KenPom coverage  : {kp_coverage:.1%}")
print(f"  Torvik coverage  : {tv_coverage:.1%}")
print(f"  Elo coverage     : {elo_coverage:.1%}")
print(f"  Conf tourney     : {ct_coverage:.1%} played in conf tourney")
for ext_col in ["kp_ff_eFG%", "kp_ht_AHgt.", "kp_ms_3PA%"]:
    if ext_col in features.columns:
        print(f"  {ext_col:20s}: {features[ext_col].notna().mean():.1%}")


# ── 8. Build matchup training dataset ─────────────────────────────────────────
print("\nBuilding matchup dataset...")

EXCLUDE    = {"Season","TeamID","Region"}
feat_cols  = [c for c in features.columns if c not in EXCLUDE]

# Drop columns with >50% missing across all rows
miss = features[feat_cols].isnull().mean()
feat_cols = list(miss[miss < 0.5].index)

# Deduplicate: keep first occurrence per Season/TeamID
features = features.drop_duplicates(subset=["Season","TeamID"], keep="first")
feat_map = features.set_index(["Season","TeamID"])

rows = []
for _, g in tourney.iterrows():
    s, w, l = g.Season, g.WTeamID, g.LTeamID
    t1, t2  = (w, l) if w < l else (l, w)
    label   = 1 if w < l else 0  # 1 = lower-ID team won

    if (s, t1) not in feat_map.index or (s, t2) not in feat_map.index:
        continue

    f1, f2 = feat_map.loc[(s, t1)], feat_map.loc[(s, t2)]
    row = {"Season": s, "T1": t1, "T2": t2, "Label": label}

    for col in feat_cols:
        v1, v2 = f1.get(col, np.nan), f2.get(col, np.nan)
        row[f"d_{col}"]  = v1 - v2  # differential (main signal)
        row[f"t1_{col}"] = v1        # raw values (trees benefit from these)
        row[f"t2_{col}"] = v2

    rows.append(row)

matchup_df = pd.DataFrame(rows)
print(f"  Matchup rows : {len(matchup_df):,}")
print(f"  Feature cols : {matchup_df.shape[1]}")
matchup_df.to_parquet(OUT_DIR / "matchup_train.parquet", index=False)
print(f"  Saved → data/matchup_train.parquet")
print(f"\n  Seasons covered: {sorted(matchup_df['Season'].unique())}")

# Coverage check for matchup rows
for col in ["d_adjEM","d_barthag","d_elo_pre_tourney","d_rank_POM"]:
    if col in matchup_df.columns:
        cov = matchup_df[col].notna().mean()
        print(f"  {col:25s}  coverage: {cov:.1%}")

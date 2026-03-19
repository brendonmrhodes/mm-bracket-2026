# March Madness Prediction Model — Project Reference

## Project Goal

Build a competitive NCAA Men's Basketball Tournament prediction algorithm that:
1. Generates calibrated win probabilities for every possible matchup
2. Produces a Kaggle-format submission for the March Machine Learning Mania 2026 competition
3. Supports bracket pool strategy via expected value (EV) analysis
4. Is the most accurate model possible using publicly available data

Current performance (v4, pre-tuning): **86.4% accuracy, 0.333 log-loss** via walk-forward cross-validation across 2008–2025 tournaments. Baseline (seed-only) is ~68%. v3 was 86.6%/0.320 (smaller feature set). Optuna tuning in progress — see outputs/models_tuned.pkl when done.

---

## Environment

```bash
# Activate environment
micromamba activate mm-predict

# Run full pipeline (from project root)
micromamba run -n mm-predict python src/01_feature_engineering.py
micromamba run -n mm-predict python src/02_model.py
micromamba run -n mm-predict python src/03_predict.py
micromamba run -n mm-predict python src/04_tune.py   # long — ~30 min
```

Python 3.11 via micromamba. Key packages: pandas, numpy, xgboost, lightgbm, optuna, scikit-learn, shap, playwright, browser-cookie3, curl_cffi, beautifulsoup4, rapidfuzz, pyarrow.

---

## Directory Structure

```
MM-Prediction/
├── CLAUDE.md                          ← this file
├── environment.yml                    ← micromamba env spec
│
├── march-machine-learning-mania-2026/ ← raw Kaggle competition data (DO NOT MODIFY)
│   ├── MRegularSeasonDetailedResults.csv   (124k regular season games, 2003–2026)
│   ├── MNCAATourneyDetailedResults.csv     (1,449 tournament games with box scores)
│   ├── MNCAATourneyCompactResults.csv      (tournament outcomes, 1985–2026)
│   ├── MRegularSeasonCompactResults.csv    (regular season results, 1985–2026)
│   ├── MMasseyOrdinals.csv                 (5.8M rows — ordinal rankings from 40+ systems)
│   ├── MNCAATourneySeeds.csv               (seeds 1985–2026)
│   ├── MTeams.csv                          (TeamID → TeamName mapping)
│   ├── MTeamSpellings.csv                  (alternate team name spellings)
│   ├── MNCAATourneySlots.csv               (bracket structure by season)
│   └── SampleSubmissionStage1.csv          (submission format: ID, Pred)
│
├── data/
│   ├── team_season_features.parquet   ← merged feature table (8,363 team-seasons × 57 cols)
│   ├── matchup_train.parquet          ← training matchups (1,449 rows × 157 features)
│   └── raw/
│       ├── kenpom_YYYY.csv            ← per-year KenPom ratings (2002–2026)
│       ├── kenpom_all.parquet         ← combined KenPom (8,679 rows)
│       ├── torvik_YYYY.csv            ← per-year Torvik/T-Rank (2008–2026)
│       ├── torvik_all.parquet         ← combined Torvik (6,689 rows, 45 cols)
│       ├── torvik_slim.parquet        ← key Torvik metrics only (17 cols)
│       ├── elo_game_by_game.parquet   ← Elo rating after every game (198k rows)
│       ├── elo_pre_tourney.parquet    ← pre-tournament Elo snapshot per team/season
│       ├── elo_momentum.parquet       ← Elo + recency features (last10, momentum, peak)
│       ├── elo_end_of_season.parquet  ← post-tournament Elo
│       ├── 538_ncaa_forecasts.csv     ← FiveThirtyEight historical forecasts 2011–2014
│       └── 538_2014_*.parquet         ← FiveThirtyEight 2014 bracket snapshots
│
├── src/
│   ├── 01_feature_engineering.py     ← STEP 1: builds team_season_features + matchup_train
│   ├── 02_model.py                   ← STEP 2: train XGB+LGB+LR ensemble, walk-forward CV
│   ├── 03_predict.py                 ← STEP 3: 2026 predictions + Monte Carlo simulation
│   ├── 04_tune.py                    ← STEP 4 (optional): Optuna hyperparameter tuning
│   └── data/
│       ├── fetch_kenpom.py           ← scrape KenPom (uses Firefox session cookies)
│       ├── fetch_kenpom_extended.py  ← scrape KenPom extended pages (Four Factors, height, etc.)
│       ├── fetch_torvik.py           ← download Torvik JSON data (free, no auth)
│       ├── fetch_538.py              ← download FiveThirtyEight archived data
│       ├── fetch_warrennolan.py      ← scrape NET/RPI rankings from WarrenNolan
│       ├── fetch_sports_reference.py ← scrape Sports-Reference CBB (rate-limited)
│       ├── fetch_all.py              ← master runner for all data fetchers
│       ├── compute_elo.py            ← build Elo ratings from Kaggle game history
│       ├── kenpom_session.py         ← shared Playwright Firefox session for KenPom
│       └── team_name_map.py          ← normalize team names → Kaggle TeamIDs
│
└── outputs/
    ├── models.pkl                    ← trained XGB + LGB + LR ensemble
    ├── models_tuned.pkl              ← Optuna-tuned + calibrated models (after 04_tune.py)
    ├── best_params.json              ← best hyperparameters from Optuna
    ├── submission_2026.csv           ← Kaggle submission (2,278 matchups)
    ├── round_probs_2026.csv          ← P(reaching each round) for all 68 teams
    ├── championship_probs_2026.csv   ← championship probability per team
    ├── pool_strategy_2026.csv        ← EV ratio = model% / expected public pick%
    ├── cv_predictions.parquet        ← out-of-fold predictions from walk-forward CV
    ├── feature_importance_xgb.csv    ← XGBoost feature importances
    ├── feature_importance_lgb.csv    ← LightGBM feature importances
    └── tuning_log.txt                ← Optuna tuning output log
```

---

## Data Sources

### Active / Integrated

| Source | Coverage | Key Features | Access |
|--------|----------|--------------|--------|
| **Kaggle March Mania** | 1985–2026 | Box scores, seeds, tournament results, Massey ordinals | Free download |
| **KenPom** | 2002–2026 | AdjEM, AdjO, AdjD, AdjT, Luck, SOS, OppO, OppD | ~$20/yr subscription; scraped via Firefox cookies + Playwright |
| **Bart Torvik / T-Rank** | 2008–2026 | BARTHAG, WAB, qual_barthag, elite_sos, adjOE, adjDE | Free JSON endpoint |
| **Elo (computed)** | 1985–2026 | elo_pre_tourney, elo_last10, elo_momentum, elo_peak, elo_late_winpct | Built from Kaggle game logs |
| **Massey Ordinals** | 2003–2026 | Rankings from POM, BPI, NET, SAG, MOR, KPI, DOK, WOL, COL, RPI | Included in Kaggle dataset |
| **FiveThirtyEight** | 2011–2014 | Historical tournament forecasts | Free GitHub archive |

### Pending (scrapers built, waiting on Cloudflare rate limit to clear)

| Source | Key Features | Script |
|--------|--------------|--------|
| **KenPom Four Factors** | eFG%, TO%, OR%, FTR (off + def) per team per year | `fetch_kenpom_extended.py` |
| **KenPom Height/Experience** | Avg height, minutes-weighted experience | `fetch_kenpom_extended.py --pages height_exp` |
| **KenPom Home Court** | Per-arena home court advantage rating | `fetch_kenpom_extended.py --pages hca` |
| **KenPom Player Stats** | Player efficiency for roster quality index | `fetch_kenpom_extended.py --pages players` |
| **WarrenNolan NET/RPI** | Official NET rankings 2019–2026, RPI 2003–2018 | `fetch_warrennolan.py` |
| **Sports-Reference** | SRS, advanced stats, coach data 1985–2026 | `fetch_sports_reference.py` |

**Note on KenPom extended:** kenpom.com is behind Cloudflare. The scraper uses Playwright Firefox which normally bypasses it, but rapid probing triggered an IP-level rate limit (Error 1015). Wait several hours before running `fetch_kenpom_extended.py`. Use a 5s delay between pages (already set in the script).

---

## Feature Engineering (src/01_feature_engineering.py)

### Pipeline

1. **Box-score Four Factors** — computed from `MRegularSeasonDetailedResults.csv` per game, then aggregated to season averages with **recency weighting** (last 35 days of regular season = 2x weight):
   - eFG%, TO%, OR%, FTR (offense and defense)
   - Points per possession (PPP), Score differential, Win%
   - FGA3 rate (3-point attempt rate)
   - Neutral site win%

2. **Massey Ordinals** — take the final pre-tournament snapshot (day ≤ 133) for 10 systems, pivot to one column per system, add composite mean rank.

3. **KenPom** — match team names via `team_name_map.py` (97.3% match rate), join on Season+TeamID:
   - `adjEM` (efficiency margin), `adjO` (offensive efficiency), `adjD` (defensive efficiency)
   - `adjT` (tempo), `luck`, `sos_adjEM`, `opp_O`, `opp_D`, `nc_sos_adjEM`

4. **Torvik** — match names (98.8% match rate):
   - `barthag` (expected win% vs avg D1), `wab` (wins above bubble)
   - `qual_barthag` (efficiency vs quality opponents), `elite_sos`

5. **Elo momentum** — from `compute_elo.py`, pre-tournament snapshot:
   - `elo_pre_tourney`, `elo_last10` (avg Elo last 10 games)
   - `elo_momentum` (Elo change over last 10 games — hot/cold signal)
   - `elo_peak` (season high), `elo_consistency` (std of game-to-game changes)
   - `elo_late_winpct` (win% in last 10 games)

6. **Seeds** — numeric seed (1–16), region

### Matchup Construction

For each tournament game, create one row with:
- `d_{feature}` = Team1 value − Team2 value (differentials — primary signal)
- `t1_{feature}` = Team1 raw value
- `t2_{feature}` = Team2 raw value
- `Label` = 1 if lower TeamID won, 0 otherwise (canonical ordering avoids leakage)

Result: **1,449 matchups × 157 features** spanning 2003–2025.

---

## Model Architecture (src/02_model.py)

### Walk-Forward Cross-Validation

Train on all seasons < N, test on season N. Starts at 2008 (need ≥5 seasons training data). This is the correct validation approach — no data leakage from future seasons.

### Ensemble

```
XGBoost (45%) + LightGBM (45%) + Logistic Regression (10%)
              ↓
    Calibrated probability output
```

- XGBoost/LGB use early stopping (30 rounds) on a 15% validation slice
- LogisticRegression uses StandardScaler pipeline
- Final probabilities clipped to [0.01, 0.99] to avoid log(0)

### Hyperparameters (defaults, before Optuna tuning)

```python
XGBoost:  n_estimators=500, max_depth=4, learning_rate=0.03,
          subsample=0.8, colsample_bytree=0.7, min_child_weight=3
LightGBM: n_estimators=500, max_depth=4, learning_rate=0.03,
          subsample=0.8, colsample_bytree=0.7, min_child_samples=10
```

### Performance History

| Version | What Changed | Accuracy | Log-Loss |
|---------|-------------|----------|----------|
| v1 | Box scores + Massey ordinal ranks only | 70.2% | 0.582 |
| v2 | + Real KenPom AdjEM/AdjO/AdjD + Torvik BARTHAG | 76.6% | 0.487 |
| v3 | + Elo momentum (last10, momentum, peak, late_winpct) | **86.6%** | **0.320** |
| v4 | + 21 Massey systems, MLP ensemble, KenPom Four Factors/Height/Misc Stats | 86.4% | 0.333 |
| v4-tuned | + Optuna tuning + isotonic calibration (in progress) | TBD | TBD |

---

## Top Predictive Features (XGBoost importance, v3)

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `d_elo_pre_tourney` | Elo differential entering tournament |
| 2 | `d_adjEM` | KenPom efficiency margin differential |
| 3 | `d_AvgScoreDiff` | Season avg scoring margin differential |
| 4 | `d_elo_last10` | Avg Elo over last 10 games differential |
| 5 | `d_elo_peak` | Season peak Elo differential |
| 6 | `d_sos_adjEM` | Strength of schedule differential |
| 7 | `d_qual_games` | Quality games played differential |
| 8 | `d_opp_D` | Opponent defensive efficiency differential |
| 9 | `t1_elo_late_winpct` | Late-season win rate (raw, not diff) |
| 10 | `d_elo_momentum` | Elo trajectory (hot/cold) differential |

---

## Prediction Pipeline (src/03_predict.py)

1. Loads tuned model if `outputs/models_tuned.pkl` exists, else falls back to `outputs/models.pkl`
2. Generates win probabilities for all 2,278 possible matchup pairs
3. Runs 100,000 Monte Carlo bracket simulations:
   - Handles First Four play-in games (X16a/b, Y11a/b, Z11a/b, Z16)
   - Proper region structure (W vs X in Final Four, Y vs Z)
4. Outputs:
   - `submission_2026.csv` — Kaggle format
   - `round_probs_2026.csv` — per-team probability for every round
   - `pool_strategy_2026.csv` — EV ratio vs estimated public pick rates

### EV Ratio (Pool Strategy)

```
EV Ratio = Model Championship % / Expected Public Pick %
```

Public pick rates are approximated by seed (1-seeds ~24%, 2-seeds ~12%, etc.). EV > 1.3x means the model gives a team more credit than the public — high value in large bracket pools.

---

## Elo System (src/data/compute_elo.py)

Follows FiveThirtyEight's NCAAB methodology:
- **K-factor = 20** × margin-of-victory multiplier (log scale, diminishing returns)
- **Home court = +100 Elo points** for home team
- **Offseason regression**: each team's Elo moves 1/3 of the way back to 1500 between seasons
- **Initial Elo = 1500** for all teams
- **Recency features** computed from last 10 regular-season games per team per season

---

## Team Name Matching (src/data/team_name_map.py)

Team names differ across sources (KenPom uses "Duke1", Torvik uses "Duke", Kaggle uses TeamID 1181). Resolution:
1. Load `MTeamSpellings.csv` (Kaggle's own alternate spellings — 1,669 entries)
2. Normalize: lowercase, strip punctuation, expand abbreviations
3. Fuzzy match via `rapidfuzz` (threshold: 85) for remaining unmatched names

Current match rates: KenPom 97.3%, Torvik 98.8%.

---

## KenPom Scraping Notes

**Authentication**: KenPom is behind Cloudflare. Only approach that works:
- Log into kenpom.com in **Firefox**
- Run `fetch_kenpom.py` — reads session cookies via `browser_cookie3`
- Uses Playwright Firefox for extended pages (bot-detection resistant)

**Rate limiting**: Cloudflare triggers a temporary IP ban (Error 1015) after ~5 rapid requests. Minimum 3–5s delay between pages. If banned, wait several hours before retrying. Never probe multiple pages in a quick loop.

**Extended pages to fetch** (run after Cloudflare ban clears):
```bash
micromamba run -n mm-predict python src/data/fetch_kenpom_extended.py
# Or specific pages:
micromamba run -n mm-predict python src/data/fetch_kenpom_extended.py --pages height_exp hca players
```

---

## Kaggle Competition

- **Competition**: March Machine Learning Mania 2026
- **Scoring**: Log-loss on predicted win probabilities for all possible matchup pairs
- **Submission format**: `SEASON_TEAM1ID_TEAM2ID, probability` (lower TeamID always first)
- **Stage 1**: Predict all possible matchups (submitted before tournament)
- **Stage 2**: Predict actual tournament games as they happen

Submission file: `outputs/submission_2026.csv`

---

## Roadmap / Next Steps

### High Priority
- [ ] Run `fetch_kenpom_extended.py` once Cloudflare ban clears (adds Height/Experience — big signal)
- [ ] Re-run `03_predict.py` after `04_tune.py` finishes to get tuned model predictions
- [ ] Run `fetch_warrennolan.py` for historical NET rankings

### Medium Priority
- [ ] Run `fetch_sports_reference.py` for SRS ratings (covers pre-2002, fills KenPom gaps)
- [ ] Add KenPom player stats → build roster quality index (minutes-weighted player efficiency)
- [ ] Add coaching experience feature (coach tenure, tournament experience)

### Stretch Goals
- [ ] Neural network layer (LSTM on game-by-game sequence) as additional ensemble member
- [ ] Live in-tournament updating (re-run predictions after each round with actual results)
- [ ] Upset detector: model specifically trained to identify 5v12, 6v11 upsets
- [ ] Women's tournament model (W* files in Kaggle data are untouched)

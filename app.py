"""
2026 NCAA Tournament Prediction Dashboard
Florida Gators themed — XGBoost + LightGBM + Logistic Regression ensemble
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="2026 NCAA Bracket Predictions",
    page_icon="https://a.espncdn.com/i/teamlogos/ncaa/500/57.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ORANGE    = "#FA4616"
BLUE      = "#0021A5"
DARK_BLUE = "#001580"
LIGHT_BG  = "#F4F6FF"
GREEN     = "#16a34a"
GREEN_BG  = "#f0fdf4"

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_seed(seed_str: str) -> str:
    """'W01' -> '1', 'X11a' -> '11a', 'Y16b' -> '16b'"""
    s = seed_str[1:]
    if s and s[-1] in "ab":
        num = s[:-1].lstrip("0") or "0"
        return num + s[-1]
    return s.lstrip("0") or "0"

REGION_NAMES = {"W": "East", "X": "South", "Y": "Midwest", "Z": "West"}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     Helvetica, Arial, sans-serif;
    }}

    /* ── Hero ── */
    .hero {{
        background: linear-gradient(135deg, {BLUE} 0%, {DARK_BLUE} 60%, {ORANGE} 100%);
        border-radius: 16px; padding: 32px 40px; margin-bottom: 24px;
        display: flex; align-items: center; gap: 28px;
        box-shadow: 0 8px 32px rgba(0,33,165,0.2);
    }}
    .hero-logo img {{ width:90px; height:90px; object-fit:contain;
        filter:drop-shadow(0 2px 8px rgba(0,0,0,0.3)); }}
    .hero-text h1 {{ color:white; font-size:2.1rem; font-weight:800;
        margin:0 0 4px; letter-spacing:-0.5px; }}
    .hero-text p  {{ color:rgba(255,255,255,0.82); font-size:0.92rem; margin:0; }}

    /* ── Stat cards ── */
    .stat-card {{
        background:white; border-radius:12px; padding:18px 22px;
        border-left:5px solid {ORANGE};
        box-shadow:0 2px 10px rgba(0,33,165,0.07);
    }}
    .stat-card .lbl {{ font-size:0.7rem; font-weight:700; color:#888;
        text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px; }}
    .stat-card .val {{ font-size:1.85rem; font-weight:800; color:{BLUE}; }}
    .stat-card .sub {{ font-size:0.78rem; color:#999; margin-top:2px; }}

    /* ── Gator accent cards ── */
    .gator-card {{
        background:linear-gradient(135deg, {BLUE}, {DARK_BLUE});
        border-radius:12px; padding:18px 22px; color:white;
        box-shadow:0 4px 16px rgba(0,33,165,0.22);
    }}
    .gator-card .lbl {{ font-size:0.7rem; font-weight:700;
        color:rgba(255,255,255,0.65); text-transform:uppercase;
        letter-spacing:0.8px; margin-bottom:4px; }}
    .gator-card .val {{ font-size:1.85rem; font-weight:800; color:{ORANGE}; }}
    .gator-card .sub {{ font-size:0.78rem; color:rgba(255,255,255,0.7); margin-top:2px; }}

    /* ── Section title ── */
    .stitle {{
        font-size:1.1rem; font-weight:800; color:{BLUE};
        border-bottom:3px solid {ORANGE};
        padding-bottom:6px; margin-bottom:16px; display:inline-block;
    }}

    /* ── Info banner ── */
    .info-banner {{
        background:{LIGHT_BG}; border-radius:10px;
        padding:14px 18px; margin-bottom:16px;
        border-left:4px solid {ORANGE};
        font-size:0.88rem;
    }}

    /* ── Bracket matchup card ── */
    .mc {{
        background:white; border-radius:8px;
        box-shadow:0 1px 6px rgba(0,33,165,0.07);
        margin-bottom:7px; overflow:hidden;
        border:1px solid #eaeef8;
    }}
    /* Winner row */
    .mc-win {{
        display:flex; align-items:center; padding:8px 12px; gap:10px;
        background:{GREEN_BG};
        border-left:4px solid {GREEN};
    }}
    /* Loser row */
    .mc-lose {{
        display:flex; align-items:center; padding:8px 12px; gap:10px;
        background:white;
        border-left:4px solid transparent;
        border-top:1px solid #f0f2f8;
    }}
    .mc-seed-win {{
        font-size:0.68rem; font-weight:800; color:white;
        background:{GREEN}; border-radius:4px;
        padding:2px 5px; min-width:22px; text-align:center; flex-shrink:0;
    }}
    .mc-seed-lose {{
        font-size:0.68rem; font-weight:700; color:#bbb;
        background:#f0f0f0; border-radius:4px;
        padding:2px 5px; min-width:22px; text-align:center; flex-shrink:0;
    }}
    .mc-name-win  {{ font-size:0.85rem; font-weight:700; color:#15803d; flex:1; }}
    .mc-name-lose {{ font-size:0.85rem; font-weight:500; color:#bbb; flex:1; }}
    .mc-name-gator-win  {{ color:{ORANGE} !important; }}
    .mc-name-gator-lose {{ color:{ORANGE} !important; opacity:0.7; }}
    .mc-prob-win  {{ font-size:0.85rem; font-weight:800; color:{GREEN}; min-width:36px; text-align:right; }}
    .mc-prob-lose {{ font-size:0.82rem; font-weight:500; color:#ccc; min-width:36px; text-align:right; }}
    .mc-wins-badge {{
        font-size:0.62rem; font-weight:800; color:white;
        background:{GREEN}; border-radius:3px;
        padding:1px 5px; letter-spacing:0.5px;
    }}
    .mc-bar-wrap {{ height:3px; background:#e8f5e9; }}
    .mc-bar {{ height:3px; background:{GREEN}; }}

    /* ── Region header ── */
    .region-hdr {{
        background:linear-gradient(90deg, {BLUE}, {DARK_BLUE});
        color:white; border-radius:8px 8px 0 0;
        padding:9px 14px; font-size:0.78rem; font-weight:800;
        text-transform:uppercase; letter-spacing:1px;
        display:flex; justify-content:space-between; align-items:center;
        margin-bottom:4px;
    }}
    .region-hdr .fav {{ color:{ORANGE}; font-size:0.72rem; font-weight:700; }}

    /* ── Matchup explorer boxes ── */
    .mx-box {{ border-radius:12px; padding:22px; text-align:center; }}
    .mx-win  {{ background:linear-gradient(135deg,{ORANGE},#FF7A50); color:white; }}
    .mx-neutral {{ background:{LIGHT_BG}; border:2px solid #dde; }}
    .mx-box .seed-lbl {{ font-size:0.75rem; font-weight:600; opacity:0.8; margin-bottom:4px; }}
    .mx-box h2 {{ font-size:1.75rem; font-weight:800; margin:0; }}
    .mx-box .pct {{ font-size:2.8rem; font-weight:900; margin:8px 0; }}
    .mx-win h2, .mx-win .pct {{ color:white; }}
    .mx-neutral h2, .mx-neutral .pct {{ color:{BLUE}; }}
    .mx-box .sub {{ font-size:0.83rem; opacity:0.8; }}

    /* ── Pool strategy EV cards ── */
    .ev-card {{
        background:white; border-radius:10px; padding:14px 16px;
        border-left:5px solid #ddd;
        box-shadow:0 1px 6px rgba(0,33,165,0.06);
        margin-bottom:8px;
        display:flex; align-items:center; gap:12px;
    }}
    .ev-card.ev-value  {{ border-left-color:{GREEN}; background:{GREEN_BG}; }}
    .ev-card.ev-fair   {{ border-left-color:{BLUE}; }}
    .ev-card.ev-over   {{ border-left-color:#e5e7eb; }}
    .ev-card .ev-team  {{ flex:1; font-size:0.88rem; font-weight:700; color:{BLUE}; }}
    .ev-card .ev-seed  {{ font-size:0.72rem; color:#999; font-weight:600; }}
    .ev-card .ev-ratio {{
        font-size:1rem; font-weight:800;
        min-width:48px; text-align:right;
    }}
    .ev-card.ev-value  .ev-ratio {{ color:{GREEN}; }}
    .ev-card.ev-fair   .ev-ratio {{ color:{BLUE}; }}
    .ev-card.ev-over   .ev-ratio {{ color:#ccc; }}
    .ev-card .ev-pcts  {{ font-size:0.72rem; color:#999; min-width:80px; text-align:right; }}

    /* ── TABS: underline navigation style ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: white;
        border-bottom: 2px solid #E0E4F5;
        padding: 0;
        border-radius: 0;
        margin-bottom: 20px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        border-radius: 0 !important;
        font-weight: 600;
        font-size: 0.92rem;
        color: #9099bb;
        padding: 14px 26px;
        border-bottom: 3px solid transparent;
        margin-bottom: -2px;
        letter-spacing: 0.1px;
    }}
    .stTabs [aria-selected="true"] {{
        background: transparent !important;
        color: {ORANGE} !important;
        border-bottom: 3px solid {ORANGE} !important;
        font-weight: 800 !important;
    }}

    /* ── Full bracket CSS ── */
    .fb-wrap {{ overflow-x: auto; padding: 4px 0; }}
    .fb-bracket {{ display: flex; gap: 0; min-width: 900px; }}
    .fb-col {{ display: flex; flex-direction: column; flex: 1; min-width: 170px; }}
    .fb-col-hdr {{ font-size: 0.68rem; font-weight: 800; color: {BLUE};
                  text-transform: uppercase; letter-spacing: 0.8px;
                  text-align: center; padding: 6px 4px; background: {LIGHT_BG};
                  border-radius: 6px; margin: 0 2px 6px; }}
    .fb-slot {{ display: flex; align-items: center; padding: 0 2px; box-sizing: border-box; }}
    .fb-mc {{ border-radius: 6px; border: 1px solid #eaeef8; overflow: hidden;
              width: 100%; background: white; box-shadow: 0 1px 4px rgba(0,33,165,0.05); }}
    .fb-row-w {{ display:flex; align-items:center; gap:5px; padding:4px 7px;
                 background: #f0fdf4; border-left: 3px solid #16a34a; }}
    .fb-row-l {{ display:flex; align-items:center; gap:5px; padding:4px 7px;
                 background: white; border-left: 3px solid transparent;
                 border-top: 1px solid #f0f2f8; }}
    .fb-badge-w {{ font-size:0.6rem; font-weight:800; color:white; background:#16a34a;
                   border-radius:3px; padding:1px 4px; min-width:16px; text-align:center; flex-shrink:0; }}
    .fb-badge-l {{ font-size:0.6rem; font-weight:700; color:#ccc; background:#f0f0f0;
                   border-radius:3px; padding:1px 4px; min-width:16px; text-align:center; flex-shrink:0; }}
    .fb-name-w {{ font-size:0.76rem; font-weight:700; color:#15803d; flex:1; white-space:nowrap;
                  overflow:hidden; text-overflow:ellipsis; }}
    .fb-name-w-gator {{ color: {ORANGE} !important; }}
    .fb-name-l {{ font-size:0.74rem; font-weight:400; color:#bbb; flex:1; white-space:nowrap;
                  overflow:hidden; text-overflow:ellipsis; }}
    .fb-prob {{ font-size:0.65rem; font-weight:700; color:#16a34a; white-space:nowrap; }}
    /* F4/Championship bracket */
    .ff-section {{ background: {LIGHT_BG}; border-radius: 12px; padding: 16px 20px; margin-top: 20px; }}
    .ff-game {{ background: white; border-radius: 8px; padding: 10px 14px;
                box-shadow: 0 1px 6px rgba(0,33,165,0.07); border: 1px solid #eaeef8; }}

    /* ── Key Factors CSS ── */
    .kf-row {{ display:flex; align-items:center; gap:10px; padding:8px 0;
               border-bottom: 1px solid #f0f2f8; }}
    .kf-row:last-child {{ border-bottom: none; }}
    .kf-label {{ font-size:0.8rem; color:#666; min-width:200px; }}
    .kf-val {{ font-size:0.82rem; font-weight:700; min-width:60px; text-align:right; }}
    .kf-bar-wrap {{ flex:1; height:8px; background:#f0f0f0; border-radius:4px; overflow:hidden; }}
    .kf-bar {{ height:8px; border-radius:4px; }}
    .kf-edge {{ font-size:0.72rem; font-weight:700; min-width:80px; }}

    /* Footer */
    .footer {{
        text-align:center; color:#bbb; font-size:0.77rem;
        padding:20px 0 8px; border-top:1px solid #eee; margin-top:36px;
    }}
    #MainMenu {{visibility:hidden;}} footer {{visibility:hidden;}} header {{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_data(ttl=0)
def load_data():
    round_df = pd.read_csv(BASE / "outputs" / "round_probs_2026.csv")
    sub_df   = pd.read_csv(BASE / "outputs" / "submission_2026.csv")
    fi_xgb   = pd.read_csv(BASE / "outputs" / "feature_importance_xgb.csv",
                            header=None, names=["feature","importance"])
    fi_lgb   = pd.read_csv(BASE / "outputs" / "feature_importance_lgb.csv",
                            header=None, names=["feature","importance"])
    stats_df     = pd.read_csv(BASE / "outputs" / "team_stats_2026.csv")
    champ_hist   = pd.read_csv(BASE / "outputs" / "champion_history.csv")
    upset_df     = pd.read_csv(BASE / "outputs" / "upset_rates.csv")
    champ_prof   = pd.read_csv(BASE / "outputs" / "champion_profile.csv")
    cv_preds     = pd.read_csv(BASE / "outputs" / "cv_predictions.csv")

    # Pre-computed historical tournament records (generated by src/compute_historical_records.py)
    hist_records_path = BASE / "outputs" / "historical_records.csv"
    if hist_records_path.exists():
        hist_records = pd.read_csv(hist_records_path)
    else:
        hist_records = pd.DataFrame()

    # KenPom extended data for player spotlight (pre-exported CSVs)
    try:
        height_exp = pd.read_csv(BASE / "outputs" / "kenpom_height_2026.csv")
        # Convert positional heights (stored as "+X.X" strings) to float
        for pos_col in ["Avg Hgt", "C Hgt", "PF Hgt", "SF Hgt", "SG Hgt", "PG Hgt"]:
            if pos_col in height_exp.columns:
                height_exp[pos_col] = height_exp[pos_col].apply(
                    lambda x: float(str(x).replace("+", "")) if pd.notna(x) else np.nan
                )
        height_exp["Experience"] = pd.to_numeric(height_exp["Experience"], errors="coerce")
    except Exception:
        height_exp = pd.DataFrame()

    try:
        misc_stats = pd.read_csv(BASE / "outputs" / "kenpom_misc_2026.csv")
        misc_stats["kp_3pt_pct"]  = pd.to_numeric(misc_stats["3P%"],  errors="coerce")
        misc_stats["kp_2pt_pct"]  = pd.to_numeric(misc_stats["FT%"],  errors="coerce")
        misc_stats["kp_3pa_rate"] = pd.to_numeric(misc_stats["3PA%"], errors="coerce")
        import re as _re
        misc_stats["TeamClean"] = misc_stats["Team"].str.replace(r"\s*\d+$", "", regex=True).str.strip()
    except Exception:
        misc_stats = pd.DataFrame()

    for fi in [fi_xgb, fi_lgb]:
        fi.drop(fi[fi["feature"].isna() | (fi["feature"] == "")].index, inplace=True)
        fi["importance"] = pd.to_numeric(fi["importance"], errors="coerce")
        fi.dropna(inplace=True)
        fi.sort_values("importance", ascending=False, inplace=True)

    prob_lookup = {}
    for _, row in sub_df.iterrows():
        _, t1, t2 = row["ID"].split("_")
        prob_lookup[(int(t1), int(t2))] = float(row["Pred"])

    return (round_df, fi_xgb, fi_lgb, prob_lookup, stats_df, champ_hist, upset_df,
            champ_prof, cv_preds, hist_records, height_exp, misc_stats)

(round_df, fi_xgb, fi_lgb, prob_lookup, stats_df, champ_hist, upset_df,
 champ_prof, cv_preds, hist_records, height_exp, misc_stats) = load_data()

round_df["SeedDisplay"] = round_df["Seed"].apply(fmt_seed)
round_df["SeedNum"]     = pd.to_numeric(round_df["SeedNum"], errors="coerce").fillna(17).astype(int)

seed_to_id        = dict(zip(round_df["Seed"], round_df["TeamID"]))
id_to_name        = dict(zip(round_df["TeamID"], round_df["TeamName"]))
id_to_seeddisplay = dict(zip(round_df["TeamID"], round_df["SeedDisplay"]))
id_to_seednum     = dict(zip(round_df["TeamID"], round_df["SeedNum"]))
team_names_sorted = sorted(round_df["TeamName"].tolist())

def win_prob(id_a: int, id_b: int) -> float:
    key = (min(id_a, id_b), max(id_a, id_b))
    p = prob_lookup.get(key, 0.5)
    return p if id_a < id_b else 1 - p

def team_row(name: str) -> pd.Series:
    return round_df[round_df["TeamName"] == name].iloc[0]

# ── Feature labels ────────────────────────────────────────────────────────────
FEAT_LABELS = {
    "d_elo_pre_tourney": "Pre-Tournament Elo Gap",
    "d_adjEM": "Adj. Efficiency Margin Gap",
    "d_AvgScoreDiff": "Avg Score Margin Gap",
    "d_avg_ScoreDiff": "Avg Score Margin Gap (wtd)",
    "d_elo_last10": "Elo — Last 10 Games",
    "d_sos_adjEM": "Strength of Schedule Gap",
    "t1_elo_pre_tourney": "Team Elo Rating",
    "d_elo_peak": "Peak Elo Gap",
    "d_qual_games": "Quality Games Gap",
    "d_opp_D": "Opp Defensive Efficiency",
    "d_rank_composite": "Composite Ranking Gap",
    "d_WinPct": "Win Percentage Gap",
    "d_wab": "Wins Above Bubble Gap",
    "d_elo_momentum": "Late-Season Elo Momentum",
    "d_luck": "Luck Factor Gap",
    "d_elo_consistency": "Elo Consistency",
    "d_barthag": "BARTHAG Power Rating Gap",
    "d_adjO": "Adj. Offensive Efficiency Gap",
    "d_adjD": "Adj. Defensive Efficiency Gap",
    "t1_adjO": "Team Adj. Offense",
    "t1_SeedNum": "Team Seed",
    "t1_avg_ScoreDiff": "Team Avg Score Margin",
    "t1_elo_late_winpct": "Team Late-Season Win %",
    "t2_elo_pre_tourney": "Opponent Elo Rating",
}
def feat_label(f):
    return FEAT_LABELS.get(f,
        f.replace("d_","").replace("t1_","").replace("t2_","")
         .replace("_"," ").title())

# ── Key Factor stat labels ─────────────────────────────────────────────────────
STAT_LABELS = {
    "elo_pre_tourney": "Elo Rating",
    "adjEM": "KenPom Efficiency Margin",
    "adjO": "Offensive Efficiency",
    "adjD": "Defensive Efficiency (lower=better)",
    "avg_ScoreDiff": "Avg Score Margin",
    "WinPct": "Win Percentage",
    "barthag": "Power Rating (Torvik)",
    "sos_adjEM": "Strength of Schedule",
}

# ── Round-specific public pick priors (seed-based, based on historical bracket data) ──
F4_PRIOR = {
    1:0.70, 2:0.50, 3:0.32, 4:0.20, 5:0.14, 6:0.09, 7:0.07, 8:0.05,
    9:0.04, 10:0.03, 11:0.03, 12:0.025, 13:0.012, 14:0.006, 15:0.003, 16:0.001,
}
NCG_PRIOR = {
    1:0.38, 2:0.18, 3:0.11, 4:0.07, 5:0.05, 6:0.03, 7:0.025, 8:0.015,
    9:0.012, 10:0.010, 11:0.009, 12:0.007, 13:0.003, 14:0.002, 15:0.001, 16:0.0005,
}
CHAMP_PRIOR = {
    1:0.24, 2:0.12, 3:0.07, 4:0.05, 5:0.03, 6:0.02, 7:0.015, 8:0.010,
    9:0.008, 10:0.007, 11:0.006, 12:0.005, 13:0.003, 14:0.002, 15:0.001, 16:0.0005,
}

def prior(seed_num: int, priors: dict) -> float:
    return priors.get(min(seed_num, 16), 0.001)

def build_ev_df():
    rows = []
    for _, r in round_df.iterrows():
        sn = int(r["SeedNum"])
        rows.append({
            "Seed":     r["SeedDisplay"],
            "Team":     r["TeamName"],
            "SeedNum":  sn,
            "F4_model":    round(r["prob_F4"] * 100, 1),
            "F4_public":   round(prior(sn, F4_PRIOR) * 100, 1),
            "F4_ev":       round(r["prob_F4"] / prior(sn, F4_PRIOR), 2) if prior(sn, F4_PRIOR) > 0 else 0,
            "NCG_model":   round(r["prob_NCG"] * 100, 1),
            "NCG_public":  round(prior(sn, NCG_PRIOR) * 100, 1),
            "NCG_ev":      round(r["prob_NCG"] / prior(sn, NCG_PRIOR), 2) if prior(sn, NCG_PRIOR) > 0 else 0,
            "Champ_model":  round(r["prob_Champion"] * 100, 1),
            "Champ_public": round(prior(sn, CHAMP_PRIOR) * 100, 1),
            "Champ_ev":     round(r["prob_Champion"] / prior(sn, CHAMP_PRIOR), 2) if prior(sn, CHAMP_PRIOR) > 0 else 0,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-logo">
        <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/57.png"
             onerror="this.style.display='none'" />
    </div>
    <div class="hero-text">
        <h1>2026 NCAA Tournament Predictions</h1>
        <p>ML ensemble &nbsp;·&nbsp; XGBoost + LightGBM + Logistic Regression
           &nbsp;·&nbsp; 100,000 bracket simulations</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Florida spotlight
gator = round_df[round_df["TeamName"] == "Florida"]
if not gator.empty:
    g = gator.iloc[0]
    st.markdown('<div class="stitle">Florida Gators — Tournament Outlook</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val, sub) in zip(
        [c1, c2, c3, c4],
        [("Seed", g["SeedDisplay"], "South Region"),
         ("Final Four", f"{g['prob_F4']*100:.1f}%", "Probability of reaching F4"),
         ("Champ. Game", f"{g['prob_NCG']*100:.1f}%", "Probability of reaching NCG"),
         ("National Champ.", f"{g['prob_Champion']*100:.1f}%", "Probability of winning it all")]
    ):
        with col:
            st.markdown(f"""
            <div class="gator-card">
                <div class="lbl">{lbl}</div><div class="val">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)
    st.write("")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_odds, tab_bracket, tab_matchup, tab_pool, tab_dna, tab_upset, tab_model, tab_optimizer, tab_deepdive, tab_hot, tab_calibration, tab_spotlight = st.tabs([
    "Championship Odds",
    "Full Bracket",
    "Matchup Explorer",
    "Pool Strategy",
    "Championship DNA",
    "Upset Picker",
    "Model Insights",
    "Bracket Optimizer",
    "Team Deep-Dive",
    "Hot Streaks & Busters",
    "Model Calibration",
    "Player Spotlight",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Championship Odds
# ════════════════════════════════════════════════════════════════════════════
with tab_odds:
    st.markdown('<div class="stitle">Championship Probability Rankings</div>',
                unsafe_allow_html=True)

    disp = round_df.sort_values("prob_Champion", ascending=False).reset_index(drop=True)

    # Top 5 cards
    cols5 = st.columns(5)
    medals = ["1st", "2nd", "3rd", "4th", "5th"]
    for i, (_, row) in enumerate(disp.head(5).iterrows()):
        with cols5[i]:
            cls = "gator-card" if row["TeamName"] == "Florida" else "stat-card"
            st.markdown(f"""
            <div class="{cls}">
                <div class="lbl">{medals[i]} &nbsp;·&nbsp; Seed {row['SeedDisplay']}</div>
                <div class="val">{row['prob_Champion']*100:.1f}%</div>
                <div class="sub">{row['TeamName']}</div>
            </div>""", unsafe_allow_html=True)
    st.write("")

    # Bar chart — top 20
    top20 = disp.head(20)
    labels = [f"({r['SeedDisplay']}) {r['TeamName']}" for _,r in top20.iterrows()]
    colors = [ORANGE if t == "Florida" else BLUE for t in top20["TeamName"]]

    fig = go.Figure(go.Bar(
        x=labels, y=top20["prob_Champion"]*100,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in top20["prob_Champion"]*100],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Championship: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Top 20 Championship Probabilities", font=dict(color=BLUE, size=15)),
        xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
        yaxis=dict(title="Championship %", gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, margin=dict(t=50,b=90,l=40,r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full table — all rounds with sort controls
    st.markdown('<div class="stitle">All Teams — Round-by-Round Probabilities</div>',
                unsafe_allow_html=True)

    sort_col_options = ["Champion", "Champ. Game", "Final Four", "Elite 8", "Sweet 16", "Round of 32"]
    sc1, sc2, _ = st.columns([2, 2, 4])
    with sc1:
        sort_col = st.selectbox("Sort by", sort_col_options, key="odds_sort_col")
    with sc2:
        sort_dir = st.selectbox("Direction", ["High → Low", "Low → High"], key="odds_sort_dir")

    tbl = disp[["SeedDisplay","TeamName",
                "prob_R32","prob_S16","prob_E8","prob_F4","prob_NCG","prob_Champion"]].copy()
    tbl.columns = ["Seed","Team","Round of 32","Sweet 16","Elite 8",
                   "Final Four","Champ. Game","Champion"]
    for c in tbl.columns[2:]:
        tbl[c] = (tbl[c] * 100).round(1)

    ascending = (sort_dir == "Low → High")
    tbl = tbl.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    def hl_gator(row):
        s = f"background-color:{BLUE};color:white;font-weight:bold"
        return [s]*len(row) if row["Team"] == "Florida" else [""]*len(row)

    fmt = {c: "{:.1f}%" for c in tbl.columns[2:]}
    st.dataframe(
        tbl.style.apply(hl_gator, axis=1).format(fmt),
        use_container_width=True, height=480,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Full Bracket
# ════════════════════════════════════════════════════════════════════════════
with tab_bracket:

    BRACKET_PAIRS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

    # Detect First Four slots
    first_four = {}
    for seed_code, tid in seed_to_id.items():
        if seed_code and seed_code[-1] in "ab":
            first_four.setdefault(seed_code[:-1], []).append((seed_code, tid))

    def weighted_prob(id_high, id_a, id_b):
        """P(id_high wins R1) weighted over both possible play-in outcomes."""
        p_a = win_prob(id_a, id_b)
        return p_a * win_prob(id_high, id_a) + (1 - p_a) * win_prob(id_high, id_b)

    def matchup_html(id_top, id_bot, ff_id_a=None, ff_id_b=None):
        """Return HTML for one first-round matchup card."""
        if ff_id_a is not None and ff_id_b is not None:
            p = win_prob(ff_id_a, ff_id_b) * 100
            n_a = id_to_name.get(ff_id_a, "TBD")
            n_b = id_to_name.get(ff_id_b, "TBD")
            s_a = id_to_seeddisplay.get(ff_id_a, "?")
            s_b = id_to_seeddisplay.get(ff_id_b, "?")
            a_wins = p >= 50
            w_id, l_id, p_w, p_l = (ff_id_a, ff_id_b, p, 100-p) if a_wins else (ff_id_b, ff_id_a, 100-p, p)
            nw = id_to_name.get(w_id,"TBD"); nl = id_to_name.get(l_id,"TBD")
            sw = id_to_seeddisplay.get(w_id,"?"); sl = id_to_seeddisplay.get(l_id,"?")
            gw = "mc-name-gator-win" if nw=="Florida" else ""
            gl = "mc-name-gator-lose" if nl=="Florida" else ""
            return f"""
            <div class="mc">
              <div class="mc-win">
                <span class="mc-seed-win">{sw}</span>
                <span class="mc-name-win {gw}">{nw}</span>
                <span class="mc-wins-badge">WINS</span>
                <span class="mc-prob-win">{p_w:.0f}%</span>
              </div>
              <div class="mc-bar-wrap"><div class="mc-bar" style="width:{p_w:.1f}%"></div></div>
              <div class="mc-lose">
                <span class="mc-seed-lose">{sl}</span>
                <span class="mc-name-lose {gl}">{nl}</span>
                <span class="mc-prob-lose">{p_l:.0f}%</span>
              </div>
            </div>"""

        if id_top is None or id_bot is None:
            return ""

        p = win_prob(id_top, id_bot) * 100
        top_wins = p >= 50
        w_id, l_id = (id_top, id_bot) if top_wins else (id_bot, id_top)
        p_w, p_l   = (p, 100-p) if top_wins else (100-p, p)
        nw = id_to_name.get(w_id,"TBD"); nl = id_to_name.get(l_id,"TBD")
        sw = id_to_seeddisplay.get(w_id,"?"); sl = id_to_seeddisplay.get(l_id,"?")
        gw = "mc-name-gator-win"  if nw=="Florida" else ""
        gl = "mc-name-gator-lose" if nl=="Florida" else ""
        return f"""
        <div class="mc">
          <div class="mc-win">
            <span class="mc-seed-win">{sw}</span>
            <span class="mc-name-win {gw}">{nw}</span>
            <span class="mc-wins-badge">WINS</span>
            <span class="mc-prob-win">{p_w:.0f}%</span>
          </div>
          <div class="mc-bar-wrap"><div class="mc-bar" style="width:{p_w:.1f}%"></div></div>
          <div class="mc-lose">
            <span class="mc-seed-lose">{sl}</span>
            <span class="mc-name-lose {gl}">{nl}</span>
            <span class="mc-prob-lose">{p_l:.0f}%</span>
          </div>
        </div>"""

    def render_region(letter: str) -> str:
        name = REGION_NAMES[letter]
        reg_teams = round_df[round_df["Seed"].str.startswith(letter)]
        fav = reg_teams.loc[reg_teams["prob_F4"].idxmax()] if not reg_teams.empty else None
        fav_str = f"Model favorite: {fav['TeamName']} ({fav['prob_F4']*100:.0f}% F4)" if fav is not None else ""

        html = f"""
        <div style="margin-bottom:16px;">
          <div class="region-hdr">
            <span>{name} Region</span>
            <span class="fav">{fav_str}</span>
          </div>"""

        for high_s, low_s in BRACKET_PAIRS:
            low_code  = f"{letter}{str(low_s).zfill(2)}"
            high_code = f"{letter}{str(high_s).zfill(2)}"

            if low_code in first_four:
                pair = first_four[low_code]
                id_a, id_b = pair[0][1], pair[1][1]
                id_high = seed_to_id.get(high_code)

                html += f"""
                <div style="font-size:0.68rem;font-weight:700;color:#aaa;
                    text-transform:uppercase;letter-spacing:0.6px;
                    margin:6px 0 3px 2px;">Play-In Game</div>"""
                html += matchup_html(None, None, ff_id_a=id_a, ff_id_b=id_b)

                if id_high:
                    p_a = win_prob(id_a, id_b)
                    likely_opp = id_a if p_a >= 0.5 else id_b
                    p_high = weighted_prob(id_high, id_a, id_b)
                    html += f"""
                    <div style="font-size:0.68rem;font-weight:700;color:#aaa;
                        text-transform:uppercase;letter-spacing:0.6px;
                        margin:6px 0 3px 2px;">Round 1 (projected)</div>"""
                    html += matchup_html(id_high, likely_opp)
            else:
                id_high = seed_to_id.get(high_code)
                id_low  = seed_to_id.get(low_code)
                html += matchup_html(id_high, id_low)

        html += "</div>"
        return html

    # ── build_predicted_bracket ──────────────────────────────────────────────
    def build_predicted_bracket():
        """
        Returns a dict keyed by region letter, each containing:
          {
            'r64':  [(winner_id, loser_id, prob_winner), ...],  # 8 games
            'r32':  [(winner_id, loser_id, prob_winner), ...],  # 4 games
            's16':  [(winner_id, loser_id, prob_winner), ...],  # 2 games
            'e8':   [(winner_id, loser_id, prob_winner), ...],  # 1 game
            'winner': winner_id
          }
        Also returns:
          'f4':   [(winner_id, loser_id, prob_winner), ...]  # 2 games (W vs X, Y vs Z)
          'ncg':  [(winner_id, loser_id, prob_winner), ...]  # 1 game
          'champion': winner_id
        """
        SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        bracket_order = [(SEED_ORDER[i], SEED_ORDER[i+1]) for i in range(0, 16, 2)]

        result = {}

        for letter in ["W", "X", "Y", "Z"]:
            # Resolve First Four — pick winner with win_prob > 0.5
            resolved = {}  # base_code -> team_id
            for base_code, pair in first_four.items():
                if base_code[0] == letter:
                    id_a, id_b = pair[0][1], pair[1][1]
                    p = win_prob(id_a, id_b)
                    resolved[base_code] = id_a if p >= 0.5 else id_b

            # Build 16-team lineup in bracket order
            teams_16 = []
            for seed_num in SEED_ORDER:
                code = f"{letter}{str(seed_num).zfill(2)}"
                if code in first_four:
                    # Use resolved First Four winner
                    teams_16.append(resolved.get(code))
                else:
                    teams_16.append(seed_to_id.get(code))

            # Play R64 — 8 matchups
            r64 = []
            r32_field = []
            for i in range(0, 16, 2):
                t1, t2 = teams_16[i], teams_16[i+1]
                if t1 is None or t2 is None:
                    winner = t1 if t2 is None else t2
                    r64.append((winner, None, 1.0))
                    r32_field.append(winner)
                else:
                    p = win_prob(t1, t2)
                    if p >= 0.5:
                        r64.append((t1, t2, p))
                        r32_field.append(t1)
                    else:
                        r64.append((t2, t1, 1-p))
                        r32_field.append(t2)

            # Play R32 — 4 matchups
            r32 = []
            s16_field = []
            for i in range(0, 8, 2):
                t1, t2 = r32_field[i], r32_field[i+1]
                if t1 is None or t2 is None:
                    winner = t1 if t2 is None else t2
                    r32.append((winner, None, 1.0))
                    s16_field.append(winner)
                else:
                    p = win_prob(t1, t2)
                    if p >= 0.5:
                        r32.append((t1, t2, p))
                        s16_field.append(t1)
                    else:
                        r32.append((t2, t1, 1-p))
                        s16_field.append(t2)

            # Play S16 — 2 matchups
            s16 = []
            e8_field = []
            for i in range(0, 4, 2):
                t1, t2 = s16_field[i], s16_field[i+1]
                if t1 is None or t2 is None:
                    winner = t1 if t2 is None else t2
                    s16.append((winner, None, 1.0))
                    e8_field.append(winner)
                else:
                    p = win_prob(t1, t2)
                    if p >= 0.5:
                        s16.append((t1, t2, p))
                        e8_field.append(t1)
                    else:
                        s16.append((t2, t1, 1-p))
                        e8_field.append(t2)

            # Play E8 — 1 matchup
            t1, t2 = e8_field[0], e8_field[1]
            if t1 is None or t2 is None:
                reg_winner = t1 if t2 is None else t2
                e8 = [(reg_winner, None, 1.0)]
            else:
                p = win_prob(t1, t2)
                if p >= 0.5:
                    e8 = [(t1, t2, p)]
                    reg_winner = t1
                else:
                    e8 = [(t2, t1, 1-p)]
                    reg_winner = t2

            result[letter] = {
                "r64": r64,
                "r32": r32,
                "s16": s16,
                "e8":  e8,
                "winner": reg_winner,
            }

        # Final Four: W vs X, Y vs Z
        f4 = []
        f4_winners = []
        for l1, l2 in [("W", "X"), ("Y", "Z")]:
            t1 = result[l1]["winner"]
            t2 = result[l2]["winner"]
            if t1 is None or t2 is None:
                fw = t1 if t2 is None else t2
                f4.append((fw, None, 1.0))
                f4_winners.append(fw)
            else:
                p = win_prob(t1, t2)
                if p >= 0.5:
                    f4.append((t1, t2, p))
                    f4_winners.append(t1)
                else:
                    f4.append((t2, t1, 1-p))
                    f4_winners.append(t2)

        # Championship
        t1, t2 = f4_winners[0], f4_winners[1]
        if t1 is None or t2 is None:
            champ = t1 if t2 is None else t2
            ncg = [(champ, None, 1.0)]
        else:
            p = win_prob(t1, t2)
            if p >= 0.5:
                ncg = [(t1, t2, p)]
                champ = t1
            else:
                ncg = [(t2, t1, 1-p)]
                champ = t2

        result["f4"] = f4
        result["ncg"] = ncg
        result["champion"] = champ
        return result

    def fb_matchup_card(winner_id, loser_id, prob_winner):
        """Render a compact full-bracket matchup card."""
        nw = id_to_name.get(winner_id, "TBD") if winner_id else "TBD"
        nl = id_to_name.get(loser_id, "TBD") if loser_id else "TBD"
        sw = id_to_seeddisplay.get(winner_id, "?") if winner_id else "?"
        sl = id_to_seeddisplay.get(loser_id, "?") if loser_id else "?"
        gator_cls = "fb-name-w-gator" if nw == "Florida" else ""
        prob_str = f"{prob_winner*100:.0f}%"
        return f"""<div class="fb-mc">
  <div class="fb-row-w">
    <span class="fb-badge-w">{sw}</span>
    <span class="fb-name-w {gator_cls}">{nw}</span>
    <span class="fb-prob">{prob_str}</span>
  </div>
  <div class="fb-row-l">
    <span class="fb-badge-l">{sl}</span>
    <span class="fb-name-l">{nl}</span>
  </div>
</div>"""

    def render_full_bracket_region(letter: str, region_data: dict) -> str:
        """Render one region as a 4-column HTML bracket with vertical spacing."""
        BASE_H = 58
        GAP    = 4

        round_keys   = ["r64", "r32", "s16", "e8"]
        round_labels = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
        # slot heights for each round column
        slot_heights = [
            BASE_H,
            2 * BASE_H + GAP,
            4 * BASE_H + 3 * GAP,
            8 * BASE_H + 7 * GAP,
        ]

        cols_html = ""
        for col_idx, (rkey, rlabel, slot_h) in enumerate(
            zip(round_keys, round_labels, slot_heights)
        ):
            games = region_data[rkey]
            slots_html = ""
            for w_id, l_id, prob in games:
                card = fb_matchup_card(w_id, l_id, prob)
                slots_html += (
                    f'<div class="fb-slot" style="height:{slot_h}px;">'
                    f'{card}'
                    f'</div>'
                )
            cols_html += (
                f'<div class="fb-col">'
                f'<div class="fb-col-hdr">{rlabel}</div>'
                f'{slots_html}'
                f'</div>'
            )

        region_name = REGION_NAMES[letter]
        winner_id = region_data["winner"]
        winner_name = id_to_name.get(winner_id, "TBD") if winner_id else "TBD"
        winner_seed = id_to_seeddisplay.get(winner_id, "?") if winner_id else "?"

        return f"""
<div style="margin-bottom:8px;">
  <div class="region-hdr" style="border-radius:8px;margin-bottom:6px;">
    <span>{region_name} Region</span>
    <span class="fav">Elite 8 Winner: ({winner_seed}) {winner_name}</span>
  </div>
  <div class="fb-wrap">
    <div class="fb-bracket">{cols_html}</div>
  </div>
</div>"""

    def render_ff_game_html(w_id, l_id, prob, label):
        """Render a Final Four or Championship game card."""
        nw = id_to_name.get(w_id, "TBD") if w_id else "TBD"
        nl = id_to_name.get(l_id, "TBD") if l_id else "TBD"
        sw = id_to_seeddisplay.get(w_id, "?") if w_id else "?"
        sl = id_to_seeddisplay.get(l_id, "?") if l_id else "?"
        gator_w = f"color:{ORANGE};" if nw == "Florida" else ""
        gator_l = f"color:{ORANGE};opacity:0.7;" if nl == "Florida" else ""
        prob_str = f"{prob*100:.0f}%"
        return f"""
<div style="margin-bottom:8px;">
  <div style="font-size:0.65rem;font-weight:700;color:#999;text-transform:uppercase;
              letter-spacing:0.6px;margin-bottom:4px;">{label}</div>
  <div class="ff-game">
    <div style="display:flex;align-items:center;gap:8px;padding:4px 0;">
      <span style="font-size:0.65rem;font-weight:800;color:white;background:{GREEN};
                   border-radius:3px;padding:1px 5px;min-width:18px;text-align:center;">{sw}</span>
      <span style="font-size:0.82rem;font-weight:700;color:#15803d;flex:1;{gator_w}">{nw}</span>
      <span style="font-size:0.72rem;font-weight:800;color:{GREEN};">{prob_str}</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;padding:4px 0;
                border-top:1px solid #f0f2f8;">
      <span style="font-size:0.65rem;font-weight:700;color:#ccc;background:#f0f0f0;
                   border-radius:3px;padding:1px 5px;min-width:18px;text-align:center;">{sl}</span>
      <span style="font-size:0.82rem;font-weight:400;color:#bbb;flex:1;{gator_l}">{nl}</span>
    </div>
  </div>
</div>"""

    # ── First Four section ───────────────────────────────────────────────────
    # Completed play-in results (updated as games are played)
    FF_COMPLETED = {
        "Y16": {"winner": "Howard",   "w_score": 86, "loser": "UMBC",     "l_score": 83, "date": "Mar 17"},
        "Z11": {"winner": "Texas",    "w_score": 68, "loser": "NC State",  "l_score": 66, "date": "Mar 17"},
    }

    def render_ff_result_card(result, region_name, seed_num):
        """Render a completed First Four result card."""
        return f"""
<div style="background:white;border-radius:10px;border:1px solid #e2e8f0;
            box-shadow:0 1px 4px rgba(0,33,165,0.06);padding:10px 12px;margin-bottom:4px;">
  <div style="font-size:0.65rem;font-weight:700;color:{BLUE};text-transform:uppercase;
              letter-spacing:0.8px;margin-bottom:6px;">
    {region_name} · Seed {seed_num} · <span style="color:{GREEN};">FINAL — {result['date']}</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;padding:5px 0;">
    <span style="font-size:0.65rem;font-weight:800;color:white;background:{GREEN};
                 border-radius:3px;padding:1px 5px;min-width:18px;text-align:center;">{seed_num}</span>
    <span style="font-size:0.85rem;font-weight:700;color:#15803d;flex:1;">{result['winner']}</span>
    <span style="font-size:0.85rem;font-weight:800;color:#15803d;">{result['w_score']}</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-top:1px solid #f0f2f8;">
    <span style="font-size:0.65rem;font-weight:700;color:#ccc;background:#f0f0f0;
                 border-radius:3px;padding:1px 5px;min-width:18px;text-align:center;">{seed_num}</span>
    <span style="font-size:0.85rem;font-weight:400;color:#9ca3af;flex:1;text-decoration:line-through;">{result['loser']}</span>
    <span style="font-size:0.85rem;font-weight:600;color:#9ca3af;">{result['l_score']}</span>
  </div>
</div>"""

    ff_items = sorted(first_four.items())
    all_ff_bases = sorted(list(FF_COMPLETED.keys()) + [b for b, _ in ff_items])
    if all_ff_bases:
        st.markdown('<div class="stitle">First Four — Play-In Games</div>',
                    unsafe_allow_html=True)
        ncols = len(all_ff_bases)
        ff_cols = st.columns(ncols)
        col_idx = 0
        # Show completed results first
        for base in sorted(FF_COMPLETED.keys()):
            with ff_cols[col_idx]:
                res = FF_COMPLETED[base]
                region_name = REGION_NAMES.get(base[0], "")
                seed_num = base[1:].lstrip("0")
                st.markdown(render_ff_result_card(res, region_name, seed_num),
                            unsafe_allow_html=True)
            col_idx += 1
        # Show pending play-ins
        for base, pair in ff_items:
            with ff_cols[col_idx]:
                id_a, id_b = pair[0][1], pair[1][1]
                region_name = REGION_NAMES.get(base[0], "")
                seed_num    = base[1:].lstrip("0")
                st.markdown(f"""
                <div style="font-size:0.7rem;font-weight:700;color:{BLUE};
                    text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">
                  {region_name} · Seed {seed_num} · <span style="color:#f59e0b;">Tonight</span>
                </div>
                {matchup_html(None, None, ff_id_a=id_a, ff_id_b=id_b)}
                """, unsafe_allow_html=True)
            col_idx += 1

    st.write("")
    st.markdown('<div class="stitle">First Round by Region</div>', unsafe_allow_html=True)
    st.caption("Green = model's predicted winner. Probability shown for the predicted winning team.")

    col_e, col_s = st.columns(2)
    col_m, col_w = st.columns(2)
    with col_e: st.markdown(render_region("W"), unsafe_allow_html=True)
    with col_s: st.markdown(render_region("X"), unsafe_allow_html=True)
    with col_m: st.markdown(render_region("Y"), unsafe_allow_html=True)
    with col_w: st.markdown(render_region("Z"), unsafe_allow_html=True)

    # ── Full predicted bracket ───────────────────────────────────────────────
    st.write("")
    st.markdown('<div class="stitle">Model\'s Predicted Full Bracket</div>',
                unsafe_allow_html=True)
    st.caption("Deterministic bracket — model always picks the higher-probability team in each round.")

    bracket_data = build_predicted_bracket()

    # Top row: East (W) and South (X)
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.markdown(render_full_bracket_region("W", bracket_data["W"]),
                    unsafe_allow_html=True)
    with row1_c2:
        st.markdown(render_full_bracket_region("X", bracket_data["X"]),
                    unsafe_allow_html=True)

    # Bottom row: Midwest (Y) and West (Z)
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.markdown(render_full_bracket_region("Y", bracket_data["Y"]),
                    unsafe_allow_html=True)
    with row2_c2:
        st.markdown(render_full_bracket_region("Z", bracket_data["Z"]),
                    unsafe_allow_html=True)

    # Final Four and Championship
    st.markdown('<div class="ff-section">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:1rem;font-weight:800;color:{BLUE};margin-bottom:12px;">Final Four &amp; Championship</div>',
                unsafe_allow_html=True)

    ff_c1, ff_c2, ff_c3 = st.columns([2, 2, 2])

    f4_games = bracket_data["f4"]
    ncg_games = bracket_data["ncg"]
    champ_id  = bracket_data["champion"]

    with ff_c1:
        if len(f4_games) > 0:
            w, l, p = f4_games[0]
            st.markdown(render_ff_game_html(w, l, p, "Semifinal 1 — East vs South"),
                        unsafe_allow_html=True)
    with ff_c2:
        if len(f4_games) > 1:
            w, l, p = f4_games[1]
            st.markdown(render_ff_game_html(w, l, p, "Semifinal 2 — Midwest vs West"),
                        unsafe_allow_html=True)
    with ff_c3:
        if ncg_games:
            w, l, p = ncg_games[0]
            st.markdown(render_ff_game_html(w, l, p, "National Championship"),
                        unsafe_allow_html=True)
            champ_name = id_to_name.get(champ_id, "TBD") if champ_id else "TBD"
            champ_seed = id_to_seeddisplay.get(champ_id, "?") if champ_id else "?"
            gator_style = f"color:{ORANGE};" if champ_name == "Florida" else "color:white;"
            st.markdown(f"""
<div style="text-align:center;margin-top:10px;padding:10px;
            background:linear-gradient(135deg,{BLUE},{DARK_BLUE});
            border-radius:8px;">
  <div style="font-size:0.65rem;font-weight:700;color:rgba(255,255,255,0.7);
              text-transform:uppercase;letter-spacing:0.8px;">Model's Predicted Champion</div>
  <div style="font-size:1.1rem;font-weight:900;{gator_style}margin-top:4px;">
    ({champ_seed}) {champ_name}
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Matchup Explorer
# ════════════════════════════════════════════════════════════════════════════
with tab_matchup:
    st.markdown('<div class="stitle">Head-to-Head Matchup Explorer</div>',
                unsafe_allow_html=True)
    st.caption("Select any two tournament teams to see win probability and bracket path comparison.")

    c_l, c_r = st.columns(2)
    with c_l:
        default_a = "Florida" if "Florida" in team_names_sorted else team_names_sorted[0]
        team_a = st.selectbox("Team A", team_names_sorted,
                              index=team_names_sorted.index(default_a), key="ta")
    with c_r:
        default_b = "Duke" if "Duke" in team_names_sorted else team_names_sorted[1]
        team_b = st.selectbox("Team B", team_names_sorted,
                              index=team_names_sorted.index(default_b), key="tb")

    if team_a == team_b:
        st.warning("Select two different teams.")
    else:
        ra, rb = team_row(team_a), team_row(team_b)
        p_a = win_prob(int(ra["TeamID"]), int(rb["TeamID"]))
        p_b = 1 - p_a

        st.write("")
        ca, cv, cb = st.columns([5,1,5])
        with ca:
            cls = "mx-box mx-win" if p_a > 0.5 else "mx-box mx-neutral"
            st.markdown(f"""
            <div class="{cls}">
              <div class="seed-lbl">
                Seed {ra['SeedDisplay']} · {REGION_NAMES.get(ra['Seed'][0],'')}
              </div>
              <h2>{team_a}</h2>
              <div class="pct">{p_a*100:.1f}%</div>
              <div class="sub">win probability</div>
            </div>""", unsafe_allow_html=True)
        with cv:
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:170px;font-size:1.2rem;font-weight:800;color:#ccc;">vs</div>
            """, unsafe_allow_html=True)
        with cb:
            cls = "mx-box mx-win" if p_b > 0.5 else "mx-box mx-neutral"
            st.markdown(f"""
            <div class="{cls}">
              <div class="seed-lbl">
                Seed {rb['SeedDisplay']} · {REGION_NAMES.get(rb['Seed'][0],'')}
              </div>
              <h2>{team_b}</h2>
              <div class="pct">{p_b*100:.1f}%</div>
              <div class="sub">win probability</div>
            </div>""", unsafe_allow_html=True)

        st.write("")

        fig_bar = go.Figure(go.Bar(
            x=[p_a*100, p_b*100], y=[team_a, team_b], orientation="h",
            marker_color=[ORANGE if team_a=="Florida" else BLUE,
                          ORANGE if team_b=="Florida" else "#5566CC"],
            text=[f"{p_a*100:.1f}%", f"{p_b*100:.1f}%"],
            textposition="inside", textfont=dict(color="white", size=13),
        ))
        fig_bar.update_layout(
            xaxis=dict(range=[0,100], ticksuffix="%", gridcolor="#eee"),
            yaxis=dict(tickfont=dict(size=13, color=BLUE)),
            plot_bgcolor="white", paper_bgcolor="white",
            height=120, margin=dict(t=8,b=8,l=10,r=10), showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<div class="stitle">Bracket Path Comparison</div>',
                    unsafe_allow_html=True)
        rounds_labels = ["Final Four","Championship Game","Champion"]
        rounds_cols   = ["prob_F4","prob_NCG","prob_Champion"]

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name=team_a, x=rounds_labels,
            y=[ra[c]*100 for c in rounds_cols],
            marker_color=ORANGE if team_a=="Florida" else BLUE,
            text=[f"{ra[c]*100:.1f}%" for c in rounds_cols], textposition="outside",
        ))
        fig_cmp.add_trace(go.Bar(
            name=team_b, x=rounds_labels,
            y=[rb[c]*100 for c in rounds_cols],
            marker_color=ORANGE if team_b=="Florida" else "#5566CC",
            text=[f"{rb[c]*100:.1f}%" for c in rounds_cols], textposition="outside",
        ))
        fig_cmp.update_layout(
            barmode="group",
            yaxis=dict(title="Probability (%)", gridcolor="#eee"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=320, margin=dict(t=20,b=20,l=40,r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color=BLUE),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Key Factors section ──────────────────────────────────────────────
        STAT_COLS = list(STAT_LABELS.keys())

        # Look up both teams in stats_df
        id_a = int(ra["TeamID"])
        id_b = int(rb["TeamID"])

        stats_a = None
        stats_b = None
        if "TeamID" in stats_df.columns:
            row_a_stats = stats_df[stats_df["TeamID"] == id_a]
            row_b_stats = stats_df[stats_df["TeamID"] == id_b]
            if not row_a_stats.empty:
                stats_a = row_a_stats.iloc[0]
            if not row_b_stats.empty:
                stats_b = row_b_stats.iloc[0]
        elif "TeamName" in stats_df.columns:
            row_a_stats = stats_df[stats_df["TeamName"] == team_a]
            row_b_stats = stats_df[stats_df["TeamName"] == team_b]
            if not row_a_stats.empty:
                stats_a = row_a_stats.iloc[0]
            if not row_b_stats.empty:
                stats_b = row_b_stats.iloc[0]

        if stats_a is not None and stats_b is not None:
            winner_name = team_a if p_a > 0.5 else team_b
            st.write("")
            st.markdown(f'<div class="stitle">Why the Model Favors {winner_name}</div>',
                        unsafe_allow_html=True)

            # Compute differentials for available stats
            diffs = []
            for stat in STAT_COLS:
                if stat in stats_a.index and stat in stats_b.index:
                    try:
                        val_a = float(stats_a[stat])
                        val_b = float(stats_b[stat])
                        diff = abs(val_a - val_b)
                        diffs.append((stat, val_a, val_b, diff))
                    except (ValueError, TypeError):
                        pass

            if diffs:
                # Sort by absolute differential, take top 3
                diffs.sort(key=lambda x: x[3], reverse=True)
                top_factors = diffs[:3]

                # Build bar max for scaling
                max_diff = max(d[3] for d in top_factors) if top_factors else 1.0

                kf_rows_html = ""
                for stat, val_a, val_b, diff in top_factors:
                    label = STAT_LABELS.get(stat, stat.replace("_", " ").title())

                    # For adjD lower is better — flip advantage logic
                    lower_is_better = (stat == "adjD")
                    if lower_is_better:
                        a_has_edge = val_a < val_b
                    else:
                        a_has_edge = val_a > val_b

                    bar_color = ORANGE if a_has_edge else BLUE
                    edge_team = team_a if a_has_edge else team_b
                    edge_color = ORANGE if a_has_edge else BLUE

                    # Bar width as % of max_diff
                    bar_pct = min(100, int((diff / max_diff) * 100)) if max_diff > 0 else 0

                    # Format values
                    if stat == "WinPct":
                        fmt_a = f"{val_a*100:.1f}%"
                        fmt_b = f"{val_b*100:.1f}%"
                    elif stat in ("adjEM", "sos_adjEM", "avg_ScoreDiff"):
                        fmt_a = f"{val_a:+.1f}"
                        fmt_b = f"{val_b:+.1f}"
                    elif stat == "barthag":
                        fmt_a = f"{val_a:.3f}"
                        fmt_b = f"{val_b:.3f}"
                    else:
                        fmt_a = f"{val_a:.1f}"
                        fmt_b = f"{val_b:.1f}"

                    kf_rows_html += f"""
<div class="kf-row">
  <div class="kf-label">{label}</div>
  <div class="kf-val" style="color:{ORANGE if a_has_edge else '#666'};">{fmt_a}</div>
  <div class="kf-bar-wrap">
    <div class="kf-bar" style="width:{bar_pct}%;background:{bar_color};"></div>
  </div>
  <div class="kf-val" style="color:{BLUE if not a_has_edge else '#666'};">{fmt_b}</div>
  <div class="kf-edge" style="color:{edge_color};">{edge_team} edge</div>
</div>"""

                # Header row
                header_html = f"""
<div style="display:flex;align-items:center;gap:10px;padding:6px 0 4px;
            border-bottom:2px solid #eaeef8;margin-bottom:4px;">
  <div style="font-size:0.72rem;font-weight:800;color:#999;text-transform:uppercase;
              letter-spacing:0.6px;min-width:200px;">Factor</div>
  <div style="font-size:0.72rem;font-weight:800;color:{ORANGE};text-transform:uppercase;
              letter-spacing:0.6px;min-width:60px;text-align:right;">{team_a}</div>
  <div style="flex:1;font-size:0.72rem;font-weight:800;color:#999;text-transform:uppercase;
              letter-spacing:0.6px;text-align:center;">Advantage</div>
  <div style="font-size:0.72rem;font-weight:800;color:{BLUE};text-transform:uppercase;
              letter-spacing:0.6px;min-width:60px;text-align:right;">{team_b}</div>
  <div style="min-width:80px;"></div>
</div>"""

                st.markdown(f"""
<div style="background:white;border-radius:12px;padding:16px 20px;
            box-shadow:0 2px 10px rgba(0,33,165,0.07);border:1px solid #eaeef8;">
  {header_html}
  {kf_rows_html}
</div>""", unsafe_allow_html=True)
            else:
                st.info("Stat data not available for factor comparison.")
        else:
            st.info("Team stats not available for one or both selected teams.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Pool Strategy (per-round EV)
# ════════════════════════════════════════════════════════════════════════════
with tab_pool:
    st.markdown('<div class="stitle">Bracket Pool Strategy — Round-by-Round Value</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-banner">
        <b style="color:{BLUE}">How to read this:</b> Each round has its own EV ratio —
        the model's probability divided by the estimated public pick rate for that round.
        <b>EV &gt; 1.2 = undervalued</b> (pick more often than the public expects).
        <b>EV &lt; 0.8 = overvalued</b> (public picks them too much relative to model).
        Use these picks to differentiate your bracket from the field.
    </div>""", unsafe_allow_html=True)

    ev_df = build_ev_df()

    def ev_cards(df_sorted, model_col, public_col, ev_col, min_model_pct=1.0, top_n=10):
        """Render EV pick cards for one round."""
        shown = df_sorted[df_sorted[model_col] >= min_model_pct].head(top_n)
        html = ""
        for _, r in shown.iterrows():
            ev = r[ev_col]
            cls = "ev-value" if ev > 1.2 else ("ev-fair" if ev > 0.8 else "ev-over")
            label = "Value" if ev > 1.2 else ("Fair" if ev > 0.8 else "Avoid")
            html += f"""
            <div class="ev-card {cls}">
              <div>
                <div class="ev-team">{r['Team']}</div>
                <div class="ev-seed">Seed {r['Seed']}</div>
              </div>
              <div class="ev-pcts">
                Model: {r[model_col]:.1f}%<br>
                Public: {r[public_col]:.1f}%
              </div>
              <div class="ev-ratio">{ev:.2f}x<br>
                <span style="font-size:0.65rem;font-weight:600;">{label}</span>
              </div>
            </div>"""
        return html

    col_f4, col_ncg, col_champ = st.columns(3)

    with col_f4:
        st.markdown(f'<div class="stitle">Final Four</div>', unsafe_allow_html=True)
        df_f4 = ev_df.sort_values("F4_ev", ascending=False)
        st.markdown(ev_cards(df_f4, "F4_model", "F4_public", "F4_ev", min_model_pct=5.0),
                    unsafe_allow_html=True)

    with col_ncg:
        st.markdown(f'<div class="stitle">Championship Game</div>', unsafe_allow_html=True)
        df_ncg = ev_df.sort_values("NCG_ev", ascending=False)
        st.markdown(ev_cards(df_ncg, "NCG_model", "NCG_public", "NCG_ev", min_model_pct=2.0),
                    unsafe_allow_html=True)

    with col_champ:
        st.markdown(f'<div class="stitle">Champion</div>', unsafe_allow_html=True)
        df_champ = ev_df.sort_values("Champ_ev", ascending=False)
        st.markdown(ev_cards(df_champ, "Champ_model", "Champ_public", "Champ_ev", min_model_pct=1.0),
                    unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="stitle">Full EV Table — All Rounds</div>',
                unsafe_allow_html=True)
    st.caption("EV = Model probability ÷ estimated public pick rate for that round. Green = undervalued, red = overvalued.")

    full_tbl = ev_df[["Seed","Team",
                       "F4_model","F4_public","F4_ev",
                       "NCG_model","NCG_public","NCG_ev",
                       "Champ_model","Champ_public","Champ_ev"]].copy()
    full_tbl.columns = ["Seed","Team",
                         "F4 Model%","F4 Public%","F4 EV",
                         "NCG Model%","NCG Public%","NCG EV",
                         "Champ Model%","Champ Public%","Champ EV"]

    def color_ev(v):
        try:
            f = float(v)
            if f > 1.2: return "color:#15803d;font-weight:bold"
            if f < 0.8: return "color:#dc2626"
        except: pass
        return ""

    st.dataframe(
        full_tbl.sort_values("Champ EV", ascending=False)
               .style.map(color_ev, subset=["F4 EV","NCG EV","Champ EV"]),
        use_container_width=True, height=480,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Insights
# ════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="stitle">What the Model Looks At</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-banner">
        A weighted ensemble of <b>XGBoost</b> (47%), <b>LightGBM</b> (48%), and
        <b>Logistic Regression</b> (5%), trained on every NCAA tournament game since 2002
        using walk-forward cross-validation (no future data leakage). 292 features per matchup
        spanning efficiency ratings, Elo momentum, strength of schedule, shot quality,
        height &amp; experience, conference tournament performance, and historical tournament DNA.
    </div>""", unsafe_allow_html=True)

    def fi_chart(df, title, color):
        top = df[df["feature"] != "feature"].head(15).copy()
        top["label"] = top["feature"].apply(feat_label)
        top = top.sort_values("importance")
        fig = go.Figure(go.Bar(
            x=top["importance"], y=top["label"], orientation="h",
            marker_color=color,
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(color=BLUE, size=13)),
            xaxis=dict(title="Importance", gridcolor="#eee"),
            yaxis=dict(tickfont=dict(size=11)),
            plot_bgcolor="white", paper_bgcolor="white",
            height=460, margin=dict(t=40,b=20,l=10,r=10),
        )
        return fig

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fi_chart(fi_xgb, "XGBoost — Top 15 Features", BLUE),
                        use_container_width=True)
    with c2:
        st.plotly_chart(fi_chart(fi_lgb, "LightGBM — Top 15 Features", ORANGE),
                        use_container_width=True)

    st.markdown('<div class="stitle">Model Performance</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    for col, (lbl, val, sub) in zip([p1,p2,p3], [
        ("Walk-Forward CV Accuracy", "86.6%", "Tested on 2010–2025 tournaments"),
        ("Log-Loss (CV)", "0.320", "Lower is better — Kaggle scoring metric"),
        ("Bracket Simulations", "100,000", "Monte Carlo runs per prediction set"),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="lbl">{lbl}</div><div class="val">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-banner" style="margin-top:18px;border-left-color:{BLUE};">
        <b style="color:{BLUE}">Data sources:</b> KenPom (2002–2026) · Torvik T-Rank (2008–2026) ·
        Kaggle NCAA box scores (1985–2026) · Massey Ordinals from 10 rating systems ·
        FiveThirtyEight-style Elo with margin-of-victory multiplier and recency momentum.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Championship DNA
# ════════════════════════════════════════════════════════════════════════════
with tab_dna:
    st.markdown('<div class="stitle">Championship DNA</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-banner">
        What separates champions from the field? Every NCAA champion since 2003 analyzed
        across efficiency, power ratings, Elo momentum, and historical upset patterns.
        See how 2026 contenders stack up against the champion blueprint.
    </div>""", unsafe_allow_html=True)

    # ── Merge stats with round probs ─────────────────────────────────────────
    dna_df = (round_df[["TeamID","TeamName","SeedNum","prob_Champion","prob_F4","prob_NCG"]]
              .merge(stats_df.drop(columns=["TeamName"], errors="ignore"),
                     on="TeamID", how="left"))

    DNA_STATS = [
        ("adjEM",          "Adj. Efficiency Margin",  True,  32.24, "%+.1f"),
        ("adjO",           "Offensive Efficiency",    True,  121.60, "%.1f"),
        ("adjD",           "Defensive Efficiency",    False, 91.10, "%.1f"),
        ("barthag",        "Power Rating (Torvik)",   True,  0.970, "%.3f"),
        ("elo_pre_tourney","Pre-Tourney Elo",          True,  2013.78, "%.0f"),
        ("AvgScoreDiff",   "Avg Score Margin",        True,  14.61, "+.1f"),
        ("wab",            "Wins Above Bubble",       True,  8.97, "+.1f"),
    ]

    # ── Section 1: Champion DNA Radar ────────────────────────────────────────
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;margin-top:8px;">Radar: How Contenders Compare to the Champion Blueprint</div>',
                unsafe_allow_html=True)
    st.caption("Radar dimensions are percentile-ranked against all 2026 tournament teams. "
               "The dashed line shows the median champion's percentile.")

    radar_cols = ["adjEM", "adjO", "adjD", "barthag", "elo_pre_tourney", "wab"]
    radar_labels = ["Efficiency\nMargin", "Offense", "Defense\n(lower=better)",
                    "Power\nRating", "Elo\nRating", "Wins Above\nBubble"]

    # Compute percentile of each 2026 team on each stat (vs all 68 teams)
    pct_df = dna_df[["TeamID", "TeamName"] + radar_cols].copy()
    for col in radar_cols:
        vals = pct_df[col].fillna(pct_df[col].median())
        if col == "adjD":  # lower is better
            pct_df[f"{col}_pct"] = (vals.rank(ascending=True) / len(vals) * 100)
        else:
            pct_df[f"{col}_pct"] = (vals.rank(ascending=False) / len(vals) * 100)
            # Flip: rank 1 = 100th pct
            pct_df[f"{col}_pct"] = 100 - pct_df[f"{col}_pct"] + (100 / len(vals))

    # Champion median profile (compute against all 2026 teams using champ_hist medians)
    champ_medians = {row["stat"]: row["median"]
                     for _, row in champ_prof.iterrows() if row["stat"] in radar_cols}
    champ_radar = []
    for col in radar_cols:
        med = champ_medians.get(col)
        if med is None:
            champ_radar.append(50)
            continue
        vals = dna_df[col].fillna(dna_df[col].median())
        if col == "adjD":
            pct = (vals > med).mean() * 100  # fraction of teams with worse (higher) defense
        else:
            pct = (vals < med).mean() * 100  # fraction of teams below champ median
        champ_radar.append(round(pct, 1))

    # Pick top contenders by model champ%
    top_teams = dna_df.nlargest(4, "prob_Champion")["TeamName"].tolist()
    team_colors = [BLUE, ORANGE, "#16a34a", "#9333ea"]

    fig_radar = go.Figure()
    theta = radar_labels + [radar_labels[0]]  # close the loop

    # Champion blueprint zone (shaded)
    cr = champ_radar + [champ_radar[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=cr, theta=theta, fill="toself",
        fillcolor="rgba(250,70,22,0.12)", line=dict(color=ORANGE, dash="dash", width=2),
        name="Champion Median", mode="lines",
    ))

    for i, tname in enumerate(top_teams):
        row = pct_df[pct_df["TeamName"] == tname]
        if row.empty:
            continue
        vals = [float(row[f"{col}_pct"].iloc[0]) for col in radar_cols]
        vals_closed = vals + [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=theta, fill="toself",
            fillcolor=f"rgba({int(team_colors[i][1:3],16)},{int(team_colors[i][3:5],16)},{int(team_colors[i][5:7],16)},0.08)",
            line=dict(color=team_colors[i], width=2.5),
            name=tname, mode="lines+markers",
            marker=dict(size=6, color=team_colors[i]),
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="th", gridcolor="#e5e7eb",
                            tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=10)),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5,
                    font=dict(size=11)),
        margin=dict(l=60, r=60, t=30, b=60),
        paper_bgcolor="white",
        height=440,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Section 2: Champion DNA Checklist ────────────────────────────────────
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;margin-top:4px;">Championship DNA Checklist</div>',
                unsafe_allow_html=True)
    st.caption("Criteria derived from all champions since 2003. A check = team meets the threshold. "
               "No team is guaranteed — Virginia (2019) was the first 1-seed ever to lose to a 16 before winning it all.")

    champ_thresholds = {
        "adjEM":           (28.0,  True,  "Adj. Efficiency Margin ≥ 28",   "Only 2 of 22 champions had adjEM < 28"),
        "adjO":            (119.0, True,  "Offensive Efficiency ≥ 119",    "Outlier: KU 2022 had 119.2"),
        "adjD":            (93.0,  False, "Defensive Efficiency ≤ 93",     "Virginia 2019 had 89.2 — elite defense wins titles"),
        "barthag":         (0.945, True,  "Power Rating (Torvik) ≥ 0.945", "UConn 2014 (7-seed) had 0.914 — only exception"),
        "elo_pre_tourney": (1960,  True,  "Pre-Tourney Elo ≥ 1960",        "UConn 2011 (1924) the only exception below 1960"),
        "AvgScoreDiff":    (7.0,   True,  "Avg Score Margin ≥ 7.0 pts",    "Every champion since 2003 won by 7+ on average"),
        "wab":             (4.5,   True,  "Wins Above Bubble ≥ 4.5",       "UConn 2014 (4.8) was the lowest recent champion"),
        "SeedNum":         (7,     False, "Seed ≤ 7 (no worse than a 7-seed)", "Only one double-digit seed has won — ever"),
    }

    checklist_rows = []
    for stat, (thresh, higher_is_better, label, note) in champ_thresholds.items():
        if stat not in dna_df.columns and stat != "SeedNum":
            continue
        col_data = dna_df[stat] if stat in dna_df.columns else dna_df["SeedNum"]
        if higher_is_better:
            passing = dna_df[col_data >= thresh]["TeamName"].tolist()
        else:
            passing = dna_df[col_data <= thresh]["TeamName"].tolist()
        # Fraction of historical champions who passed
        if stat in champ_hist.columns:
            cv = champ_hist[stat].dropna()
            if higher_is_better:
                champ_pass_rate = (cv >= thresh).mean()
            else:
                champ_pass_rate = (cv <= thresh).mean()
        else:
            champ_pass_rate = 1.0
        n_pass = len(passing)
        checklist_rows.append({
            "label": label, "note": note,
            "champ_pass_rate": champ_pass_rate,
            "n_pass": n_pass, "passing": passing,
        })

    for cr in checklist_rows:
        rate_bar = int(cr["champ_pass_rate"] * 100)
        rate_color = GREEN if rate_bar >= 80 else ORANGE if rate_bar >= 60 else "#ef4444"
        n = cr["n_pass"]
        teams_str = ", ".join(cr["passing"][:6]) + ("…" if n > 6 else "")
        st.markdown(f"""
<div style="background:white;border-radius:10px;padding:14px 18px;margin-bottom:8px;
            box-shadow:0 1px 6px rgba(0,33,165,0.07);border:1px solid #eaeef8;">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
    <div style="flex:2;">
      <div style="font-size:0.88rem;font-weight:700;color:#1e1e2e;">{cr['label']}</div>
      <div style="font-size:0.73rem;color:#888;margin-top:2px;">{cr['note']}</div>
    </div>
    <div style="flex:1;text-align:center;">
      <div style="font-size:0.68rem;color:#999;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">% of Champions</div>
      <div style="height:6px;background:#f0f0f0;border-radius:3px;margin:4px 0;overflow:hidden;">
        <div style="height:6px;width:{rate_bar}%;background:{rate_color};border-radius:3px;"></div>
      </div>
      <div style="font-size:0.8rem;font-weight:800;color:{rate_color};">{rate_bar}%</div>
    </div>
    <div style="flex:2;">
      <div style="font-size:0.68rem;color:#999;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">{n} team{'s' if n!=1 else ''} qualify</div>
      <div style="font-size:0.78rem;font-weight:600;color:{BLUE};margin-top:2px;">{teams_str if teams_str else "—"}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── "Meets All Criteria" summary ─────────────────────────────────────────
    all_pass_sets = [set(cr["passing"]) for cr in checklist_rows]
    if all_pass_sets:
        meets_all = sorted(all_pass_sets[0].intersection(*all_pass_sets[1:]))
    else:
        meets_all = []
    if meets_all:
        badges = "".join(
            f'<span style="display:inline-block;background:{BLUE};color:white;'
            f'font-size:0.82rem;font-weight:700;border-radius:20px;'
            f'padding:4px 14px;margin:4px 4px;">{t}</span>'
            for t in meets_all
        )
        st.markdown(f"""
<div style="background:linear-gradient(135deg,{BLUE},{DARK_BLUE});border-radius:12px;
            padding:18px 22px;margin:12px 0;">
  <div style="font-size:0.7rem;font-weight:800;color:rgba(255,255,255,0.7);
              text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
    Meets All {len(checklist_rows)} Championship DNA Criteria
  </div>
  <div>{badges}</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style="background:#fff7ed;border:1px solid {ORANGE};border-radius:10px;
            padding:14px 18px;margin:12px 0;">
  <div style="font-size:0.85rem;font-weight:700;color:{ORANGE};">
    No team meets all {len(checklist_rows)} criteria — upsets always happen.
  </div>
  <div style="font-size:0.78rem;color:#666;margin-top:4px;">
    UConn's 7-seed title in 2014 cleared only 5 of the 8 thresholds.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Section 3: Efficiency Scatter ────────────────────────────────────────
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;margin-top:16px;">The Champion Zone: Efficiency vs Elo</div>',
                unsafe_allow_html=True)
    st.caption("Grey = all 2026 tournament teams  ·  Colored = top contenders  ·  "
               "Orange zone = where all champions since 2003 have landed.")

    fig_sc = go.Figure()

    # Champion zone shading (bounding box of all historical champions)
    ch_x = champ_hist["adjEM"].dropna()
    ch_y = champ_hist["elo_pre_tourney"].dropna()
    fig_sc.add_shape(type="rect",
        x0=ch_x.quantile(0.10), x1=ch_x.max() + 1,
        y0=ch_y.quantile(0.10), y1=ch_y.max() + 10,
        fillcolor="rgba(250,70,22,0.07)", line=dict(color=ORANGE, width=1.5, dash="dot"),
    )
    fig_sc.add_annotation(
        x=ch_x.quantile(0.10), y=ch_y.max() + 10,
        text="Champion Zone", showarrow=False,
        font=dict(size=10, color=ORANGE), xanchor="left", yanchor="bottom",
    )

    # All 2026 teams (grey)
    others = dna_df[~dna_df["TeamName"].isin(top_teams)]
    fig_sc.add_trace(go.Scatter(
        x=others["adjEM"], y=others["elo_pre_tourney"],
        mode="markers+text", text=others["SeedNum"].astype(str),
        textposition="middle center",
        marker=dict(size=22, color="#d1d5db", line=dict(color="#9ca3af", width=1)),
        textfont=dict(size=8, color="#555"),
        name="Field", hovertemplate="<b>%{customdata}</b><br>adjEM: %{x:.1f}<br>Elo: %{y:.0f}<extra></extra>",
        customdata=others["TeamName"],
    ))

    # Historical champions (small orange dots)
    fig_sc.add_trace(go.Scatter(
        x=champ_hist["adjEM"], y=champ_hist["elo_pre_tourney"],
        mode="markers",
        marker=dict(size=9, color=ORANGE, symbol="star", opacity=0.6,
                    line=dict(color="white", width=1)),
        name="Past Champions",
        hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>adjEM: %{x:.1f}<br>Elo: %{y:.0f}<extra></extra>",
        customdata=list(zip(champ_hist["TeamName"].fillna(""), champ_hist["Season"].astype(str))),
    ))

    # Top contenders (colored)
    for i, tname in enumerate(top_teams):
        row = dna_df[dna_df["TeamName"] == tname]
        if row.empty:
            continue
        sn = int(row["SeedNum"].iloc[0])
        champ_pct = float(row["prob_Champion"].iloc[0]) * 100
        fig_sc.add_trace(go.Scatter(
            x=row["adjEM"], y=row["elo_pre_tourney"],
            mode="markers+text",
            text=[f"{sn}"],
            textposition="middle center",
            marker=dict(size=32, color=team_colors[i], line=dict(color="white", width=2)),
            textfont=dict(size=11, color="white", family="Arial Black"),
            name=f"{tname} ({champ_pct:.1f}%)",
            hovertemplate=f"<b>{tname}</b><br>adjEM: %{{x:.1f}}<br>Elo: %{{y:.0f}}<br>Champ: {champ_pct:.1f}%<extra></extra>",
        ))

    fig_sc.update_layout(
        xaxis=dict(title=dict(text="Adjusted Efficiency Margin (KenPom)", font=dict(size=11)),
                   gridcolor="#f0f0f0"),
        yaxis=dict(title=dict(text="Pre-Tournament Elo Rating", font=dict(size=11)),
                   gridcolor="#f0f0f0"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5,
                    font=dict(size=10)),
        margin=dict(l=50, r=30, t=30, b=80),
        height=450,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Section 4: Upset Rates ────────────────────────────────────────────────
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;margin-top:4px;">Historical Upset Rates by Seed Matchup</div>',
                unsafe_allow_html=True)
    st.caption("Based on all NCAA Tournament first and second round games since 2003.")

    upset_sorted = upset_df.sort_values("FavSeed")
    colors_upset = [GREEN if u < 20 else ORANGE if u < 40 else "#ef4444"
                    for u in upset_sorted["UpsetPct"]]

    fig_up = go.Figure()
    fig_up.add_trace(go.Bar(
        x=upset_sorted["matchup"],
        y=upset_sorted["UpsetPct"],
        marker_color=colors_upset,
        text=[f"{v:.0f}%" for v in upset_sorted["UpsetPct"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Upset rate: %{y:.1f}%<br>Games played: %{customdata}<extra></extra>",
        customdata=upset_sorted["Games"],
    ))
    fig_up.update_layout(
        xaxis=dict(title="Seed Matchup", tickfont=dict(size=12)),
        yaxis=dict(title="Upset Rate (%)", range=[0, 60], gridcolor="#f0f0f0"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=20, t=30, b=50),
        height=320,
        showlegend=False,
    )
    fig_up.add_hline(y=50, line_dash="dot", line_color="#999",
                     annotation_text="50% (coin flip)", annotation_position="top right",
                     annotation_font_size=10)
    st.plotly_chart(fig_up, use_container_width=True)

    # ── Section 5: Past Champions Gallery ────────────────────────────────────
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;margin-top:4px;">Past Champions Since 2003</div>',
                unsafe_allow_html=True)

    champ_disp = champ_hist[["Season","TeamName","SeedNum","adjEM","adjO","adjD",
                              "barthag","elo_pre_tourney","AvgScoreDiff","wab"]].copy()
    champ_disp = champ_disp.sort_values("Season", ascending=False).reset_index(drop=True)
    champ_disp.columns = ["Year","Champion","Seed","AdjEM","Off Eff","Def Eff",
                          "BARTHAG","Elo","Avg Margin","WAB"]
    champ_disp["Def Eff"] = champ_disp["Def Eff"].round(1)
    champ_disp["AdjEM"]   = champ_disp["AdjEM"].round(1)
    champ_disp["BARTHAG"] = champ_disp["BARTHAG"].round(3)
    champ_disp["Elo"]     = champ_disp["Elo"].round(0).astype(int)
    champ_disp["Avg Margin"] = champ_disp["Avg Margin"].round(1)
    champ_disp["WAB"]     = champ_disp["WAB"].round(1)

    def color_adjEM(val):
        if pd.isna(val): return ""
        if val >= 32: return f"background-color:{BLUE};color:white;font-weight:bold"
        if val >= 27: return f"background-color:#dbeafe;color:{BLUE};font-weight:bold"
        return ""

    styled = (champ_disp.style
              .map(color_adjEM, subset=["AdjEM"])
              .format({"AdjEM": "{:+.1f}", "Avg Margin": "{:+.1f}", "WAB": "{:+.1f}",
                       "BARTHAG": "{:.3f}", "Elo": "{:,}"})
              .set_properties(**{"font-size": "0.82rem"}))
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Upset Picker
# ════════════════════════════════════════════════════════════════════════════
with tab_upset:
    st.markdown('<div class="stitle">Upset Picker</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-banner">
        The model's best upset candidates for each round and region — ranked by upset probability
        with a stat-by-stat breakdown of <em>why</em> the lower seed has a chance.
        Green stats = underdog advantage. Red stats = favorite advantage.
    </div>""", unsafe_allow_html=True)

    REGIONS = {"W": "East", "X": "South", "Y": "Midwest", "Z": "West"}
    HISTORICAL_UPSET = {(5,12):38.6,(6,11):44.3,(7,10):37.9,(8,9):50.0,
                        (4,13):20.5,(3,14):11.4,(2,15):8.0,(1,16):2.3}

    UPSET_STATS = [
        ("adjEM",           "Efficiency Margin",  True,  "{:+.1f}"),
        ("adjO",            "Offense (AdjO)",      True,  "{:.1f}"),
        ("adjD",            "Defense (AdjD)",      False, "{:.1f}"),
        ("elo_pre_tourney", "Elo Rating",          True,  "{:.0f}"),
        ("elo_momentum",    "Elo Momentum",        True,  "{:+.0f}"),
        ("elo_late_winpct", "Late Win%",           True,  "{:.0%}"),
        ("barthag",         "Power Rating",        True,  "{:.3f}"),
        ("wab",             "Wins Above Bubble",   True,  "{:+.1f}"),
        ("AvgScoreDiff",    "Avg Margin",          True,  "{:+.1f}"),
        ("sos_adjEM",       "Strength of Schedule",True,  "{:.1f}"),
    ]

    def get_stat(team_id, col):
        r = stats_df[stats_df["TeamID"] == int(team_id)]
        if r.empty or col not in r.columns: return None
        v = r[col].iloc[0]
        return float(v) if pd.notna(v) else None

    def build_r64_matchups():
        matchups = []
        for reg_code, reg_name in REGIONS.items():
            reg = round_df[round_df["Seed"].str.startswith(reg_code)].copy()
            seed_to_teams = {}
            for _, r in reg.iterrows():
                sn = int(r["SeedNum"])
                seed_to_teams.setdefault(sn, []).append(r)
            for s_fav, s_dog in [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)]:
                favs = seed_to_teams.get(s_fav, [])
                dogs = seed_to_teams.get(s_dog, [])
                if not favs or not dogs: continue
                fav = favs[0]; dog = dogs[0]
                p_dog = win_prob(int(dog["TeamID"]), int(fav["TeamID"]))
                hist  = HISTORICAL_UPSET.get((s_fav, s_dog), 50.0)
                matchups.append({
                    "round": "R64", "region": reg_name, "region_code": reg_code,
                    "fav_name": fav["TeamName"], "fav_seed": s_fav, "fav_id": int(fav["TeamID"]),
                    "dog_name": dog["TeamName"], "dog_seed": s_dog, "dog_id": int(dog["TeamID"]),
                    "p_dog": p_dog, "hist_upset": hist,
                    "value": round(p_dog * 100 / hist, 2) if hist > 0 else 1.0,
                })
        return matchups

    def build_r32_matchups():
        """Project R32 upsets: best-seed opponent vs next-round opponents from model."""
        matchups = []
        for reg_code, reg_name in REGIONS.items():
            reg = round_df[round_df["Seed"].str.startswith(reg_code)].copy()
            seed_to_teams = {}
            for _, r in reg.iterrows():
                seed_to_teams.setdefault(int(r["SeedNum"]), []).append(r)
            # R32 pairings after R64: winners of (1v16) vs (8v9), (5v12) vs (4v13), (6v11) vs (3v14), (2v15) vs (7v10)
            r32_pairs = [
                ((1,16),(8,9)),   # bottom of bracket
                ((5,12),(4,13)),  # 4/5 line
                ((6,11),(3,14)),  # 3/6 line
                ((2,15),(7,10)),  # top of bracket
            ]
            for (s1a, s1b), (s2a, s2b) in r32_pairs:
                # Most likely survivor from each first-round game
                t1_pool = [(seed_to_teams.get(s1a,[]) or [None])[0],
                           (seed_to_teams.get(s1b,[]) or [None])[0]]
                t2_pool = [(seed_to_teams.get(s2a,[]) or [None])[0],
                           (seed_to_teams.get(s2b,[]) or [None])[0]]
                t1_pool = [t for t in t1_pool if t is not None]
                t2_pool = [t for t in t2_pool if t is not None]
                if len(t1_pool) < 2 or len(t2_pool) < 2: continue
                # Determine favourite by R64 win prob
                p_t1a = win_prob(int(t1_pool[0]["TeamID"]), int(t1_pool[1]["TeamID"]))
                fav1  = t1_pool[0] if p_t1a >= 0.5 else t1_pool[1]
                dog1  = t1_pool[1] if p_t1a >= 0.5 else t1_pool[0]
                p_t2a = win_prob(int(t2_pool[0]["TeamID"]), int(t2_pool[1]["TeamID"]))
                fav2  = t2_pool[0] if p_t2a >= 0.5 else t2_pool[1]
                dog2  = t2_pool[1] if p_t2a >= 0.5 else t2_pool[0]
                # R32 upset = lower seed beating higher seed
                for (fav, dog) in [(fav1, fav2), (fav2, fav1)]:
                    if int(fav["SeedNum"]) >= int(dog["SeedNum"]): continue
                    p_dog = win_prob(int(dog["TeamID"]), int(fav["TeamID"]))
                    if p_dog < 0.15: continue  # skip near-impossible
                    matchups.append({
                        "round": "R32 (projected)", "region": reg_name, "region_code": reg_code,
                        "fav_name": fav["TeamName"], "fav_seed": int(fav["SeedNum"]),
                        "fav_id": int(fav["TeamID"]),
                        "dog_name": dog["TeamName"], "dog_seed": int(dog["SeedNum"]),
                        "dog_id": int(dog["TeamID"]),
                        "p_dog": p_dog, "hist_upset": 35.0, "value": p_dog * 100 / 35.0,
                    })
        return matchups

    r64_matchups = build_r64_matchups()
    r32_matchups = build_r32_matchups()
    all_matchups = r64_matchups + r32_matchups

    # ── Filters ──────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        round_filter = st.selectbox("Round", ["All Rounds", "R64 (First Round)", "R32 (Projected)"],
                                    key="upset_round")
    with col_f2:
        region_filter = st.selectbox("Region", ["All Regions"] + list(REGIONS.values()),
                                     key="upset_region")
    with col_f3:
        sort_by = st.selectbox("Sort by", ["Upset Probability", "Model vs History (Value)"],
                               key="upset_sort")

    filtered = [m for m in all_matchups
                if (round_filter == "All Rounds" or
                    (round_filter == "R64 (First Round)" and m["round"] == "R64") or
                    (round_filter == "R32 (Projected)" and "R32" in m["round"]))
                and (region_filter == "All Regions" or m["region"] == region_filter)]

    if sort_by == "Upset Probability":
        filtered.sort(key=lambda x: x["p_dog"], reverse=True)
    else:
        filtered.sort(key=lambda x: x["value"], reverse=True)

    if not filtered:
        st.info("No matchups found for selected filters.")
    else:
        st.markdown(f"**{len(filtered)} matchups** — showing top upset candidates first")

    for m in filtered[:20]:  # cap at 20 cards
        p_pct    = m["p_dog"] * 100
        hist_pct = m["hist_upset"]
        value    = m["value"]
        val_color = GREEN if value >= 1.2 else ORANGE if value >= 0.8 else "#ef4444"
        val_label = "Value Pick ▲" if value >= 1.2 else ("Lean ▶" if value >= 0.8 else "Fade ▼")
        bar_pct  = min(int(p_pct), 100)

        # Stat comparison
        stat_html = ""
        dog_edge_count = 0
        for skey, slabel, higher_better, fmt in UPSET_STATS:
            fv = get_stat(m["fav_id"], skey)
            dv = get_stat(m["dog_id"], skey)
            if fv is None or dv is None: continue
            dog_wins = (dv > fv) if higher_better else (dv < fv)
            diff = dv - fv
            if dog_wins:
                dog_edge_count += 1
            arrow = "▲" if dog_wins else "▼"
            c = GREEN if dog_wins else "#ef4444"
            try:
                fv_str = fmt.format(fv)
                dv_str = fmt.format(dv)
            except Exception:
                fv_str = f"{fv:.2f}"; dv_str = f"{dv:.2f}"
            stat_html += (
                f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;'
                f'border-bottom:1px solid #f5f5f5;">'
                f'<span style="min-width:130px;font-size:0.71rem;color:#666;">{slabel}</span>'
                f'<span style="min-width:55px;text-align:right;font-size:0.73rem;'
                f'font-weight:{"700" if not dog_wins else "400"};color:{"#333" if not dog_wins else "#bbb"};">{fv_str}</span>'
                f'<span style="font-size:0.7rem;color:{c};font-weight:800;margin:0 4px;">{arrow}</span>'
                f'<span style="min-width:55px;font-size:0.73rem;'
                f'font-weight:{"700" if dog_wins else "400"};color:{c if dog_wins else "#bbb"};">{dv_str}</span>'
                f'</div>'
            )

        # Expand/collapse via st.expander
        with st.expander(
            f"**{m['dog_name']}** ({m['dog_seed']}-seed) over **{m['fav_name']}** "
            f"({m['fav_seed']}-seed)  ·  {m['region']}  ·  {p_pct:.1f}% chance",
            expanded=(p_pct >= 35)
        ):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                st.markdown(f"""
<div style="text-align:center;padding:10px;">
  <div style="font-size:0.65rem;color:#999;text-transform:uppercase;font-weight:700;letter-spacing:0.5px;">
    Upset Probability</div>
  <div style="font-size:2rem;font-weight:900;color:{BLUE};">{p_pct:.1f}%</div>
  <div style="height:6px;background:#f0f0f0;border-radius:3px;margin:4px 0;overflow:hidden;">
    <div style="height:6px;width:{bar_pct}%;background:{BLUE};border-radius:3px;"></div>
  </div>
  <div style="font-size:0.7rem;color:#888;">Historical avg: {hist_pct:.1f}%</div>
</div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
<div style="text-align:center;padding:10px;">
  <div style="font-size:0.65rem;color:#999;text-transform:uppercase;font-weight:700;letter-spacing:0.5px;">
    Model vs History</div>
  <div style="font-size:1.8rem;font-weight:900;color:{val_color};">{value:.2f}x</div>
  <div style="font-size:0.78rem;font-weight:700;color:{val_color};">{val_label}</div>
  <div style="font-size:0.68rem;color:#888;margin-top:2px;">{m['round']} · {m['region']}</div>
</div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
<div style="text-align:center;padding:10px;">
  <div style="font-size:0.65rem;color:#999;text-transform:uppercase;font-weight:700;letter-spacing:0.5px;">
    Dog Edges</div>
  <div style="font-size:2rem;font-weight:900;color:{GREEN if dog_edge_count >= 5 else ORANGE};">
    {dog_edge_count}/{len(UPSET_STATS)}</div>
  <div style="font-size:0.7rem;color:#888;">stats favoring<br>the underdog</div>
</div>""", unsafe_allow_html=True)

            st.markdown(f"""
<div style="background:#fafafa;border-radius:8px;padding:12px 14px;margin-top:4px;">
  <div style="font-size:0.68rem;font-weight:800;color:#999;text-transform:uppercase;
              letter-spacing:0.5px;margin-bottom:6px;">
    Stat Comparison &nbsp;
    <span style="color:{BLUE}">← {m['fav_name']} ({m['fav_seed']})</span>
    &nbsp;&nbsp;
    <span style="color:{GREEN}">→ {m['dog_name']} ({m['dog_seed']})</span>
  </div>
  {stat_html}
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — Bracket Optimizer
# ════════════════════════════════════════════════════════════════════════════
with tab_optimizer:
    st.markdown('<div class="stitle">Bracket Optimizer</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-banner">
        Find the bracket that maximizes your expected score under your pool's scoring rules.
        The optimizer picks the team with the highest expected points at each bracket slot,
        using the model's Monte Carlo-computed round probabilities.
    </div>""", unsafe_allow_html=True)

    SCORING_PRESETS = {
        "ESPN Standard (10-20-40-80-160-320)":   {"R64":10,"R32":20,"S16":40,"E8":80,"F4":160,"NCG":160,"Champion":320},
        "CBS Sports (2-4-8-16-32-64)":           {"R64":2, "R32":4, "S16":8, "E8":16,"F4":32, "NCG":32, "Champion":64},
        "Yahoo Sports (1-2-4-8-16-32)":          {"R64":1, "R32":2, "S16":4, "E8":8, "F4":16, "NCG":16, "Champion":32},
        "Upset-Heavy (2x pts for upsets)":       {"R64":10,"R32":20,"S16":40,"E8":80,"F4":160,"NCG":160,"Champion":320},
        "Flat (1 pt per win)":                   {"R64":1, "R32":1, "S16":1, "E8":1, "F4":1, "NCG":1, "Champion":1},
        "Custom":                                None,
    }

    opt_c1, opt_c2 = st.columns([2, 2])
    with opt_c1:
        preset_name = st.selectbox("Scoring System", list(SCORING_PRESETS.keys()), key="opt_preset")

    pts_cfg = SCORING_PRESETS[preset_name]
    if pts_cfg is None:
        st.markdown("**Enter custom points per round:**")
        cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
        pts_cfg = {
            "R64":      cc1.number_input("R64", value=10, min_value=0, key="opt_r64"),
            "R32":      cc2.number_input("R32", value=20, min_value=0, key="opt_r32"),
            "S16":      cc3.number_input("S16", value=40, min_value=0, key="opt_s16"),
            "E8":       cc4.number_input("E8",  value=80, min_value=0, key="opt_e8"),
            "F4":       cc5.number_input("F4",  value=160, min_value=0, key="opt_f4"),
            "Champion": cc6.number_input("Champ", value=320, min_value=0, key="opt_champ"),
        }
        pts_cfg["NCG"] = pts_cfg["F4"]  # NCG win = F4 pts toward championship

    ROUND_COLS = ["prob_R32","prob_S16","prob_E8","prob_F4","prob_NCG","prob_Champion"]
    ROUND_KEYS = ["R64","R32","S16","E8","F4","Champion"]

    # Compute expected score for each team
    def team_expected_score(team_id, pts):
        row = round_df[round_df["TeamID"] == team_id]
        if row.empty: return 0.0
        r = row.iloc[0]
        # prob_R32 = P(winning R64), prob_S16 = P(winning R32), etc.
        val = (r["prob_R32"]      * pts.get("R64", 0) +
               r["prob_S16"]      * pts.get("R32", 0) +
               r["prob_E8"]       * pts.get("S16", 0) +
               r["prob_F4"]       * pts.get("E8",  0) +
               r["prob_NCG"]      * pts.get("F4",  0) +
               r["prob_Champion"] * pts.get("Champion", 0))
        return float(val)

    exp_scores = {int(r["TeamID"]): team_expected_score(int(r["TeamID"]), pts_cfg)
                  for _, r in round_df.iterrows()}

    # Build chalk expected score for comparison (always pick lower seed num = better seed)
    def chalk_pick(t1_id, t2_id):
        s1 = id_to_seednum.get(t1_id, 99)
        s2 = id_to_seednum.get(t2_id, 99)
        return t1_id if s1 <= s2 else t2_id

    def model_pick(t1_id, t2_id):
        return t1_id if exp_scores.get(t1_id, 0) >= exp_scores.get(t2_id, 0) else t2_id

    def simulate_bracket_picks(pick_fn):
        """Simulate a bracket using pick_fn(t1_id, t2_id) -> winner."""
        BRACKET_ORDER_LOCAL = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
        all_picks = {}  # round_name -> list of (fav_name, dog_name, pick_name)
        region_e8 = []

        for reg in ["W","X","Y","Z"]:
            reg_df = round_df[round_df["Seed"].str.startswith(reg)].copy()
            seed_to_team = {}
            for _, row in reg_df.iterrows():
                sn = int(row["SeedNum"])
                if sn not in seed_to_team:
                    seed_to_team[sn] = int(row["TeamID"])

            # Handle First Four: if two teams share same seed num (a/b), pick the survivor
            for sn in list(seed_to_team.keys()):
                a_row = reg_df[(reg_df["SeedNum"] == sn) & (reg_df["Seed"].str.endswith("a"))]
                b_row = reg_df[(reg_df["SeedNum"] == sn) & (reg_df["Seed"].str.endswith("b"))]
                if not a_row.empty and not b_row.empty:
                    seed_to_team[sn] = pick_fn(int(a_row.iloc[0]["TeamID"]), int(b_row.iloc[0]["TeamID"]))

            current = []
            for s_fav, s_dog in BRACKET_ORDER_LOCAL:
                t_fav = seed_to_team.get(s_fav)
                t_dog = seed_to_team.get(s_dog)
                if t_fav is None or t_dog is None: continue
                winner = pick_fn(t_fav, t_dog)
                all_picks.setdefault("R64", []).append({
                    "game": f"({s_fav}) {id_to_name.get(t_fav,'')} vs ({s_dog}) {id_to_name.get(t_dog,'')}",
                    "pick": id_to_name.get(winner, ""), "seed": id_to_seednum.get(winner, ""),
                })
                current.append(winner)

            for rname in ["R32","S16","E8"]:
                next_r = []
                for i in range(0, len(current)-1, 2):
                    t1, t2 = current[i], current[i+1]
                    winner = pick_fn(t1, t2)
                    all_picks.setdefault(rname, []).append({
                        "game": f"({id_to_seednum.get(t1,'?')}) {id_to_name.get(t1,'')} vs ({id_to_seednum.get(t2,'?')}) {id_to_name.get(t2,'')}",
                        "pick": id_to_name.get(winner, ""), "seed": id_to_seednum.get(winner, ""),
                    })
                    next_r.append(winner)
                current = next_r
            if current:
                region_e8.append(current[0])

        # Final Four
        ff_pairs = [(0,1),(2,3)]
        ncg_teams = []
        for i, j in ff_pairs:
            if i < len(region_e8) and j < len(region_e8):
                t1, t2 = region_e8[i], region_e8[j]
                winner = pick_fn(t1, t2)
                all_picks.setdefault("F4", []).append({
                    "game": f"({id_to_seednum.get(t1,'?')}) {id_to_name.get(t1,'')} vs ({id_to_seednum.get(t2,'?')}) {id_to_name.get(t2,'')}",
                    "pick": id_to_name.get(winner, ""), "seed": id_to_seednum.get(winner, ""),
                })
                ncg_teams.append(winner)

        if len(ncg_teams) == 2:
            champ = pick_fn(ncg_teams[0], ncg_teams[1])
            all_picks["Champion"] = [{
                "game": f"({id_to_seednum.get(ncg_teams[0],'?')}) {id_to_name.get(ncg_teams[0],'')} vs ({id_to_seednum.get(ncg_teams[1],'?')}) {id_to_name.get(ncg_teams[1],'')}",
                "pick": id_to_name.get(champ, ""), "seed": id_to_seednum.get(champ, ""),
            }]

        return all_picks

    def compute_bracket_expected_score(picks_by_round, pts):
        """Given picks, compute expected score summing prob × points for each pick."""
        round_map = {"R64":"prob_R32","R32":"prob_S16","S16":"prob_E8","E8":"prob_F4",
                     "F4":"prob_NCG","Champion":"prob_Champion"}
        total = 0.0
        for rname, game_picks in picks_by_round.items():
            prob_col = round_map.get(rname)
            round_pts = pts.get(rname, 0)
            if prob_col is None: continue
            for gp in game_picks:
                team_name = gp["pick"]
                row = round_df[round_df["TeamName"] == team_name]
                if not row.empty and prob_col in row.columns:
                    total += float(row.iloc[0][prob_col]) * round_pts
        return round(total, 1)

    model_picks   = simulate_bracket_picks(model_pick)
    chalk_picks   = simulate_bracket_picks(chalk_pick)
    model_exp_pts = compute_bracket_expected_score(model_picks, pts_cfg)
    chalk_exp_pts = compute_bracket_expected_score(chalk_picks, pts_cfg)

    # Summary cards
    st.write("")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        champ_pick = model_picks.get("Champion", [{}])[0].get("pick", "—")
        champ_seed = model_picks.get("Champion", [{}])[0].get("seed", "")
        st.markdown(f"""
        <div class="gator-card">
            <div class="lbl">Optimal Champion Pick</div>
            <div class="val" style="font-size:1.4rem;">{champ_pick}</div>
            <div class="sub">Seed {champ_seed} · {preset_name.split('(')[0].strip()}</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="lbl">Expected Score (Optimal)</div>
            <div class="val">{model_exp_pts}</div>
            <div class="sub">pts using model probabilities</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        delta = round(model_exp_pts - chalk_exp_pts, 1)
        delta_color = GREEN if delta >= 0 else "#ef4444"
        st.markdown(f"""
        <div class="stat-card">
            <div class="lbl">vs Chalk Bracket</div>
            <div class="val" style="color:{delta_color};">{'+' if delta>=0 else ''}{delta}</div>
            <div class="sub">Chalk expected: {chalk_exp_pts} pts</div>
        </div>""", unsafe_allow_html=True)

    st.write("")

    # Round-by-round picks comparison
    DISPLAY_ROUNDS = ["Champion","F4","E8","S16","R32","R64"]
    ROUND_DISPLAY_NAMES = {"Champion":"Championship","F4":"Final Four","E8":"Elite Eight",
                           "S16":"Sweet Sixteen","R32":"Round of 32","R64":"Round of 64"}

    for rname in DISPLAY_ROUNDS:
        m_picks = model_picks.get(rname, [])
        c_picks = chalk_picks.get(rname, [])
        if not m_picks: continue
        rnd_pts = pts_cfg.get(rname, 0)
        with st.expander(
            f"**{ROUND_DISPLAY_NAMES.get(rname, rname)}** — {len(m_picks)} pick(s)  ·  {rnd_pts} pts each",
            expanded=(rname in ["Champion","F4","E8"])
        ):
            # Side-by-side: model vs chalk
            hdr1, hdr2 = st.columns(2)
            hdr1.markdown(f"**Model Optimal** *(maximizes expected score)*")
            hdr2.markdown(f"**Chalk Bracket** *(always pick the better seed)*")
            for i, (mp, cp) in enumerate(zip(m_picks, c_picks)):
                col1, col2 = st.columns(2)
                diff_color = GREEN if mp["pick"] != cp["pick"] else "#666"
                with col1:
                    marker = "**" if mp["pick"] != cp["pick"] else ""
                    st.markdown(
                        f'<span style="color:{diff_color};font-weight:{"800" if mp["pick"]!=cp["pick"] else "400"};">'
                        f'({mp["seed"]}) {mp["pick"]}</span>',
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f'<span style="color:#666;">({cp["seed"]}) {cp["pick"]}</span>',
                        unsafe_allow_html=True
                    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 9 — Team Deep-Dive
# ════════════════════════════════════════════════════════════════════════════
with tab_deepdive:
    st.markdown('<div class="stitle">Team Deep-Dive</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-banner">
        Select a team to see their full statistical profile, champion comparison, round-by-round
        probabilities, historical tournament record, projected bracket path, and roster profile.
    </div>""", unsafe_allow_html=True)

    dd_team = st.selectbox("Select Team", team_names_sorted, key="dd_team_select")
    dd_row  = round_df[round_df["TeamName"] == dd_team]
    if dd_row.empty:
        st.warning("Team data not found.")
    else:
        dd_r    = dd_row.iloc[0]
        dd_id   = int(dd_r["TeamID"])
        dd_seed = dd_r["SeedDisplay"]
        dd_seednum = int(dd_r["SeedNum"])

        # ── Header cards ─────────────────────────────────────────────────
        hc1, hc2, hc3, hc4, hc5 = st.columns(5)
        for col, (lbl, val, sub) in zip([hc1,hc2,hc3,hc4,hc5], [
            ("Seed",           dd_seed,                           "Tournament seed"),
            ("Champ %",        f"{dd_r['prob_Champion']*100:.1f}%","Win it all"),
            ("Final Four %",   f"{dd_r['prob_F4']*100:.1f}%",    "Reach Final Four"),
            ("Elite Eight %",  f"{dd_r['prob_E8']*100:.1f}%",    "Reach Elite Eight"),
            ("First Round Win",f"{dd_r['prob_R32']*100:.1f}%",   "Win first game"),
        ]):
            with col:
                cls = "gator-card" if dd_team == "Florida" else "stat-card"
                st.markdown(f"""
                <div class="{cls}">
                    <div class="lbl">{lbl}</div>
                    <div class="val" style="font-size:1.5rem;">{val}</div>
                    <div class="sub">{sub}</div>
                </div>""", unsafe_allow_html=True)
        st.write("")

        # ── Round probability chart ───────────────────────────────────────
        st.markdown(f'<div class="stitle" style="font-size:1.05rem;">Round-by-Round Probabilities</div>',
                    unsafe_allow_html=True)
        rnd_lbls = ["R32","S16","E8","F4","NCG","Champion"]
        rnd_display = ["Win R64","Sweet 16","Elite 8","Final Four","NCG","Champion"]
        rnd_probs = [dd_r[f"prob_{r}"] * 100 for r in rnd_lbls]
        seed_hist_probs = [
            {1:97,2:93,3:86,4:79,5:65,6:62,7:61,8:49,9:51,10:39,11:38,12:35,13:21,14:14,15:7,16:3}.get(dd_seednum,50),
            {1:71,2:50,3:37,4:27,5:17,6:14,7:12,8:10,9:8,10:7,11:10,12:9,13:4,14:2,15:1,16:0}.get(dd_seednum,10),
            {1:42,2:23,3:17,4:12,5:7,6:6,7:5,8:4,9:3,10:3,11:5,12:4,13:2,14:1,15:0,16:0}.get(dd_seednum,5),
            {1:22,2:10,3:7,4:4,5:3,6:2,7:2,8:1,9:1,10:1,11:2,12:1,13:0,14:0,15:0,16:0}.get(dd_seednum,2),
            {1:11,2:5,3:3,4:2,5:1,6:1,7:1,8:0,9:0,10:0,11:1,12:0,13:0,14:0,15:0,16:0}.get(dd_seednum,1),
            {1:6, 2:2,3:1,4:1,5:0,6:0,7:0,8:0,9:0,10:0,11:1,12:0,13:0,14:0,15:0,16:0}.get(dd_seednum,0),
        ]
        fig_rnd = go.Figure()
        fig_rnd.add_trace(go.Bar(
            name="Historical avg for this seed",
            x=rnd_display, y=seed_hist_probs,
            marker_color="#d1d5db", opacity=0.7,
            hovertemplate="%{x}: %{y:.0f}% (hist avg)<extra></extra>",
        ))
        fig_rnd.add_trace(go.Bar(
            name=dd_team,
            x=rnd_display, y=rnd_probs,
            marker_color=BLUE if dd_team != "Florida" else ORANGE,
            text=[f"{v:.1f}%" for v in rnd_probs],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig_rnd.update_layout(
            barmode="group",
            plot_bgcolor="white", paper_bgcolor="white",
            height=320, margin=dict(t=20,b=20,l=40,r=20),
            yaxis=dict(title="Probability (%)", gridcolor="#eee", range=[0, max(rnd_probs+seed_hist_probs)*1.25 or 10]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
            xaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_rnd, use_container_width=True)

        # ── Key Stats vs Champion Median ──────────────────────────────────
        st.markdown(f'<div class="stitle" style="font-size:1.05rem;">Key Stats vs Champion Median</div>',
                    unsafe_allow_html=True)
        dd_stats = stats_df[stats_df["TeamID"] == dd_id]
        champ_medians_map = {row["stat"]: row["median"] for _, row in champ_prof.iterrows()}

        DEEP_STATS = [
            ("adjEM",           "Adj. Efficiency Margin",   True,  "%+.1f"),
            ("adjO",            "Offensive Efficiency",     True,  "%.1f"),
            ("adjD",            "Defensive Efficiency",     False, "%.1f"),
            ("barthag",         "Power Rating (Torvik)",    True,  "%.3f"),
            ("elo_pre_tourney", "Pre-Tourney Elo",          True,  "%.0f"),
            ("wab",             "Wins Above Bubble",        True,  "%+.1f"),
            ("AvgScoreDiff",    "Avg Score Margin",         True,  "%+.1f"),
            ("sos_adjEM",       "Strength of Schedule",     True,  "%.1f"),
        ]

        stat_rows_html = ""
        for skey, slabel, higher_better, fmt in DEEP_STATS:
            if dd_stats.empty or skey not in dd_stats.columns: continue
            val = dd_stats[skey].iloc[0]
            if pd.isna(val): continue
            med = champ_medians_map.get(skey)
            if med is None: continue
            exceeds = (val >= med) if higher_better else (val <= med)
            color = GREEN if exceeds else ORANGE
            mark = "Above" if exceeds else "Below"
            try:
                val_str = fmt % val
                med_str = fmt % med
            except Exception:
                val_str = f"{val:.2f}"; med_str = f"{med:.2f}"
            bar_w = min(int(abs(val - med) / (abs(med) + 0.01) * 50 + 50), 100)
            stat_rows_html += f"""
<div style="display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #f0f2f8;">
  <div style="min-width:160px;font-size:0.82rem;color:#555;">{slabel}</div>
  <div style="min-width:70px;text-align:right;font-size:0.85rem;font-weight:800;color:{color};">{val_str}</div>
  <div style="flex:1;height:8px;background:#f0f0f0;border-radius:4px;overflow:hidden;">
    <div style="height:8px;width:{bar_w}%;background:{color};border-radius:4px;"></div>
  </div>
  <div style="min-width:80px;font-size:0.75rem;color:#888;">Champ median: {med_str}</div>
  <div style="min-width:55px;font-size:0.72rem;font-weight:700;color:{color};">{mark}</div>
</div>"""

        st.markdown(f'<div style="background:white;border-radius:10px;padding:14px 18px;'
                    f'box-shadow:0 1px 6px rgba(0,33,165,0.07);border:1px solid #eaeef8;">'
                    f'{stat_rows_html}</div>', unsafe_allow_html=True)
        st.write("")

        # ── Historical Tournament Record ──────────────────────────────────
        st.markdown(f'<div class="stitle" style="font-size:1.05rem;">Historical Tournament Record (since 2003)</div>',
                    unsafe_allow_html=True)
        if hist_records.empty:
            st.caption("Historical record data not available.")
        else:
            team_hist = hist_records[hist_records["TeamName"] == dd_team].copy()
            if team_hist.empty:
                st.caption("No tournament appearances on record since 2003.")
            else:
                hist_tbl = team_hist.sort_values("Season", ascending=False)
                total_apps = len(hist_tbl)
                total_wins = hist_tbl["Wins"].sum()
                final_fours = (hist_tbl["Wins"] >= 4).sum()
                titles = (hist_tbl["Wins"] == 6).sum()

                hh1, hh2, hh3, hh4 = st.columns(4)
                for col, (lbl, val) in zip([hh1,hh2,hh3,hh4], [
                    ("Tournament Apps", total_apps), ("Total Tourney Wins", total_wins),
                    ("Final Fours", final_fours), ("Championships", titles),
                ]):
                    with col:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="lbl">{lbl}</div>
                            <div class="val">{val}</div>
                        </div>""", unsafe_allow_html=True)
                st.write("")

                def color_round(val):
                    colors = {"Champion":"background:#0021A5;color:white",
                              "NCG":"background:#1d4ed8;color:white",
                              "Final Four":"background:#2563eb;color:white",
                              "Elite Eight":"background:#3b82f6;color:white",
                              "Sweet 16":"background:#93c5fd;color:#1e3a8a",
                              "Round of 32":"background:#dbeafe;color:#1e3a8a"}
                    return colors.get(val, "")

                st.dataframe(
                    hist_tbl[["Season","Seed","Record","Deepest Round"]]
                    .style.map(color_round, subset=["Deepest Round"])
                    .format({"Season": "{:.0f}"}),
                    use_container_width=True, hide_index=True, height=320,
                )

        # ── Roster Profile ───────────────────────────────────────────────
        if not height_exp.empty:
            st.write("")
            st.markdown(f'<div class="stitle" style="font-size:1.05rem;">Roster Profile</div>',
                        unsafe_allow_html=True)
            # Match team name (KenPom names may have seed suffix stripped already)
            he_row = height_exp[height_exp["Team"].str.lower().str.strip() == dd_team.lower().strip()]
            if he_row.empty:
                # Try partial match
                he_row = height_exp[height_exp["Team"].str.lower().str.contains(
                    dd_team.lower().split()[0], na=False
                )].head(1)
            if not he_row.empty:
                hr = he_row.iloc[0]
                rp_cols = st.columns(4)
                roster_items = [
                    ("Avg Height",        hr.get("Avg Hgt",    "N/A"), "inches"),
                    ("PG Height",         hr.get("PG Hgt",     "N/A"), "inches"),
                    ("Experience Rank",   hr.get("Experience", "N/A"), "1=most exp"),
                    ("Bench Quality",     hr.get("Bench",      "N/A"), "pts above avg"),
                ]
                for col, (lbl, val, sub) in zip(rp_cols, roster_items):
                    with col:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="lbl">{lbl}</div>
                            <div class="val" style="font-size:1.4rem;">{val}</div>
                            <div class="sub">{sub}</div>
                        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 10 — Hot Streaks & Busters
# ════════════════════════════════════════════════════════════════════════════
with tab_hot:
    st.markdown('<div class="stitle">Hot Streaks & Bracket Busters</div>', unsafe_allow_html=True)

    hot_df = round_df[["TeamID","TeamName","SeedNum","SeedDisplay","prob_Champion","prob_F4"]].merge(
        stats_df[["TeamID","elo_momentum","elo_late_winpct","elo_pre_tourney","adjEM","WinPct"]],
        on="TeamID", how="left"
    )

    # ── Section 1: Hot Teams ─────────────────────────────────────────────
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;margin-top:4px;">Hot Teams Entering the Tournament</div>',
                unsafe_allow_html=True)
    st.caption("Teams ranked by Elo momentum (rating change over last 10 games) and late-season win rate.")

    hot_sorted = hot_df.sort_values("elo_momentum", ascending=False).head(20)
    hot_sorted["late_pct"] = (hot_sorted["elo_late_winpct"] * 100).round(0)
    hot_sorted["momentum_disp"] = hot_sorted["elo_momentum"].apply(lambda v: f"{v:+.0f}" if pd.notna(v) else "N/A")

    fig_hot = go.Figure()
    colors_hot = [ORANGE if r["TeamName"]=="Florida" else BLUE
                  for _,r in hot_sorted.iterrows()]
    fig_hot.add_trace(go.Bar(
        x=[f"({r['SeedDisplay']}) {r['TeamName']}" for _,r in hot_sorted.iterrows()],
        y=hot_sorted["elo_momentum"],
        marker_color=colors_hot,
        text=[f"{v:+.0f}" for v in hot_sorted["elo_momentum"].fillna(0)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Elo Momentum: %{y:+.0f}<extra></extra>",
        name="Elo Momentum",
    ))
    fig_hot.update_layout(
        title=dict(text="Elo Momentum — Top 20 Teams (Last 10 Games)", font=dict(color=BLUE, size=13)),
        xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(title="Elo Change (last 10 games)", gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=380, margin=dict(t=40,b=90,l=40,r=20),
    )
    st.plotly_chart(fig_hot, use_container_width=True)

    # Late win % vs Elo momentum scatter
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;">Momentum vs Late-Season Win Rate</div>',
                unsafe_allow_html=True)
    st.caption("Top-right quadrant = hot teams with high win rate. Size = championship probability.")

    plot_df = hot_df.dropna(subset=["elo_momentum","elo_late_winpct"])
    top_label_teams = set(plot_df.nlargest(8,"elo_momentum")["TeamName"].tolist() +
                          plot_df.nlargest(8,"elo_late_winpct")["TeamName"].tolist() +
                          plot_df.nlargest(5,"prob_Champion")["TeamName"].tolist())

    fig_scat = go.Figure()
    # Quadrant lines
    med_mom = float(plot_df["elo_momentum"].median())
    med_win = float(plot_df["elo_late_winpct"].median())
    for x_val in [med_mom]:
        fig_scat.add_vline(x=x_val, line=dict(color="#e5e7eb", dash="dash", width=1))
    for y_val in [med_win]:
        fig_scat.add_hline(y=y_val, line=dict(color="#e5e7eb", dash="dash", width=1))

    # Annotation for quadrant
    fig_scat.add_annotation(x=plot_df["elo_momentum"].max()*0.85,
                             y=plot_df["elo_late_winpct"].max()*0.97,
                             text="Hot & Winning", font=dict(size=9,color=GREEN),
                             showarrow=False)

    for _, row in plot_df.iterrows():
        is_highlight = row["TeamName"] in top_label_teams
        sz = max(10, min(40, row["prob_Champion"] * 2000))
        clr = ORANGE if row["TeamName"] == "Florida" else (BLUE if is_highlight else "#d1d5db")
        lbl = f"({row['SeedDisplay']}) {row['TeamName']}" if is_highlight else ""
        fig_scat.add_trace(go.Scatter(
            x=[row["elo_momentum"]], y=[row["elo_late_winpct"]],
            mode="markers+text" if lbl else "markers",
            text=[lbl] if lbl else None,
            textposition="top right",
            textfont=dict(size=9),
            marker=dict(size=sz, color=clr, opacity=0.75,
                        line=dict(color="white" if is_highlight else "rgba(0,0,0,0)", width=1.5)),
            name=row["TeamName"] if is_highlight else "",
            hovertemplate=f"<b>({row['SeedDisplay']}) {row['TeamName']}</b><br>"
                          f"Momentum: {row['elo_momentum']:+.0f}<br>"
                          f"Late Win%: {row['elo_late_winpct']*100:.0f}%<br>"
                          f"Champ: {row['prob_Champion']*100:.1f}%<extra></extra>",
            showlegend=is_highlight,
        ))
    fig_scat.update_layout(
        xaxis=dict(title=dict(text="Elo Momentum (last 10 games)", font=dict(size=11)),
                   gridcolor="#f0f0f0"),
        yaxis=dict(title=dict(text="Late-Season Win Rate", font=dict(size=11)),
                   tickformat=".0%", gridcolor="#f0f0f0"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=440, margin=dict(l=50,r=30,t=20,b=50),
        showlegend=False,
    )
    st.plotly_chart(fig_scat, use_container_width=True)

    # ── Section 2: Bracket Busters — High Variance Picks ─────────────────
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;margin-top:4px;">Bracket Busters — High Variance Picks</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-banner">
        Teams whose model probability significantly exceeds seed expectations —
        high-upside picks that could bust chalk brackets. EV = model champ% / seed historical champ%.
        Pool differentiation picks: teams others won't pick, but our model likes.
    </div>""", unsafe_allow_html=True)

    # Compute "surprise factor" = how much this team outperforms their seed baseline
    seed_champ_baseline = {
        1:6.0, 2:2.0, 3:1.0, 4:0.8, 5:0.4, 6:0.3, 7:0.2, 8:0.15,
        9:0.12, 10:0.10, 11:0.10, 12:0.07, 13:0.03, 14:0.01, 15:0.005, 16:0.001,
    }
    buster_rows = []
    for _, r in round_df.iterrows():
        sn = int(r["SeedNum"])
        baseline = seed_champ_baseline.get(sn, 0.001)
        model_p = float(r["prob_Champion"]) * 100
        ev = round(model_p / baseline, 2) if baseline > 0 else 0
        # Variance proxy: spread between F4 and Champion probs (wider = more volatile)
        f4_p = float(r["prob_F4"]) * 100
        variance_score = round(f4_p / max(model_p, 0.01), 1)  # F4/Champion ratio
        buster_rows.append({
            "Seed": r["SeedDisplay"], "Team": r["TeamName"], "SeedNum": sn,
            "Model Champ%": round(model_p, 1),
            "Seed Baseline%": round(baseline, 1),
            "EV Ratio": ev,
            "F4%": round(f4_p, 1),
            "Variance": variance_score,
        })

    buster_df = pd.DataFrame(buster_rows)
    # Show teams with EV > 1.5 and not 1-seeds (1-seeds are expected to do well)
    busters = buster_df[(buster_df["EV Ratio"] >= 1.5) & (buster_df["SeedNum"] >= 3)].sort_values("EV Ratio", ascending=False)

    if not busters.empty:
        for _, b in busters.head(12).iterrows():
            ev_c = GREEN if b["EV Ratio"] >= 2.0 else ORANGE
            f4_disp = f"F4: {b['F4%']:.1f}%"
            st.markdown(f"""
<div class="ev-card {"ev-value" if b["EV Ratio"] >= 2.0 else "ev-fair"}">
  <div>
    <div class="ev-seed">Seed {b['Seed']}</div>
    <div class="ev-team">{b['Team']}</div>
  </div>
  <div class="ev-pcts">
    Model: {b['Model Champ%']:.1f}%<br>
    Baseline: {b['Seed Baseline%']:.1f}%<br>
    {f4_disp}
  </div>
  <div class="ev-ratio" style="color:{ev_c};">{b['EV Ratio']:.1f}x</div>
</div>""", unsafe_allow_html=True)
    else:
        st.info("No bracket busters found — the model closely follows seed expectations this year.")

    # High-ceiling / low-floor section
    st.write("")
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;">High Ceiling Picks (F4% >> Champ%)</div>',
                unsafe_allow_html=True)
    st.caption("Teams likely to go deep but not necessarily win it all — good for pool strategies with large F4/E8 points.")

    high_ceil = buster_df[buster_df["Variance"] > 3.5].sort_values("Variance", ascending=False).head(10)
    if not high_ceil.empty:
        hc_cols = st.columns(min(len(high_ceil), 5))
        for i, (_, hc) in enumerate(high_ceil.head(5).iterrows()):
            with hc_cols[i]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="lbl">Seed {hc['Seed']}</div>
                    <div class="val" style="font-size:1.2rem;">{hc['Team']}</div>
                    <div class="sub">F4: {hc['F4%']:.1f}% &nbsp; Champ: {hc['Model Champ%']:.1f}%</div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 11 — Model Calibration & Player Spotlight
# ════════════════════════════════════════════════════════════════════════════
with tab_calibration:
    st.markdown('<div class="stitle">Model Calibration</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-banner">
        A well-calibrated model means: when it says 70%, the team wins ~70% of the time.
        This reliability diagram shows actual win rates vs predicted probabilities
        using out-of-fold cross-validation predictions from 2008–2025 tournaments (1,129 games).
    </div>""", unsafe_allow_html=True)

    # Compute calibration from cv_predictions
    cv = cv_preds.copy()
    # Bin predicted probabilities into deciles
    n_bins = 10
    cv["bin"] = pd.cut(cv["pred_prob"], bins=np.linspace(0, 1, n_bins+1),
                       labels=[f"{i*10}-{(i+1)*10}%" for i in range(n_bins)],
                       include_lowest=True)
    cal_data = (cv.groupby("bin", observed=True)
                  .agg(actual_rate=("Label","mean"), count=("Label","count"),
                       mean_pred=("pred_prob","mean"))
                  .reset_index())
    cal_data["bin_mid"] = [i*10 + 5 for i in range(len(cal_data))]
    cal_data["error_low"]  = cal_data["actual_rate"] * 100 - 1.96*np.sqrt(
        cal_data["actual_rate"]*(1-cal_data["actual_rate"])/cal_data["count"].clip(1))*100
    cal_data["error_high"] = cal_data["actual_rate"] * 100 + 1.96*np.sqrt(
        cal_data["actual_rate"]*(1-cal_data["actual_rate"])/cal_data["count"].clip(1))*100

    fig_cal = go.Figure()

    # Perfect calibration line
    fig_cal.add_trace(go.Scatter(
        x=[0,100], y=[0,100],
        mode="lines", name="Perfect Calibration",
        line=dict(color="#d1d5db", dash="dash", width=2),
    ))

    # Confidence intervals
    fig_cal.add_trace(go.Scatter(
        x=cal_data["bin_mid"].tolist() + cal_data["bin_mid"].tolist()[::-1],
        y=cal_data["error_high"].tolist() + cal_data["error_low"].tolist()[::-1],
        fill="toself", fillcolor="rgba(0,33,165,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        hoverinfo="skip",
    ))

    # Actual calibration
    fig_cal.add_trace(go.Scatter(
        x=cal_data["bin_mid"],
        y=cal_data["actual_rate"] * 100,
        mode="lines+markers",
        name="Model (actual win rate)",
        line=dict(color=BLUE, width=3),
        marker=dict(size=10, color=BLUE, line=dict(color="white", width=2)),
        customdata=cal_data[["count","mean_pred"]].values,
        hovertemplate="Predicted: %{x:.0f}%<br>Actual win rate: %{y:.1f}%<br>"
                      "Games: %{customdata[0]:.0f}<br>Mean pred: %{customdata[1]:.1f}%<extra></extra>",
    ))

    fig_cal.update_layout(
        xaxis=dict(title=dict(text="Predicted Win Probability (%)", font=dict(size=11)),
                   range=[0,100], gridcolor="#eee"),
        yaxis=dict(title=dict(text="Actual Win Rate (%)", font=dict(size=11)),
                   range=[0,100], gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, margin=dict(t=20,b=50,l=50,r=20),
        legend=dict(orientation="h", y=-0.2, x=0),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # Calibration summary stats
    # Brier score and ECE
    brier = float(np.mean((cv["pred_prob"] - cv["Label"])**2))
    # Expected Calibration Error
    ece = float((cal_data["count"] / cal_data["count"].sum() *
                 abs(cal_data["actual_rate"] - cal_data["mean_pred"])).sum())

    cal_c1, cal_c2, cal_c3 = st.columns(3)
    for col, (lbl, val, sub) in zip([cal_c1,cal_c2,cal_c3], [
        ("Brier Score", f"{brier:.4f}", "Lower = better (0 = perfect)"),
        ("ECE", f"{ece:.4f}", "Expected Calibration Error (lower = better)"),
        ("CV Games", f"{len(cv):,}", "Out-of-fold predictions 2008–2025"),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="lbl">{lbl}</div>
                <div class="val">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.write("")

    # ── Log-loss by season ────────────────────────────────────────────────
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;">Log-Loss by Season</div>',
                unsafe_allow_html=True)
    st.caption("Walk-forward validation: each season was held out while model trained on all prior seasons.")

    def safe_logloss(labels, preds):
        preds_c = np.clip(preds, 1e-6, 1-1e-6)
        return -np.mean(labels * np.log(preds_c) + (1-labels) * np.log(1-preds_c))

    szn_lls = (cv.groupby("Season")
                 .apply(lambda g: safe_logloss(g["Label"].values, g["pred_prob"].values),
                        include_groups=False)
                 .reset_index(name="logloss"))

    fig_ll = go.Figure(go.Bar(
        x=szn_lls["Season"], y=szn_lls["logloss"],
        marker_color=[ORANGE if ll > 0.45 else BLUE for ll in szn_lls["logloss"]],
        text=[f"{ll:.3f}" for ll in szn_lls["logloss"]],
        textposition="outside",
        hovertemplate="Season %{x}: log-loss = %{y:.3f}<extra></extra>",
    ))
    fig_ll.add_hline(y=float(szn_lls["logloss"].mean()), line=dict(color=ORANGE, dash="dash"),
                     annotation_text=f"Mean: {float(szn_lls['logloss'].mean()):.3f}",
                     annotation_font=dict(color=ORANGE))
    fig_ll.update_layout(
        xaxis=dict(title="Season", tickfont=dict(size=11), dtick=1),
        yaxis=dict(title="Log-Loss", gridcolor="#eee", range=[0, szn_lls["logloss"].max()*1.2]),
        plot_bgcolor="white", paper_bgcolor="white",
        height=320, margin=dict(t=20,b=50,l=50,r=20),
    )
    st.plotly_chart(fig_ll, use_container_width=True)

    # (Player Spotlight moved to its own tab below)


# ── Footer ────

# ════════════════════════════════════════════════════════════════════════════
# TAB 12 — Player Spotlight
# ════════════════════════════════════════════════════════════════════════════
with tab_spotlight:
    st.markdown('<div class="stitle">Player Spotlight</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-banner">
        KenPom-powered roster intelligence: positional height, shooting identity, experience,
        and what makes each team uniquely dangerous entering the tournament.
        Heights shown as inches above/below the national average at each position (KenPom 2026).
    </div>""", unsafe_allow_html=True)

    # ── Section A: Top Players by Position Across All Tournament Teams ───
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;">Elite Size by Position — Tournament Field</div>',
                unsafe_allow_html=True)
    st.caption("Which tournament teams have the biggest size advantages? Heights are inches above/below the national average at that position.")

    if not height_exp.empty:
        # Merge height_exp with tournament teams (strip seed suffix from team names)
        import re as _re2
        he_tour = height_exp.copy()
        he_tour["TeamClean"] = he_tour["Team"].str.replace(r"\s*\d+$", "", regex=True).str.strip()
        # Match against round_df team names
        tour_names = set(round_df["TeamName"].str.strip())
        he_tour_matched = he_tour[he_tour["TeamClean"].isin(tour_names)].copy()
        he_tour_matched = he_tour_matched.merge(
            round_df[["TeamName","SeedDisplay","SeedNum","prob_Champion"]].rename(columns={"TeamName":"TeamClean"}),
            on="TeamClean", how="inner"
        )

        POSITIONS = [
            ("C Hgt",  "Center",      "#0021A5"),
            ("PF Hgt", "Power Forward","#1d4ed8"),
            ("SF Hgt", "Small Forward","#16a34a"),
            ("SG Hgt", "Shooting Guard","#d97706"),
            ("PG Hgt", "Point Guard",  "#7c3aed"),
        ]

        pos_tabs = st.tabs([p[1] for p in POSITIONS])
        for tab_pos, (pos_col, pos_name, pos_color) in zip(pos_tabs, POSITIONS):
            with tab_pos:
                if pos_col not in he_tour_matched.columns:
                    st.caption("Data not available.")
                    continue
                pos_df = he_tour_matched[["TeamClean","SeedDisplay",pos_col,"Avg Hgt"]].dropna(subset=[pos_col]).copy()
                pos_df[pos_col] = pd.to_numeric(pos_df[pos_col], errors="coerce")
                # Keep only true height-delta values (−10 to +10 inches); filter out rank columns (1–365)
                pos_df = pos_df[pos_df[pos_col].abs() <= 10].dropna(subset=[pos_col])
                if pos_df.empty:
                    st.caption("Height delta data not available for this position in the current dataset.")
                    continue
                pos_df = pos_df.sort_values(pos_col, ascending=False)
                top10 = pos_df.head(10)
                labels = [f"({r['SeedDisplay']}) {r['TeamClean']}" for _,r in top10.iterrows()]
                bar_colors = [pos_color if v >= 0 else "#f87171" for v in top10[pos_col]]
                fig_pos = go.Figure(go.Bar(
                    x=labels, y=top10[pos_col],
                    marker_color=bar_colors,
                    text=[f"{v:+.1f}\"" for v in top10[pos_col]],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Size advantage: %{y:+.1f} inches<extra></extra>",
                ))
                fig_pos.add_hline(y=0, line=dict(color="#9ca3af", width=1))
                fig_pos.update_layout(
                    xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
                    yaxis=dict(title="Inches vs National Avg", gridcolor="#eee"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=360, margin=dict(t=52,b=80,l=40,r=20),
                    title=dict(text=f"Top 10 {pos_name} Size Advantages in the Tournament",
                               font=dict(color=BLUE, size=13)),
                )
                st.plotly_chart(fig_pos, use_container_width=True)

                # Show bottom (smallest) for context
                bottom5 = pos_df.tail(5).sort_values(pos_col)
                if not bottom5.empty:
                    st.caption(f"Smallest {pos_name}s: " +
                               ", ".join([f"{r['TeamClean']} ({r[pos_col]:+.1f}\")"
                                          for _, r in bottom5.iterrows()]))
    else:
        st.caption("Height data not available.")

    # ── Section B: Shooting Identity ─────────────────────────────────────
    st.write("")
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;">Shooting Identity — 3-Point Reliance</div>',
                unsafe_allow_html=True)
    st.caption("Teams in the top-right live and die by the three. Teams in bottom-right attempt threes often but shoot poorly — high variance. Data: KenPom misc stats.")

    if not misc_stats.empty and "kp_3pt_pct" in misc_stats.columns:
        ms_tour = misc_stats.copy()
        ms_tour = ms_tour.merge(
            round_df[["TeamName","SeedDisplay","SeedNum","prob_Champion"]].rename(columns={"TeamName":"TeamClean"}),
            on="TeamClean", how="inner"
        )
        ms_tour = ms_tour.dropna(subset=["kp_3pt_pct","kp_3pa_rate"])

        if not ms_tour.empty:
            # Scatter: 3PA rate vs 3P%
            fig_shoot = go.Figure()

            # Average lines
            avg_3pa = float(ms_tour["kp_3pa_rate"].mean())
            avg_3pct = float(ms_tour["kp_3pt_pct"].mean())
            fig_shoot.add_vline(x=avg_3pa, line=dict(color="#e5e7eb", dash="dash", width=1))
            fig_shoot.add_hline(y=avg_3pct, line=dict(color="#e5e7eb", dash="dash", width=1))

            # Quadrant labels
            x_max = ms_tour["kp_3pa_rate"].max()
            y_max = ms_tour["kp_3pt_pct"].max()
            fig_shoot.add_annotation(x=x_max*0.92, y=y_max*0.98,
                text="Elite 3PT teams", font=dict(size=9,color=GREEN), showarrow=False)
            fig_shoot.add_annotation(x=avg_3pa*0.4, y=avg_3pct*0.97,
                text="Interior teams", font=dict(size=9,color="#6b7280"), showarrow=False)

            top_spot = set(ms_tour.nlargest(10,"kp_3pt_pct")["TeamClean"].tolist() +
                           ms_tour.nlargest(10,"kp_3pa_rate")["TeamClean"].tolist() +
                           ms_tour.nlargest(5,"prob_Champion")["TeamClean"].tolist())

            for _, r in ms_tour.iterrows():
                is_top = r["TeamClean"] in top_spot
                sz = max(8, min(30, r["prob_Champion"] * 2000))
                clr = ORANGE if r["TeamClean"] == "Florida" else (BLUE if is_top else "#d1d5db")
                lbl = f"({r['SeedDisplay']}) {r['TeamClean']}" if is_top else ""
                fig_shoot.add_trace(go.Scatter(
                    x=[r["kp_3pa_rate"]], y=[r["kp_3pt_pct"]],
                    mode="markers+text" if lbl else "markers",
                    text=[lbl] if lbl else None,
                    textposition="top right",
                    textfont=dict(size=8),
                    marker=dict(size=sz, color=clr, opacity=0.8,
                                line=dict(color="white" if is_top else "rgba(0,0,0,0)", width=1.5)),
                    hovertemplate=f"<b>({r['SeedDisplay']}) {r['TeamClean']}</b><br>"
                                  f"3P%: {r['kp_3pt_pct']:.1f}%<br>"
                                  f"3PA Rate: {r['kp_3pa_rate']:.1f}%<br>"
                                  f"Champ: {r['prob_Champion']*100:.1f}%<extra></extra>",
                    showlegend=False,
                ))
            fig_shoot.update_layout(
                xaxis=dict(title=dict(text="3-Point Attempt Rate (%)", font=dict(size=11)),
                           gridcolor="#f0f0f0"),
                yaxis=dict(title=dict(text="3-Point Shooting % (KenPom)", font=dict(size=11)),
                           gridcolor="#f0f0f0"),
                plot_bgcolor="white", paper_bgcolor="white",
                height=420, margin=dict(l=50,r=30,t=20,b=50),
            )
            st.plotly_chart(fig_shoot, use_container_width=True)

            # Top shooters table
            st.markdown("**Top 3-Point Shooting Teams in the Tournament Field:**")
            top_shoot = ms_tour.nlargest(10, "kp_3pt_pct")[["TeamClean","SeedDisplay","kp_3pt_pct","kp_3pa_rate"]].copy()
            top_shoot.columns = ["Team","Seed","3P%","3PA Rate%"]
            top_shoot = top_shoot.reset_index(drop=True)
            top_shoot.index = top_shoot.index + 1
            st.dataframe(top_shoot.style.format({"3P%":"{:.1f}%","3PA Rate%":"{:.1f}%"}),
                         use_container_width=True, height=320)
    else:
        st.caption("Shooting data not available.")

    # ── Section C: Individual Team Spotlight ─────────────────────────────
    st.write("")
    st.markdown(f'<div class="stitle" style="font-size:1.05rem;">Individual Team Spotlight</div>',
                unsafe_allow_html=True)

    spot_team = st.selectbox("Select Team", team_names_sorted, key="spot_team_select")

    # Helper: match a tournament team name to KenPom data
    def kenpom_match(df, col="TeamClean", name=None):
        name = name or spot_team
        m = df[df[col].str.lower().str.strip() == name.lower().strip()]
        if m.empty:
            m = df[df[col].str.lower().str.contains(name.lower().split()[0], na=False)].head(1)
        return m

    spot_row  = stats_df[stats_df["TeamName"] == spot_team]
    he_match  = kenpom_match(height_exp.assign(TeamClean=height_exp["Team"].str.replace(r"\s*\d+$","",regex=True).str.strip())) if not height_exp.empty else pd.DataFrame()
    ms_match  = kenpom_match(misc_stats) if not misc_stats.empty else pd.DataFrame()

    if spot_row.empty:
        st.warning("Team data not found.")
    else:
        sr  = spot_row.iloc[0]
        rd  = round_df[round_df["TeamName"] == spot_team]
        rd_r = rd.iloc[0] if not rd.empty else None

        # ── Header metrics ─────────────────────────────────────────────
        def _safe(val):
            """Return None if val is NaN/None, else the value."""
            try:
                return None if pd.isna(val) else val
            except Exception:
                return val

        h1, h2, h3, h4 = st.columns(4)
        with h1:
            adj_em = _safe(sr.get("adjEM", None))
            em_rank = int((stats_df["adjEM"].dropna() > adj_em).sum()) + 1 if adj_em is not None else None
            st.markdown(f"""<div class="stat-card">
                <div class="lbl">Efficiency Margin</div>
                <div class="val">{f"{adj_em:+.1f}" if adj_em is not None else "N/A"}</div>
                <div class="sub">{"#"+str(em_rank)+" in tournament" if em_rank else "KenPom data N/A"}</div></div>""", unsafe_allow_html=True)
        with h2:
            adj_o = _safe(sr.get("adjO", None))
            st.markdown(f"""<div class="stat-card">
                <div class="lbl">Offense (AdjO)</div>
                <div class="val">{f"{adj_o:.1f}" if adj_o is not None else "N/A"}</div>
                <div class="sub">pts per 100 poss (adj)</div></div>""", unsafe_allow_html=True)
        with h3:
            adj_d = _safe(sr.get("adjD", None))
            st.markdown(f"""<div class="stat-card">
                <div class="lbl">Defense (AdjD)</div>
                <div class="val">{f"{adj_d:.1f}" if adj_d is not None else "N/A"}</div>
                <div class="sub">opp pts per 100 poss</div></div>""", unsafe_allow_html=True)
        with h4:
            three_pct = float(ms_match["kp_3pt_pct"].iloc[0]) if not ms_match.empty and "kp_3pt_pct" in ms_match.columns and pd.notna(ms_match["kp_3pt_pct"].iloc[0]) else None
            three_rank = None
            if three_pct is not None and not misc_stats.empty:
                three_rank = int((misc_stats["kp_3pt_pct"].dropna() > three_pct).sum()) + 1
            st.markdown(f"""<div class="stat-card">
                <div class="lbl">3-Point Shooting</div>
                <div class="val">{f"{three_pct:.1f}%" if three_pct else "N/A"}</div>
                <div class="sub">{"#"+str(three_rank)+" in nation" if three_rank else ""}</div></div>""",
                unsafe_allow_html=True)

        st.write("")
        sc1, sc2 = st.columns(2)

        # ── Positional size chart ──────────────────────────────────────
        with sc1:
            st.markdown(f"**Positional Size Profile (vs national avg)**")
            if not he_match.empty:
                hr = he_match.iloc[0]
                pos_names_short = ["PG","SG","SF","PF","C"]
                pos_cols_map = {"PG":"PG Hgt","SG":"SG Hgt","SF":"SF Hgt","PF":"PF Hgt","C":"C Hgt"}
                pos_vals, pos_labels, pos_colors_list = [], [], []
                for p in pos_names_short:
                    col = pos_cols_map[p]
                    if col in he_match.columns:
                        v = pd.to_numeric(hr.get(col, None), errors="coerce")
                        if pd.notna(v):
                            pos_vals.append(float(v))
                            pos_labels.append(p)
                            pos_colors_list.append(GREEN if v >= 0.5 else ORANGE if v >= -0.5 else "#f87171")

                if pos_vals:
                    # Add avg height overall
                    avg_hgt = pd.to_numeric(hr.get("Avg Hgt", None), errors="coerce")
                    avg_in = int(avg_hgt) // 1
                    avg_frac = round((avg_hgt % 1) * 12)
                    avg_str = f"{int(avg_hgt // 12)}'{int(avg_hgt % 12)}\"" if avg_hgt and avg_hgt > 0 else "N/A"

                    fig_pos_team = go.Figure(go.Bar(
                        x=pos_labels, y=pos_vals,
                        marker_color=pos_colors_list,
                        text=[f"{v:+.1f}\"" for v in pos_vals],
                        textposition="outside",
                    ))
                    fig_pos_team.add_hline(y=0, line=dict(color="#9ca3af", width=1.5))
                    fig_pos_team.update_layout(
                        yaxis=dict(title="In. vs national avg", gridcolor="#eee", range=[min(pos_vals)-1, max(pos_vals)+1.5]),
                        xaxis=dict(title="Position"),
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=280, margin=dict(t=10,b=30,l=40,r=10),
                        title=dict(text=f"Avg team height: {avg_str}", font=dict(size=11,color="#666")),
                    )
                    st.plotly_chart(fig_pos_team, use_container_width=True)

                    # Experience and Bench
                    exp_rank = hr.get("Experience", None)
                    bench    = hr.get("Bench", None)
                    cont     = hr.get("Continuity", None)
                    if exp_rank is not None and pd.notna(exp_rank):
                        total_teams = len(height_exp)
                        exp_pct = int((total_teams - int(exp_rank)) / total_teams * 100)
                        st.markdown(
                            f'<span style="font-size:0.82rem;color:#555;">Experience rank: '
                            f'<b style="color:{BLUE};">#{int(exp_rank)}</b> nationally '
                            f'({exp_pct}th percentile)'
                            + (f'  ·  Bench: <b>{bench}</b> pts+/-' if bench else "")
                            + (f'  ·  Continuity: <b>#{int(cont)}</b>' if cont else "")
                            + '</span>',
                            unsafe_allow_html=True
                        )
            else:
                st.caption("Height data not available for this team.")

        # ── Key player strengths ───────────────────────────────────────
        with sc2:
            st.markdown(f"**Offensive Weapons Profile**")

            # Compute percentile for each stat among ALL KenPom teams
            WEAPON_STATS = [
                ("adjO",            stats_df,    "adjO",       "Offensive Efficiency",     True),
                ("adjD",            stats_df,    "adjD",       "Defensive Efficiency",     False),
                ("wab",             stats_df,    "wab",        "Wins Above Bubble",        True),
                ("elos",            stats_df,    "elo_pre_tourney","Elo Rating",           True),
                ("AvgScoreDiff",    stats_df,    "AvgScoreDiff","Avg Scoring Margin",      True),
                ("barthag",         stats_df,    "barthag",    "Power Rating (Torvik)",    True),
            ]

            weapons_html = ""
            for key, df_src, col, label, higher_better in WEAPON_STATS:
                if col not in df_src.columns: continue
                team_val_row = df_src[df_src["TeamName"] == spot_team] if "TeamName" in df_src.columns \
                               else spot_row
                if team_val_row.empty or col not in team_val_row.columns: continue
                val = team_val_row[col].iloc[0]
                if pd.isna(val): continue
                all_vals = df_src[col].dropna()
                if higher_better:
                    pct = (all_vals < val).mean() * 100
                else:
                    pct = (all_vals > val).mean() * 100
                bar_color = GREEN if pct >= 80 else BLUE if pct >= 50 else ORANGE if pct >= 25 else "#f87171"
                bar_w = int(pct)
                try:
                    val_str = f"{val:+.1f}" if abs(val) < 1000 else f"{val:.0f}"
                except Exception:
                    val_str = str(val)
                weapons_html += f"""
    <div style="padding:5px 0;border-bottom:1px solid #f5f5f5;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
    <span style="font-size:0.78rem;color:#555;">{label}</span>
    <span style="font-size:0.78rem;font-weight:700;color:{bar_color};">{val_str} &nbsp; ({int(pct)}th%ile)</span>
      </div>
      <div style="height:6px;background:#f0f0f0;border-radius:3px;overflow:hidden;">
    <div style="height:6px;width:{bar_w}%;background:{bar_color};border-radius:3px;"></div>
      </div>
    </div>"""

            # Add 3P% if available
            if not ms_match.empty and "kp_3pt_pct" in ms_match.columns:
                val3 = ms_match["kp_3pt_pct"].iloc[0]
                if pd.notna(val3):
                    val3 = float(val3)
                    all3 = misc_stats["kp_3pt_pct"].dropna()
                    pct3 = (all3 < val3).mean() * 100
                    bar_color = GREEN if pct3 >= 80 else BLUE if pct3 >= 50 else ORANGE if pct3 >= 25 else "#f87171"
                    weapons_html += f"""
    <div style="padding:5px 0;border-bottom:1px solid #f5f5f5;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
    <span style="font-size:0.78rem;color:#555;">3-Point Shooting (KenPom)</span>
    <span style="font-size:0.78rem;font-weight:700;color:{bar_color};">{val3:.1f}% &nbsp; ({int(pct3)}th%ile)</span>
      </div>
      <div style="height:6px;background:#f0f0f0;border-radius:3px;overflow:hidden;">
    <div style="height:6px;width:{int(pct3)}%;background:{bar_color};border-radius:3px;"></div>
      </div>
    </div>"""

            st.markdown(f'<div style="background:white;border-radius:10px;padding:14px;'
                        f'box-shadow:0 1px 6px rgba(0,33,165,0.07);border:1px solid #eaeef8;">'
                        f'{weapons_html}</div>', unsafe_allow_html=True)

        # ── Biggest strengths callout ──────────────────────────────────
        strengths = []
        if not he_match.empty:
            hr = he_match.iloc[0]
            for p, pcol in [("Center","C Hgt"),("Fwd","SF Hgt"),("PG","PG Hgt")]:
                if pcol in he_match.columns:
                    v = pd.to_numeric(hr.get(pcol, None), errors="coerce")
                    if pd.notna(v) and float(v) >= 1.5:
                        strengths.append(f"Elite {p} size ({v:+.1f}\" above avg)")
        if not ms_match.empty and "kp_3pt_pct" in ms_match.columns:
            v = ms_match["kp_3pt_pct"].iloc[0]
            if pd.notna(v) and not misc_stats.empty:
                rank3 = int((misc_stats["kp_3pt_pct"].dropna() > float(v)).sum()) + 1
                if rank3 <= 20:
                    strengths.append(f"Elite 3-point shooting ({v:.1f}%, #{rank3} nationally)")
        if not spot_row.empty:
            wab = spot_row["wab"].iloc[0] if "wab" in spot_row.columns else None
            elo = spot_row["elo_pre_tourney"].iloc[0] if "elo_pre_tourney" in spot_row.columns else None
            if wab and pd.notna(wab) and float(wab) >= 10:
                strengths.append(f"Outstanding résumé ({float(wab):+.1f} wins above bubble)")
            if elo and pd.notna(elo) and float(elo) >= 2050:
                strengths.append(f"Elite Elo rating ({float(elo):.0f} — top tier)")

        if strengths:
            strengths_str = "  ·  ".join(strengths)
            st.markdown(f"""
    <div style="background:{GREEN_BG};border:1px solid {GREEN};border-radius:8px;
            padding:10px 14px;margin-top:8px;font-size:0.83rem;color:#15803d;">
      <b>Key Strengths:</b> {strengths_str}
    </div>""", unsafe_allow_html=True)
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Python · XGBoost · LightGBM · Streamlit &nbsp;|&nbsp;
    Data: KenPom, Torvik T-Rank, Massey Ordinals, Kaggle MMLM 2026 &nbsp;|&nbsp;
    Predictions are probabilistic — upsets happen.
</div>
""", unsafe_allow_html=True)

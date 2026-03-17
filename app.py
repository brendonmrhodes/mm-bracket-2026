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

    for fi in [fi_xgb, fi_lgb]:
        fi.drop(fi[fi["feature"].isna() | (fi["feature"] == "")].index, inplace=True)
        fi["importance"] = pd.to_numeric(fi["importance"], errors="coerce")
        fi.dropna(inplace=True)
        fi.sort_values("importance", ascending=False, inplace=True)

    prob_lookup = {}
    for _, row in sub_df.iterrows():
        _, t1, t2 = row["ID"].split("_")
        prob_lookup[(int(t1), int(t2))] = float(row["Pred"])

    return round_df, fi_xgb, fi_lgb, prob_lookup, stats_df, champ_hist, upset_df, champ_prof

round_df, fi_xgb, fi_lgb, prob_lookup, stats_df, champ_hist, upset_df, champ_prof = load_data()

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
tab_odds, tab_bracket, tab_matchup, tab_pool, tab_dna, tab_model = st.tabs([
    "Championship Odds",
    "Full Bracket",
    "Matchup Explorer",
    "Pool Strategy",
    "🧬 Championship DNA",
    "Model Insights",
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
    ff_items = sorted(first_four.items())
    if ff_items:
        st.markdown('<div class="stitle">First Four — Play-In Games</div>',
                    unsafe_allow_html=True)
        ff_cols = st.columns(len(ff_items))
        for col, (base, pair) in zip(ff_cols, ff_items):
            with col:
                id_a, id_b = pair[0][1], pair[1][1]
                region_name = REGION_NAMES.get(base[0], "")
                seed_num    = base[1:].lstrip("0")
                st.markdown(f"""
                <div style="font-size:0.7rem;font-weight:700;color:{BLUE};
                    text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">
                  {region_name} · Seed {seed_num} Play-In
                </div>
                {matchup_html(None, None, ff_id_a=id_a, ff_id_b=id_b)}
                """, unsafe_allow_html=True)

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
               .style.applymap(color_ev, subset=["F4 EV","NCG EV","Champ EV"]),
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
    st.markdown('<div class="stitle">🧬 Championship DNA</div>', unsafe_allow_html=True)
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
    st.caption("Criteria derived from all champions since 2003. ✅ = team meets the threshold. "
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
        xaxis=dict(title="Adjusted Efficiency Margin (KenPom)", gridcolor="#f0f0f0",
                   titlefont=dict(size=11)),
        yaxis=dict(title="Pre-Tournament Elo Rating", gridcolor="#f0f0f0",
                   titlefont=dict(size=11)),
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
              .applymap(color_adjEM, subset=["AdjEM"])
              .format({"AdjEM": "{:+.1f}", "Avg Margin": "{:+.1f}", "WAB": "{:+.1f}",
                       "BARTHAG": "{:.3f}", "Elo": "{:,}"})
              .set_properties(**{"font-size": "0.82rem"}))
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Python · XGBoost · LightGBM · Streamlit &nbsp;|&nbsp;
    Data: KenPom, Torvik T-Rank, Massey Ordinals, Kaggle MMLM 2026 &nbsp;|&nbsp;
    Predictions are probabilistic — upsets happen.
</div>
""", unsafe_allow_html=True)

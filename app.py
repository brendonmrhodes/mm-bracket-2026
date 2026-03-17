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

@st.cache_data
def load_data():
    round_df = pd.read_csv(BASE / "outputs" / "round_probs_2026.csv")
    sub_df   = pd.read_csv(BASE / "outputs" / "submission_2026.csv")
    fi_xgb   = pd.read_csv(BASE / "outputs" / "feature_importance_xgb.csv",
                            header=None, names=["feature","importance"])
    fi_lgb   = pd.read_csv(BASE / "outputs" / "feature_importance_lgb.csv",
                            header=None, names=["feature","importance"])

    for fi in [fi_xgb, fi_lgb]:
        fi.drop(fi[fi["feature"].isna() | (fi["feature"] == "")].index, inplace=True)
        fi["importance"] = pd.to_numeric(fi["importance"], errors="coerce")
        fi.dropna(inplace=True)
        fi.sort_values("importance", ascending=False, inplace=True)

    prob_lookup = {}
    for _, row in sub_df.iterrows():
        _, t1, t2 = row["ID"].split("_")
        prob_lookup[(int(t1), int(t2))] = float(row["Pred"])

    return round_df, fi_xgb, fi_lgb, prob_lookup

round_df, fi_xgb, fi_lgb, prob_lookup = load_data()

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
tab_odds, tab_bracket, tab_matchup, tab_pool, tab_model = st.tabs([
    "Championship Odds",
    "Full Bracket",
    "Matchup Explorer",
    "Pool Strategy",
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

    # Full table — all rounds
    st.markdown('<div class="stitle">All Teams — Round-by-Round Probabilities</div>',
                unsafe_allow_html=True)
    tbl = disp[["SeedDisplay","TeamName",
                "prob_R32","prob_S16","prob_E8","prob_F4","prob_NCG","prob_Champion"]].copy()
    tbl.columns = ["Seed","Team","Round of 32","Sweet 16","Elite 8",
                   "Final Four","Champ. Game","Champion"]
    for c in tbl.columns[2:]:
        tbl[c] = (tbl[c] * 100).round(1)

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
        """Return HTML for one first-round matchup card.
        Green border + badge = model's predicted winner. Gray = loser.
        ff_id_a/ff_id_b set when one slot is a First Four play-in pair.
        """
        # Play-in matchup (two teams competing for one slot)
        if ff_id_a is not None and ff_id_b is not None:
            p = win_prob(ff_id_a, ff_id_b) * 100
            n_a = id_to_name.get(ff_id_a, "TBD")
            n_b = id_to_name.get(ff_id_b, "TBD")
            s_a = id_to_seeddisplay.get(ff_id_a, "?")
            s_b = id_to_seeddisplay.get(ff_id_b, "?")
            a_wins = p >= 50
            # top is winner
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
                # The lower seed slot is a First Four game
                pair = first_four[low_code]
                id_a, id_b = pair[0][1], pair[1][1]
                id_high = seed_to_id.get(high_code)

                # ① Play-in matchup
                html += f"""
                <div style="font-size:0.68rem;font-weight:700;color:#aaa;
                    text-transform:uppercase;letter-spacing:0.6px;
                    margin:6px 0 3px 2px;">Play-In Game</div>"""
                html += matchup_html(None, None, ff_id_a=id_a, ff_id_b=id_b)

                # ② First round: high seed vs likely play-in winner
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

    # First Four
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
        A weighted ensemble of <b>XGBoost</b> (45%), <b>LightGBM</b> (45%), and
        <b>Logistic Regression</b> (10%), trained on every NCAA tournament game since 2002
        using walk-forward cross-validation (no future data leakage). 153 features per matchup
        spanning efficiency ratings, Elo momentum, strength of schedule, and shot quality.
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


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Python · XGBoost · LightGBM · Streamlit &nbsp;|&nbsp;
    Data: KenPom, Torvik T-Rank, Massey Ordinals, Kaggle MMLM 2026 &nbsp;|&nbsp;
    Predictions are probabilistic — upsets happen.
</div>
""", unsafe_allow_html=True)

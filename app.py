"""
2026 NCAA Tournament Prediction Dashboard
Florida Gators themed — powered by XGBoost + LightGBM + Logistic Regression ensemble
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="2026 NCAA Bracket Predictions",
    page_icon="🐊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ORANGE     = "#FA4616"
BLUE       = "#0021A5"
LIGHT_BG   = "#F4F6FF"
DARK_BLUE  = "#001580"
MID_ORANGE = "#FF7A50"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Header banner */
    .hero {{
        background: linear-gradient(135deg, {BLUE} 0%, {DARK_BLUE} 50%, {ORANGE} 100%);
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,33,165,0.18);
    }}
    .hero h1 {{
        color: white;
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -1px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}
    .hero p {{
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin: 0;
    }}
    .hero .gator-emoji {{
        font-size: 3rem;
        display: block;
        margin-bottom: 8px;
    }}

    /* Stat cards */
    .stat-card {{
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 5px solid {ORANGE};
        box-shadow: 0 2px 12px rgba(0,33,165,0.08);
        margin-bottom: 12px;
    }}
    .stat-card .label {{
        font-size: 0.78rem;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }}
    .stat-card .value {{
        font-size: 2rem;
        font-weight: 800;
        color: {BLUE};
    }}
    .stat-card .sub {{
        font-size: 0.85rem;
        color: #888;
        margin-top: 2px;
    }}

    /* Gator spotlight card */
    .gator-card {{
        background: linear-gradient(135deg, {BLUE}, {DARK_BLUE});
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        box-shadow: 0 4px 16px rgba(0,33,165,0.25);
    }}
    .gator-card .label {{
        font-size: 0.78rem;
        font-weight: 600;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}
    .gator-card .value {{
        font-size: 2rem;
        font-weight: 800;
        color: {ORANGE};
    }}
    .gator-card .sub {{
        font-size: 0.85rem;
        color: rgba(255,255,255,0.75);
    }}

    /* Section headers */
    .section-title {{
        font-size: 1.4rem;
        font-weight: 800;
        color: {BLUE};
        border-bottom: 3px solid {ORANGE};
        padding-bottom: 8px;
        margin-bottom: 20px;
        display: inline-block;
    }}

    /* Value badge */
    .value-badge {{
        background: {ORANGE};
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 12px;
        margin-left: 6px;
    }}

    /* Tab styling override */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: {LIGHT_BG};
        padding: 6px;
        border-radius: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        font-weight: 600;
        color: {BLUE};
    }}
    .stTabs [aria-selected="true"] {{
        background: {ORANGE} !important;
        color: white !important;
    }}

    /* Matchup result box */
    .matchup-win {{
        background: linear-gradient(135deg, {ORANGE}, {MID_ORANGE});
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
    }}
    .matchup-win h2 {{
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }}
    .matchup-neutral {{
        background: {LIGHT_BG};
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 2px solid #dde;
    }}
    .matchup-neutral h2 {{
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: {BLUE};
    }}

    /* Footer */
    .footer {{
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        padding: 24px 0 8px;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }}

    /* Hide streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_data
def load_data():
    round_df   = pd.read_csv(BASE / "outputs" / "round_probs_2026.csv")
    pool_df    = pd.read_csv(BASE / "outputs" / "pool_strategy_2026.csv")
    sub_df     = pd.read_csv(BASE / "outputs" / "submission_2026.csv")
    fi_xgb     = pd.read_csv(BASE / "outputs" / "feature_importance_xgb.csv",
                              header=None, names=["feature", "importance"])
    fi_lgb     = pd.read_csv(BASE / "outputs" / "feature_importance_lgb.csv",
                              header=None, names=["feature", "importance"])

    # Drop header row if present
    fi_xgb = fi_xgb[fi_xgb["feature"] != ""].dropna()
    fi_lgb = fi_lgb[fi_lgb["feature"] != ""].dropna()
    fi_xgb["importance"] = pd.to_numeric(fi_xgb["importance"], errors="coerce")
    fi_lgb["importance"]  = pd.to_numeric(fi_lgb["importance"],  errors="coerce")
    fi_xgb = fi_xgb.dropna().sort_values("importance", ascending=False)
    fi_lgb = fi_lgb.dropna().sort_values("importance", ascending=False)

    # Build matchup probability lookup: (min_id, max_id) -> prob(min wins)
    prob_lookup = {}
    for _, row in sub_df.iterrows():
        parts = row["ID"].split("_")
        t1, t2 = int(parts[1]), int(parts[2])
        prob_lookup[(t1, t2)] = float(row["Pred"])

    return round_df, pool_df, fi_xgb, fi_lgb, prob_lookup

round_df, pool_df, fi_xgb, fi_lgb, prob_lookup = load_data()

# Team name → ID map
team_map = dict(zip(round_df["TeamName"], round_df["TeamID"]))
team_names_sorted = sorted(round_df["TeamName"].tolist())

def get_win_prob(team_a_name, team_b_name):
    """Return P(team_a beats team_b)."""
    id_a = team_map.get(team_a_name)
    id_b = team_map.get(team_b_name)
    if id_a is None or id_b is None:
        return 0.5
    key = (min(id_a, id_b), max(id_a, id_b))
    p = prob_lookup.get(key, 0.5)
    return p if id_a < id_b else 1 - p

def get_team_row(team_name):
    return round_df[round_df["TeamName"] == team_name].iloc[0]

# Friendly feature name mapping
FEAT_LABELS = {
    "d_elo_pre_tourney":  "Pre-Tournament Elo Gap",
    "d_adjEM":            "Adj. Efficiency Margin Gap",
    "d_AvgScoreDiff":     "Avg Score Margin Gap",
    "d_avg_ScoreDiff":    "Avg Score Margin Gap (wtd)",
    "d_elo_last10":       "Elo – Last 10 Games Gap",
    "d_sos_adjEM":        "Strength of Schedule Gap",
    "t1_elo_pre_tourney": "Team Elo Rating",
    "d_elo_peak":         "Peak Elo Gap",
    "d_qual_games":       "Quality Games Gap",
    "d_opp_D":            "Opp Defensive Efficiency Gap",
    "d_rank_composite":   "Composite Ranking Gap",
    "d_WinPct":           "Win % Gap",
    "d_wab":              "Wins Above Bubble Gap",
    "d_elo_momentum":     "Elo Momentum Gap",
    "d_luck":             "Luck Factor Gap",
    "d_elo_consistency":  "Elo Consistency Gap",
    "d_barthag":          "BARTHAG Gap",
    "d_adjO":             "Adj. Offensive Efficiency Gap",
    "d_adjD":             "Adj. Defensive Efficiency Gap",
    "d_rank_POM":         "KenPom Ranking Gap",
    "d_rank_BPI":         "ESPN BPI Ranking Gap",
    "d_rank_NET":         "NCAA NET Ranking Gap",
    "t1_adjO":            "Team Adj. Offense",
    "t1_SeedNum":         "Team Seed",
    "t1_avg_ScoreDiff":   "Team Avg Score Margin",
    "t1_elo_late_winpct": "Team Late-Season Win%",
    "t2_elo_pre_tourney": "Opponent Elo Rating",
    "t2_rank_DOK":        "Opp Dokter Ranking",
    "t2_sos_adjEM":       "Opp Strength of Schedule",
}

def label(feat):
    return FEAT_LABELS.get(feat, feat.replace("d_", "").replace("t1_", "").replace("t2_", "").replace("_", " ").title() + " Factor")


# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="gator-emoji">🐊</span>
    <h1>2026 NCAA Tournament Predictions</h1>
    <p>ML ensemble model · XGBoost + LightGBM + Logistic Regression · 100,000 bracket simulations</p>
</div>
""", unsafe_allow_html=True)


# ── Florida Gator spotlight ───────────────────────────────────────────────────
gator_row = round_df[round_df["TeamName"] == "Florida"]
if not gator_row.empty:
    g = gator_row.iloc[0]
    champ_pct  = g["prob_Champion"] * 100
    final_pct  = g["prob_NCG"]      * 100
    f4_pct     = g["prob_F4"]       * 100

    st.markdown('<div class="section-title">🐊 Gator Spotlight</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="gator-card">
            <div class="label">Seed</div>
            <div class="value">{g['Seed']}</div>
            <div class="sub">Florida Gators</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="gator-card">
            <div class="label">Final Four %</div>
            <div class="value">{f4_pct:.1f}%</div>
            <div class="sub">Probability of reaching F4</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="gator-card">
            <div class="label">Championship Game %</div>
            <div class="value">{final_pct:.1f}%</div>
            <div class="sub">Probability of reaching NCG</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="gator-card">
            <div class="label">Championship %</div>
            <div class="value">{champ_pct:.1f}%</div>
            <div class="sub">Probability of winning it all</div>
        </div>""", unsafe_allow_html=True)
    st.write("")

st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆  Championship Odds",
    "⚔️  Matchup Explorer",
    "💰  Pool Strategy",
    "🔬  Model Insights",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Championship Odds
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Championship Probability Rankings</div>', unsafe_allow_html=True)

    display_df = round_df.copy()
    display_df = display_df.sort_values("prob_Champion", ascending=False).reset_index(drop=True)

    # Top 5 spotlight cards
    top5 = display_df.head(5)
    cols = st.columns(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        with cols[i]:
            is_gator = row["TeamName"] == "Florida"
            card_class = "gator-card" if is_gator else "stat-card"
            rank_label = ["🥇","🥈","🥉","4th","5th"][i]
            st.markdown(f"""
            <div class="{card_class}">
                <div class="label">{rank_label} — {row['Seed']}</div>
                <div class="value">{row['prob_Champion']*100:.1f}%</div>
                <div class="sub">{row['TeamName']}</div>
            </div>""", unsafe_allow_html=True)
    st.write("")

    # Bar chart — top 20
    top20 = display_df.head(20)
    colors = [ORANGE if t == "Florida" else BLUE for t in top20["TeamName"]]

    fig = go.Figure(go.Bar(
        x=top20["TeamName"],
        y=top20["prob_Champion"] * 100,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in top20["prob_Champion"] * 100],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Championship: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Top 20 Championship Probabilities", font=dict(color=BLUE, size=16)),
        xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
        yaxis=dict(title="Championship %", gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(t=50, b=80, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.markdown('<div class="section-title">Full Round-by-Round Probabilities</div>', unsafe_allow_html=True)

    table_df = display_df[["Seed", "TeamName", "prob_F4", "prob_NCG", "prob_Champion"]].copy()
    table_df.columns = ["Seed", "Team", "Final Four %", "Championship Game %", "Champion %"]
    for col in ["Final Four %", "Championship Game %", "Champion %"]:
        table_df[col] = (table_df[col] * 100).round(1)

    # Highlight Florida row
    def highlight_gator(row):
        if row["Team"] == "Florida":
            return [f"background-color: {BLUE}; color: white; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        table_df.style.apply(highlight_gator, axis=1).format({
            "Final Four %": "{:.1f}%",
            "Championship Game %": "{:.1f}%",
            "Champion %": "{:.1f}%",
        }),
        use_container_width=True,
        height=480,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Matchup Explorer
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Head-to-Head Matchup Explorer</div>', unsafe_allow_html=True)
    st.caption("Pick any two tournament teams to see the model's win probability and compare their paths through the bracket.")

    c_left, c_right = st.columns(2)
    with c_left:
        team_a = st.selectbox("Team A", team_names_sorted,
                               index=team_names_sorted.index("Florida") if "Florida" in team_names_sorted else 0,
                               key="team_a")
    with c_right:
        default_b = "Duke" if "Duke" in team_names_sorted else team_names_sorted[1]
        team_b = st.selectbox("Team B", team_names_sorted,
                               index=team_names_sorted.index(default_b),
                               key="team_b")

    if team_a == team_b:
        st.warning("Please select two different teams.")
    else:
        p_a = get_win_prob(team_a, team_b)
        p_b = 1 - p_a

        row_a = get_team_row(team_a)
        row_b = get_team_row(team_b)

        st.write("")
        col_a, col_vs, col_b = st.columns([5, 1, 5])

        with col_a:
            box = "matchup-win" if p_a > 0.5 else "matchup-neutral"
            st.markdown(f"""
            <div class="{box}">
                <div style="font-size:0.85rem; font-weight:600; opacity:0.8; margin-bottom:4px;">
                    {row_a['Seed']}
                </div>
                <h2>{team_a}</h2>
                <div style="font-size:3rem; font-weight:900; margin:8px 0;">
                    {p_a*100:.1f}%
                </div>
                <div style="font-size:0.9rem; opacity:0.85;">win probability</div>
            </div>""", unsafe_allow_html=True)

        with col_vs:
            st.markdown("""
            <div style="display:flex; align-items:center; justify-content:center;
                        height:160px; font-size:1.5rem; font-weight:800; color:#aaa;">
                VS
            </div>""", unsafe_allow_html=True)

        with col_b:
            box = "matchup-win" if p_b > 0.5 else "matchup-neutral"
            st.markdown(f"""
            <div class="{box}">
                <div style="font-size:0.85rem; font-weight:600; opacity:0.8; margin-bottom:4px;">
                    {row_b['Seed']}
                </div>
                <h2>{team_b}</h2>
                <div style="font-size:3rem; font-weight:900; margin:8px 0;">
                    {p_b*100:.1f}%
                </div>
                <div style="font-size:0.9rem; opacity:0.85;">win probability</div>
            </div>""", unsafe_allow_html=True)

        st.write("")

        # Probability gauge
        fig_gauge = go.Figure(go.Bar(
            x=[p_a * 100, p_b * 100],
            y=[team_a, team_b],
            orientation="h",
            marker_color=[ORANGE if team_a == "Florida" else BLUE,
                          ORANGE if team_b == "Florida" else "#5566CC"],
            text=[f"{p_a*100:.1f}%", f"{p_b*100:.1f}%"],
            textposition="inside",
            textfont=dict(color="white", size=14, family="Inter"),
            hovertemplate="%{y}: %{x:.1f}% win probability<extra></extra>",
        ))
        fig_gauge.update_layout(
            xaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="#eee"),
            yaxis=dict(tickfont=dict(size=14, color=BLUE)),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=130,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Side-by-side bracket path comparison
        st.markdown('<div class="section-title">Bracket Path Comparison</div>', unsafe_allow_html=True)

        rounds = ["Final Four", "Championship Game", "Champion"]
        cols_r = ["prob_F4", "prob_NCG", "prob_Champion"]
        comp_data = {
            "Round": rounds,
            team_a: [f"{row_a[c]*100:.1f}%" for c in cols_r],
            team_b: [f"{row_b[c]*100:.1f}%" for c in cols_r],
        }
        comp_df = pd.DataFrame(comp_data)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name=team_a,
            x=rounds,
            y=[row_a[c] * 100 for c in cols_r],
            marker_color=ORANGE if team_a == "Florida" else BLUE,
            text=[f"{row_a[c]*100:.1f}%" for c in cols_r],
            textposition="outside",
        ))
        fig_comp.add_trace(go.Bar(
            name=team_b,
            x=rounds,
            y=[row_b[c] * 100 for c in cols_r],
            marker_color=ORANGE if team_b == "Florida" else "#5566CC",
            text=[f"{row_b[c]*100:.1f}%" for c in cols_r],
            textposition="outside",
        ))
        fig_comp.update_layout(
            barmode="group",
            yaxis=dict(title="Probability (%)", gridcolor="#eee"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=340,
            margin=dict(t=20, b=20, l=40, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color=BLUE),
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Pool Strategy
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Bracket Pool Strategy — Expected Value Analysis</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{LIGHT_BG}; border-radius:10px; padding:16px 20px; margin-bottom:20px;
                border-left:4px solid {ORANGE};">
        <b style="color:{BLUE}">How to read this:</b>
        The <b>EV Ratio</b> compares the model's championship probability against
        the estimated public pick percentage. An EV > 1.0 means the model thinks this team
        is <i>undervalued</i> by the public — they win more often than people expect.
        <b>EV > 1.3 = high value pick</b> for large bracket pools.
    </div>
    """, unsafe_allow_html=True)

    pool_display = pool_df.copy().sort_values("EV_ratio", ascending=False)
    pool_display["Value"] = pool_display["EV_ratio"].apply(
        lambda x: "🔥 High Value" if x > 1.3 else ("✅ Fair" if x > 0.9 else "⚠️ Overvalued")
    )

    # EV chart
    top_pool = pool_display[pool_display["Model%"] >= 1.0].head(16)
    bar_colors = []
    for _, row in top_pool.iterrows():
        if row["Team"] == "Florida":
            bar_colors.append(ORANGE)
        elif row["EV_ratio"] > 1.3:
            bar_colors.append("#22c55e")  # green for value
        elif row["EV_ratio"] < 0.9:
            bar_colors.append("#ef4444")  # red for overvalued
        else:
            bar_colors.append(BLUE)

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Bar(
        x=top_pool["Team"],
        y=top_pool["EV_ratio"],
        marker_color=bar_colors,
        text=[f"{v:.2f}x" for v in top_pool["EV_ratio"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>EV Ratio: %{y:.2f}x<extra></extra>",
    ))
    fig_ev.add_hline(y=1.0, line_dash="dash", line_color="#aaa",
                     annotation_text="Breakeven (1.0x)", annotation_position="top right")
    fig_ev.add_hline(y=1.3, line_dash="dot", line_color="#22c55e",
                     annotation_text="High Value (1.3x)", annotation_position="top right")
    fig_ev.update_layout(
        title=dict(text="Expected Value Ratio by Team", font=dict(color=BLUE, size=16)),
        xaxis=dict(tickangle=-30, tickfont=dict(size=11)),
        yaxis=dict(title="EV Ratio", gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(t=50, b=80, l=40, r=20),
    )
    st.plotly_chart(fig_ev, use_container_width=True)

    # Table
    st.markdown('<div class="section-title">Full EV Table</div>', unsafe_allow_html=True)

    def color_ev(val):
        try:
            v = float(val)
            if v > 1.3:
                return f"color: #15803d; font-weight: bold"
            elif v < 0.9:
                return f"color: #dc2626; font-weight: bold"
        except:
            pass
        return ""

    pool_table = pool_display[["Seed", "Team", "Model%", "Field%", "EV_ratio", "F4%", "Value"]].copy()
    pool_table.columns = ["Seed", "Team", "Model Champ%", "Est. Public Pick%", "EV Ratio", "Final Four%", "Assessment"]

    st.dataframe(
        pool_table.style.applymap(color_ev, subset=["EV Ratio"]),
        use_container_width=True,
        height=480,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Insights
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">What Does the Model Look At?</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{LIGHT_BG}; border-radius:10px; padding:16px 20px; margin-bottom:24px;
                border-left:4px solid {ORANGE};">
        The model is a weighted ensemble of <b>XGBoost</b> (45%), <b>LightGBM</b> (45%),
        and <b>Logistic Regression</b> (10%), trained on every NCAA tournament game since 2002
        using walk-forward cross-validation (no data leakage).
        Features include KenPom efficiency ratings, Torvik T-Rank stats, Elo ratings, Massey ordinals,
        and box-score Four Factors from 40+ years of game logs.
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)

    def make_fi_chart(df, title, color):
        top = df[df["feature"] != "feature"].head(15).copy()
        top["label"] = top["feature"].apply(label)
        top["importance"] = pd.to_numeric(top["importance"], errors="coerce")
        top = top.dropna().sort_values("importance")

        fig = go.Figure(go.Bar(
            x=top["importance"],
            y=top["label"],
            orientation="h",
            marker_color=color,
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(color=BLUE, size=14)),
            xaxis=dict(title="Feature Importance", gridcolor="#eee"),
            yaxis=dict(tickfont=dict(size=11)),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=460,
            margin=dict(t=40, b=20, l=10, r=10),
        )
        return fig

    with col_m1:
        st.plotly_chart(make_fi_chart(fi_xgb, "XGBoost — Top 15 Features", BLUE),
                        use_container_width=True)
    with col_m2:
        st.plotly_chart(make_fi_chart(fi_lgb, "LightGBM — Top 15 Features", ORANGE),
                        use_container_width=True)

    # Model performance callout
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Walk-Forward CV Accuracy</div>
            <div class="value">86.6%</div>
            <div class="sub">Tested on 2010–2025 tournaments</div>
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Log-Loss</div>
            <div class="value">0.320</div>
            <div class="sub">Lower is better (Kaggle metric)</div>
        </div>""", unsafe_allow_html=True)
    with p3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Bracket Simulations</div>
            <div class="value">100,000</div>
            <div class="sub">Monte Carlo runs per prediction</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{LIGHT_BG}; border-radius:10px; padding:16px 20px; margin-top:20px;
                border-left:4px solid {BLUE};">
        <b style="color:{BLUE}">Data sources:</b> KenPom (2002–2026), Torvik T-Rank (2008–2026),
        Kaggle NCAA box scores (1985–2026), Massey Ordinals composite ranking (10 systems),
        Elo ratings with margin-of-victory multiplier & recency momentum features.
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🐊 Built with Python · XGBoost · LightGBM · Streamlit &nbsp;|&nbsp;
    Data: KenPom, Torvik T-Rank, Massey Ordinals, Kaggle MMLM 2026 &nbsp;|&nbsp;
    Predictions are probabilistic — upsets happen! &nbsp;🏀
</div>
""", unsafe_allow_html=True)

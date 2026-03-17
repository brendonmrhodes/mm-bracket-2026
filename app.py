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
FAINT     = "#EAEEF8"

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_seed(seed_str: str) -> str:
    """'W01' -> '1', 'X11a' -> '11a', 'Y16b' -> '16b'"""
    s = seed_str[1:]  # strip region letter
    if s and s[-1] in "ab":
        return s.lstrip("0")[:-1].lstrip("0") + s[-1] if len(s) > 1 else s
    return s.lstrip("0") or "0"

def fmt_seed_clean(seed_str: str) -> str:
    """Return just the numeric part, ignoring a/b suffix."""
    s = seed_str[1:]
    if s and s[-1] in "ab":
        s = s[:-1]
    return s.lstrip("0") or "0"

REGION_NAMES = {"W": "East", "X": "South", "Y": "Midwest", "Z": "West"}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    html, body, [class*="css"] {{ font-family: -apple-system, BlinkMacSystemFont,
        'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}

    /* Hero */
    .hero {{
        background: linear-gradient(135deg, {BLUE} 0%, {DARK_BLUE} 60%, {ORANGE} 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 28px;
        box-shadow: 0 8px 32px rgba(0,33,165,0.2);
    }}
    .hero-logo img {{
        width: 90px; height: 90px; object-fit: contain;
        filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3));
    }}
    .hero-text h1 {{
        color: white; font-size: 2.2rem; font-weight: 800;
        margin: 0 0 4px; letter-spacing: -0.5px;
    }}
    .hero-text p {{ color: rgba(255,255,255,0.82); font-size: 0.95rem; margin: 0; }}

    /* Stat cards */
    .stat-card {{
        background: white; border-radius: 12px; padding: 18px 22px;
        border-left: 5px solid {ORANGE};
        box-shadow: 0 2px 10px rgba(0,33,165,0.07);
    }}
    .stat-card .lbl {{ font-size: 0.72rem; font-weight: 700; color: #888;
        text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }}
    .stat-card .val {{ font-size: 1.9rem; font-weight: 800; color: {BLUE}; }}
    .stat-card .sub {{ font-size: 0.8rem; color: #999; margin-top: 2px; }}

    /* Gator accent cards */
    .gator-card {{
        background: linear-gradient(135deg, {BLUE}, {DARK_BLUE});
        border-radius: 12px; padding: 18px 22px; color: white;
        box-shadow: 0 4px 16px rgba(0,33,165,0.22);
    }}
    .gator-card .lbl {{ font-size: 0.72rem; font-weight: 700;
        color: rgba(255,255,255,0.65); text-transform: uppercase;
        letter-spacing: 0.8px; margin-bottom: 4px; }}
    .gator-card .val {{ font-size: 1.9rem; font-weight: 800; color: {ORANGE}; }}
    .gator-card .sub {{ font-size: 0.8rem; color: rgba(255,255,255,0.7); margin-top: 2px; }}

    /* Section title */
    .stitle {{
        font-size: 1.15rem; font-weight: 800; color: {BLUE};
        border-bottom: 3px solid {ORANGE};
        padding-bottom: 6px; margin-bottom: 18px; display: inline-block;
    }}

    /* Info banner */
    .info-banner {{
        background: {LIGHT_BG}; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 18px;
        border-left: 4px solid {ORANGE};
    }}

    /* Bracket matchup card */
    .mc {{
        background: white; border-radius: 8px;
        box-shadow: 0 1px 6px rgba(0,33,165,0.07);
        margin-bottom: 6px; overflow: hidden;
    }}
    .mc-team {{
        display: flex; align-items: center; padding: 7px 12px; gap: 10px;
        border-bottom: 1px solid #f0f2f8;
    }}
    .mc-team:last-child {{ border-bottom: none; }}
    .mc-seed {{
        font-size: 0.7rem; font-weight: 700; color: white;
        background: {BLUE}; border-radius: 4px;
        padding: 2px 5px; min-width: 22px; text-align: center;
        flex-shrink: 0;
    }}
    .mc-seed-hi {{ background: {ORANGE}; }}
    .mc-name {{ font-size: 0.85rem; font-weight: 600; color: #222; flex: 1; }}
    .mc-name-gator {{ color: {ORANGE} !important; }}
    .mc-prob {{ font-size: 0.85rem; font-weight: 700; color: {BLUE};
        min-width: 38px; text-align: right; }}
    .mc-bar-wrap {{
        height: 3px; background: #eef;
    }}
    .mc-bar {{ height: 3px; background: {ORANGE}; transition: width 0.3s; }}

    /* Region header */
    .region-hdr {{
        background: linear-gradient(90deg, {BLUE}, {DARK_BLUE});
        color: white; border-radius: 8px 8px 0 0;
        padding: 8px 14px; font-size: 0.8rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px;
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 4px;
    }}
    .region-hdr .fav {{ color: {ORANGE}; font-size: 0.75rem; }}

    /* Matchup explorer */
    .mx-box {{
        border-radius: 12px; padding: 22px; text-align: center;
    }}
    .mx-win {{ background: linear-gradient(135deg, {ORANGE}, #FF7A50); color: white; }}
    .mx-neutral {{ background: {LIGHT_BG}; border: 2px solid #dde; }}
    .mx-box .seed-lbl {{
        font-size: 0.78rem; font-weight: 600; opacity: 0.8; margin-bottom: 4px;
    }}
    .mx-box h2 {{ font-size: 1.8rem; font-weight: 800; margin: 0; }}
    .mx-box .pct {{ font-size: 2.8rem; font-weight: 900; margin: 8px 0; }}
    .mx-win h2, .mx-win .pct {{ color: white; }}
    .mx-neutral h2, .mx-neutral .pct {{ color: {BLUE}; }}
    .mx-box .sub {{ font-size: 0.85rem; opacity: 0.8; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; background: {LIGHT_BG};
        padding: 5px; border-radius: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 7px; font-weight: 600; color: {BLUE};
        font-size: 0.9rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: {ORANGE} !important; color: white !important;
    }}

    /* Footer */
    .footer {{
        text-align: center; color: #bbb; font-size: 0.78rem;
        padding: 20px 0 8px; border-top: 1px solid #eee; margin-top: 36px;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

@st.cache_data
def load_data():
    round_df = pd.read_csv(BASE / "outputs" / "round_probs_2026.csv")
    pool_df  = pd.read_csv(BASE / "outputs" / "pool_strategy_2026.csv")
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

    return round_df, pool_df, fi_xgb, fi_lgb, prob_lookup

round_df, pool_df, fi_xgb, fi_lgb, prob_lookup = load_data()

# Clean seed display — add to round_df
round_df["SeedDisplay"] = round_df["Seed"].apply(fmt_seed)

# Seed → TeamID lookup
seed_to_id   = dict(zip(round_df["Seed"], round_df["TeamID"]))
id_to_name   = dict(zip(round_df["TeamID"], round_df["TeamName"]))
id_to_seed   = dict(zip(round_df["TeamID"], round_df["Seed"]))
id_to_seeddisplay = dict(zip(round_df["TeamID"], round_df["SeedDisplay"]))
team_names_sorted = sorted(round_df["TeamName"].tolist())

def win_prob(id_a: int, id_b: int) -> float:
    """P(team_a beats team_b)"""
    key = (min(id_a, id_b), max(id_a, id_b))
    p = prob_lookup.get(key, 0.5)
    return p if id_a < id_b else 1 - p

def team_row(name: str) -> pd.Series:
    return round_df[round_df["TeamName"] == name].iloc[0]

# ── Feature label mapping ─────────────────────────────────────────────────────
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
    "d_rank_POM": "KenPom Ranking Gap",
    "d_rank_BPI": "ESPN BPI Ranking Gap",
    "d_rank_NET": "NCAA NET Ranking Gap",
    "t1_adjO": "Team Adj. Offense",
    "t1_SeedNum": "Team Seed",
    "t1_avg_ScoreDiff": "Team Avg Score Margin",
    "t1_elo_late_winpct": "Team Late-Season Win %",
    "t2_elo_pre_tourney": "Opponent Elo Rating",
    "t2_rank_DOK": "Opponent Dokter Ranking",
    "t2_sos_adjEM": "Opponent Strength of Schedule",
}
def feat_label(f):
    return FEAT_LABELS.get(f,
        f.replace("d_","").replace("t1_","").replace("t2_","")
         .replace("_"," ").title())


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-logo">
        <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/57.png"
             onerror="this.style.display='none'" />
    </div>
    <div class="hero-text">
        <h1>2026 NCAA Tournament Predictions</h1>
        <p>ML ensemble model &nbsp;·&nbsp; XGBoost + LightGBM + Logistic Regression
           &nbsp;·&nbsp; 100,000 bracket simulations</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Florida Gator spotlight ───────────────────────────────────────────────────
gator = round_df[round_df["TeamName"] == "Florida"]
if not gator.empty:
    g = gator.iloc[0]
    st.markdown('<div class="stitle">Florida Gators — Tournament Outlook</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Seed", g["SeedDisplay"], "South Region"),
        ("Final Four", f"{g['prob_F4']*100:.1f}%", "Probability of reaching F4"),
        ("Championship Game", f"{g['prob_NCG']*100:.1f}%", "Probability of reaching NCG"),
        ("National Champion", f"{g['prob_Champion']*100:.1f}%", "Probability of winning it all"),
    ]
    for col, (lbl, val, sub) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""
            <div class="gator-card">
                <div class="lbl">{lbl}</div>
                <div class="val">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)
    st.write("")

st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────
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
    top5 = disp.head(5)
    cols5 = st.columns(5)
    medals = ["1st", "2nd", "3rd", "4th", "5th"]
    for i, (_, row) in enumerate(top5.iterrows()):
        with cols5[i]:
            is_g = row["TeamName"] == "Florida"
            cls  = "gator-card" if is_g else "stat-card"
            st.markdown(f"""
            <div class="{cls}">
                <div class="lbl">{medals[i]} &nbsp;·&nbsp; Seed {row['SeedDisplay']}</div>
                <div class="val">{row['prob_Champion']*100:.1f}%</div>
                <div class="sub">{row['TeamName']}</div>
            </div>""", unsafe_allow_html=True)
    st.write("")

    # Bar chart — top 20
    top20 = disp.head(20)
    colors = [ORANGE if t == "Florida" else BLUE for t in top20["TeamName"]]
    labels = [f"({row['SeedDisplay']}) {row['TeamName']}" for _, row in top20.iterrows()]

    fig = go.Figure(go.Bar(
        x=labels,
        y=top20["prob_Champion"] * 100,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in top20["prob_Champion"] * 100],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Championship: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Top 20 Championship Probabilities", font=dict(color=BLUE, size=15)),
        xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
        yaxis=dict(title="Championship %", gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, margin=dict(t=50, b=90, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.markdown('<div class="stitle">All Teams — Round-by-Round Probabilities</div>',
                unsafe_allow_html=True)

    tbl = disp[["SeedDisplay","TeamName","prob_F4","prob_NCG","prob_Champion"]].copy()
    tbl.columns = ["Seed","Team","Final Four %","Championship Game %","Champion %"]
    for c in ["Final Four %","Championship Game %","Champion %"]:
        tbl[c] = (tbl[c] * 100).round(1)

    def hl_gator(row):
        style = f"background-color:{BLUE};color:white;font-weight:bold"
        return [style]*len(row) if row["Team"] == "Florida" else [""]*len(row)

    st.dataframe(
        tbl.style.apply(hl_gator, axis=1).format(
            {"Final Four %":"{:.1f}%","Championship Game %":"{:.1f}%","Champion %":"{:.1f}%"}
        ),
        use_container_width=True, height=480,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Full Bracket
# ════════════════════════════════════════════════════════════════════════════
with tab_bracket:

    # Bracket order: (high_seed_num, low_seed_num)
    BRACKET_PAIRS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

    # Identify First Four seeds (have a/b variants in this year's field)
    first_four_seeds = {}
    for seed_code, tid in seed_to_id.items():
        if seed_code and seed_code[-1] in "ab":
            base = seed_code[:-1]  # e.g. "X16"
            first_four_seeds.setdefault(base, []).append((seed_code, tid))

    def get_team_id(region_letter: str, seed_num: int):
        """Look up TeamID for a region+seed, handling First Four slots."""
        code = f"{region_letter}{str(seed_num).zfill(2)}"
        if code in seed_to_id:
            return seed_to_id[code], None   # direct lookup
        # First Four placeholder
        key = code
        if key in first_four_seeds:
            a, b = first_four_seeds[key]
            return a[1], b[1]   # (team_a_id, team_b_id)
        return None, None

    def matchup_card_html(id_top, id_bot, id_top2=None, id_bot2=None):
        """
        Render one first-round matchup as HTML.
        id_top2/id_bot2 are set when the slot is a First Four play-in game.
        """
        def team_line(tid, seed_disp, is_fav, is_playin_winner=False):
            name  = id_to_name.get(tid, "TBD")
            is_g  = name == "Florida"
            nc    = "mc-name-gator" if is_g else ""
            sc    = "mc-seed-hi" if is_fav else ""
            return (name, seed_disp, nc, sc)

        # Play-in slot: show probability between the two play-in teams
        if id_top2 is not None:
            p = win_prob(id_top, id_top2) * 100
            n1, n2 = id_to_name.get(id_top,"TBD"), id_to_name.get(id_top2,"TBD")
            s1 = id_to_seeddisplay.get(id_top,"?")
            s2 = id_to_seeddisplay.get(id_top2,"?")
            g1 = "mc-name-gator" if n1=="Florida" else ""
            g2 = "mc-name-gator" if n2=="Florida" else ""
            return f"""
            <div class="mc">
              <div class="mc-team">
                <span class="mc-seed">{s1}</span>
                <span class="mc-name {g1}">{n1}</span>
                <span class="mc-prob">{p:.0f}%</span>
              </div>
              <div class="mc-bar-wrap"><div class="mc-bar" style="width:{p:.1f}%"></div></div>
              <div class="mc-team">
                <span class="mc-seed">{s2}</span>
                <span class="mc-name {g2}">{n2}</span>
                <span class="mc-prob">{100-p:.0f}%</span>
              </div>
            </div>"""

        if id_top is None or id_bot is None:
            return ""

        p = win_prob(id_top, id_bot) * 100
        n_top, n_bot = id_to_name.get(id_top,"TBD"), id_to_name.get(id_bot,"TBD")
        s_top = id_to_seeddisplay.get(id_top,"?")
        s_bot = id_to_seeddisplay.get(id_bot,"?")
        gt = "mc-name-gator" if n_top=="Florida" else ""
        gb = "mc-name-gator" if n_bot=="Florida" else ""
        sc_top = "mc-seed-hi" if p >= 50 else ""
        sc_bot = "mc-seed-hi" if p < 50 else ""

        return f"""
        <div class="mc">
          <div class="mc-team">
            <span class="mc-seed {sc_top}">{s_top}</span>
            <span class="mc-name {gt}">{n_top}</span>
            <span class="mc-prob">{p:.0f}%</span>
          </div>
          <div class="mc-bar-wrap"><div class="mc-bar" style="width:{p:.1f}%"></div></div>
          <div class="mc-team">
            <span class="mc-seed {sc_bot}">{s_bot}</span>
            <span class="mc-name {gb}">{n_bot}</span>
            <span class="mc-prob">{100-p:.0f}%</span>
          </div>
        </div>"""

    def render_region(region_letter: str):
        region_name = REGION_NAMES[region_letter]
        # Find projected region champion (highest F4 prob within region)
        region_teams = round_df[round_df["Seed"].str.startswith(region_letter)]
        if not region_teams.empty:
            fav = region_teams.loc[region_teams["prob_F4"].idxmax()]
            fav_str = f"Model favorite: {fav['TeamName']} ({fav['prob_F4']*100:.0f}% F4)"
        else:
            fav_str = ""

        cards_html = f"""
        <div style="margin-bottom:12px;">
          <div class="region-hdr">
            <span>{region_name} Region</span>
            <span class="fav">{fav_str}</span>
          </div>"""

        # Check for First Four games in this region
        region_ff = {k: v for k, v in first_four_seeds.items() if k.startswith(region_letter)}

        for high_s, low_s in BRACKET_PAIRS:
            slot_code = f"{region_letter}{str(low_s).zfill(2)}"
            if slot_code in first_four_seeds:
                # This is a First Four slot
                pair = first_four_seeds[slot_code]
                id_a, id_b = pair[0][1], pair[1][1]
                # High seed is direct
                high_code = f"{region_letter}{str(high_s).zfill(2)}"
                id_high = seed_to_id.get(high_code)

                # Show play-in card + main matchup label
                ff_card = matchup_card_html(id_a, None, id_b, None)
                main_card = ""
                if id_high:
                    n_high = id_to_name.get(id_high,"TBD")
                    gh = "mc-name-gator" if n_high=="Florida" else ""
                    s_high = id_to_seeddisplay.get(id_high,"?")
                    main_card = f"""
                    <div class="mc">
                      <div class="mc-team">
                        <span class="mc-seed mc-seed-hi">{s_high}</span>
                        <span class="mc-name {gh}">{n_high}</span>
                        <span class="mc-prob" style="color:#aaa;font-size:0.75rem;">vs winner</span>
                      </div>
                    </div>"""
                cards_html += ff_card + main_card
            else:
                # Normal matchup
                high_code = f"{region_letter}{str(high_s).zfill(2)}"
                low_code  = f"{region_letter}{str(low_s).zfill(2)}"
                id_high = seed_to_id.get(high_code)
                id_low  = seed_to_id.get(low_code)
                cards_html += matchup_card_html(id_high, id_low)

        cards_html += "</div>"
        return cards_html

    # First Four section
    ff_items = sorted(first_four_seeds.items())
    if ff_items:
        st.markdown('<div class="stitle">First Four — Play-In Games</div>',
                    unsafe_allow_html=True)
        ff_cols = st.columns(len(ff_items))
        for col, (base_code, pair) in zip(ff_cols, ff_items):
            with col:
                id_a, id_b = pair[0][1], pair[1][1]
                p = win_prob(id_a, id_b) * 100
                n_a = id_to_name.get(id_a,"TBD")
                n_b = id_to_name.get(id_b,"TBD")
                s_a = id_to_seeddisplay.get(id_a,"?")
                s_b = id_to_seeddisplay.get(id_b,"?")
                region_name = REGION_NAMES.get(base_code[0], "")
                seed_num    = base_code[1:].lstrip("0")
                ga = "mc-name-gator" if n_a=="Florida" else ""
                gb = "mc-name-gator" if n_b=="Florida" else ""
                sc_a = "mc-seed-hi" if p >= 50 else ""
                sc_b = "mc-seed-hi" if p < 50 else ""
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                  <div style="font-size:0.72rem;font-weight:700;color:{BLUE};
                              text-transform:uppercase;letter-spacing:0.8px;
                              margin-bottom:4px;">{region_name} · Seed {seed_num} Play-In</div>
                  <div class="mc">
                    <div class="mc-team">
                      <span class="mc-seed {sc_a}">{s_a}</span>
                      <span class="mc-name {ga}">{n_a}</span>
                      <span class="mc-prob">{p:.0f}%</span>
                    </div>
                    <div class="mc-bar-wrap"><div class="mc-bar" style="width:{p:.1f}%"></div></div>
                    <div class="mc-team">
                      <span class="mc-seed {sc_b}">{s_b}</span>
                      <span class="mc-name {gb}">{n_b}</span>
                      <span class="mc-prob">{100-p:.0f}%</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="stitle">First Round Matchups by Region</div>',
                unsafe_allow_html=True)
    st.caption("Seed highlighted in orange = model's predicted winner. Florida Gators shown in orange.")

    col_ew, col_sx = st.columns(2)
    col_mw, col_wt = st.columns(2)

    with col_ew:
        st.markdown(render_region("W"), unsafe_allow_html=True)
    with col_sx:
        st.markdown(render_region("X"), unsafe_allow_html=True)
    with col_mw:
        st.markdown(render_region("Y"), unsafe_allow_html=True)
    with col_wt:
        st.markdown(render_region("Z"), unsafe_allow_html=True)


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
        id_a = team_row(team_a)["TeamID"]
        id_b = team_row(team_b)["TeamID"]
        p_a = win_prob(int(id_a), int(id_b))
        p_b = 1 - p_a
        ra, rb = team_row(team_a), team_row(team_b)

        st.write("")
        ca, cv, cb = st.columns([5,1,5])
        with ca:
            cls = "mx-box mx-win" if p_a > 0.5 else "mx-box mx-neutral"
            st.markdown(f"""
            <div class="{cls}">
                <div class="seed-lbl">Seed {ra['SeedDisplay']} · {REGION_NAMES.get(ra['Seed'][0],'')}</div>
                <h2>{team_a}</h2>
                <div class="pct">{p_a*100:.1f}%</div>
                <div class="sub">win probability</div>
            </div>""", unsafe_allow_html=True)
        with cv:
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:170px;font-size:1.3rem;font-weight:800;color:#bbb;">
                vs
            </div>""", unsafe_allow_html=True)
        with cb:
            cls = "mx-box mx-win" if p_b > 0.5 else "mx-box mx-neutral"
            st.markdown(f"""
            <div class="{cls}">
                <div class="seed-lbl">Seed {rb['SeedDisplay']} · {REGION_NAMES.get(rb['Seed'][0],'')}</div>
                <h2>{team_b}</h2>
                <div class="pct">{p_b*100:.1f}%</div>
                <div class="sub">win probability</div>
            </div>""", unsafe_allow_html=True)

        st.write("")

        # Probability bar
        fig_bar = go.Figure(go.Bar(
            x=[p_a*100, p_b*100], y=[team_a, team_b], orientation="h",
            marker_color=[ORANGE if team_a=="Florida" else BLUE,
                          ORANGE if team_b=="Florida" else "#5566CC"],
            text=[f"{p_a*100:.1f}%", f"{p_b*100:.1f}%"],
            textposition="inside",
            textfont=dict(color="white", size=13),
        ))
        fig_bar.update_layout(
            xaxis=dict(range=[0,100], ticksuffix="%", gridcolor="#eee"),
            yaxis=dict(tickfont=dict(size=13, color=BLUE)),
            plot_bgcolor="white", paper_bgcolor="white",
            height=120, margin=dict(t=8,b=8,l=10,r=10), showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Bracket path comparison
        st.markdown('<div class="stitle">Bracket Path Comparison</div>',
                    unsafe_allow_html=True)
        rounds_labels = ["Final Four", "Championship Game", "Champion"]
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
# TAB 4 — Pool Strategy
# ════════════════════════════════════════════════════════════════════════════
with tab_pool:
    st.markdown('<div class="stitle">Bracket Pool Strategy — Expected Value Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-banner">
        <b style="color:{BLUE}">How to read this:</b>
        The <b>EV Ratio</b> compares the model's championship probability against the estimated
        public pick percentage (based on historical seed selection patterns). An EV above 1.0
        means the team wins more often than the public expects —
        <b>EV &gt; 1.3 is a high-value pick</b> in large pools.
    </div>""", unsafe_allow_html=True)

    pool_disp = pool_df.copy().sort_values("EV_ratio", ascending=False)
    pool_disp["Assessment"] = pool_disp["EV_ratio"].apply(
        lambda x: "High Value" if x > 1.3 else ("Fair" if x > 0.9 else "Overvalued")
    )

    top_pool = pool_disp[pool_disp["Model%"] >= 1.0].head(16)
    bar_colors = []
    for _, row in top_pool.iterrows():
        if row["Team"] == "Florida":        bar_colors.append(ORANGE)
        elif row["EV_ratio"] > 1.3:         bar_colors.append("#16a34a")
        elif row["EV_ratio"] < 0.9:         bar_colors.append("#dc2626")
        else:                               bar_colors.append(BLUE)

    fig_ev = go.Figure(go.Bar(
        x=top_pool["Team"], y=top_pool["EV_ratio"],
        marker_color=bar_colors,
        text=[f"{v:.2f}x" for v in top_pool["EV_ratio"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>EV: %{y:.2f}x<extra></extra>",
    ))
    fig_ev.add_hline(y=1.0, line_dash="dash", line_color="#bbb",
                     annotation_text="Breakeven", annotation_position="top right")
    fig_ev.add_hline(y=1.3, line_dash="dot", line_color="#16a34a",
                     annotation_text="High value", annotation_position="top right")
    fig_ev.update_layout(
        title=dict(text="Expected Value by Team (model pick% ÷ estimated public pick%)",
                   font=dict(color=BLUE, size=14)),
        xaxis=dict(tickangle=-30, tickfont=dict(size=11)),
        yaxis=dict(title="EV Ratio", gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, margin=dict(t=50,b=80,l=40,r=20),
    )
    st.plotly_chart(fig_ev, use_container_width=True)

    st.markdown('<div class="stitle">Full EV Table</div>', unsafe_allow_html=True)

    def color_ev_cell(val):
        try:
            v = float(val)
            if v > 1.3: return "color:#15803d;font-weight:bold"
            if v < 0.9: return "color:#dc2626;font-weight:bold"
        except: pass
        return ""

    tbl_pool = pool_disp[["Seed","Team","Model%","Field%","EV_ratio","F4%","Assessment"]].copy()
    tbl_pool.columns = ["Seed","Team","Model Champ %","Est. Public %","EV Ratio","Final Four %","Assessment"]
    st.dataframe(
        tbl_pool.style.applymap(color_ev_cell, subset=["EV Ratio"]),
        use_container_width=True, height=480,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Insights
# ════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="stitle">What the Model Looks At</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-banner">
        The model is a weighted ensemble of <b>XGBoost</b> (45%), <b>LightGBM</b> (45%), and
        <b>Logistic Regression</b> (10%), trained on every NCAA tournament game since 2002 using
        walk-forward cross-validation. Features include KenPom efficiency ratings, Torvik T-Rank,
        Elo ratings with recency momentum, Massey composite rankings, and Four Factors from
        40+ years of box scores — totaling 153 features per matchup.
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
    perf = [
        ("Walk-Forward CV Accuracy", "86.6%", "Tested on 2010–2025 tournaments"),
        ("Log-Loss (CV)", "0.320", "Lower is better — Kaggle scoring metric"),
        ("Bracket Simulations", "100,000", "Monte Carlo runs per prediction set"),
    ]
    for col, (lbl,val,sub) in zip([p1,p2,p3], perf):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="lbl">{lbl}</div>
                <div class="val">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-banner" style="margin-top:18px; border-left-color:{BLUE};">
        <b style="color:{BLUE}">Data sources:</b>
        KenPom (2002–2026) · Torvik T-Rank (2008–2026) · Kaggle NCAA box scores (1985–2026) ·
        Massey Ordinals composite from 10 rating systems · FiveThirtyEight-style Elo with
        margin-of-victory multiplier and recency momentum features.
    </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Python · XGBoost · LightGBM · Streamlit &nbsp;|&nbsp;
    Data: KenPom, Torvik T-Rank, Massey Ordinals, Kaggle MMLM 2026 &nbsp;|&nbsp;
    Predictions are probabilistic — upsets happen.
</div>
""", unsafe_allow_html=True)

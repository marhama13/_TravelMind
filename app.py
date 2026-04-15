"""
app.py  –  Personalized Travel Recommendation System
GLS University | Recommender Systems Lab | B.Tech CS&E Sem VI

Run with:  streamlit run app.py
"""

import os, io, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium

from data_loader import (
    load_attractions, load_restaurants, load_local_food,
    load_cost_estimation, load_user_ratings,
    get_country_city_map, BUDGET_MULTIPLIER,
)
from recommender import hybrid_recommend, build_itinerary, budget_filter_cost

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌍 TravelMind – Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.main { background: #F7F4EF; }
[data-testid="stAppViewContainer"] { background: #F7F4EF; }
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1B2A4A 0%, #243855 60%, #2D4A6B 100%);
}
[data-testid="stSidebar"] * { color: #E8EDF5 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFD580 !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1B2A4A 0%, #2D4A6B 50%, #3A6186 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: #FFFFFF !important;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,213,128,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
    color: #FFD580 !important;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #E8EDF5 !important;
    margin-top: 0.5rem;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,213,128,0.18);
    border: 1px solid rgba(255,213,128,0.4);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    color: #FFD580 !important;
    margin-bottom: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #0F1E3A !important;
    border-left: 4px solid #FFD580;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}

/* ── Cards ── */
.place-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 12px rgba(27,42,74,0.08);
    border-left: 4px solid #3A6186;
    transition: transform 0.2s ease;
}
.place-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(27,42,74,0.13); }
.place-card.gem { border-left-color: #E8955A; }
.place-card.restaurant { border-left-color: #4CAF82; }

.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #0F1E3A !important;
    margin-bottom: 0.2rem;
}
.card-meta {
    font-size: 0.82rem;
    color: #0F1E3A !important;
}
.score-badge {
    display: inline-block;
    background: #EEF2FF;
    color: #3A6186;
    border-radius: 8px;
    padding: 0.1rem 0.5rem;
    font-size: 0.75rem;
    font-weight: 600;
    float: right;
}

/* ── Metric Cards ── */
.metric-row {
    display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1; min-width: 140px;
    background: white;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(27,42,74,0.07);
    
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0F1E3A !important;
    
}
.metric-label {
    font-size: 0.78rem;
    color: #3A4B6A !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
    
}

/* ── Itinerary Table ── */
.itin-table { width: 100%; border-collapse: collapse; }
.itin-table th {
    background: #1B2A4A; color: #FFD580;
    padding: 0.7rem 1rem; text-align: left;
    font-size: 0.85rem; font-weight: 600;
}
.itin-table td {
    padding: 0.7rem 1rem;
    border-bottom: 1px solid #EEF0F6;
    font-size: 0.88rem;
    color: #1B2A4A !important;
}
.itin-table tr:nth-child(even) td { background: #F9F8F5; }
.itin-table tr:hover td { background: #EEF4FF; }

/* ── Food chips ── */
.food-chip {
    display: inline-block;
    background: linear-gradient(135deg, #FF8C42, #E8955A);
    color: white;
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    margin: 0.3rem 0.3rem 0.3rem 0;
    font-size: 0.83rem;
    font-weight: 500;
}

/* ── Info Box ── */
.info-box {
    background: linear-gradient(135deg, #EBF4FF, #F0F7FF);
    border: 1px solid #C5DCF5;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: #1F3A5F !important;
}

/* ── Algorithm Badge ── */
.algo-badge {
    display: inline-block;
    background: #1B2A4A;
    color: #FFD580;
    border-radius: 8px;
    padding: 0.2rem 0.7rem;
    font-size: 0.73rem;
    font-weight: 600;
    margin: 0.2rem 0.2rem 0.2rem 0;
    letter-spacing: 0.04em;
}

/* ── Sidebar inputs ── */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { font-weight: 500; font-size: 0.9rem; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #FFD580, #E8A020);
    color: #FFFFFF !important;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
    font-family: 'DM Sans', sans-serif;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Tab ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    color: #2C3E6B !important;
    
}
/* 🔥 FIX DROPDOWN TEXT (FINAL WORKING) */
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* Dropdown popup */
[data-baseweb="popover"] {
    background-color: #0F1E3A !important;
}

/* Options */
[data-baseweb="option"] {
    color: #FFFFFF !important;
    background-color: #0F1E3A !important;
}

/* Hover */
[data-baseweb="option"]:hover {
    background-color: #1B2A4A !important;
    color: #FFFFFF !important;
}

/* Selected item */
[aria-selected="true"] {
    color: #FFFFFF !important;
}
.stButton > button {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)


# ── Load Data (cached) ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_data():
    attractions  = load_attractions()
    restaurants  = load_restaurants()
    local_food   = load_local_food()
    cost_est     = load_cost_estimation()
    user_ratings = load_user_ratings()
    country_city = get_country_city_map(attractions)
    return attractions, restaurants, local_food, cost_est, user_ratings, country_city


with st.spinner("Loading travel data…"):
    attractions, restaurants, local_food, cost_est, user_ratings, country_city = load_all_data()

COUNTRIES = sorted(country_city.keys())

# ── Session State Init ────────────────────────────────────────────────────────
for key in ["results_ready", "recommended", "itinerary", "country", "budget",
            "mood", "num_days", "cost_info", "rest_results", "food_results"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "results_ready" not in st.session_state or st.session_state["results_ready"] is None:
    st.session_state["results_ready"] = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ TravelMind")
    st.markdown("*Your travel planner*")
    st.markdown("---")

    st.markdown("### 🌍 Destination")
    country = st.selectbox("Country", COUNTRIES, key="sel_country")

    st.markdown("### 💰 Budget")
    budget = st.selectbox("Budget Level", ["Low", "Medium", "High"], index=1, key="sel_budget")

    st.markdown("### 🎭 Travel Mood")
    mood = st.selectbox("Mood", ["Relax", "Adventure", "Culture", "Romantic"], key="sel_mood")

    st.markdown("### 📅 Duration")
    num_days = st.slider("Number of Days", 1, 14, 5, key="sel_days")

    st.markdown("### 👤 User ID (CF)")
    user_ids = sorted(user_ratings["user_id"].unique().tolist()) if not user_ratings.empty else [1]
    cf_user = st.selectbox("Select User Profile", user_ids, key="sel_user",
                            help="Used for Collaborative Filtering personalisation")

    st.markdown("---")
    generate_btn = st.button("🚀 Generate My Trip Plan")

    st.markdown("---")
    st.markdown("##### 🧠 Algorithms Used")
    for algo in ["Rule-Based Filtering", "Content-Based (TF-IDF)", "KNN Similarity", "Collaborative Filtering", "Hybrid Scoring"]:
        st.markdown(f'<span class="algo-badge">{algo}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem;color:#8899BB;">GLS University · RS Lab · B.Tech CS&amp;E Sem VI</p>', unsafe_allow_html=True)


# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">✈️ &nbsp; Personalized Travel Recommendations</div>
  <p class="hero-title">🌍 TravelMind</p>
  <p class="hero-subtitle">
    Hybrid Recommender System combining Content-Based Filtering, KNN, and Collaborative Filtering
    to build your perfect personalized trip itinerary.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Generate Recommendations ──────────────────────────────────────────────────
if generate_btn:
    with st.spinner(f"🔍 Crafting your {mood.lower()} trip to {country}…"):
        recommended = hybrid_recommend(
            attractions, user_ratings, country, mood, budget,
            top_n=30, cf_user_id=cf_user
        )

        rest_df = restaurants[restaurants["country"] == country].copy()
        food_df = local_food[local_food["country"] == country]

        cost_row_df = cost_est[cost_est["country"] == country]
        if cost_row_df.empty:
            cost_info = {"stay_cost": 2000, "food_cost": 1000, "local_travel_cost": 600}
        else:
            cost_info = budget_filter_cost(cost_row_df.iloc[0], budget)

        itinerary = build_itinerary(recommended, restaurants, local_food, country, num_days, mood)

        # Store in session state
        st.session_state.update({
            "results_ready": True,
            "recommended": recommended,
            "itinerary": itinerary,
            "country": country,
            "budget": budget,
            "mood": mood,
            "num_days": num_days,
            "cost_info": cost_info,
            "rest_results": rest_df,
            "food_results": food_df,
        })


# ── Display Results ───────────────────────────────────────────────────────────
if st.session_state["results_ready"]:
    recommended = st.session_state["recommended"]
    itinerary   = st.session_state["itinerary"]
    country_    = st.session_state["country"]
    budget_     = st.session_state["budget"]
    mood_       = st.session_state["mood"]
    num_days_   = st.session_state["num_days"]
    cost_info   = st.session_state["cost_info"]
    rest_df     = st.session_state["rest_results"]
    food_df     = st.session_state["food_results"]

    # ── Summary Metrics ───────────────────────────────────────────────────────
    total_cost = (cost_info["stay_cost"] + cost_info["food_cost"] + cost_info["local_travel_cost"]) / 30 * num_days_
    must_visit = recommended.head(5)
    hidden_gems = recommended.tail(min(5, max(1, len(recommended) - 5)))

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value">🌍</div>
        <div class="metric-label">{country_}</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{num_days_}</div>
        <div class="metric-label">Days Planned</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{len(recommended)}</div>
        <div class="metric-label">Places Found</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">💰</div>
        <div class="metric-label">Est. ${total_cost:,.0f} Total</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{mood_}</div>
        <div class="metric-label">Trip Mood</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⭐ Places", "🍽️ Food & Dining", "💰 Costs",
        "🗓️ Itinerary", "🗺️ Map", "📥 Download"
    ])

    # ─────────────────────── TAB 1: PLACES ───────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">⭐ Must-Visit Places</div>', unsafe_allow_html=True)
            if recommended.empty:
                st.warning("No attractions found for this selection.")
            else:
                for _, row in must_visit.iterrows():
                    score = row.get("hybrid_score", 0)
                    st.markdown(f"""
                    <div class="place-card">
                      <span class="score-badge">★ {score:.2f}</span>
                      <div class="card-title">📍 {row['name']}</div>
                      <div class="card-meta">🏙️ {row['city']} &nbsp;|&nbsp; 🏷️ {row['category']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-header">💎 Hidden Gems</div>', unsafe_allow_html=True)
            if len(recommended) < 6:
                st.info("Explore all places in the Must-Visit section — these are your gems!")
            else:
                for _, row in hidden_gems.iterrows():
                    score = row.get("hybrid_score", 0)
                    st.markdown(f"""
                    <div class="place-card gem">
                      <span class="score-badge">💎 {score:.2f}</span>
                      <div class="card-title">🔍 {row['name']}</div>
                      <div class="card-meta">🏙️ {row['city']} &nbsp;|&nbsp; 🏷️ {row['category']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # All places expandable
        with st.expander(f"📋 View all {len(recommended)} recommended places"):
            display_cols = ["name", "city", "category", "hybrid_score", "cb_score", "knn_score", "cf_score"]
            display_cols = [c for c in display_cols if c in recommended.columns]
            st.dataframe(
                recommended[display_cols].rename(columns={
                    "name": "Place", "city": "City", "category": "Category",
                    "hybrid_score": "Hybrid Score", "cb_score": "Content Score",
                    "knn_score": "KNN Score", "cf_score": "CF Score"
                }).round(3),
                use_container_width=True, hide_index=True
            )

    # ─────────────────────── TAB 2: FOOD & DINING ────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">🍽️ Top Restaurants</div>', unsafe_allow_html=True)
            if rest_df.empty:
                st.warning("No restaurant data for this country.")
            else:
                show_rest = rest_df.dropna(subset=["name"]).head(10)
                for _, row in show_rest.iterrows():
                    cuisine = str(row.get("cuisine", "local")).replace(";", ", ")
                    st.markdown(f"""
                    <div class="place-card restaurant">
                      <div class="card-title">🍴 {row['name']}</div>
                      <div class="card-meta">🍽️ {cuisine} &nbsp;|&nbsp; 🏙️ {row['city']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-header">🍝 Local Food to Try</div>', unsafe_allow_html=True)
            if food_df.empty:
                st.info(f"Try local street food and ask locals for hidden food spots in {country_}!")
            else:
                for _, row in food_df.iterrows():
                    st.markdown(f'<span class="food-chip">🍜 {row["dish"]}</span> <small style="color:#888">{row.get("type","")}</small><br>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="info-box">💡 <strong>Tip:</strong> Ask locals for their favourite hidden spots — the best food is rarely on tourist maps!</div>', unsafe_allow_html=True)

    # ─────────────────────── TAB 3: COSTS ────────────────────────────────────
    # ───────── TAB 3: COSTS ─────────
    with tab3:
        st.markdown('<div class="section-header">💰 Price Estimation</div>', unsafe_allow_html=True)

        daily_stay  = cost_info["stay_cost"] / 30
        daily_food  = cost_info["food_cost"] / 30
        daily_travel = cost_info["local_travel_cost"] / 30
        daily_total = daily_stay + daily_food + daily_travel
        trip_total  = daily_total * num_days_

        # ---- METRICS ----
        c1, c2, c3, c4 = st.columns(4)

        def custom_metric(label, value):
            return f"""
            <div style="
                background: white;
                padding: 1rem;
                border-radius: 14px;
                text-align: center;
                box-shadow: 0 2px 12px rgba(0,0,0,0.05);
            ">
                <div style="font-size: 0.9rem; color:#3A4B6A; margin-bottom:5px;">
                    {label}
                </div>
                <div style="font-size: 1.8rem; font-weight:700; color:#0F1E3A;">
                    {value}
                </div>
            </div>
            """

        with c1:
            st.markdown(custom_metric("🏨 Stay / Day", f"${daily_stay:,.0f}"), unsafe_allow_html=True)

        with c2:
            st.markdown(custom_metric("🍽️ Food / Day", f"${daily_food:,.0f}"), unsafe_allow_html=True)

        with c3:
            st.markdown(custom_metric("🚌 Travel / Day", f"${daily_travel:,.0f}"), unsafe_allow_html=True)

        with c4:
            st.markdown(custom_metric("📅 Total Trip", f"${trip_total:,.0f}"), unsafe_allow_html=True)

        # ---- INFO BOX (OUTSIDE COLUMNS) ----
        st.markdown(f"""
        <div class="info-box">
        <strong>Budget Level:</strong> {budget_} &nbsp;|&nbsp;
        <strong>Country:</strong> {country_} &nbsp;|&nbsp;
        <strong>Duration:</strong> {num_days_} days<br><br>
        These are estimated per-person costs in USD. Prices may vary by season and exact location.
        </div>
        """, unsafe_allow_html=True)

        # ---- TABLE DATA ----
        breakdown = pd.DataFrame({
            "Category": ["🏨 Accommodation", "🍽️ Food & Dining", "🚌 Local Transport", "💰 Total"],
            "Per Day (USD)": [f"${daily_stay:.0f}", f"${daily_food:.0f}", f"${daily_travel:.0f}", f"${daily_total:.0f}"],
            f"For {num_days_} Days (USD)": [
                f"${daily_stay*num_days_:.0f}",
                f"${daily_food*num_days_:.0f}",
                f"${daily_travel*num_days_:.0f}",
                f"${trip_total:.0f}"
            ]
        })

        st.markdown("<br><br>", unsafe_allow_html=True)

        # ---- FINAL TABLE (FULL WIDTH) ----
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    # ─────────────────────── TAB 4: ITINERARY ────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">🗓️ Day-Wise Itinerary</div>', unsafe_allow_html=True)
        if not itinerary:
            st.warning("Could not generate itinerary — no attractions found.")
        else:
            itin_df = pd.DataFrame(itinerary)
            # HTML table rendering
            rows_html = ""
            for _, row in itin_df.iterrows():
                rows_html += f"""
                <tr>
                  <td><strong>Day {row['Day']}</strong></td>
                  <td>🌅 {row['Morning']}</td>
                  <td>☀️ {row['Afternoon']}</td>
                  <td>🍴 {row['Dinner At']}</td>
                  <td>🍜 {row['Local Food to Try']}</td>
                </tr>"""
            st.markdown(f"""
            <table class="itin-table">
              <thead>
                <tr>
                  <th>📅 Day</th><th>🌅 Morning</th><th>☀️ Afternoon</th>
                  <th>🍴 Dinner</th><th>🍜 Try This Food</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

    # ─────────────────────── TAB 5: MAP ──────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">🗺️ Interactive Map</div>', unsafe_allow_html=True)

        map_data = recommended[["name", "city", "latitude", "longitude", "category"]].dropna().head(25)
        rest_map = rest_df[["name", "city", "latitude", "longitude"]].dropna().head(10) if not rest_df.empty else pd.DataFrame()

        if map_data.empty:
            st.warning("No coordinate data available for mapping.")
        else:
            center_lat = map_data["latitude"].mean()
            center_lon = map_data["longitude"].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                           tiles="CartoDB positron")

            # Attraction markers — blue
            for _, row in map_data.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=8,
                    color="#1B2A4A",
                    fill=True,
                    fill_color="#3A6186",
                    fill_opacity=0.85,
                    tooltip=folium.Tooltip(f"📍 {row['name']} — {row['category']}"),
                    popup=folium.Popup(f"<b>{row['name']}</b><br>{row['city']}", max_width=200),
                ).add_to(m)

            # Restaurant markers — orange
            if not rest_map.empty:
                for _, row in rest_map.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=6,
                        color="#E8955A",
                        fill=True,
                        fill_color="#FF8C42",
                        fill_opacity=0.85,
                        tooltip=folium.Tooltip(f"🍴 {row['name']}"),
                        popup=folium.Popup(f"<b>{row['name']}</b><br>{row['city']}", max_width=200),
                    ).add_to(m)

            st_folium(m, width=None, height=500, returned_objects=[])

            st.markdown("""
            <div class="info-box">
            🔵 <strong>Blue</strong> = Attractions &nbsp;|&nbsp; 🟠 <strong>Orange</strong> = Restaurants<br>
            Click any marker for more details. Scroll to zoom.
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────── TAB 6: DOWNLOAD ─────────────────────────────────
    with tab6:
        st.markdown('<div class="section-header">📥 Download Your Trip Plan</div>', unsafe_allow_html=True)

        if itinerary:
            itin_df = pd.DataFrame(itinerary)
            csv_buf = io.StringIO()
            itin_df.to_csv(csv_buf, index=False)

            st.download_button(
                label="📥 Download Itinerary as CSV",
                data=csv_buf.getvalue(),
                file_name=f"TravelMind_{country_}_{num_days_}days.csv",
                mime="text/csv",
            )

        if not recommended.empty:
            rec_buf = io.StringIO()
            save_cols = [c for c in ["name", "city", "category", "hybrid_score"] if c in recommended.columns]
            recommended[save_cols].to_csv(rec_buf, index=False)
            st.download_button(
                label="📥 Download Recommended Places as CSV",
                data=rec_buf.getvalue(),
                file_name=f"TravelMind_{country_}_places.csv",
                mime="text/csv",
            )

        st.markdown(f"""
        <div class="info-box">
        <strong>📋 Trip Summary</strong><br>
        🌍 Country: <strong>{country_}</strong> &nbsp;|&nbsp;
        🎭 Mood: <strong>{mood_}</strong> &nbsp;|&nbsp;
        💰 Budget: <strong>{budget_}</strong> &nbsp;|&nbsp;
        📅 Days: <strong>{num_days_}</strong><br>
        ⭐ Places Recommended: <strong>{len(recommended)}</strong> &nbsp;|&nbsp;
        💡 Algorithms: <strong>Hybrid (CB + KNN + CF)</strong>
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #6B7A99;">
      <div style="font-size: 4rem; margin-bottom: 1rem;">🗺️</div>
      <h2 style="font-family:'Playfair Display',serif; color:#1B2A4A; font-size:1.8rem;">
        Your personalized journey starts here
      </h2>
      <p style="max-width:500px; margin:0 auto; line-height:1.7;">
        Select your destination, budget, mood and trip duration in the sidebar,
        then click <strong>Generate My Trip Plan</strong> to get recommendations
        with an interactive map and downloadable itinerary.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="place-card" style="text-align:center; border-left-color:#3A6186;">
          <div style="font-size:2rem">🧠</div>
          <div class="card-title">Hybrid Engine</div>
          <div class="card-meta">Content-Based + KNN + Collaborative Filtering combined for best results</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="place-card" style="text-align:center; border-left-color:#E8955A;">
          <div style="font-size:2rem">🗺️</div>
          <div class="card-title">Interactive Maps</div>
          <div class="card-meta">Visualise all recommended places and restaurants on a live folium map</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="place-card" style="text-align:center; border-left-color:#4CAF82;">
          <div style="font-size:2rem">📥</div>
          <div class="card-title">Download Itinerary</div>
          <div class="card-meta">Export your full day-by-day trip plan as a CSV to keep and share</div>
        </div>""", unsafe_allow_html=True)

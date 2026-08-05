from datetime import date
import gzip

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="FareCast AI - Flight Price & Decision Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. PREMIUM CHEAPFLIGHTS-INSPIRED UI
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --yellow: #FFD400;
        --yellow-dark: #F4C400;
        --ink: #202635;
        --ink-soft: #343B4D;
        --slate: #60687A;
        --line: #E7E9EF;
        --surface: #FFFFFF;
        --surface-soft: #F7F8FB;
        --success: #12805C;
        --warning: #A46700;
        --danger: #C63E45;
    }

    html {
        scroll-behavior: smooth;
    }

    body,
    [class*="css"] {
        font-family: "DM Sans", sans-serif;
    }

    .stApp {
        background:
            linear-gradient(
                180deg,
                rgba(6, 51, 87, 0.05) 0%,
                rgba(255, 255, 255, 0.08) 43%,
                #F4F6FA 76%,
                #F4F6FA 100%
            ),
            url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?auto=format&fit=crop&w=2200&q=90");
        background-size: cover;
        background-position: center top;
        background-attachment: fixed;
        color: var(--ink);
    }

    #MainMenu,
    footer,
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    .block-container {
        max-width: 1180px;
        padding: 0 1.15rem 4rem 1.15rem;
    }

    .top-shell {
        width: 100vw;
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        background: var(--yellow);
        box-shadow: 0 3px 14px rgba(18, 25, 39, 0.16);
        overflow: hidden;
    }

    .brand-row {
        max-width: 1180px;
        min-height: 82px;
        margin: 0 auto;
        padding: 0 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 12px;
        color: var(--ink);
        font-family: "Manrope", sans-serif;
        font-weight: 800;
        font-size: clamp(1.65rem, 3vw, 2.35rem);
        letter-spacing: -1.5px;
        white-space: nowrap;
    }

    .brand-lockup small {
        display: block;
        font-family: "DM Sans", sans-serif;
        font-size: 0.68rem;
        letter-spacing: 2.8px;
        text-transform: uppercase;
        font-weight: 800;
        margin-top: -3px;
        color: rgba(32, 38, 53, 0.68);
    }

    .brand-copy {
        display: flex;
        flex-direction: column;
        line-height: 1;
    }

    .logo-flight-track {
        width: 78px;
        height: 36px;
        position: relative;
        overflow: hidden;
        flex: 0 0 78px;
    }

    .logo-plane {
        position: absolute;
        left: -36px;
        top: 2px;
        font-size: 1.85rem;
        color: white;
        filter: drop-shadow(0 4px 3px rgba(0,0,0,.18));
        animation: logoFly 3.8s cubic-bezier(.45,.05,.55,.95) infinite;
        transform-origin: center;
    }

    .logo-flight-track::after {
        content: "";
        position: absolute;
        width: 42px;
        height: 3px;
        left: 8px;
        top: 20px;
        border-radius: 999px;
        background: rgba(255,255,255,.72);
        animation: trailPulse 3.8s ease-in-out infinite;
    }

    @keyframes logoFly {
        0%   { left: -38px; transform: translateY(5px) rotate(-5deg); opacity: 0; }
        12%  { opacity: 1; }
        48%  { transform: translateY(-3px) rotate(2deg); }
        85%  { opacity: 1; }
        100% { left: 88px; transform: translateY(-8px) rotate(6deg); opacity: 0; }
    }

    @keyframes trailPulse {
        0%, 14% { transform: scaleX(.1); opacity: 0; }
        32%     { transform: scaleX(1); opacity: .75; }
        72%     { transform: translateX(22px) scaleX(.45); opacity: .45; }
        100%    { transform: translateX(60px) scaleX(.1); opacity: 0; }
    }

    .trust-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 14px;
        border-radius: 999px;
        border: 1px solid rgba(32,38,53,.16);
        background: rgba(255,255,255,.42);
        color: var(--ink);
        font-size: .83rem;
        font-weight: 800;
        backdrop-filter: blur(8px);
    }

    .nav-shell {
        width: 100vw;
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        background: rgba(31, 36, 49, .97);
        box-shadow: 0 6px 18px rgba(0,0,0,.16);
    }

    .nav-row {
        max-width: 1180px;
        margin: 0 auto;
        min-height: 54px;
        padding: 0 20px;
        display: flex;
        align-items: center;
        gap: 28px;
        color: #F8FAFC;
        font-size: .91rem;
        font-weight: 700;
        overflow-x: auto;
        scrollbar-width: none;
    }

    .nav-row::-webkit-scrollbar {
        display: none;
    }

    .nav-item {
        white-space: nowrap;
        opacity: .82;
        position: relative;
        padding: 17px 0 14px;
    }

    .nav-item.active {
        color: var(--yellow);
        opacity: 1;
    }

    .nav-item.active::after {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        bottom: 7px;
        height: 3px;
        border-radius: 999px;
        background: var(--yellow);
    }

    .hero-copy {
        text-align: center;
        padding: clamp(2.2rem, 5vw, 4.5rem) 0 1.4rem;
        color: white;
        text-shadow: 0 3px 18px rgba(7, 31, 53, .35);
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 13px;
        margin-bottom: 15px;
        border: 1px solid rgba(255,255,255,.42);
        background: rgba(11, 39, 66, .30);
        backdrop-filter: blur(10px);
        border-radius: 999px;
        font-size: .8rem;
        font-weight: 800;
        letter-spacing: .7px;
        text-transform: uppercase;
    }

    .hero-copy h1 {
        color: white !important;
        font-family: "Manrope", sans-serif;
        font-size: clamp(2.15rem, 5vw, 4.15rem);
        line-height: 1.05;
        max-width: 930px;
        margin: 0 auto 15px;
        letter-spacing: -2.5px;
        font-weight: 800;
    }

    .hero-copy p {
        max-width: 720px;
        margin: 0 auto;
        color: rgba(255,255,255,.93);
        font-size: clamp(1rem, 2vw, 1.18rem);
        line-height: 1.7;
        font-weight: 600;
    }

    .search-card-heading {
        background: rgba(39, 45, 59, .96);
        border-radius: 22px 22px 0 0;
        padding: 18px 24px;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
        box-shadow: 0 18px 45px rgba(18, 28, 46, .25);
    }

    .search-card-heading h2 {
        color: white !important;
        margin: 0 !important;
        font-family: "Manrope", sans-serif;
        font-size: 1.13rem;
        letter-spacing: -.2px;
    }

    .search-card-heading span {
        color: rgba(255,255,255,.72);
        font-size: .82rem;
        font-weight: 600;
    }

    .step-pills {
        display: flex;
        gap: 7px;
    }

    .step-pill {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: rgba(255,255,255,.22);
    }

    .step-pill.active {
        width: 28px;
        border-radius: 99px;
        background: var(--yellow);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        border-radius: 0 0 22px 22px !important;
        background: rgba(255,255,255,.98) !important;
        box-shadow: 0 22px 50px rgba(22, 33, 51, .23);
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 1.25rem 1.35rem 1.45rem !important;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }

    label,
    [data-testid="stWidgetLabel"] p {
        color: var(--ink-soft) !important;
        font-size: .84rem !important;
        font-weight: 800 !important;
        letter-spacing: .05px;
    }

    div[data-baseweb="select"] > div,
    [data-testid="stDateInput"] > div > div,
    [data-testid="stDateInput"] input {
        min-height: 50px !important;
        border-radius: 12px !important;
        border-color: #DFE3EA !important;
        background: var(--surface-soft) !important;
        color: var(--ink) !important;
        transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
    }

    div[data-baseweb="select"] > div:hover,
    [data-testid="stDateInput"] > div > div:hover {
        border-color: #C7CDD8 !important;
        box-shadow: 0 6px 14px rgba(31, 41, 55, .07);
    }

    div[data-baseweb="select"] > div:focus-within,
    [data-testid="stDateInput"] > div > div:focus-within {
        border-color: var(--yellow-dark) !important;
        box-shadow: 0 0 0 3px rgba(255,212,0,.20) !important;
    }

    [data-testid="stRadio"] {
        background: var(--surface-soft);
        border: 1px solid #DFE3EA;
        padding: 8px 12px;
        border-radius: 12px;
        min-height: 50px;
        display: flex;
        align-items: center;
    }

    [data-testid="stRadio"] label {
        padding-right: 14px !important;
    }

    .route-summary {
        margin: 1.15rem 0 1rem;
        padding: 16px 18px;
        border-radius: 15px;
        border: 1px solid #E5E8EF;
        background: linear-gradient(90deg, rgba(255,212,0,.13), rgba(255,255,255,.9));
        display: grid;
        grid-template-columns: 1.45fr 1fr 1fr;
        gap: 10px;
        align-items: center;
    }

    .route-main {
        font-size: 1.04rem;
        font-weight: 800;
        color: var(--ink);
    }

    .route-main b {
        font-family: "Manrope", sans-serif;
    }

    .route-meta {
        color: var(--slate);
        font-size: .86rem;
        font-weight: 650;
    }

    .route-meta strong {
        color: var(--ink);
        display: block;
        margin-top: 2px;
        font-size: .95rem;
    }

    .stButton > button {
        min-height: 54px;
        width: 100%;
        border: 0 !important;
        border-radius: 13px !important;
        background: linear-gradient(135deg, #FFD400 0%, #FFC400 100%) !important;
        color: #202635 !important;
        font-family: "Manrope", sans-serif !important;
        font-size: .98rem !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 24px rgba(219, 166, 0, .30) !important;
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(219, 166, 0, .38) !important;
        filter: brightness(1.02);
    }

    .partner-strip {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid var(--line);
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px 22px;
        color: #788092;
        font-size: .78rem;
        font-weight: 750;
    }

    .partner-name {
        color: #4E5668;
        font-family: "Manrope", sans-serif;
        font-weight: 800;
    }

    .section-title {
        text-align: center;
        margin: 3.5rem 0 1.5rem;
    }

    .section-title h2 {
        font-family: "Manrope", sans-serif;
        margin: 0 0 8px;
        color: var(--ink) !important;
        font-size: clamp(1.65rem, 3vw, 2.2rem);
        letter-spacing: -1px;
    }

    .section-title p {
        margin: 0;
        color: var(--slate);
    }

    .benefit-card {
        background: rgba(255,255,255,.94);
        border: 1px solid rgba(222,226,234,.9);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 12px 30px rgba(28,38,55,.08);
        transition: transform .22s ease, box-shadow .22s ease;
        height: 100%;
    }

    .benefit-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 38px rgba(28,38,55,.13);
    }

    .benefit-icon {
        width: 44px;
        height: 44px;
        border-radius: 13px;
        display: grid;
        place-items: center;
        background: rgba(255,212,0,.18);
        font-size: 1.25rem;
        margin-bottom: 14px;
    }

    .benefit-card h3 {
        font-family: "Manrope", sans-serif;
        color: var(--ink) !important;
        margin: 0 0 7px;
        font-size: 1rem;
    }

    .benefit-card p {
        color: var(--slate);
        margin: 0;
        font-size: .88rem;
        line-height: 1.55;
    }

    .result-shell {
        margin-top: 1.25rem;
        background: rgba(255,255,255,.98);
        border: 1px solid #E1E5EC;
        border-radius: 22px;
        box-shadow: 0 20px 45px rgba(24,34,52,.14);
        overflow: hidden;
    }

    .result-top {
        padding: 22px 24px;
        background: linear-gradient(135deg, #252B39, #343C4E);
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
    }

    .result-top p {
        margin: 0 0 5px;
        color: rgba(255,255,255,.68);
        font-size: .77rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-price {
        font-family: "Manrope", sans-serif;
        font-size: clamp(1.9rem, 4vw, 2.8rem);
        font-weight: 800;
        letter-spacing: -1.3px;
    }

    .result-route {
        text-align: right;
        color: rgba(255,255,255,.82);
        font-weight: 700;
        font-size: .9rem;
    }

    .recommendation-box {
        margin: 18px 20px 8px;
        padding: 17px 18px;
        border-radius: 15px;
        border-left: 5px solid;
    }

    .recommendation-box.buy-urgent {
        background: #FFF1F2;
        border-color: var(--danger);
    }

    .recommendation-box.wait {
        background: #FFF8E8;
        border-color: #D79800;
    }

    .recommendation-box.buy-good {
        background: #ECFDF5;
        border-color: var(--success);
    }

    .recommendation-title {
        color: var(--ink);
        font-family: "Manrope", sans-serif;
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .recommendation-copy {
        color: #51596A;
        line-height: 1.55;
        font-size: .9rem;
    }

    .factor-title {
        padding: 14px 20px 7px;
        font-family: "Manrope", sans-serif;
        color: var(--ink);
        font-weight: 800;
    }

    .factor-grid {
        padding: 8px 20px 22px;
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
    }

    .factor-card {
        border: 1px solid #E4E7ED;
        background: #FAFBFC;
        border-radius: 14px;
        padding: 14px 15px;
        color: #555E70;
        font-size: .86rem;
        line-height: 1.5;
    }

    .factor-card b {
        display: block;
        color: var(--ink);
        margin-bottom: 3px;
    }

    @media (max-width: 820px) {
        .trust-chip {
            display: none;
        }

        .brand-row {
            min-height: 70px;
        }

        .nav-row {
            gap: 22px;
        }

        .hero-copy {
            padding-top: 2.3rem;
        }

        .search-card-heading {
            align-items: flex-start;
            flex-direction: column;
            gap: 9px;
        }

        .route-summary,
        .factor-grid {
            grid-template-columns: 1fr;
        }

        .result-top {
            align-items: flex-start;
            flex-direction: column;
        }

        .result-route {
            text-align: left;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD TRAINED MODEL AND FEATURES
# =========================================================
@st.cache_resource
def load_assets():
    with gzip.open("model.pkl.gz", "rb") as f:
        model = joblib.load(f)

    features = joblib.load("model_features.pkl")
    return model, features


try:
    model, feature_names = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# =========================================================
# 4. HEADER, NAVIGATION AND HERO
# =========================================================
st.markdown(
    """
    <div class="top-shell">
        <div class="brand-row">
            <div class="brand-lockup">
                <div class="brand-copy">
                    <div>FareCast AI</div>
                    <small>Smart flight decisions</small>
                </div>
                <div class="logo-flight-track">
                    <span class="logo-plane">✈</span>
                </div>
            </div>
            <div class="trust-chip">🛡️ AI-powered fare guidance</div>
        </div>
    </div>

    <div class="nav-shell">
        <div class="nav-row">
            <div class="nav-item active">Fare predictor</div>
            <div class="nav-item">Route insights</div>
            <div class="nav-item">Price advice</div>
            <div class="nav-item">Smart savings</div>
            <div class="nav-item">Travel confidence</div>
        </div>
    </div>

    <section class="hero-copy">
        <div class="hero-eyebrow">✦ Predict smarter · Book better</div>
        <h1>Know the likely fare before you book.</h1>
        <p>
            Compare route factors, estimate your ticket price and receive an
            instant AI recommendation on whether to buy now or wait.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 5. ROUTE DURATION MATRIX
# =========================================================
city_durations = {
    ("Delhi", "Mumbai"): 2.2,
    ("Delhi", "Bangalore"): 2.8,
    ("Delhi", "Kolkata"): 2.2,
    ("Delhi", "Hyderabad"): 2.2,
    ("Delhi", "Chennai"): 2.8,
    ("Mumbai", "Delhi"): 2.2,
    ("Mumbai", "Bangalore"): 1.8,
    ("Mumbai", "Kolkata"): 2.6,
    ("Mumbai", "Hyderabad"): 1.5,
    ("Mumbai", "Chennai"): 2.0,
    ("Bangalore", "Delhi"): 2.8,
    ("Bangalore", "Mumbai"): 1.8,
    ("Bangalore", "Kolkata"): 2.5,
    ("Bangalore", "Hyderabad"): 1.2,
    ("Bangalore", "Chennai"): 1.0,
    ("Kolkata", "Delhi"): 2.2,
    ("Kolkata", "Mumbai"): 2.6,
    ("Kolkata", "Bangalore"): 2.5,
    ("Kolkata", "Hyderabad"): 2.0,
    ("Kolkata", "Chennai"): 2.2,
    ("Hyderabad", "Delhi"): 2.2,
    ("Hyderabad", "Mumbai"): 1.5,
    ("Hyderabad", "Bangalore"): 1.2,
    ("Hyderabad", "Kolkata"): 2.0,
    ("Hyderabad", "Chennai"): 1.2,
    ("Chennai", "Delhi"): 2.8,
    ("Chennai", "Mumbai"): 2.0,
    ("Chennai", "Bangalore"): 1.0,
    ("Chennai", "Kolkata"): 2.2,
    ("Chennai", "Hyderabad"): 1.2,
}


# =========================================================
# 6. FLIGHT SEARCH UI
# =========================================================
st.markdown(
    """
    <div class="search-card-heading">
        <div>
            <h2>✈️ Build your flight estimate</h2>
            <span>Enter the same trip details you would use when booking.</span>
        </div>
        <div class="step-pills">
            <div class="step-pill active"></div>
            <div class="step-pill"></div>
            <div class="step-pill"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox(
            "Airline",
            ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"],
        )

        source_city = st.selectbox(
            "Departure city",
            ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"],
            index=0,
        )

        dest_options = [
            city
            for city in [
                "Mumbai",
                "Delhi",
                "Bangalore",
                "Kolkata",
                "Hyderabad",
                "Chennai",
            ]
            if city != source_city
        ]

        destination_city = st.selectbox(
            "Destination city",
            dest_options,
            index=0,
        )

        flight_class = st.radio(
            "Cabin class",
            ["Economy", "Business"],
            horizontal=True,
        )

    with col2:
        departure_time = st.selectbox(
            "Departure time",
            [
                "Morning",
                "Early_Morning",
                "Evening",
                "Afternoon",
                "Night",
                "Late_Night",
            ],
        )

        arrival_time = st.selectbox(
            "Arrival time",
            [
                "Morning",
                "Evening",
                "Night",
                "Early_Morning",
                "Afternoon",
                "Late_Night",
            ],
        )

        stops = st.selectbox(
            "Flight type",
            ["Non-stop (Direct)", "1 Stop", "2 or more stops"],
            index=0,
        )

        today = date.today()

        travel_date = st.date_input(
            "Travel date",
            min_value=today,
            value=today,
        )

    estimated_duration = city_durations.get(
        (source_city, destination_city),
        2.5,
    )

    if stops == "1 Stop":
        estimated_duration += 3.0
    elif stops == "2 or more stops":
        estimated_duration += 6.0

    days_left = (travel_date - today).days

    if days_left <= 0:
        days_left = 1

    route_summary_html = f"""
    <div class="route-summary">
        <div class="route-main">
            <span style="opacity:.55;">Route</span><br>
            <b>{source_city}</b>
            <span style="padding:0 7px; color:#A4AAB7;">→</span>
            <b>{destination_city}</b>
        </div>
        <div class="route-meta">
            Estimated journey
            <strong>⏱ {estimated_duration:.1f} hours</strong>
        </div>
        <div class="route-meta">
            Booking lead time
            <strong>📅 {days_left} day(s)</strong>
        </div>
    </div>
    """
    st.markdown(route_summary_html, unsafe_allow_html=True)

    analyze_clicked = st.button(
        "Analyze fare & get decision advice",
        type="primary",
        use_container_width=True,
    )

   # =========================================================
    # 7. PREDICTION & DECISION LOGIC
    # =========================================================
    if analyze_clicked:
        stops_num = (
            0
            if stops == "Non-stop (Direct)"
            else 1
            if stops == "1 Stop"
            else 2
        )

        class_num = 0 if flight_class == "Economy" else 1

        input_dict = {column: 0 for column in feature_names}

        input_dict["stops"] = stops_num
        input_dict["class"] = class_num
        input_dict["duration"] = estimated_duration
        input_dict["days_left"] = days_left

        if f"airline_{airline}" in input_dict:
            input_dict[f"airline_{airline}"] = 1

        if f"source_city_{source_city}" in input_dict:
            input_dict[f"source_city_{source_city}"] = 1

        if f"destination_city_{destination_city}" in input_dict:
            input_dict[f"destination_city_{destination_city}"] = 1

        if f"departure_time_{departure_time}" in input_dict:
            input_dict[f"departure_time_{departure_time}"] = 1

        if f"arrival_time_{arrival_time}" in input_dict:
            input_dict[f"arrival_time_{arrival_time}"] = 1

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Predict price
        raw_prediction = model.predict(input_df)[0]
        predicted_price = max(raw_prediction, 1850.0)

        if days_left <= 7:
            recommendation_class = "buy-urgent"
            recommendation_title = "🚨 BUY NOW"
            recommendation_copy = (
                f"Your flight is only {days_left} day(s) away. "
                "Fares often rise sharply during the final week, so booking now is the safer decision."
            )
        elif days_left > 20:
            recommendation_class = "wait"
            recommendation_title = "⏳ WAIT & MONITOR"
            recommendation_copy = (
                f"Your flight is {days_left} days away. "
                "The fare may remain stable, so you can monitor prices before confirming your booking."
            )
        else:
            recommendation_class = "buy-good"
            recommendation_title = "✅ GOOD TIME TO BUY"
            recommendation_copy = (
                f"Your flight is {days_left} days away. "
                "You are within a practical 1–3 week booking window, so booking now is a sensible choice."
            )

        if flight_class == "Economy":
            class_factor_title = "Economy class savings"
            class_factor_text = "Economy seating keeps the predicted fare lower than Business Class."
        else:
            class_factor_title = "Business class premium"
            class_factor_text = "Business Class adds a significant premium because of upgraded services and seating."

        if days_left <= 7:
            timing_factor_title = "Short lead-time pressure"
            timing_factor_text = "The departure date is close, which can increase the estimated ticket fare."
        elif days_left > 20:
            timing_factor_title = "Early planning advantage"
            timing_factor_text = "You are searching early, which gives you more time to monitor fare changes."
        else:
            timing_factor_title = "Balanced booking window"
            timing_factor_text = "Your selected date falls within a commonly practical domestic booking period."

        if airline in ["SpiceJet", "AirAsia", "Indigo", "GO_FIRST"]:
            airline_factor_title = "Low-cost carrier"
            airline_factor_text = f"{airline} is treated as a budget-oriented carrier in the model inputs."
        else:
            airline_factor_title = "Full-service carrier"
            airline_factor_text = f"{airline} is treated as a full-service carrier, which may increase the fare."

        route_factor_title = "Route and duration"
        route_factor_text = f"The selected {stops.lower()} journey has an estimated duration of approximately {estimated_duration:.1f} hours."

        results_html = f"""<section class="result-shell">
<div class="result-top">
<div>
<p>Estimated ticket fare</p>
<div class="result-price">₹{predicted_price:,.2f}</div>
</div>
<div class="result-route">
{source_city} → {destination_city}<br>
{travel_date.strftime("%d %b %Y")} · {flight_class}
</div>
</div>
<div class="recommendation-box {recommendation_class}">
<div class="recommendation-title">{recommendation_title}</div>
<div class="recommendation-copy">{recommendation_copy}</div>
</div>
<div class="factor-title">Why this fare was estimated</div>
<div class="factor-grid">
<div class="factor-card">
<b>🪑 {class_factor_title}</b>
{class_factor_text}
</div>
<div class="factor-card">
<b>📅 {timing_factor_title}</b>
{timing_factor_text}
</div>
<div class="factor-card">
<b>✈️ {airline_factor_title}</b>
{airline_factor_text}
</div>
<div class="factor-card">
<b>🗺️ {route_factor_title}</b>
{route_factor_text}
</div>
</div>
</section>"""
        st.markdown(results_html, unsafe_allow_html=True)


# =========================================================
# 8. VALUE CARDS
# =========================================================
st.markdown(
    """
    <div class="section-title">
        <h2>Make your next booking with more confidence</h2>
        <p>A focused decision assistant built around the factors that affect airfare.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        """
        <article class="benefit-card">
            <div class="benefit-icon">⚡</div>
            <h3>Instant fare estimate</h3>
            <p>Turn your route and travel choices into a clear estimated ticket price in seconds.</p>
        </article>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <article class="benefit-card">
            <div class="benefit-icon">🧭</div>
            <h3>Simple buy-or-wait advice</h3>
            <p>Get a direct recommendation based on how soon your selected departure date is.</p>
        </article>
        """,
        unsafe_allow_html=True,
    )

with col_c:
    st.markdown(
        """
        <article class="benefit-card">
            <div class="benefit-icon">🔎</div>
            <h3>Transparent fare factors</h3>
            <p>See the key trip choices that contributed to the model's price estimate.</p>
        </article>
        """,
        unsafe_allow_html=True,
    )
from datetime import date
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="FareCast AI - Flight Price & Decision Assistant",
    layout="wide",
    page_icon="✈️",
)

# 2. Bright Light Theme with High-Quality Flight Sky Background Image
st.markdown(
    """
    <style>
    /* Full-screen Light Sky Aviation Background Image */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), 
                    url('https://static1.simpleflyingimages.com/wordpress/wp-content/uploads/2024/04/indigo-a350-900-april-2024.png');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Clean dark text styling for optimal readability */
    .stApp h1, .stApp h2, .stApp h3, .stApp label, .stMarkdown {
        color: #0f172a !important;
    }

    /* Soft White Glassmorphism Cards for Input Sections */
    div[data-testid="stBlock"] {
        background: rgba(255, 255, 255, 0.75) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* Animated Flight Jet Icon */
    .animated-plane {
        display: inline-block;
        font-size: 2.8rem;
        animation: floatPlane 3s ease-in-out infinite alternate;
    }

    @keyframes floatPlane {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(3deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }

    /* Pulse Money Savings Icon */
    .money-pulse {
        display: inline-block;
        font-size: 2.2rem;
        animation: pulseMoney 1.8s infinite;
    }

    @keyframes pulseMoney {
        0% { transform: scale(1); }
        50% { transform: scale(1.18); }
        100% { transform: scale(1); }
    }

    /* Custom Header Box */
    .header-box {
        text-align: center;
        padding: 15px 0 10px 0;
        margin-bottom: 20px;
    }

    .info-card {
        background: rgba(241, 245, 249, 0.9) !important;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 14px 20px;
        color: #1e293b !important;
        margin-bottom: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 3. Load Trained Model and Features
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features


try:
    model, feature_names = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 4. Hero Title Section
st.markdown(
    """
    <div class="header-box">
        <h1>
            <span class="animated-plane">✈️</span> 
            FareCast AI 
            <span class="money-pulse">💰</span>
        </h1>
        <p style="font-size: 1.2rem; color: #334155; margin-top: -5px;">
            <b>Predict Flight Fares & Get Instant Money-Saving Decision Advice 💸🛡️</b>
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

# 5. Route Duration Matrix
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

# 6. User Inputs Form
st.subheader("📋 Flight Search Details")
col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox(
        "Select Airline 🛫",
        ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"],
    )
    source_city = st.selectbox(
        "Departure City 📍",
        ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"],
        index=0,
    )
    dest_options = [
        c
        for c in [
            "Mumbai",
            "Delhi",
            "Bangalore",
            "Kolkata",
            "Hyderabad",
            "Chennai",
        ]
        if c != source_city
    ]
    destination_city = st.selectbox("Destination City 🎯", dest_options, index=0)
    flight_class = st.radio(
        "Cabin Class 🪑", ["Economy", "Business"], horizontal=True
    )

with col2:
    departure_time = st.selectbox(
        "Departure Time ⏰",
        ["Morning", "Early_Morning", "Evening", "Afternoon", "Night", "Late_Night"],
    )
    arrival_time = st.selectbox(
        "Arrival Time 🛬",
        ["Morning", "Evening", "Night", "Early_Morning", "Afternoon", "Late_Night"],
    )
    stops = st.selectbox(
        "Flight Type 🛑",
        ["Non-stop (Direct)", "1 Stop", "2 or more stops"],
        index=0,
    )

    today = date.today()
    travel_date = st.date_input(
        "Select Travel Date 📅",
        min_value=today,
        value=today,
    )

    # Compute duration & lead time
    estimated_duration = city_durations.get(
        (source_city, destination_city), 2.5
    )
    if stops == "1 Stop":
        estimated_duration += 3.0
    elif stops == "2 or more stops":
        estimated_duration += 6.0

    days_left = (travel_date - today).days
    if days_left == 0:
        days_left = 1

st.markdown(
    f"""
    <div class="info-card">
        🗺️ Route Summary: <b>{source_city}</b> ➔ <b>{destination_city}</b> &nbsp;|&nbsp; 
        ⏱️ Duration: <b>~{estimated_duration:.1f} hrs</b> &nbsp;|&nbsp; 
        📅 Lead Time: <b>{days_left} day(s)</b>
    </div>
""",
    unsafe_allow_html=True,
)

# 7. Prediction Trigger Button
if st.button("🚀 Analyze Fare & Get Decision Advice", type="primary"):
    stops_num = (
        0
        if stops == "Non-stop (Direct)"
        else (1 if stops == "1 Stop" else 2)
    )
    class_num = 0 if flight_class == "Economy" else 1

    input_dict = {col: 0 for col in feature_names}
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
    predicted_price = model.predict(input_df)[0]

    st.write("---")

    # Estimated Price Output
    st.subheader(f"💰 Estimated Ticket Fare: ₹{predicted_price:,.2f}")

    # Recommendation Output
    if days_left <= 7:
        st.subheader("📊 Recommendation: BUY NOW 🛒")
        st.error(
            f"🚨 **Travel Date: {travel_date.strftime('%d %b %Y')} ({days_left} day(s) away)**\n\n"
            f"Fares surge significantly in the final 7 days before departure. We strongly advise booking now before prices jump higher!"
        )
    elif days_left > 20:
        st.subheader("📊 Recommendation: WAIT ⏳")
        st.warning(
            f"🟡 **Travel Date: {travel_date.strftime('%d %b %Y')} ({days_left} days away)**\n\n"
            f"Fares are currently steady. You can safely wait and monitor prices for a week or two before committing."
        )
    else:
        st.subheader("📊 Recommendation: BUY NOW 🛒")
        st.success(
            f"🟢 **Travel Date: {travel_date.strftime('%d %b %Y')} ({days_left} days away)**\n\n"
            f"You are in the optimal 1–3 week booking window for domestic flights to get the best standard fare."
        )

    # Explanation Reasons
    st.subheader("💡 Key Fare Factors (Why is this price calculated this way?)")

    if flight_class == "Economy":
        st.write(
            "✅ **Economy Class Savings:** Standard economy seating keeps your fare substantially cheaper than Business Class."
        )
    else:
        st.write(
            "⭐ **Business Class Premium:** Premium seating adds a major fare increase."
        )

    if days_left <= 7:
        st.write(
            "⚠️ **Short Lead-Time Penalty:** Fares are higher because departure is less than a week away."
        )
    elif days_left > 20:
        st.write(
            "✅ **Early Planning Advantage:** Booking early protects you against sudden last-minute surges."
        )
    else:
        st.write(
            "✅ **Optimal Booking Timing:** Booking 1–3 weeks out puts you in the sweet spot for flight pricing."
        )

    if airline in ["SpiceJet", "AirAsia", "Indigo", "GO_FIRST"]:
        st.write(
            f"✅ **Low-Cost Carrier Rates:** Flying with **{airline}** keeps base fares competitive."
        )
    else:
        st.write(
            f"⭐ **Full-Service Carrier:** Flying with **{airline}** includes full-service amenities reflected in the fare."
        )

    st.write(
        f"📍 **Distance & Duration:** A **{stops}** flight (~{estimated_duration:.1f} hrs) sets the base route cost."
    )
"""
EcoShield AI v2.1
India-wide Forest Fire Risk Intelligence Dashboard

Run:
    streamlit run app_v2.py
"""

from pathlib import Path
import base64

import pandas as pd
import requests
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ============================================================
# PAGE SETTINGS
# ============================================================
st.set_page_config(
    page_title="EcoShield AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SAFE HTML RENDERER
# All custom HTML in this application goes through this helper.
# This fixes the issue where <div>, <h2>, <p>, etc. appeared
# literally on the screen.
# ============================================================
def render_html(html: str):
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# BACKGROUND IMAGE
# ============================================================
def get_background_css():
    possible_files = [
        Path("forest_background.jpg"),
        Path("forest_background.jpeg"),
        Path("assets/forest_background.jpg"),
        Path("assets/forest_background.jpeg"),
    ]

    for image_path in possible_files:
        if image_path.exists():
            try:
                encoded = base64.b64encode(image_path.read_bytes()).decode()
                return f"""
                background-image:
                    linear-gradient(
                        rgba(2, 18, 10, 0.86),
                        rgba(2, 18, 10, 0.93)
                    ),
                    url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                """
            except Exception:
                pass

    return """
        background:
            radial-gradient(
                circle at top right,
                rgba(255, 104, 0, 0.12),
                transparent 35%
            ),
            linear-gradient(135deg, #02120a, #062516 55%, #03150c);
    """


BACKGROUND_CSS = get_background_css()


# ============================================================
# GLOBAL CSS
# ============================================================
render_html(f"""
    <style>

    html, body, [class*="css"] {{
        font-family: "Times New Roman", Times, serif !important;
    }}

    .stApp {{
        {BACKGROUND_CSS}
        color: #f4f7f5;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0.18);
    }}

    [data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                rgba(1, 22, 12, 0.98),
                rgba(0, 35, 18, 0.96)
            );
        border-right: 1px solid rgba(125, 255, 174, 0.16);
    }}

    [data-testid="stSidebar"] * {{
        font-family: "Times New Roman", Times, serif !important;
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* Sidebar */
    .side-brand {{
        text-align: center;
        padding: 18px 8px 25px;
    }}

    .side-brand-title {{
        color: #d9ff73;
        font-size: 30px;
        font-weight: 700;
        line-height: 1.1;
    }}

    .side-brand-subtitle {{
        color: #7fffa9;
        font-size: 15px;
        margin-top: 8px;
    }}

    /* Hero */
    .hero {{
        text-align: center;
        padding: 18px 20px 26px;
        margin-bottom: 18px;
    }}

    .hero-title {{
        color: #e5ff83;
        font-size: 48px;
        font-weight: 700;
        line-height: 1.05;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,.45);
    }}

    .hero-subtitle {{
        color: #d8eee0;
        font-size: 20px;
        margin-top: 10px;
    }}

    .hero-status {{
        color: #7fffa9;
        font-size: 14px;
        margin-top: 14px;
        font-weight: 700;
        letter-spacing: .3px;
    }}

    /* Cards */
    .glass-card {{
        background: rgba(5, 24, 15, 0.78);
        border: 1px solid rgba(132, 255, 171, 0.17);
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 14px 40px rgba(0,0,0,.22);
        backdrop-filter: blur(8px);
    }}

    .info-title {{
        color: #e6ff91;
        font-size: 23px;
        font-weight: 700;
        margin-bottom: 12px;
    }}

    .info-text {{
        color: #d1ded5;
        font-size: 16px;
        line-height: 1.7;
    }}

    .metric-card {{
        background: rgba(5, 27, 17, .82);
        border: 1px solid rgba(127, 255, 169, .16);
        border-radius: 18px;
        padding: 18px;
        min-height: 125px;
    }}

    .metric-label {{
        color: #b7cabe;
        font-size: 16px;
    }}

    .metric-value {{
        color: #ffffff;
        font-size: 31px;
        font-weight: 700;
        margin-top: 8px;
    }}

    .metric-note {{
        color: #7fffa9;
        font-size: 13px;
        margin-top: 5px;
    }}

    /* Monitoring */
    .monitor {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        background: rgba(3, 26, 15, .80);
        border: 1px solid rgba(128, 255, 170, .16);
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }}

    .monitor-left {{
        color: #e8f4eb;
        font-size: 17px;
    }}

    .monitor-right {{
        color: #b9c8be;
        font-size: 14px;
        text-align: right;
    }}

    .live-badge {{
        display: inline-block;
        background: #d92929;
        color: white;
        padding: 5px 11px;
        border-radius: 20px;
        font-size: 12px;
        margin-left: 10px;
        font-weight: 700;
    }}

    /* Risk */
    .risk-card {{
        border-radius: 22px;
        padding: 26px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, .12);
        box-shadow: 0 16px 45px rgba(0,0,0,.30);
    }}

    .risk-title {{
        color: #ffffff;
        font-size: 20px;
        font-weight: 700;
    }}

    .risk-number {{
        color: #ffffff;
        font-size: 62px;
        font-weight: 700;
        line-height: 1;
        margin: 18px 0 8px;
    }}

    .risk-level {{
        font-size: 25px;
        font-weight: 700;
        letter-spacing: .6px;
    }}

    .risk-description {{
        color: #d5ddd7;
        margin-top: 10px;
        font-size: 14px;
    }}

    .low {{
        background: linear-gradient(135deg, rgba(11, 83, 46, .92), rgba(8, 55, 34, .92));
    }}

    .moderate {{
        background: linear-gradient(135deg, rgba(117, 78, 5, .90), rgba(66, 45, 4, .92));
    }}

    .high {{
        background: linear-gradient(135deg, rgba(128, 49, 13, .92), rgba(72, 27, 8, .92));
    }}

    .critical {{
        background: linear-gradient(135deg, rgba(128, 24, 24, .94), rgba(58, 12, 12, .96));
    }}

    .footer {{
        text-align: center;
        color: #9db0a2;
        font-size: 13px;
        padding: 30px 0 10px;
    }}

    @media (max-width: 900px) {{
        .hero-title {{
            font-size: 36px;
        }}

        .monitor {{
            flex-direction: column;
            align-items: flex-start;
        }}

        .monitor-right {{
            text-align: left;
        }}
    }}

    </style>
    """)


# ============================================================
# INDIA-WIDE FOREST REGIONS
# ============================================================
FOREST_REGIONS = {
    "Dachigam — Jammu & Kashmir": {
        "lat": 34.0837,
        "lon": 74.8836,
        "state": "Jammu & Kashmir",
    },
    "Jim Corbett — Uttarakhand": {
        "lat": 29.5300,
        "lon": 78.7747,
        "state": "Uttarakhand",
    },
    "Ranthambore — Rajasthan": {
        "lat": 26.0173,
        "lon": 76.5026,
        "state": "Rajasthan",
    },
    "Kaziranga — Assam": {
        "lat": 26.5775,
        "lon": 93.1711,
        "state": "Assam",
    },
    "Gir — Gujarat": {
        "lat": 21.1243,
        "lon": 70.8242,
        "state": "Gujarat",
    },
    "Kanha — Madhya Pradesh": {
        "lat": 22.3345,
        "lon": 80.6115,
        "state": "Madhya Pradesh",
    },
    "Tadoba — Maharashtra": {
        "lat": 20.2510,
        "lon": 79.3580,
        "state": "Maharashtra",
    },
    "Similipal — Odisha": {
        "lat": 21.9497,
        "lon": 86.3700,
        "state": "Odisha",
    },
    "Sundarbans — West Bengal": {
        "lat": 21.9497,
        "lon": 88.9000,
        "state": "West Bengal",
    },
    "Bandipur — Karnataka": {
        "lat": 11.6821,
        "lon": 76.6300,
        "state": "Karnataka",
    },
    "Nallamala — Andhra Pradesh": {
        "lat": 15.3793,
        "lon": 78.4800,
        "state": "Andhra Pradesh",
    },
    "Nagarjunsagar — Telangana": {
        "lat": 16.5427,
        "lon": 79.3110,
        "state": "Telangana",
    },
}


# ============================================================
# WEATHER
# ============================================================
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Moderate rain showers",
    82: "Heavy rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


@st.cache_data(ttl=900, show_spinner=False)
def get_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "precipitation,"
            "weather_code"
        ),
        "hourly": (
            "temperature_2m," "relative_humidity_2m," "wind_speed_10m," "precipitation"
        ),
        "forecast_days": 1,
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()

    payload = response.json()
    current = payload["current"]

    hourly = pd.DataFrame(payload["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])

    current_time = pd.to_datetime(current["time"])
    future = hourly[hourly["time"] >= current_time].head(24).copy()

    return {
        "temperature": float(current["temperature_2m"]),
        "humidity": float(current["relative_humidity_2m"]),
        "wind": float(current["wind_speed_10m"]),
        "rain": float(current["precipitation"]),
        "weather_code": int(current["weather_code"]),
        "weather": WEATHER_CODES.get(
            int(current["weather_code"]),
            "Unknown conditions",
        ),
        "time": current["time"],
        "timezone": payload.get("timezone", "Local"),
        "hourly": future,
    }


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================
@st.cache_resource
def train_model():
    data = pd.read_csv("forestfires.csv")

    required = ["temp", "RH", "wind", "area"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            "forestfires.csv is missing required columns: " + ", ".join(missing)
        )

    data = data.copy()
    data["risk"] = (data["area"] > 0).astype(int)

    features = ["temp", "RH", "wind"]

    if "rain" in data.columns:
        features.append("rain")

    X = data[features]
    y = data["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }

    return data, model, features, metrics


# ============================================================
# RISK FUNCTIONS
# ============================================================
def get_risk(model, features, temperature, humidity, wind, rain):
    values = {
        "temp": temperature,
        "RH": humidity,
        "wind": wind,
        "rain": rain,
    }

    input_data = pd.DataFrame(
        [[values[f] for f in features]],
        columns=features,
    )

    probabilities = model.predict_proba(input_data)[0]
    positive_index = list(model.classes_).index(1)

    score = float(probabilities[positive_index] * 100)

    if score < 25:
        level = "LOW"
        css_class = "low"
    elif score < 50:
        level = "MODERATE"
        css_class = "moderate"
    elif score < 75:
        level = "HIGH"
        css_class = "high"
    else:
        level = "CRITICAL"
        css_class = "critical"

    return score, level, css_class


def environmental_factors(temp, humidity, wind, rain):
    factors = []

    if temp >= 35:
        factors.append("High temperature is increasing dryness.")
    elif temp >= 30:
        factors.append("Warm temperature may increase fire susceptibility.")

    if humidity <= 30:
        factors.append("Very low humidity indicates dry atmospheric conditions.")
    elif humidity <= 45:
        factors.append("Moderately low humidity can support ignition.")

    if wind >= 25:
        factors.append("Strong winds can accelerate fire spread.")
    elif wind >= 15:
        factors.append("Moderate winds may support fire spread.")

    if rain <= 0.1:
        factors.append("Little or no current rainfall is present.")
    else:
        factors.append("Current rainfall may reduce immediate fire susceptibility.")

    return factors


def recommendations(score):
    if score >= 75:
        return [
            "Increase monitoring frequency.",
            "Avoid unnecessary open-fire activities.",
            "Keep forest-response teams prepared.",
            "Check local official fire and emergency advisories.",
        ]

    if score >= 50:
        return [
            "Maintain enhanced monitoring.",
            "Avoid burning dry vegetation.",
            "Watch wind and humidity changes.",
            "Review local forest safety procedures.",
        ]

    if score >= 25:
        return [
            "Continue routine monitoring.",
            "Avoid uncontrolled fires near dry vegetation.",
            "Recheck conditions if temperature rises or humidity falls.",
        ]

    return [
        "Continue normal forest monitoring.",
        "Maintain preventive fire-safety practices.",
        "Reassess if weather conditions change significantly.",
    ]


# ============================================================
# LOAD MODEL
# ============================================================
try:
    data, model, FEATURES, MODEL_METRICS = train_model()
except Exception as exc:
    st.error(f"Project startup error: {exc}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    render_html("""
        <div class="side-brand">
            <div class="side-brand-title">🌲 EcoShield AI</div>
            <div class="side-brand-subtitle">
                India's Forest Fire Intelligence
            </div>
        </div>
        """)

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🌲 Forest Map",
            "🔥 Risk Prediction",
            "🌤️ Live Weather",
            "📊 Analytics",
            "🛡️ Safety & Awareness",
            "ℹ️ About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📍 MONITORING REGION")

    selected_region = st.selectbox(
        "Select Forest Region",
        list(FOREST_REGIONS.keys()),
    )

    st.markdown("---")

    if st.button("🔄 Refresh Live Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption("🌐 Weather: Open-Meteo")
    st.caption("🤖 Model: Random Forest")
    st.caption(" Coverage: India")


# ============================================================
# SELECTED REGION WEATHER
# ============================================================
region_info = FOREST_REGIONS[selected_region]

try:
    weather = get_weather(
        region_info["lat"],
        region_info["lon"],
    )

    temperature = weather["temperature"]
    humidity = weather["humidity"]
    wind = weather["wind"]
    rain = weather["rain"]

    score, level, risk_css = get_risk(
        model,
        FEATURES,
        temperature,
        humidity,
        wind,
        rain,
    )

except Exception as exc:
    st.error(
        "Live weather could not be loaded. "
        "Check your internet connection and press Refresh Live Data."
    )
    st.caption(f"Details: {exc}")
    st.stop()


# ============================================================
# DASHBOARD
# ============================================================
if page == "🏠 Dashboard":

    render_html("""
        <div class="hero">
            <div class="hero-title">🌲 EcoShield AI</div>
            <div class="hero-subtitle">
                India's Forest Fire Risk Intelligence Platform
            </div>
            <div class="hero-status">
                🟢 LIVE MONITORING
                &nbsp; | &nbsp;
                🤖 AI-POWERED PREDICTION
                &nbsp; | &nbsp;
                FOREST COVERAGE
            </div>
        </div>
        """)

    render_html(f"""
        <div class="monitor">
            <div class="monitor-left">
                📍 Monitoring:
                <b>{selected_region}</b>
                <span class="live-badge">● LIVE</span>
            </div>
            <div class="monitor-right">
                🕒 Updated: <b>{weather["time"]}</b>
            </div>
        </div>
        """)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_html(f"""
            <div class="metric-card">
                <div class="metric-label">🌡️ Temperature</div>
                <div class="metric-value">{temperature:.1f} °C</div>
                <div class="metric-note">Live weather</div>
            </div>
            """)

    with c2:
        render_html(f"""
            <div class="metric-card">
                <div class="metric-label">💧 Humidity</div>
                <div class="metric-value">{humidity:.0f}%</div>
                <div class="metric-note">Relative humidity</div>
            </div>
            """)

    with c3:
        render_html(f"""
            <div class="metric-card">
                <div class="metric-label">💨 Wind Speed</div>
                <div class="metric-value">{wind:.1f} km/h</div>
                <div class="metric-note">Current wind</div>
            </div>
            """)

    with c4:
        render_html(f"""
            <div class="metric-card">
                <div class="metric-label">🌧️ Rainfall</div>
                <div class="metric-value">{rain:.1f} mm</div>
                <div class="metric-note">Current precipitation</div>
            </div>
            """)

    st.write("")

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        render_html(f"""
            <div class="risk-card {risk_css}">
                <div class="risk-title">🔥 FOREST FIRE RISK</div>
                <div class="risk-number">
                    {score:.0f}
                    <span style="font-size:24px;"> / 100</span>
                </div>
                <div class="risk-level">{level} RISK</div>
                <div class="risk-description">
                    Random Forest environmental risk estimate
                </div>
            </div>
            """)

    with right:
        render_html(f"""
            <div class="glass-card">
                <div class="info-title">🌤️ Live Weather Report</div>
                <div class="info-text">
                    🌡️ <b>{temperature:.1f} °C</b> — Temperature<br>
                    💧 <b>{humidity:.0f}%</b> — Relative Humidity<br>
                    💨 <b>{wind:.1f} km/h</b> — Wind Speed<br>
                    🌧️ <b>{rain:.1f} mm</b> — Rainfall<br>
                    ☁️ <b>{weather["weather"]}</b><br><br>
                    📍 {selected_region}<br>
                    🌐 Source: Open-Meteo
                </div>
            </div>
            """)

    st.markdown("### 🔎 Environmental Factors")

    for factor in environmental_factors(
        temperature,
        humidity,
        wind,
        rain,
    ):
        st.info("• " + factor)

    st.markdown("### 📈 Next 24-Hour Risk Trend")

    hourly = weather["hourly"].copy()

    if not hourly.empty:
        scores = []

        for _, row in hourly.iterrows():
            s, _, _ = get_risk(
                model,
                FEATURES,
                float(row["temperature_2m"]),
                float(row["relative_humidity_2m"]),
                float(row["wind_speed_10m"]),
                float(row["precipitation"]),
            )
            scores.append(s)

        trend = pd.DataFrame(
            {
                "Time": hourly["time"],
                "Risk Score": scores,
            }
        )

        st.line_chart(
            trend.set_index("Time"),
            y="Risk Score",
        )

    render_html("""
        <div class="footer">
            EcoShield AI • Academic ML Prototype •
            Weather data supplied by Open-Meteo
        </div>
        """)


# ============================================================
# FOREST MAP
# ============================================================
elif page == "🌲 Forest Map":

    render_html("""
        <div class="hero">
            <div class="hero-title">🌲 India Forest Risk Map</div>
            <div class="hero-subtitle">
                Live environmental monitoring across major Indian forest regions
            </div>
        </div>
        """)

    st.info(
        "Markers represent selected forest monitoring regions across India. "
        "Risk values are calculated from current weather inputs."
    )

    rows = []
    failed = []

    with st.spinner("Fetching live weather for Indian forest regions..."):

        for name, info in FOREST_REGIONS.items():
            try:
                w = get_weather(info["lat"], info["lon"])

                s, lvl, _ = get_risk(
                    model,
                    FEATURES,
                    w["temperature"],
                    w["humidity"],
                    w["wind"],
                    w["rain"],
                )

                rows.append(
                    {
                        "Region": name,
                        "State": info["state"],
                        "Latitude": info["lat"],
                        "Longitude": info["lon"],
                        "Temperature °C": round(w["temperature"], 1),
                        "Humidity %": round(w["humidity"], 0),
                        "Wind km/h": round(w["wind"], 1),
                        "Rain mm": round(w["rain"], 1),
                        "Risk /100": round(s, 1),
                        "Level": lvl,
                    }
                )

            except Exception:
                failed.append(name)

    map_df = pd.DataFrame(rows)

    if not map_df.empty:

        st.map(
            map_df,
            latitude="Latitude",
            longitude="Longitude",
            zoom=4,
            size=120,
        )

        st.markdown("### 🔥 Regional Risk Overview")

        display_df = map_df[
            [
                "Region",
                "Temperature °C",
                "Humidity %",
                "Wind km/h",
                "Rain mm",
                "Risk /100",
                "Level",
            ]
        ].sort_values("Risk /100", ascending=False)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    if failed:
        st.warning(f"Weather was unavailable for {len(failed)} region(s).")


# ============================================================
# RISK PREDICTION
# ============================================================
elif page == "🔥 Risk Prediction":

    render_html("""
        <div class="hero">
            <div class="hero-title">🔥 Fire Risk Prediction</div>
            <div class="hero-subtitle">
                Estimate fire risk from environmental conditions
            </div>
        </div>
        """)

    col1, col2 = st.columns(2)

    with col1:
        temp_input = st.slider(
            "🌡️ Temperature °C",
            0.0,
            50.0,
            float(temperature),
            0.1,
        )

        humidity_input = st.slider(
            "💧 Relative Humidity %",
            0.0,
            100.0,
            float(humidity),
            1.0,
        )

    with col2:
        wind_input = st.slider(
            "💨 Wind Speed km/h",
            0.0,
            100.0,
            float(wind),
            0.1,
        )

        rain_input = st.slider(
            "🌧️ Rainfall mm",
            0.0,
            50.0,
            float(min(rain, 50)),
            0.1,
        )

    if st.button(
        "🔥 Calculate Fire Risk",
        type="primary",
        use_container_width=True,
    ):

        predicted_score, predicted_level, predicted_css = get_risk(
            model,
            FEATURES,
            temp_input,
            humidity_input,
            wind_input,
            rain_input,
        )

        render_html(f"""
            <div class="risk-card {predicted_css}">
                <div class="risk-title">🔥 AI PREDICTION</div>
                <div class="risk-number">
                    {predicted_score:.0f}
                    <span style="font-size:24px;"> / 100</span>
                </div>
                <div class="risk-level">
                    {predicted_level} RISK
                </div>
                <div class="risk-description">
                    Based on temperature, humidity, wind and rainfall
                </div>
            </div>
            """)

        st.markdown("### 🧠 Environmental Explanation")

        for item in environmental_factors(
            temp_input,
            humidity_input,
            wind_input,
            rain_input,
        ):
            st.write("• " + item)


# ============================================================
# LIVE WEATHER
# ============================================================
elif page == "🌤️ Live Weather":

    render_html("""
        <div class="hero">
            <div class="hero-title">🌤️ Live Weather</div>
            <div class="hero-subtitle">
                Current atmospheric conditions for the selected forest region
            </div>
        </div>
        """)

    render_html(f"""
        <div class="glass-card">
            <div class="info-title">
                📍 {selected_region}
            </div>

            <div class="info-text">
                🌡️ Temperature: <b>{temperature:.1f} °C</b><br>
                💧 Relative Humidity: <b>{humidity:.0f}%</b><br>
                💨 Wind Speed: <b>{wind:.1f} km/h</b><br>
                🌧️ Rainfall: <b>{rain:.1f} mm</b><br>
                ☁️ Condition: <b>{weather["weather"]}</b><br>
                🕒 API Time: <b>{weather["time"]}</b><br>
                🌍 Timezone: <b>{weather["timezone"]}</b>
            </div>
        </div>
        """)

    st.markdown("### 📈 Next 24 Hours")

    hourly = weather["hourly"].copy()

    if not hourly.empty:

        weather_chart = hourly[
            [
                "time",
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "precipitation",
            ]
        ].copy()

        weather_chart = weather_chart.rename(
            columns={
                "time": "Time",
                "temperature_2m": "Temperature °C",
                "relative_humidity_2m": "Humidity %",
                "wind_speed_10m": "Wind km/h",
                "precipitation": "Rain mm",
            }
        )

        st.line_chart(weather_chart.set_index("Time"))


# ============================================================
# ANALYTICS
# ============================================================
elif page == "📊 Analytics":

    render_html("""
        <div class="hero">
            <div class="hero-title">📊 Model Analytics</div>
            <div class="hero-subtitle">
                Performance information for the Random Forest classifier
            </div>
        </div>
        """)

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Accuracy",
            f"{MODEL_METRICS['accuracy'] * 100:.1f}%",
        )

    with b:
        st.metric(
            "Precision",
            f"{MODEL_METRICS['precision'] * 100:.1f}%",
        )

    with c:
        st.metric(
            "Recall",
            f"{MODEL_METRICS['recall'] * 100:.1f}%",
        )

    with d:
        st.metric(
            "F1 Score",
            f"{MODEL_METRICS['f1'] * 100:.1f}%",
        )

    left, right = st.columns(2)

    with left:
        render_html(f"""
            <div class="glass-card">
                <div class="info-title">🤖 Model Information</div>
                <div class="info-text">
                    <b>Algorithm:</b> Random Forest Classifier<br><br>
                    <b>Dataset Records:</b> {len(data)}<br><br>
                    <b>Features:</b> {", ".join(FEATURES)}<br><br>
                    <b>Target:</b> Historical fire occurrence
                </div>
            </div>
            """)

    with right:
        cm = MODEL_METRICS["confusion_matrix"]

        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Fire", "Actual: Fire"],
            columns=["Predicted: No Fire", "Predicted: Fire"],
        )

        st.markdown("### 🧮 Confusion Matrix")

        st.dataframe(
            cm_df,
            use_container_width=True,
        )

    st.info(
        "Metrics are calculated on a held-out test split of the historical "
        "dataset. They are not an official operational fire-warning accuracy."
    )


# ============================================================
# SAFETY & AWARENESS
# ============================================================
elif page == "🛡️ Safety & Awareness":

    render_html("""
        <div class="hero">
            <div class="hero-title">🛡️ Forest Safety & Awareness</div>
            <div class="hero-subtitle">
                Preventive awareness for forest-fire risk conditions
            </div>
        </div>
        """)

    st.markdown("### 🔥 Current Recommendations")

    for recommendation in recommendations(score):
        st.success("• " + recommendation)

    st.markdown("### 🌲 General Prevention")

    prevention = [
        "Avoid uncontrolled fires and burning activities near dry vegetation.",
        "Do not leave campfires or other heat sources unattended.",
        "Report visible smoke or suspected forest fires to appropriate local authorities.",
        "Avoid throwing cigarette ends or other ignition sources in forest areas.",
        "During hot, dry and windy conditions, increase monitoring and preparedness.",
    ]

    for item in prevention:
        st.info("• " + item)


# ============================================================
# ABOUT
# ============================================================
else:

    render_html("""
        <div class="hero">
            <div class="hero-title">ℹ️ About EcoShield AI</div>
            <div class="hero-subtitle">
                India's Forest Fire Risk Intelligence Platform
            </div>
        </div>
        """)

    render_html("""
        <div class="glass-card">
            <div class="info-title">🌲 Project Vision</div>
            <div class="info-text">
                EcoShield AI is an academic machine-learning prototype
                designed to demonstrate how environmental data and
                machine learning can be combined to estimate forest-fire
                risk across selected forest regions of India.
            </div>
        </div>
        """)

    st.write("")

    c1, c2 = st.columns(2)

    with c1:
        render_html("""
            <div class="glass-card">
                <div class="info-title">⚙️ Technology Stack</div>
                <div class="info-text">
                    🐍 Python<br>
                    🧠 Scikit-learn<br>
                    🌲 Random Forest<br>
                    📊 Pandas<br>
                    🖥️ Streamlit<br>
                    🌐 Open-Meteo Weather API
                </div>
            </div>
            """)

    with c2:
        render_html("""
            <div class="glass-card">
                <div class="info-title">⚠️ Important Limitation</div>
                <div class="info-text">
                    This is an academic prototype. The model estimates
                    environmental fire risk from historical training data
                    and current weather inputs. It does not directly
                    detect active fires or replace official emergency,
                    satellite or forest-department warning systems.
                </div>
            </div>
            """)

    render_html("""
        <div class="footer">
            EcoShield AI v2.1 • Academic Project •
            Built for environmental risk awareness
        </div>
        """)

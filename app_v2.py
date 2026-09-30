
"""
EcoShield AI v2.2
Forest Fire Risk Intelligence Dashboard

Run:
    streamlit run app_v2.py
"""

from pathlib import Path
from textwrap import dedent
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

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except ImportError:
    PYDECK_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EcoShield AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HELPERS
# ============================================================
def html(content: str):
    """
    Render HTML directly.

    Streamlit's st.html() is used instead of st.markdown() so that
    HTML such as <div>, <strong>, <br> and custom cards is never
    displayed as source code.
    """
    clean = dedent(content).strip()

    if hasattr(st, "html"):
        st.html(clean)
    else:
        st.markdown(clean, unsafe_allow_html=True)


def background_css():
    candidates = [
        Path("forest_background.jpg"),
        Path("forest_background.jpeg"),
        Path("assets/forest_background.jpg"),
        Path("assets/forest_background.jpeg"),
    ]

    for image_path in candidates:
        if image_path.exists():
            try:
                encoded = base64.b64encode(image_path.read_bytes()).decode()
                return f"""
                    background-image:
                        linear-gradient(
                            rgba(3, 16, 9, 0.84),
                            rgba(3, 16, 9, 0.94)
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
                circle at 80% 10%,
                rgba(255, 99, 32, 0.10),
                transparent 30%
            ),
            linear-gradient(135deg, #02150b, #062719 55%, #03170d);
    """


html(f"""
<style>

html, body, [class*="css"] {{
    font-family: "Times New Roman", Times, serif !important;
}}

.stApp {{
    {background_css()}
    color: #f4f7f5;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0.16);
}}

[data-testid="stSidebar"] {{
    background:
        linear-gradient(
            180deg,
            rgba(1, 20, 11, 0.98),
            rgba(0, 34, 18, 0.97)
        );
    border-right: 1px solid rgba(130, 255, 170, 0.14);
}}

[data-testid="stSidebar"] * {{
    font-family: "Times New Roman", Times, serif !important;
}}

.block-container {{
    max-width: 1450px;
    padding-top: 2.0rem;
    padding-bottom: 3.0rem;
}}

.side-brand {{
    text-align: center;
    padding: 16px 8px 24px;
}}

.side-brand-title {{
    color: #ddff75;
    font-size: 31px;
    font-weight: 700;
    line-height: 1.05;
}}

.side-brand-subtitle {{
    color: #87e8a6;
    font-size: 15px;
    margin-top: 9px;
}}

.sidebar-section {{
    color: #dff1e5;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.1px;
}}

.hero {{
    text-align: center;
    padding: 10px 18px 24px;
}}

.hero-title {{
    color: #e7ff7c;
    font-size: 47px;
    font-weight: 700;
    line-height: 1.08;
    margin: 0;
    text-shadow: 0 4px 22px rgba(0,0,0,.38);
}}

.hero-subtitle {{
    color: #dcece1;
    font-size: 19px;
    margin-top: 10px;
}}

.hero-meta {{
    color: #82e9a4;
    font-size: 14px;
    font-weight: 700;
    margin-top: 13px;
}}

.monitor {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 18px;
    background: rgba(3, 25, 14, .84);
    border: 1px solid rgba(128, 255, 170, .17);
    border-radius: 17px;
    padding: 15px 20px;
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
    background: #c92828;
    color: #fff;
    padding: 4px 10px;
    border-radius: 16px;
    font-size: 11px;
    margin-left: 9px;
    font-weight: 700;
}}

.glass {{
    background: rgba(4, 25, 15, .80);
    border: 1px solid rgba(127, 255, 169, .16);
    border-radius: 20px;
    padding: 23px;
    box-shadow: 0 15px 40px rgba(0,0,0,.20);
}}

.section-title {{
    color: #e5ff8c;
    font-size: 24px;
    font-weight: 700;
    margin: 8px 0 14px;
}}

.section-text {{
    color: #d0ded4;
    font-size: 16px;
    line-height: 1.7;
}}

.metric-card {{
    background: rgba(4, 27, 16, .84);
    border: 1px solid rgba(127, 255, 169, .15);
    border-radius: 17px;
    padding: 17px;
    min-height: 120px;
}}

.metric-label {{
    color: #b9c9be;
    font-size: 15px;
}}

.metric-value {{
    color: #fff;
    font-size: 30px;
    font-weight: 700;
    margin-top: 7px;
}}

.metric-note {{
    color: #7ee8a1;
    font-size: 13px;
    margin-top: 4px;
}}

.risk-card {{
    border-radius: 22px;
    padding: 29px 24px;
    text-align: center;
    border: 1px solid rgba(255,255,255,.14);
    min-height: 275px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 18px 50px rgba(0,0,0,.30);
}}

.risk-low {{
    background:
        linear-gradient(
            135deg,
            rgba(11, 91, 49, .95),
            rgba(5, 52, 31, .96)
        );
}}

.risk-moderate {{
    background:
        linear-gradient(
            135deg,
            rgba(118, 76, 5, .95),
            rgba(70, 42, 3, .96)
        );
}}

.risk-high {{
    background:
        linear-gradient(
            135deg,
            rgba(148, 61, 10, .96),
            rgba(76, 26, 7, .97)
        );
}}

.risk-critical {{
    background:
        linear-gradient(
            135deg,
            rgba(150, 31, 24, .97),
            rgba(72, 10, 10, .98)
        );
}}

.risk-title {{
    color: #fff;
    font-size: 20px;
    font-weight: 700;
}}

.risk-number {{
    color: #fff;
    font-size: 61px;
    font-weight: 700;
    line-height: 1;
    margin: 17px 0 9px;
}}

.risk-level {{
    color: #fff;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: .5px;
}}

.risk-note {{
    color: #e4e9e5;
    font-size: 14px;
    margin-top: 10px;
}}

.band {{
    border-radius: 12px;
    padding: 12px 14px;
    margin: 7px 0;
    color: #fff;
    font-size: 14px;
}}

.band-green {{ background: rgba(28, 139, 76, .75); }}
.band-yellow {{ background: rgba(179, 132, 19, .80); }}
.band-orange {{ background: rgba(181, 78, 18, .82); }}
.band-red {{ background: rgba(177, 39, 33, .84); }}

.info-row {{
    background: rgba(10, 34, 21, .75);
    border: 1px solid rgba(127, 255, 169, .11);
    border-radius: 13px;
    padding: 13px 15px;
    margin: 7px 0;
    color: #d7e3da;
}}

.info-row strong {{
    color: #ecff9b;
}}

.review-card {{
    background: rgba(4, 24, 15, .82);
    border-left: 4px solid #9ddc42;
    border-radius: 12px;
    padding: 17px 20px;
    margin: 10px 0;
}}

.review-title {{
    color: #e9ff91;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 7px;
}}

.review-text {{
    color: #d3dfd6;
    font-size: 15px;
    line-height: 1.65;
}}

.footer {{
    text-align: center;
    color: #94aa9a;
    font-size: 13px;
    padding: 30px 0 10px;
}}

[data-testid="stMetricValue"] {{
    font-family: "Times New Roman", Times, serif !important;
}}

</style>
""")


# ============================================================
# FOREST REGIONS
# ============================================================
FOREST_REGIONS = {
    "Dachigam — Jammu & Kashmir": {
        "lat": 34.0837, "lon": 74.8836, "state": "Jammu & Kashmir"
    },
    "Jim Corbett — Uttarakhand": {
        "lat": 29.5300, "lon": 78.7747, "state": "Uttarakhand"
    },
    "Ranthambore — Rajasthan": {
        "lat": 26.0173, "lon": 76.5026, "state": "Rajasthan"
    },
    "Kaziranga — Assam": {
        "lat": 26.5775, "lon": 93.1711, "state": "Assam"
    },
    "Gir — Gujarat": {
        "lat": 21.1243, "lon": 70.8242, "state": "Gujarat"
    },
    "Kanha — Madhya Pradesh": {
        "lat": 22.3345, "lon": 80.6115, "state": "Madhya Pradesh"
    },
    "Tadoba — Maharashtra": {
        "lat": 20.2510, "lon": 79.3580, "state": "Maharashtra"
    },
    "Similipal — Odisha": {
        "lat": 21.9497, "lon": 86.3700, "state": "Odisha"
    },
    "Sundarbans — West Bengal": {
        "lat": 21.9497, "lon": 88.9000, "state": "West Bengal"
    },
    "Bandipur — Karnataka": {
        "lat": 11.6821, "lon": 76.6300, "state": "Karnataka"
    },
    "Nallamala — Andhra Pradesh": {
        "lat": 15.3793, "lon": 78.4800, "state": "Andhra Pradesh"
    },
    "Nagarjunsagar — Telangana": {
        "lat": 16.5427, "lon": 79.3110, "state": "Telangana"
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
    48: "Rime fog",
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
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "precipitation"
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
    hourly = hourly[hourly["time"] >= current_time].head(24).copy()

    return {
        "temperature": float(current["temperature_2m"]),
        "humidity": float(current["relative_humidity_2m"]),
        "wind": float(current["wind_speed_10m"]),
        "rain": float(current["precipitation"]),
        "weather_code": int(current["weather_code"]),
        "weather": WEATHER_CODES.get(
            int(current["weather_code"]),
            "Unknown",
        ),
        "time": current["time"],
        "timezone": payload.get("timezone", "Local"),
        "hourly": hourly,
    }


# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def train_model():
    data = pd.read_csv("forestfires.csv")

    required = ["temp", "RH", "wind", "area"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            "forestfires.csv is missing: " + ", ".join(missing)
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
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "cm": confusion_matrix(y_test, predictions),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return data, model, features, metrics


data, model, FEATURES, METRICS = train_model()


# ============================================================
# RISK
# ============================================================
def risk_result(temp, humidity, wind, rain):
    values = {
        "temp": temp,
        "RH": humidity,
        "wind": wind,
        "rain": rain,
    }

    row = pd.DataFrame(
        [[values[f] for f in FEATURES]],
        columns=FEATURES,
    )

    probabilities = model.predict_proba(row)[0]
    positive_index = list(model.classes_).index(1)
    score = float(probabilities[positive_index] * 100)

    if score < 25:
        level, css = "LOW", "risk-low"
    elif score < 50:
        level, css = "MODERATE", "risk-moderate"
    elif score < 75:
        level, css = "HIGH", "risk-high"
    else:
        level, css = "CRITICAL", "risk-critical"

    return score, level, css


def map_rgb(score):
    if score >= 75:
        return [190, 35, 35]
    if score >= 50:
        return [230, 100, 25]
    if score >= 25:
        return [220, 175, 40]
    return [35, 155, 80]


def environmental_factors(temp, humidity, wind, rain):
    result = []

    if temp >= 35:
        result.append(
            ("Temperature", "High temperature is contributing to dry conditions.")
        )
    elif temp >= 30:
        result.append(
            ("Temperature", "Warm conditions may increase fire susceptibility.")
        )
    else:
        result.append(
            ("Temperature", "Temperature is not currently in the high-risk range.")
        )

    if humidity <= 30:
        result.append(
            ("Humidity", "Very low humidity indicates dry atmospheric conditions.")
        )
    elif humidity <= 45:
        result.append(
            ("Humidity", "Lower humidity can support ignition and drying.")
        )
    else:
        result.append(
            ("Humidity", "Humidity is providing some resistance to rapid drying.")
        )

    if wind >= 25:
        result.append(
            ("Wind", "Strong winds can accelerate fire spread.")
        )
    elif wind >= 15:
        result.append(
            ("Wind", "Moderate winds may support fire spread.")
        )
    else:
        result.append(
            ("Wind", "Wind speed is currently relatively low.")
        )

    if rain <= 0.1:
        result.append(
            ("Rainfall", "Little or no current rainfall is present.")
        )
    else:
        result.append(
            ("Rainfall", "Current rainfall can reduce immediate fire susceptibility.")
        )

    return result


def recommendations(score):
    if score >= 75:
        return [
            "Increase monitoring frequency.",
            "Avoid unnecessary open-fire activities.",
            "Keep response teams prepared.",
            "Follow local forest and emergency advisories.",
        ]

    if score >= 50:
        return [
            "Maintain enhanced monitoring.",
            "Avoid burning dry vegetation.",
            "Watch for changes in wind and humidity.",
            "Review local forest safety procedures.",
        ]

    if score >= 25:
        return [
            "Continue routine monitoring.",
            "Avoid uncontrolled fires near dry vegetation.",
            "Recheck conditions if temperature rises or humidity falls.",
        ]

    return [
        "Continue routine forest monitoring.",
        "Maintain preventive fire-safety practices.",
        "Reassess if weather conditions change significantly.",
    ]


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    html("""
    <div class="side-brand">
        <div class="side-brand-title">EcoShield AI</div>
        <div class="side-brand-subtitle">
            India's Forest Fire Intelligence
        </div>
    </div>
    """)

    st.markdown(
        '<div class="sidebar-section">NAVIGATION</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Forest Map",
            "Risk Prediction",
            "Live Weather",
            "Model Analytics",
            "Safety & Awareness",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section">MONITORING REGION</div>',
        unsafe_allow_html=True,
    )

    selected_region = st.selectbox(
        "Select Forest Region",
        list(FOREST_REGIONS.keys()),
    )

    st.markdown("---")

    if st.button(
        "Refresh Live Data",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    st.caption("Weather source: Open-Meteo")
    st.caption("Model: Random Forest")
    st.caption("Coverage: Selected Indian forest regions")


# ============================================================
# CURRENT REGION DATA
# ============================================================
selected = FOREST_REGIONS[selected_region]

try:
    current_weather = get_weather(
        selected["lat"],
        selected["lon"],
    )

    temperature = current_weather["temperature"]
    humidity = current_weather["humidity"]
    wind = current_weather["wind"]
    rain = current_weather["rain"]

    current_score, current_level, current_css = risk_result(
        temperature,
        humidity,
        wind,
        rain,
    )

except Exception as exc:
    st.error(
        "Unable to load live weather. Check your internet connection "
        "and press Refresh Live Data."
    )
    st.caption(str(exc))
    st.stop()


# ============================================================
# PAGE: DASHBOARD
# ============================================================
if page == "Dashboard":

    html("""
    <div class="hero">
        <div class="hero-title">EcoShield AI</div>
        <div class="hero-subtitle">
            India's Forest Fire Risk Intelligence Platform
        </div>
        <div class="hero-meta">
            LIVE ENVIRONMENTAL MONITORING &nbsp; | &nbsp;
            MACHINE LEARNING RISK ESTIMATION &nbsp; | &nbsp;
            FOREST REGION COVERAGE
        </div>
    </div>
    """)

    html(f"""
    <div class="monitor">
        <div class="monitor-left">
            Monitoring: <b>{selected_region}</b>
            <span class="live-badge">LIVE</span>
        </div>
        <div class="monitor-right">
            Updated: <b>{current_weather["time"]}</b>
        </div>
    </div>
    """)

    m1, m2, m3, m4 = st.columns(4)

    metric_data = [
        ("Temperature", f"{temperature:.1f} °C", "Current condition"),
        ("Humidity", f"{humidity:.0f} %", "Relative humidity"),
        ("Wind Speed", f"{wind:.1f} km/h", "Current wind"),
        ("Rainfall", f"{rain:.1f} mm", "Current precipitation"),
    ]

    for column, (label, value, note) in zip(
        [m1, m2, m3, m4],
        metric_data,
    ):
        with column:
            html(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """)

    st.write("")

    left, right = st.columns([1.05, .95], gap="large")

    with left:
        html(f"""
        <div class="risk-card {current_css}">
            <div class="risk-title">FOREST FIRE RISK</div>
            <div class="risk-number">
                {current_score:.0f}
                <span style="font-size:24px;">/100</span>
            </div>
            <div class="risk-level">{current_level} RISK</div>
            <div class="risk-note">
                Random Forest environmental risk estimate
            </div>
        </div>
        """)

    with right:
        html(f"""
        <div class="glass">
            <div class="section-title">Live Weather Report</div>
            <div class="section-text">
                Temperature: <b>{temperature:.1f} °C</b><br>
                Relative Humidity: <b>{humidity:.0f}%</b><br>
                Wind Speed: <b>{wind:.1f} km/h</b><br>
                Rainfall: <b>{rain:.1f} mm</b><br>
                Condition: <b>{current_weather["weather"]}</b><br><br>
                Region: <b>{selected_region}</b><br>
                Weather Source: <b>Open-Meteo</b>
            </div>
        </div>
        """)

    st.markdown("### Environmental Factors")

    for name, explanation in environmental_factors(
        temperature,
        humidity,
        wind,
        rain,
    ):
        html(f"""
        <div class="info-row">
            <strong>{name}</strong> — {explanation}
        </div>
        """)

    st.markdown("### Risk Classification")

    html("""
    <div class="band band-green"><b>0–24</b> &nbsp; Low — routine monitoring</div>
    <div class="band band-yellow"><b>25–49</b> &nbsp; Moderate — increased awareness</div>
    <div class="band band-orange"><b>50–74</b> &nbsp; High — enhanced monitoring</div>
    <div class="band band-red"><b>75–100</b> &nbsp; Critical — high-priority monitoring</div>
    """)

    st.markdown("### Next 24-Hour Risk Trend")

    hourly = current_weather["hourly"].copy()

    if not hourly.empty:
        trend_scores = []

        for _, row in hourly.iterrows():
            s, _, _ = risk_result(
                float(row["temperature_2m"]),
                float(row["relative_humidity_2m"]),
                float(row["wind_speed_10m"]),
                float(row["precipitation"]),
            )
            trend_scores.append(s)

        trend = pd.DataFrame({
            "Time": hourly["time"],
            "Risk Score": trend_scores,
        })

        st.line_chart(
            trend.set_index("Time"),
            y="Risk Score",
        )


# ============================================================
# PAGE: FOREST MAP
# ============================================================
elif page == "Forest Map":

    html("""
    <div class="hero">
        <div class="hero-title">Forest Risk Map</div>
        <div class="hero-subtitle">
            Live weather and model risk across selected Indian forest regions
        </div>
    </div>
    """)

    st.info(
        "Move your cursor over a colored forest marker to see its "
        "location, risk level and live weather details."
    )

    rows = []
    failures = []

    with st.spinner("Loading live forest-region conditions..."):

        for name, info in FOREST_REGIONS.items():
            try:
                w = get_weather(info["lat"], info["lon"])

                score, level, _ = risk_result(
                    w["temperature"],
                    w["humidity"],
                    w["wind"],
                    w["rain"],
                )

                rows.append({
                    "Region": name,
                    "State": info["state"],
                    "lat": info["lat"],
                    "lon": info["lon"],
                    "Temperature": round(w["temperature"], 1),
                    "Humidity": round(w["humidity"], 0),
                    "Wind": round(w["wind"], 1),
                    "Rainfall": round(w["rain"], 1),
                    "Condition": w["weather"],
                    "Risk": round(score, 1),
                    "Level": level,
                    "Color": map_rgb(score),
                    "Updated": w["time"],
                })

            except Exception:
                failures.append(name)

    map_df = pd.DataFrame(rows)

    if map_df.empty:
        st.error("No forest-region weather data could be loaded.")
    else:

        if PYDECK_AVAILABLE:

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_fill_color="Color",
                get_line_color=[255, 255, 255],
                get_line_width=2,
                get_radius=42000,
                pickable=True,
                auto_highlight=True,
            )

            view_state = pdk.ViewState(
                latitude=21.0,
                longitude=79.0,
                zoom=4.25,
                pitch=0,
            )

            tooltip = {
                "html": """
                <div style="
                    font-family: Arial, sans-serif;
                    min-width: 250px;
                    padding: 8px;
                ">
                    <div style="
                        font-size: 17px;
                        font-weight: 700;
                        margin-bottom: 8px;
                    ">
                        {Region}
                    </div>

                    <div><b>State:</b> {State}</div>
                    <div><b>Risk:</b> {Risk}/100 ({Level})</div>
                    <hr>
                    <div><b>Temperature:</b> {Temperature} °C</div>
                    <div><b>Humidity:</b> {Humidity} %</div>
                    <div><b>Wind Speed:</b> {Wind} km/h</div>
                    <div><b>Rainfall:</b> {Rainfall} mm</div>
                    <div><b>Condition:</b> {Condition}</div>
                    <div><b>Updated:</b> {Updated}</div>
                </div>
                """
            }

            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_provider="carto",
                map_style="light",
            )

            st.pydeck_chart(
                deck,
                use_container_width=True,
            )

        else:
            st.warning(
                "Install pydeck to enable hover details on forest markers."
            )
            st.map(
                map_df,
                latitude="lat",
                longitude="lon",
                size=120,
            )

        st.markdown("### Regional Risk Details")

        table = map_df[
            [
                "Region",
                "State",
                "Temperature",
                "Humidity",
                "Wind",
                "Rainfall",
                "Condition",
                "Risk",
                "Level",
            ]
        ].sort_values(
            "Risk",
            ascending=False,
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

    if failures:
        st.warning(
            f"Live weather was unavailable for {len(failures)} region(s)."
        )


# ============================================================
# PAGE: RISK PREDICTION
# ============================================================
elif page == "Risk Prediction":

    html("""
    <div class="hero">
        <div class="hero-title">Fire Risk Prediction</div>
        <div class="hero-subtitle">
            Compare current conditions with a custom environmental scenario
        </div>
    </div>
    """)

    html(f"""
    <div class="glass">
        <div class="section-title">Current Live Conditions</div>
        <div class="section-text">
            {selected_region} &nbsp; | &nbsp;
            Temperature: <b>{temperature:.1f} °C</b> &nbsp; | &nbsp;
            Humidity: <b>{humidity:.0f}%</b> &nbsp; | &nbsp;
            Wind: <b>{wind:.1f} km/h</b> &nbsp; | &nbsp;
            Rainfall: <b>{rain:.1f} mm</b>
        </div>
    </div>
    """)

    st.markdown("### Scenario Analysis")

    s1, s2 = st.columns(2)

    with s1:
        temp_input = st.slider(
            "Temperature °C",
            0.0,
            50.0,
            float(temperature),
            0.1,
        )

        humidity_input = st.slider(
            "Relative Humidity %",
            0.0,
            100.0,
            float(humidity),
            1.0,
        )

    with s2:
        wind_input = st.slider(
            "Wind Speed km/h",
            0.0,
            100.0,
            float(wind),
            0.1,
        )

        rain_input = st.slider(
            "Rainfall mm",
            0.0,
            50.0,
            float(min(rain, 50)),
            0.1,
        )

    scenario_score, scenario_level, scenario_css = risk_result(
        temp_input,
        humidity_input,
        wind_input,
        rain_input,
    )

    st.markdown("### Prediction Result")

    a, b = st.columns([1, 1], gap="large")

    with a:
        html(f"""
        <div class="risk-card {scenario_css}">
            <div class="risk-title">AI RISK ESTIMATE</div>
            <div class="risk-number">
                {scenario_score:.0f}
                <span style="font-size:24px;">/100</span>
            </div>
            <div class="risk-level">{scenario_level} RISK</div>
            <div class="risk-note">
                Estimated from the selected environmental inputs
            </div>
        </div>
        """)

    with b:
        difference = scenario_score - current_score

        if difference > 0:
            direction = "higher"
            change_text = f"{abs(difference):.1f} points higher"
        elif difference < 0:
            direction = "lower"
            change_text = f"{abs(difference):.1f} points lower"
        else:
            direction = "the same"
            change_text = "no change"

        html(f"""
        <div class="glass">
            <div class="section-title">Live vs Scenario</div>
            <div class="info-row">
                Current live risk: <strong>{current_score:.1f}/100</strong>
            </div>
            <div class="info-row">
                Scenario risk: <strong>{scenario_score:.1f}/100</strong>
            </div>
            <div class="info-row">
                Scenario is <strong>{direction}</strong> by
                <strong>{change_text}</strong>.
            </div>
        </div>
        """)

    st.markdown("### Environmental Impact Review")

    for name, explanation in environmental_factors(
        temp_input,
        humidity_input,
        wind_input,
        rain_input,
    ):
        html(f"""
        <div class="info-row">
            <strong>{name}</strong> — {explanation}
        </div>
        """)

    st.markdown("### Suggested Action")

    for item in recommendations(scenario_score):
        st.write("• " + item)


# ============================================================
# PAGE: LIVE WEATHER
# ============================================================
elif page == "Live Weather":

    html("""
    <div class="hero">
        <div class="hero-title">Live Weather</div>
        <div class="hero-subtitle">
            Current atmospheric conditions for the selected forest region
        </div>
    </div>
    """)

    html(f"""
    <div class="glass">
        <div class="section-title">{selected_region}</div>

        <div class="info-row">
            <strong>Temperature:</strong> {temperature:.1f} °C
        </div>

        <div class="info-row">
            <strong>Relative Humidity:</strong> {humidity:.0f} %
        </div>

        <div class="info-row">
            <strong>Wind Speed:</strong> {wind:.1f} km/h
        </div>

        <div class="info-row">
            <strong>Rainfall:</strong> {rain:.1f} mm
        </div>

        <div class="info-row">
            <strong>Condition:</strong> {current_weather["weather"]}
        </div>

        <div class="info-row">
            <strong>API Update:</strong> {current_weather["time"]}
        </div>

        <div class="info-row">
            <strong>Timezone:</strong> {current_weather["timezone"]}
        </div>
    </div>
    """)

    st.markdown("### Next 24 Hours")

    hourly = current_weather["hourly"].copy()

    if not hourly.empty:
        chart_df = hourly[
            [
                "time",
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "precipitation",
            ]
        ].rename(
            columns={
                "time": "Time",
                "temperature_2m": "Temperature °C",
                "relative_humidity_2m": "Humidity %",
                "wind_speed_10m": "Wind km/h",
                "precipitation": "Rain mm",
            }
        )

        st.line_chart(
            chart_df.set_index("Time")
        )


# ============================================================
# PAGE: MODEL ANALYTICS
# ============================================================
elif page == "Model Analytics":

    html("""
    <div class="hero">
        <div class="hero-title">Model Analytics</div>
        <div class="hero-subtitle">
            How the Random Forest model was trained and evaluated
        </div>
    </div>
    """)

    st.markdown("### Model Performance")

    a, b, c, d = st.columns(4)

    metrics_to_show = [
        (a, "Accuracy", METRICS["accuracy"]),
        (b, "Precision", METRICS["precision"]),
        (c, "Recall", METRICS["recall"]),
        (d, "F1 Score", METRICS["f1"]),
    ]

    for col, label, value in metrics_to_show:
        with col:
            st.metric(label, f"{value * 100:.1f}%")

    st.markdown("### What These Metrics Mean")

    html("""
    <div class="review-card">
        <div class="review-title">Accuracy</div>
        <div class="review-text">
            Percentage of test records classified correctly overall.
        </div>
    </div>

    <div class="review-card">
        <div class="review-title">Precision</div>
        <div class="review-text">
            Among records predicted as fire-risk, the proportion that
            actually belonged to the fire class.
        </div>
    </div>

    <div class="review-card">
        <div class="review-title">Recall</div>
        <div class="review-text">
            Among actual fire-class records, the proportion detected by
            the model. Recall is especially useful when missed fire-risk
            cases matter.
        </div>
    </div>

    <div class="review-card">
        <div class="review-title">F1 Score</div>
        <div class="review-text">
            A combined measure of precision and recall using their
            harmonic mean.
        </div>
    </div>
    """)

    st.markdown("### Dataset and Training Setup")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric("Total Records", len(data))

    with d2:
        st.metric("Training Records", METRICS["train_size"])

    with d3:
        st.metric("Testing Records", METRICS["test_size"])

    st.markdown("### Confusion Matrix")

    cm = METRICS["cm"]

    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Fire", "Actual: Fire"],
        columns=["Predicted: No Fire", "Predicted: Fire"],
    )

    st.dataframe(
        cm_df,
        use_container_width=True,
    )

    tn, fp, fn, tp = cm.ravel()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("True Negative", int(tn))

    with c2:
        st.metric("False Positive", int(fp))

    with c3:
        st.metric("False Negative", int(fn))

    with c4:
        st.metric("True Positive", int(tp))

    html("""
    <div class="review-card">
        <div class="review-title">How to read the confusion matrix</div>
        <div class="review-text">
            True Negative: model correctly predicted no fire.
            False Positive: model predicted fire when the test record
            was no fire.
            False Negative: model missed a fire-class record.
            True Positive: model correctly predicted the fire class.
        </div>
    </div>
    """)

    st.markdown("### Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values(
        "Importance",
        ascending=False,
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True,
    )

    html("""
    <div class="review-card">
        <div class="review-title">Project review point</div>
        <div class="review-text">
            The model uses temperature, relative humidity, wind speed and,
            when available in the dataset, rainfall. The target variable
            is historical fire occurrence derived from whether the
            recorded burned area is greater than zero.
        </div>
    </div>

    <div class="review-card">
        <div class="review-title">Important limitation</div>
        <div class="review-text">
            The model is an academic prototype. Its score is an estimate
            based on historical training data and current weather inputs;
            it is not an official wildfire warning or active-fire detector.
        </div>
    </div>
    """)


# ============================================================
# PAGE: SAFETY
# ============================================================
elif page == "Safety & Awareness":

    html("""
    <div class="hero">
        <div class="hero-title">Forest Safety & Awareness</div>
        <div class="hero-subtitle">
            Preventive practices for forest-fire risk conditions
        </div>
    </div>
    """)

    st.markdown("### Current Risk-Based Guidance")

    for item in recommendations(current_score):
        html(f"""
        <div class="review-card">
            <div class="review-text">• {item}</div>
        </div>
        """)

    st.markdown("### General Forest-Fire Prevention")

    prevention = [
        "Do not start uncontrolled fires or burn vegetation near dry forest areas.",
        "Never leave campfires, cooking fires or other ignition sources unattended.",
        "Do not throw cigarette ends, matches or other ignition materials in forest areas.",
        "Avoid activities that can create sparks during hot, dry and windy conditions.",
        "Report visible smoke or suspected fire to the appropriate local authorities.",
        "Follow official forest-department and emergency instructions during high-risk periods.",
    ]

    for item in prevention:
        html(f"""
        <div class="info-row">
            {item}
        </div>
        """)

    st.markdown("### Risk Response Levels")

    html("""
    <div class="band band-green">
        <b>LOW</b> — Routine monitoring and standard prevention.
    </div>

    <div class="band band-yellow">
        <b>MODERATE</b> — Increase awareness and recheck weather conditions.
    </div>

    <div class="band band-orange">
        <b>HIGH</b> — Enhanced monitoring and preventive readiness.
    </div>

    <div class="band band-red">
        <b>CRITICAL</b> — High-priority monitoring and adherence to official advisories.
    </div>
    """)


# ============================================================
# PAGE: ABOUT
# ============================================================
else:

    html("""
    <div class="hero">
        <div class="hero-title">About EcoShield AI</div>
        <div class="hero-subtitle">
            Forest Fire Risk Intelligence — Academic Project
        </div>
    </div>
    """)

    html("""
    <div class="glass">
        <div class="section-title">Project Overview</div>
        <div class="section-text">
            EcoShield AI is a machine-learning based environmental
            intelligence prototype designed to estimate forest-fire risk
            from weather and environmental conditions. The system combines
            a Random Forest classifier with live weather information and
            a forest-region monitoring interface.
        </div>
    </div>
    """)

    st.markdown("### Problem Statement")

    html("""
    <div class="review-card">
        <div class="review-text">
            Forest-fire risk can increase when environmental conditions
            become hot, dry and windy. Traditional monitoring can require
            continuous observation of large forest areas. EcoShield AI
            demonstrates a software-based approach for bringing
            environmental measurements, machine learning and visualization
            into one monitoring dashboard.
        </div>
    </div>
    """)

    st.markdown("### Proposed Solution")

    html("""
    <div class="review-card">
        <div class="review-text">
            The system obtains current weather information for selected
            forest regions, sends environmental values through a trained
            Random Forest model, converts the model output into a
            0–100 risk score, classifies the risk level, and presents
            the result through a dashboard, map and analytical views.
        </div>
    </div>
    """)

    st.markdown("### System Workflow")

    workflow = [
        "Historical forest-fire dataset",
        "Data preparation and risk-label creation",
        "Train/test split",
        "Random Forest model training",
        "Live weather retrieval",
        "Risk-score estimation",
        "Forest-region map visualization",
        "Safety and monitoring guidance",
    ]

    for i, step in enumerate(workflow, start=1):
        html(f"""
        <div class="info-row">
            <strong>Step {i}:</strong> {step}
        </div>
        """)

    st.markdown("### Technology Stack")

    html("""
    <div class="glass">
        <div class="info-row"><strong>Programming:</strong> Python</div>
        <div class="info-row"><strong>Data Processing:</strong> Pandas</div>
        <div class="info-row"><strong>Machine Learning:</strong> Scikit-learn</div>
        <div class="info-row"><strong>Algorithm:</strong> Random Forest Classifier</div>
        <div class="info-row"><strong>Web Application:</strong> Streamlit</div>
        <div class="info-row"><strong>Weather Service:</strong> Open-Meteo</div>
        <div class="info-row"><strong>Interactive Map:</strong> PyDeck</div>
    </div>
    """)

    st.markdown("### Key Modules")

    modules = [
        ("Dashboard", "Current environmental conditions and overall risk."),
        ("Forest Map", "Regional risk visualization with hover weather details."),
        ("Risk Prediction", "Scenario-based risk estimation using custom inputs."),
        ("Live Weather", "Current and next-24-hour environmental conditions."),
        ("Model Analytics", "Evaluation metrics, confusion matrix and feature importance."),
        ("Safety & Awareness", "Risk-level guidance and prevention practices."),
    ]

    for title, description in modules:
        html(f"""
        <div class="review-card">
            <div class="review-title">{title}</div>
            <div class="review-text">{description}</div>
        </div>
        """)

    st.markdown("### Future Scope")

    future_scope = [
        "Integrate satellite hotspot and vegetation information.",
        "Use geographically representative wildfire datasets.",
        "Add historical risk maps and seasonal trend analysis.",
        "Add user authentication and role-based monitoring.",
        "Deploy the dashboard as a cloud-hosted application.",
        "Evaluate the model using additional environmental variables.",
    ]

    for item in future_scope:
        html(f"""
        <div class="info-row">{item}</div>
        """)

    st.markdown("### Academic Limitation")

    html("""
    <div class="review-card">
        <div class="review-title">Important</div>
        <div class="review-text">
            This project is an academic prototype. It estimates
            environmental fire risk; it does not directly detect active
            fires and must not be treated as a replacement for official
            wildfire, emergency or forest-department warning systems.
        </div>
    </div>
    """)

    html("""
    <div class="footer">
        EcoShield AI v2.2 • Academic Project • Environmental Risk Awareness
    </div>
    """)


# ============================================================
# END
# ============================================================

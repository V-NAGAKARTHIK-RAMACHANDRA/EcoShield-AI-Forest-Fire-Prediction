import os
import base64
import requests
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

try:
    import pydeck as pdk

    PYDECK = True
except ImportError:
    PYDECK = False


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="EcoShield AI",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# BACKGROUND IMAGE
# =========================================================

BG_PATH = os.path.join("assets", "forest_background.jpg")


def load_background():

    if not os.path.exists(BG_PATH):
        return ""

    with open(BG_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


BG = load_background()


# =========================================================
# PREMIUM UI
# =========================================================

if BG:

    background = f"""
    background-image:
        linear-gradient(
            rgba(2, 12, 7, 0.78),
            rgba(2, 12, 7, 0.92)
        ),
        url("data:image/jpeg;base64,{BG}");
    """

else:

    background = """
    background:
        linear-gradient(
            135deg,
            #06140b,
            #0b2414,
            #020805
        );
    """


st.markdown(
    f"""
<style>

.stApp {{
    {background}

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}}

/* SIDEBAR */

section[data-testid="stSidebar"] {{
    background:
        linear-gradient(
            180deg,
            rgba(2,15,8,.97),
            rgba(3,28,14,.96)
        );

    border-right:
        1px solid rgba(163,230,53,.20);
}}

section[data-testid="stSidebar"] * {{
    color: #f0fdf4;
}}

/* TITLES */

.hero-title {{
    font-size: 48px;
    font-weight: 900;
    color: #d9f99d;
    line-height: 1;
    margin-bottom: 7px;
}}

.hero-subtitle {{
    color: #d1d5db;
    font-size: 17px;
    margin-bottom: 8px;
}}

.hero-line {{
    color: #86efac;
    font-size: 13px;
    margin-bottom: 22px;
}}

/* GLASS */

.glass {{
    background:
        rgba(5,25,14,.72);

    border:
        1px solid rgba(255,255,255,.11);

    border-radius:
        20px;

    padding:
        20px;

    backdrop-filter:
        blur(16px);

    box-shadow:
        0 12px 40px rgba(0,0,0,.30);

    color:
        #f8fafc;

    margin-bottom:
        16px;
}}

/* METRICS */

div[data-testid="stMetric"] {{
    background:
        rgba(4,25,13,.78);

    border:
        1px solid rgba(134,239,172,.12);

    border-radius:
        16px;

    padding:
        15px;

    box-shadow:
        0 8px 25px rgba(0,0,0,.20);
}}

div[data-testid="stMetricLabel"] {{
    color:#a7f3d0 !important;
}}

div[data-testid="stMetricValue"] {{
    color:#ffffff !important;
}}

/* SECTION */

.section {{
    color:#d9f99d;
    font-size:22px;
    font-weight:800;
    margin:18px 0 10px 0;
}}

/* RISK */

.risk-box {{
    background:
        linear-gradient(
            135deg,
            rgba(127,29,29,.78),
            rgba(20,20,10,.85)
        );

    border:
        1px solid rgba(251,146,60,.45);

    border-radius:
        24px;

    padding:
        28px;

    text-align:
        center;

    box-shadow:
        0 0 40px rgba(249,115,22,.14);
}}

.risk-score {{
    font-size:60px;
    font-weight:900;
    color:#ffffff;
}}

.risk-high {{
    color:#fb923c;
    font-size:25px;
    font-weight:900;
}}

.risk-critical {{
    color:#ef4444;
    font-size:25px;
    font-weight:900;
}}

.risk-moderate {{
    color:#facc15;
    font-size:25px;
    font-weight:900;
}}

.risk-low {{
    color:#4ade80;
    font-size:25px;
    font-weight:900;
}}

/* STATUS */

.live {{
    display:inline-block;
    background:#dc2626;
    color:white;
    padding:5px 11px;
    border-radius:20px;
    font-size:12px;
    font-weight:800;
}}

/* BUTTON */

.stButton > button {{
    border-radius:12px;
    background:
        linear-gradient(
            135deg,
            #166534,
            #15803d
        );
    color:white;
    border:1px solid #4ade80;
    font-weight:700;
}}

.stButton > button:hover {{
    background:#22c55e;
    color:#052e16;
}}

/* FOOTER */

.footer {{
    text-align:center;
    color:#94a3b8;
    font-size:12px;
    padding:25px 0;
}}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# MODEL
# =========================================================


@st.cache_resource
def train_model():

    data = pd.read_csv("forestfires.csv")

    required = ["temp", "RH", "wind", "area"]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data["risk"] = data["area"].apply(lambda x: 1 if x > 0 else 0)

    features = ["temp", "RH", "wind"]

    if "rain" in data.columns:
        features.append("rain")

    X = data[features]
    y = data["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
    }

    matrix = confusion_matrix(y_test, prediction)

    return (data, model, features, metrics, matrix)


try:

    data, model, FEATURES, METRICS, CM = train_model()

except Exception as e:

    st.error("Model loading failed.")

    st.code(str(e))

    st.stop()


# =========================================================
# INDIA FOREST REGIONS
# =========================================================

FOREST_REGIONS = {
    "Dachigam — Jammu & Kashmir": (34.0837, 74.9257),
    "Great Himalayan — Himachal Pradesh": (31.7500, 77.4500),
    "Jim Corbett — Uttarakhand": (29.5300, 78.7747),
    "Ranthambore — Rajasthan": (26.0173, 76.5026),
    "Gir — Gujarat": (21.1243, 70.8242),
    "Kanha — Madhya Pradesh": (22.3345, 80.6115),
    "Bandhavgarh — Madhya Pradesh": (23.6850, 81.0300),
    "Tadoba — Maharashtra": (20.2489, 79.2997),
    "Similipal — Odisha": (21.9497, 86.3700),
    "Sundarbans — West Bengal": (21.9497, 89.1833),
    "Kaziranga — Assam": (26.5775, 93.1711),
    "Manas — Assam": (26.6594, 91.0011),
    "Nallamala — Andhra Pradesh": (15.3793, 78.4800),
    "Nagarjunsagar — Andhra Pradesh": (16.0850, 79.3000),
    "Bandipur — Karnataka": (11.7401, 76.6800),
    "Mudumalai — Tamil Nadu": (11.5731, 76.5450),
    "Periyar — Kerala": (9.4620, 77.2360),
    "Nilgiri — South India": (11.4064, 76.6932),
}


# =========================================================
# WEATHER
# =========================================================


def weather_name(code):

    names = {
        0: "Clear Sky",
        1: "Mainly Clear",
        2: "Partly Cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Fog",
        51: "Light Drizzle",
        53: "Moderate Drizzle",
        55: "Heavy Drizzle",
        61: "Light Rain",
        63: "Moderate Rain",
        65: "Heavy Rain",
        71: "Snow",
        73: "Snow",
        75: "Heavy Snow",
        80: "Rain Showers",
        81: "Rain Showers",
        82: "Heavy Rain Showers",
        95: "Thunderstorm",
        96: "Thunderstorm",
        99: "Thunderstorm",
    }

    return names.get(int(code), "Unknown")


@st.cache_data(ttl=600)
def get_weather(lat, lon):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,"
        "relative_humidity_2m,"
        "wind_speed_10m,"
        "precipitation,"
        "weather_code",
        "hourly": "temperature_2m,"
        "relative_humidity_2m,"
        "wind_speed_10m,"
        "precipitation",
        "forecast_days": 1,
        "timezone": "auto",
    }

    r = requests.get(url, params=params, timeout=15)

    r.raise_for_status()

    result = r.json()

    current = result["current"]

    hourly = pd.DataFrame(result["hourly"])

    hourly["time"] = pd.to_datetime(hourly["time"])

    return {
        "temp": float(current["temperature_2m"]),
        "humidity": float(current["relative_humidity_2m"]),
        "wind": float(current["wind_speed_10m"]),
        "rain": float(current["precipitation"]),
        "weather": weather_name(current["weather_code"]),
        "time": current["time"],
        "hourly": hourly,
    }


# =========================================================
# RISK
# =========================================================


def risk_score(temp, humidity, wind, rain):

    values = {"temp": temp, "RH": humidity, "wind": wind, "rain": rain}

    frame = pd.DataFrame([values])

    frame = frame[FEATURES]

    probabilities = model.predict_proba(frame)[0]

    classes = list(model.classes_)

    fire_index = classes.index(1)

    return round(probabilities[fire_index] * 100, 1)


def risk_level(score):

    if score >= 75:
        return "CRITICAL"

    if score >= 50:
        return "HIGH"

    if score >= 25:
        return "MODERATE"

    return "LOW"


def risk_color(score):

    if score >= 75:
        return [239, 68, 68]

    if score >= 50:
        return [249, 115, 22]

    if score >= 25:
        return [250, 204, 21]

    return [34, 197, 94]


# =========================================================
# EXPLANATION
# =========================================================


def risk_reasons(temp, humidity, wind, rain):

    reasons = []

    if temp >= 35:
        reasons.append("🌡️ High temperature")

    elif temp >= 30:
        reasons.append("🌡️ Warm conditions")

    if humidity <= 30:
        reasons.append("💧 Very low humidity")

    elif humidity <= 40:
        reasons.append("💧 Low humidity")

    if wind >= 25:
        reasons.append("💨 Strong wind")

    elif wind >= 15:
        reasons.append("💨 Moderate wind")

    if rain <= 0.1:
        reasons.append("🌧️ Little/no rainfall")

    if not reasons:
        reasons.append("🌿 Relatively stable conditions")

    return reasons


def recommendations(score):

    if score >= 75:

        return [
            "🚨 Increase monitoring frequency",
            "🔥 Avoid activities that create sparks",
            "🌲 Monitor dry vegetation closely",
            "📞 Follow official emergency guidance",
        ]

    if score >= 50:

        return [
            "⚠️ Increase environmental monitoring",
            "🔥 Avoid unnecessary burning",
            "🌲 Monitor dry vegetation",
            "💨 Watch wind conditions",
        ]

    if score >= 25:

        return [
            "👀 Continue regular monitoring",
            "🌿 Watch temperature changes",
            "💧 Monitor rainfall and humidity",
        ]

    return [
        "✅ Current modeled risk is relatively low",
        "🌿 Continue routine monitoring",
    ]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
<div style="
font-size:29px;
font-weight:900;
color:#d9f99d;
">
🌲 EcoShield AI
</div>

<div style="
font-size:12px;
color:#86efac;
margin-bottom:20px;
">
India's Forest Fire Intelligence
</div>
""",
    unsafe_allow_html=True,
)


page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🇮🇳 India Forest Map",
        "🔥 Risk Prediction",
        "🌤️ Live Weather",
        "📊 Analytics",
        "🛡️ Safety & Awareness",
        "ℹ️ About Project",
    ],
)


st.sidebar.markdown("---")


selected_region = st.sidebar.selectbox(
    "📍 MONITORING REGION", list(FOREST_REGIONS.keys())
)


st.sidebar.markdown("---")


if st.sidebar.button("🔄 Refresh Live Data", use_container_width=True):

    st.cache_data.clear()
    st.rerun()


st.sidebar.markdown(
    """
<div style="
font-size:12px;
color:#94a3b8;
line-height:1.8;
margin-top:25px;
">
🟢 LIVE WEATHER<br>
🌐 Open-Meteo API<br>
🤖 Random Forest ML<br>
🇮🇳 Pan-India Coverage
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SELECTED LOCATION
# =========================================================

lat, lon = FOREST_REGIONS[selected_region]


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    try:

        weather = get_weather(lat, lon)

    except Exception as e:

        st.error("Live weather connection failed.")

        st.code(str(e))

        st.stop()

    temp = weather["temp"]
    humidity = weather["humidity"]
    wind = weather["wind"]
    rain = weather["rain"]

    score = risk_score(temp, humidity, wind, rain)

    level = risk_level(score)

    # HERO

    st.markdown(
        '<div class="hero-title">' "🌲 EcoShield AI" "</div>", unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "India's Forest Fire Risk Intelligence Platform"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-line">'
        "🟢 LIVE MONITORING &nbsp; | &nbsp; "
        "🤖 AI-POWERED PREDICTION &nbsp; | &nbsp; "
        "🇮🇳 PAN-INDIA FOREST COVERAGE"
        "</div>",
        unsafe_allow_html=True,
    )

    # TOP STATUS

    st.markdown(
        f"""
<div class="glass">
<b>📍 Monitoring:</b> {selected_region}
&nbsp;&nbsp;&nbsp;
<span class="live">● LIVE</span>
&nbsp;&nbsp;&nbsp;
<b>🕒 Updated:</b> {weather['time']}
</div>
""",
        unsafe_allow_html=True,
    )

    # METRICS

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🌡️ Temperature", f"{temp:.1f} °C")

    with c2:
        st.metric("💧 Humidity", f"{humidity:.0f} %")

    with c3:
        st.metric("💨 Wind Speed", f"{wind:.1f} km/h")

    with c4:
        st.metric("🌧️ Rainfall", f"{rain:.1f} mm")

    st.markdown("")

    # RISK + WEATHER

    left, right = st.columns([1, 1])

    with left:

        level_class = (
            "risk-critical"
            if level == "CRITICAL"
            else (
                "risk-high"
                if level == "HIGH"
                else "risk-moderate" if level == "MODERATE" else "risk-low"
            )
        )

        st.markdown(
            f"""
<div class="risk-box">
<div style="
font-size:16px;
color:#fed7aa;
font-weight:700;
">
🔥 FOREST FIRE RISK
</div>

<div class="risk-score">
{score:.0f}<span style="font-size:22px;"> / 100</span>
</div>

<div class="{level_class}">
{level} RISK
</div>

<div style="
color:#cbd5e1;
font-size:13px;
margin-top:10px;
">
Random Forest environmental risk estimate
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.progress(int(score))

    with right:

        st.markdown(
            f"""
<div class="glass">
<h3>🌤️ Live Weather Report</h3>

<p>🌡️ <b>{temp:.1f} °C</b> — Temperature</p>
<p>💧 <b>{humidity:.0f}%</b> — Relative Humidity</p>
<p>💨 <b>{wind:.1f} km/h</b> — Wind Speed</p>
<p>🌧️ <b>{rain:.1f} mm</b> — Rainfall</p>
<p>☁️ <b>{weather['weather']}</b></p>

<hr>

<p style="color:#86efac;">
📍 {selected_region}
</p>

<p style="color:#94a3b8;font-size:12px;">
Source: Open-Meteo API
</p>
</div>
""",
            unsafe_allow_html=True,
        )

    # TREND + EXPLANATION

    left2, right2 = st.columns([1.2, 0.8])

    with left2:

        st.markdown(
            '<div class="section">' "📈 24-Hour Fire Risk Trend" "</div>",
            unsafe_allow_html=True,
        )

        hourly = weather["hourly"].copy()

        trend_rows = []

        for _, row in hourly.iterrows():

            s = risk_score(
                row["temperature_2m"],
                row["relative_humidity_2m"],
                row["wind_speed_10m"],
                row["precipitation"],
            )

            trend_rows.append({"Time": row["time"], "Risk Score": s})

        trend = pd.DataFrame(trend_rows)

        if not trend.empty:

            st.line_chart(trend.set_index("Time")["Risk Score"], height=310)

    with right2:

        st.markdown(
            '<div class="section">' "💡 Why is the Risk at This Level?" "</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        for reason in risk_reasons(temp, humidity, wind, rain):
            st.write(reason)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section">' "🛡️ Recommended Actions" "</div>",
            unsafe_allow_html=True,
        )

        for action in recommendations(score):
            st.write(action)

    # INDIA SUMMARY

    st.markdown(
        '<div class="section">' "🇮🇳 EcoShield Monitoring Network" "</div>",
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric("🌲 Forest Regions", len(FOREST_REGIONS))

    with s2:
        st.metric("🤖 ML Algorithm", "Random Forest")

    with s3:
        st.metric("🌐 Weather Source", "Open-Meteo")

    with s4:
        st.metric("📡 Monitoring", "Live")

    # DOWNLOAD

    report = f"""
ECOSHIELD AI
INDIA FOREST FIRE RISK REPORT
====================================

Region:
{selected_region}

Temperature:
{temp:.1f} °C

Humidity:
{humidity:.1f} %

Wind:
{wind:.1f} km/h

Rainfall:
{rain:.1f} mm

Weather:
{weather['weather']}

Risk Score:
{score:.1f}/100

Risk Level:
{level}

Generated:
{weather['time']}

Weather Source:
Open-Meteo

Model:
Random Forest

NOTE:
Academic prototype only.
Not an official wildfire warning system.
"""

    st.download_button(
        "📥 Download Current Risk Report",
        report,
        file_name="EcoShield_Risk_Report.txt",
        mime="text/plain",
    )


# =========================================================
# INDIA MAP
# =========================================================

elif page == "🇮🇳 India Forest Map":

    st.markdown(
        '<div class="hero-title">' "🇮🇳 India Forest Fire Risk Map" "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "Live environmental risk across major forest regions"
        "</div>",
        unsafe_allow_html=True,
    )

    rows = []

    failed = []

    with st.spinner("🌐 Fetching live weather across India..."):

        for name, coords in FOREST_REGIONS.items():

            try:

                w = get_weather(coords[0], coords[1])

                s = risk_score(w["temp"], w["humidity"], w["wind"], w["rain"])

                rows.append(
                    {
                        "region": name,
                        "lat": coords[0],
                        "lon": coords[1],
                        "temperature": round(w["temp"], 1),
                        "humidity": round(w["humidity"], 1),
                        "wind": round(w["wind"], 1),
                        "rain": round(w["rain"], 1),
                        "risk": round(s, 1),
                        "level": risk_level(s),
                        "color": risk_color(s),
                    }
                )

            except Exception:
                failed.append(name)

    map_df = pd.DataFrame(rows)

    if not map_df.empty:

        # SUMMARY

        low = len(map_df[map_df["level"] == "LOW"])

        moderate = len(map_df[map_df["level"] == "MODERATE"])

        high = len(map_df[map_df["level"] == "HIGH"])

        critical = len(map_df[map_df["level"] == "CRITICAL"])

        a, b, c, d = st.columns(4)

        with a:
            st.metric("🟢 Low", low)

        with b:
            st.metric("🟡 Moderate", moderate)

        with c:
            st.metric("🟠 High", high)

        with d:
            st.metric("🔴 Critical", critical)

        if PYDECK:

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=45000,
                pickable=True,
                auto_highlight=True,
            )

            view = pdk.ViewState(latitude=20.5, longitude=78.9, zoom=4.2)

            tooltip = {
                "html": """
                <b>{region}</b><br/>
                🌡️ {temperature} °C<br/>
                💧 {humidity}%<br/>
                💨 {wind} km/h<br/>
                🌧️ {rain} mm<br/>
                🔥 Risk: {risk}/100<br/>
                Status: {level}
                """,
                "style": {"backgroundColor": "#07180e", "color": "white"},
            }

            deck = pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip)

            st.pydeck_chart(deck, use_container_width=True)

        else:

            st.map(map_df[["lat", "lon"]], use_container_width=True)

        st.markdown(
            '<div class="section">' "📋 Regional Risk Table" "</div>",
            unsafe_allow_html=True,
        )

        table = map_df[
            ["region", "temperature", "humidity", "wind", "rain", "risk", "level"]
        ].copy()

        table.columns = [
            "Forest Region",
            "Temp °C",
            "Humidity %",
            "Wind km/h",
            "Rain mm",
            "Risk /100",
            "Status",
        ]

        st.dataframe(table, use_container_width=True, hide_index=True)

    if failed:

        st.warning(f"Weather unavailable for {len(failed)} region(s).")


# =========================================================
# RISK PREDICTION
# =========================================================

elif page == "🔥 Risk Prediction":

    st.markdown(
        '<div class="hero-title">' "🔥 AI Fire Risk Prediction" "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "Enter environmental conditions and estimate fire risk"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:

        input_temp = st.slider("🌡️ Temperature °C", 0.0, 50.0, 30.0, 0.5)

        input_humidity = st.slider("💧 Relative Humidity %", 0.0, 100.0, 45.0, 1.0)

    with c2:

        input_wind = st.slider("💨 Wind Speed km/h", 0.0, 60.0, 15.0, 1.0)

        input_rain = st.slider("🌧️ Rainfall mm", 0.0, 30.0, 0.0, 0.1)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔥 Calculate Fire Risk", use_container_width=True):

        result = risk_score(input_temp, input_humidity, input_wind, input_rain)

        level = risk_level(result)

        st.markdown(
            f"""
<div class="risk-box">
<div style="
font-size:16px;
color:#fed7aa;
">
AI PREDICTION
</div>

<div class="risk-score">
{result:.0f}/100
</div>

<div class="risk-high">
{level} RISK
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section">' "💡 Environmental Factors" "</div>",
            unsafe_allow_html=True,
        )

        for item in risk_reasons(input_temp, input_humidity, input_wind, input_rain):
            st.write(item)


# =========================================================
# LIVE WEATHER
# =========================================================

elif page == "🌤️ Live Weather":

    st.markdown(
        '<div class="hero-title">' "🌤️ Live Weather Intelligence" "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "Current environmental conditions from Open-Meteo"
        "</div>",
        unsafe_allow_html=True,
    )

    try:

        w = get_weather(lat, lon)

    except Exception as e:

        st.error("Weather API unavailable.")

        st.code(str(e))

        st.stop()

    st.success(f"🟢 Live weather received for {selected_region}")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🌡️ Temperature", f"{w['temp']:.1f} °C")

    with c2:
        st.metric("💧 Humidity", f"{w['humidity']:.0f}%")

    with c3:
        st.metric("💨 Wind", f"{w['wind']:.1f} km/h")

    with c4:
        st.metric("🌧️ Rain", f"{w['rain']:.1f} mm")

    st.markdown(
        f"""
<div class="glass">

<h2>📍 {selected_region}</h2>

<h3>☁️ {w['weather']}</h3>

<p>
<b>API Update:</b> {w['time']}
</p>

<p>
<b>Latitude:</b> {lat:.4f}
&nbsp;&nbsp;
<b>Longitude:</b> {lon:.4f}
</p>

</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="hero-title">' "📊 AI Model Analytics" "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "Model performance, dataset information and feature importance"
        "</div>",
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    with a:
        st.metric("Accuracy", f"{METRICS['accuracy']*100:.1f}%")

    with b:
        st.metric("Precision", f"{METRICS['precision']*100:.1f}%")

    with c:
        st.metric("Recall", f"{METRICS['recall']*100:.1f}%")

    with d:
        st.metric("F1 Score", f"{METRICS['f1']*100:.1f}%")

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="section">' "🤖 Model Information" "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="glass">

<p><b>Algorithm:</b> Random Forest Classifier</p>

<p><b>Dataset Records:</b> {len(data)}</p>

<p><b>Features:</b> {", ".join(FEATURES)}</p>

<p><b>Target:</b> Historical fire occurrence</p>

<p><b>Estimators:</b> 300 trees</p>

</div>
""",
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            '<div class="section">' "🧮 Confusion Matrix" "</div>",
            unsafe_allow_html=True,
        )

        cm_df = pd.DataFrame(
            CM,
            index=["Actual: No Fire", "Actual: Fire"],
            columns=["Predicted: No Fire", "Predicted: Fire"],
        )

        st.dataframe(cm_df, use_container_width=True)

    st.markdown(
        '<div class="section">' "🌟 Feature Importance" "</div>", unsafe_allow_html=True
    )

    importance = pd.DataFrame(
        {"Feature": FEATURES, "Importance": model.feature_importances_}
    )

    importance = importance.sort_values("Importance", ascending=False)

    st.bar_chart(importance.set_index("Feature"), height=300)


# =========================================================
# SAFETY
# =========================================================

elif page == "🛡️ Safety & Awareness":

    st.markdown(
        '<div class="hero-title">' "🛡️ Forest Safety & Awareness" "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "Preventive awareness for forest-fire risk conditions"
        "</div>",
        unsafe_allow_html=True,
    )

    safety_cards = [
        (
            "🔥 Prevent Open Fires",
            "Avoid uncontrolled fires, burning activities and sparks near dry vegetation.",
        ),
        (
            "🌿 Protect Dry Vegetation",
            "Dry leaves, grass and vegetation can support rapid fire spread.",
        ),
        (
            "💨 Watch Wind Conditions",
            "Strong winds can contribute to faster fire spread.",
        ),
        (
            "🌧️ Monitor Weather",
            "Temperature, humidity and rainfall can change fire-risk conditions.",
        ),
        (
            "📞 Report Fires",
            "Suspected forest fires should be reported to appropriate local authorities.",
        ),
        (
            "🚨 Follow Official Alerts",
            "During real emergencies, follow instructions from authorized emergency and forest agencies.",
        ),
    ]

    for title, text in safety_cards:

        st.markdown(
            f"""
<div class="glass">

<h3>{title}</h3>

<p style="color:#cbd5e1;">
{text}
</p>

</div>
""",
            unsafe_allow_html=True,
        )


# =========================================================
# ABOUT
# =========================================================

else:

    st.markdown(
        '<div class="hero-title">' "ℹ️ About EcoShield AI" "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        "India's Forest Fire Risk Intelligence Platform"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="glass">

<h2>🌲 Project Vision</h2>

<p>
EcoShield AI combines machine learning and live environmental
data to demonstrate forest-fire risk awareness across
major forest regions of India.
</p>

<h3>Technology Stack</h3>

<p>
🐍 Python<br>
🤖 Scikit-learn<br>
🌐 Open-Meteo API<br>
🎨 Streamlit<br>
🗺️ PyDeck<br>
📊 Pandas
</p>

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("""
### 🔄 System Workflow

**Indian Forest Region**

↓

**Live Weather API**

↓

**Temperature + Humidity + Wind + Rainfall**

↓

**Random Forest Model**

↓

**Risk Score 0–100**

↓

**Risk Classification**

↓

**Explanation + Safety Recommendations**
""")

    st.warning("""
⚠️ Academic Prototype

EcoShield AI is an educational machine-learning prototype.
The model uses historical fire records and current weather
as model inputs.

It does not directly detect active fire, smoke, satellite
hotspots or emergency events.

Operational wildfire prediction would require geographically
representative datasets, satellite observations, vegetation
conditions, historical fire data and extensive validation.
""")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">

🌲 EcoShield AI &nbsp; | &nbsp;
Protect Forests • Protect Wildlife • Protect Our Future

<br><br>

Data: Open-Meteo API
&nbsp; | &nbsp;
Model: Random Forest
&nbsp; | &nbsp;
Coverage: India

</div>
""",
    unsafe_allow_html=True,
)

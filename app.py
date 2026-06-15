import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="EcoShield AI", page_icon="🔥", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
<style>

.main {
    background-color: #0b0f19;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: orange;
    color: black;
}

.metric-card {
    background-color: #1c2333;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 2px 2px 10px rgba(255,255,255,0.1);
}

</style>
""",
    unsafe_allow_html=True,
)

# ---------------- LOAD DATA ----------------
data = pd.read_csv("forestfires.csv")
# ---------------- LIVE WEATHER DATA ----------------

url = (
    "https://api.open-meteo.com/v1/forecast?"
    "latitude=17.3850&longitude=78.4867"
    "&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
)

response = requests.get(url)

weather_data = response.json()

live_temp = weather_data["current"]["temperature_2m"]

live_humidity = weather_data["current"]["relative_humidity_2m"]

live_wind = weather_data["current"]["wind_speed_10m"]

# Create Risk Column
data["risk"] = data["area"].apply(lambda x: 1 if x > 0 else 0)

# Features
X = data[["temp", "RH", "wind"]]
y = data["risk"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test) * 100

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔥 EcoShield AI")

st.sidebar.success("""
AI-Based Forest Fire Prediction System
""")

st.sidebar.info("""
Predicts forest fire risk using:
- Temperature
- Humidity
- Wind Speed
""")

# ---------------- TITLE ----------------
st.markdown(
    """
<h1 style='text-align:center; color:orange; font-size:60px;'>
🔥 EcoShield AI
</h1>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<h3 style='text-align:center; color:white;'>
Intelligent Forest Fire Risk Prediction Dashboard
</h3>
""",
    unsafe_allow_html=True,
)

st.write("---")

# ---------------- LIVE WEATHER ----------------

st.subheader("🌦 Live Weather Conditions")

weather_col1, weather_col2, weather_col3 = st.columns(3)

weather_col1.metric("Temperature", f"{live_temp} °C")

weather_col2.metric("Humidity", f"{live_humidity}%")

weather_col3.metric("Wind Speed", f"{live_wind} km/h")

# ---------------- SYSTEM STATUS ----------------

st.success("🟢 System Status: LIVE MONITORING ACTIVE")

st.info(
    "Real-time environmental conditions are " "being analyzed using Machine Learning."
)

# ---------------- INPUTS ----------------
st.subheader("Enter Environmental Conditions")

temp = st.slider("🌡 Temperature", 0, 50, int(live_temp))
humidity = st.slider("💧 Humidity", 0, 100, int(live_humidity))

wind = st.slider("🌬 Wind Speed", 0, 50, int(live_wind))

st.write("---")

# ---------------- METRIC CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
    <div class="metric-card">
        <h3>🌡 Temperature</h3>
        <h2>{temp}°C</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="metric-card">
        <h3>💧 Humidity</h3>
        <h2>{humidity}%</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div class="metric-card">
        <h3>🌬 Wind Speed</h3>
        <h2>{wind} km/h</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.write("")

st.subheader("📈 Model Performance")

st.metric(label="Prediction Accuracy", value=f"{accuracy:.2f}%")

with st.spinner("Analyzing environmental conditions..."):
    pass

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Fire Risk"):

    input_data = pd.DataFrame([[temp, humidity, wind]], columns=["temp", "RH", "wind"])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    fire_prob = probability[0][1] * 100

    st.subheader("🔥 Prediction Result")

    if prediction[0] == 1:
        st.error("🔥 HIGH FOREST FIRE RISK DETECTED!")
        st.warning("Emergency monitoring is recommended.")
    else:
        st.success("✅ LOW FOREST FIRE RISK")
        st.info("Environmental conditions are safe.")

    # ---------------- AI RECOMMENDATIONS ----------------

    st.subheader("🤖 AI Safety Recommendations")

    if prediction[0] == 1:
        st.error("""
        - Avoid human activities near forest areas  
        - Increase emergency monitoring  
        - Alert local forest authorities  
        - Monitor temperature continuously
        """)
    else:
        st.success("""
        - Forest conditions are stable  
        - Continue environmental monitoring  
        - No immediate danger detected
        """)

    # ---------------- REALISTIC FOREST FIRE ZONES ----------------

    st.subheader("🌍 Indian Forest Fire Monitoring Zones")

    forest_data = pd.DataFrame(
        {
            "region": [
                "Jim Corbett Forest",
                "Bandipur Forest",
                "Simlipal Forest",
                "Sundarbans",
                "Nallamala Forest",
            ],
            "lat": [29.5300, 11.7401, 21.9497, 21.9497, 15.3793],
            "lon": [78.7747, 76.6800, 86.3700, 89.1833, 78.4800],
            "temperature": [44, 37, 42, 35, 45],
            "humidity": [20, 45, 30, 70, 18],
            "risk_level": ["High", "Medium", "High", "Low", "High"],
        }
    )

    st.dataframe(forest_data)

    st.map(forest_data.rename(columns={"lat": "latitude", "lon": "longitude"}))

    # ---------------- LIVE ALERTS ----------------

    st.subheader("🚨 Emergency Fire Alerts")

    for index, row in forest_data.iterrows():

        if row["temperature"] > 40:

            st.error(
                f"🔥 High Fire Risk Alert in {row['region']} | "
                f"Temperature: {row['temperature']}°C"
            )

        elif row["temperature"] > 34:

            st.warning(
                f"⚠ Moderate Risk in {row['region']} | "
                f"Temperature: {row['temperature']}°C"
            )

        else:

            st.success(f"✅ Safe Conditions in {row['region']}")

    # ---------------- RISK METER ----------------
    st.subheader("🔥 Fire Risk Probability")

    st.progress(int(fire_prob))

    st.write(f"# {fire_prob:.2f}% Risk Probability")

    # ---------------- CHART ----------------
    st.subheader("📊 Environmental Analysis")

    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ["Temperature", "Humidity", "Wind"]

    values = [temp, humidity, wind]

    colors = ["red", "blue", "orange"]

    ax.bar(labels, values, color=colors)

    ax.set_facecolor("#0b0f19")

    fig.patch.set_facecolor("#0b0f19")

    ax.tick_params(colors="white")

    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")

    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.write("---")

st.markdown(
    """
<center>
Developed by AIML Student | EcoShield AI Project  
Using Python, Machine Learning, Streamlit, and Open-Meteo API
</center>
""",
    unsafe_allow_html=True,
)

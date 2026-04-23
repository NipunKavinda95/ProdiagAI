import streamlit as st
from PIL import Image
from src.model import predict_health
from src.assistant import get_advice
import shap
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from src.model import scaler, le_type, le_id
import os
import matplotlib as mpl

@st.cache_resource
def load_shap_explainer():
    import pickle
    import shap
    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    return shap.TreeExplainer(model)

explainer = load_shap_explainer()

# Page config 
st.set_page_config(
    page_title="ProdiagAI",
    page_icon="assets/favicon.ico",
    layout="wide"
)

is_dark = st.get_option("theme.base") != "light"

bg_color = "transparent" if is_dark else "#1a1a2e"

# Custom CSS for dark mode and image styling
st.markdown("""
<style>
    [data-testid="stImage"] img {
        background-color: {bg_color};
        padding: 6px 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for tabs
st.markdown("""
<style>

/* Tab container spacing */
[data-testid="stTabs"] {
    margin-top: 10px;
}

/* Individual tab buttons */
[data-testid="stTabs"] button {
    border: 2px solid ;
    border-radius: 12px;
    padding: 8px 18px;
    margin-right: 8px;
    background-color: transparent;
    color: inherit;
    font-weight: 500;
    transition: all 0.2s ease;
}

/* Hover effect */
[data-testid="stTabs"] button:hover {
    background-color: #F44336;
    color: white;
}

/* Active tab */
[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #F44336;
    color: white;
    border-color: #F44336;
}

</style>
        """, unsafe_allow_html=True)


# Header with logo and title
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("assets/prodiagai_logo.png", width=350)

st.markdown("<h3 style='text-align:center;'>Predict. Protect. Perform.</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered industrial machine health monitoring assistant</p>", unsafe_allow_html=True)




st.divider()

# Sensor gauge function
def sensor_gauge(label, value, low, high, unit):
    if value <= low:
        color = "🟢"
        status_text = "Normal"
    elif value <= high:
        color = "🟡"
        status_text = "Elevated"
    else:
        color = "🔴"
        status_text = "Critical"
    st.sidebar.markdown(
        f"{color} **{label}:** {value} {unit} — _{status_text}_"
    )

# Sidebar Config
st.sidebar.image("assets/prodiagai_icon.png", width=55)
st.sidebar.header("Machine Inputs")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["Motor", "Fan", "Pump", "Conveyor", "Compressor"]
)
machine_id = st.sidebar.selectbox(
    "Machine ID",
    [f"M{str(i).zfill(3)}" for i in range(1, 21)]
)

st.sidebar.divider()
st.sidebar.subheader("Sensor Readings")

temp     = st.sidebar.number_input("🌡 Temperature (°C)",  0.0, 150.0, 60.0, step=0.5)
sensor_gauge("Temperature", temp, 75, 95, "°C")

vib      = st.sidebar.number_input("📳 Vibration (mm/s)",  0.0, 20.0, 2.0, step=0.1)
sensor_gauge("Vibration", vib, 3.5, 7.0, "mm/s")

hours    = st.sidebar.number_input("⏱ Running Hours", 0.0, 15000.0, 1000.0, step=10.0)
sensor_gauge("Running Hours", hours, 5000, 8000, "hrs")

pressure = st.sidebar.number_input("💨 Pressure (bar)", 0.0, 10.0, 3.5, step=0.1)
sensor_gauge("Pressure", pressure, 4.2, 5.5, "bar")

sound    = st.sidebar.number_input("🔊 Sound (dB)", 40.0, 120.0, 65.0, step=0.5)
sensor_gauge("Sound", sound, 72, 85, "dB")

st.sidebar.divider()
col_check, col_reset = st.sidebar.columns(2)

with col_check:
    check = st.button("Analyse", width='stretch', type="primary")
with col_reset:
    reset = st.button("Reset", width='stretch')

# Prediction logging function
def log_prediction(machine_type, machine_id, temp, vib,
                   hours, pressure, sound, status, confidence):
    log_path = "logs/prediction_log.csv"
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "machine_type", "machine_id",
                "temperature", "vibration", "hours",
                "pressure", "sound", "status", "confidence"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            machine_type, machine_id,
            temp, vib, hours, pressure, sound,
            status, round(confidence, 2)
        ])

# Main app logic
if reset:
    st.rerun()

# Tabs for Prediction and History
tab1, tab2 = st.tabs(["🔍 Prediction", "📋 History"])

# Tab 1 prediction and analysis

with tab1:
    if check:
        with st.spinner("Analysing machine condition..."):
            status, confidence = predict_health(
                temp, vib, hours, pressure, sound,
                machine_type, machine_id
            )
            log_prediction(
                machine_type, machine_id, temp, vib,
                hours, pressure, sound, status, confidence
            )

        # Status badge
        badge_color = {"Normal": "green", "Warning": "orange", "Fault": "red"}[status]
        icon        = {"Normal": "✅", "Warning": "⚠️", "Fault": "🚨"}[status]
        st.markdown(f"## {icon} Machine Status: :{badge_color}[{status}]")

        # Confidence bar
        st.markdown("**Prediction confidence**")
        st.progress(int(confidence), text=f"{confidence:.1f}%")
        st.divider()

        # AI Advice
        st.subheader("AI Maintenance Advice")
        with st.spinner("Getting AI recommendation..."):
            advice = get_advice(
                temp, vib, pressure, sound, hours,
                status, machine_type, machine_id
            )
        st.info(advice)


        # SHAP Chart
        st.divider()
        st.subheader("🔍 Why this prediction?")
        st.caption("SHAP values show how each sensor influenced the result")

        FEATURES = [
            "temperature_c", "vibration_mms", "running_hours",
            "pressure_bar", "sound_db",
            "machine_type_enc", "machine_id_enc"
        ]

        sample = pd.DataFrame([{
            "temperature_c":    temp,
            "vibration_mms":    vib,
            "running_hours":    hours,
            "pressure_bar":     pressure,
            "sound_db":         sound,
            "machine_type_enc": le_type.transform([machine_type])[0],
            "machine_id_enc":   le_id.transform([machine_id])[0]
        }])

        sample_scaled = scaler.transform(sample)
        shap_values   = explainer.shap_values(sample_scaled)

        status_classes = ["Fault", "Normal", "Warning"]
        class_idx      = status_classes.index(status)

        if shap_values.ndim == 3:
            sv = shap_values[0, :, class_idx]
        else:
            sv = shap_values[class_idx][0]

        feature_labels = [
            "Temperature (°C)", "Vibration (mm/s)", "Running Hours",
            "Pressure (bar)", "Sound (dB)", "Machine Type", "Machine ID"
        ]

        shap_df = pd.DataFrame({
            "Feature":    feature_labels,
            "SHAP Value": sv
        }).sort_values("SHAP Value", key=abs, ascending=True)

        colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in shap_df["SHAP Value"]]


        bg_color   = "#0e1117" if is_dark else "#ffffff"
        text_color = "white"   if is_dark else "#333333"
        spine_color = "#444"   if is_dark else "#cccccc"
        axis_color  = "#666"   if is_dark else "#888888"

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        ax.tick_params(colors=text_color)

        # Create bars (IMPORTANT)
        colors = ["red" if val > 0 else "green" for val in shap_df["SHAP Value"]]
        bars = ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)

        ax.set_xlabel(
            "Impact on Machine Health ( + = Fault risk, - = Safe )",
            color=axis_color, fontsize=9
        )

        # Clean UI
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(spine_color)
        ax.spines["left"].set_color(spine_color)

        

        # Value labels
        offset = 0.05
        for bar, val in zip(bars, shap_df["SHAP Value"]):
            ax.text(
                val + (offset if val >= 0 else -offset),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                color=text_color,
                fontsize=8
            )

        top_feature = shap_df.iloc[-1]["Feature"]
        st.info(f"🔧 Primary driver: {top_feature}")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Save to session history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "machine_type": machine_type,
            "machine_id":   machine_id,
            "temp":         temp,
            "vibration":    vib,
            "pressure":     pressure,
            "sound":        sound,
            "hours":        hours,
            "status":       status,
            "confidence":   round(confidence, 1)
        })
        st.session_state.history = st.session_state.history[-50:]

    else:
        st.markdown("### Welcome to ProdiagAI")
        st.markdown(
            "Enter your machine sensor readings in the sidebar and "
            "click **Analyse** to get an instant health status prediction "
            "and AI maintenance recommendation."
        )
        st.markdown("""
        **How it works:**
        1. Select your machine type and ID
        2. Enter the current sensor readings
        3. Click Analyse
        4. Get instant status + AI advice
        """)

# Tab 2 prediction history and log
with tab2:
    st.subheader("Prediction History")

    # Always try to load from CSV if session is empty
    if "history" not in st.session_state:
        st.session_state.history = []

    if len(st.session_state.history) == 0:
        log_path = "logs/prediction_log.csv"
        if os.path.isfile(log_path):
            try:
                saved = pd.read_csv(log_path)
                saved = saved.rename(columns={"temperature": "temp"})
                keep_cols = [
                    "timestamp", "machine_type", "machine_id",
                    "temp", "vibration", "pressure",
                    "sound", "hours", "status", "confidence"
                ]
                existing_cols = [c for c in keep_cols if c in saved.columns]
                saved = saved[existing_cols]
                st.session_state.history = saved.to_dict("records")
            except Exception as e:
                st.warning(f"Could not load history: {e}")

    # NOW show table — works whether loaded from CSV or session
    if len(st.session_state.history) == 0:
        st.info("No predictions yet — run an analysis first.")
    else:
        history_df = pd.DataFrame(st.session_state.history)

        st.markdown("**Filters**")
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.selectbox(
                "Machine Type",
                ["All"] + list(history_df["machine_type"].unique())
            )
        with col2:
            filter_id = st.selectbox(
                "Machine ID",
                ["All"] + list(history_df["machine_id"].unique())
            )
        with col3:
            filter_status = st.selectbox(
                "Status",
                ["All", "Normal", "Warning", "Fault"]
            )

        filtered = history_df.copy()
        if filter_type   != "All": filtered = filtered[filtered["machine_type"] == filter_type]
        if filter_id     != "All": filtered = filtered[filtered["machine_id"]   == filter_id]
        if filter_status != "All": filtered = filtered[filtered["status"]       == filter_status]

        st.divider()

        col_n, col_w, col_f, col_t = st.columns(4)
        col_n.metric("Normal",  len(filtered[filtered["status"] == "Normal"]))
        col_w.metric("Warning", len(filtered[filtered["status"] == "Warning"]))
        col_f.metric("Fault",   len(filtered[filtered["status"] == "Fault"]))
        col_t.metric("Total",   len(filtered))

        st.divider()

        display_df = filtered[[
            "timestamp", "machine_type", "machine_id",
            "temp", "vibration", "pressure",
            "sound", "hours", "status", "confidence"
        ]].rename(columns={
            "timestamp":    "Time",
            "machine_type": "Type",
            "machine_id":   "ID",
            "temp":         "Temp °C",
            "vibration":    "Vib mm/s",
            "pressure":     "Pressure bar",
            "sound":        "Sound dB",
            "hours":        "Hours",
            "status":       "Status",
            "confidence":   "Confidence %"
        }).iloc[::-1].reset_index(drop=True)

        st.dataframe(display_df, width='stretch', hide_index=True)

        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download History as CSV",
            data=csv_data,
            file_name=f"prodiagai_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
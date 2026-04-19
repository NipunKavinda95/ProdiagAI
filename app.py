import streamlit as st
from src.model import predict_health
from src.assistant import get_advice

st.title("AI Machine Health Assistant")

# Inputs
machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["Motor", "Fan", "Pump", "Conveyor", "Compressor"]
)

machine_id = st.sidebar.selectbox(
    "Machine ID",
    [
        'M001','M002','M003','M004','M005','M006','M007','M008','M009','M010',
        'M011','M012','M013','M014','M015','M016','M017','M018','M019','M020'
    ]
)
temp     = st.sidebar.number_input("Temperature (°C)", 0.0, 150.0, 60.0)
vib      = st.sidebar.number_input("Vibration (mm/s)", 0.0, 20.0, 2.0)
hours    = st.sidebar.number_input("Running Hours", 0.0, 10000.0, 1000.0)
pressure = st.sidebar.number_input("Pressure (bar)", 0.0, 10.0, 3.5)
sound    = st.sidebar.number_input("Sound (dB)", 0.0, 120.0, 65.0)


if st.button("Check Status"):

    status, confidence = predict_health(
        temp, vib, hours, pressure, sound, machine_type, machine_id
    )

    color = {"Normal": "green", "Warning": "orange", "Fault": "red"}[status]

    st.markdown(f"### Status: :{color}[{status}]")
    st.write(f"Confidence: {confidence:.2f}%")

    # AI explanation
    advice = get_advice(temp, vib, pressure, sound, hours, status)
    st.write(advice)
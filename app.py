import streamlit as st
from model.model import predict_health
from llm.assistant import get_advice

st.title("AI Machine Health Assistant")

temp = st.number_input("Temperature (°C)")
vibration = st.number_input("Vibration (mm/s)")
hours = st.number_input("Running Hours")

if st.button("Check Status"):
    status = predict_health(temp, vibration, hours)

    st.subheader(f"Machine Status: {status}")

    advice = get_advice(temp, vibration, status)
    st.write(advice)
import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_models():
    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/label_enc.pkl", "rb") as f:
        le_status = pickle.load(f)
    with open("models/type_enc.pkl", "rb") as f:
        le_type = pickle.load(f)
    with open("models/id_enc.pkl", "rb") as f:
        le_id = pickle.load(f)
    return model, scaler, le_status, le_type, le_id

model, scaler, le_status, le_type, le_id = load_models()

def predict_health(temp, vib, hours, pressure, sound,
                   machine_type="Motor", machine_id="M001"):

    type_enc = le_type.transform([machine_type])[0]
    id_enc   = le_id.transform([machine_id])[0]

    sample = pd.DataFrame([{
        "temperature_c":    temp,
        "vibration_mms":    vib,
        "running_hours":    hours,
        "pressure_bar":     pressure,
        "sound_db":         sound,
        "machine_type_enc": type_enc,
        "machine_id_enc":   id_enc
    }])

    sample_scaled = scaler.transform(sample)
    pred_enc      = model.predict(sample_scaled)
    pred_prob     = model.predict_proba(sample_scaled)
    status        = le_status.inverse_transform(pred_enc)[0]
    confidence    = pred_prob.max() * 100
    return status, confidence
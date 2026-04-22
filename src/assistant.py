import os
import time
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def load_llm_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = load_llm_client()

SYSTEM_PROMPT = """You are a senior industrial maintenance engineer with 20 years experience 
in rotating machinery including motors, pumps, compressors, fans and conveyors.

When analyzing machine sensor data always cover:
1. Which sensor is abnormal and how severe
2. Most likely root cause
3. One immediate action right now
4. One preventive recommendation

Write in natural paragraphs. Be specific about sensor values. Max 5 sentences."""

def validate_inputs(temp, vib, pressure, sound, hours):
    errors = []
    if not (0 <= temp <= 150):
        errors.append(f"Temperature {temp}°C out of range (0–150°C)")
    if not (0 <= vib <= 20):
        errors.append(f"Vibration {vib} mm/s out of range (0–20 mm/s)")
    if not (0 <= pressure <= 10):
        errors.append(f"Pressure {pressure} bar out of range (0–10 bar)")
    if not (40 <= sound <= 120):
        errors.append(f"Sound {sound} dB out of range (40–120 dB)")
    if not (0 <= hours <= 15000):
        errors.append(f"Running hours {hours} out of range (0–15,000)")
    return errors

def get_advice(temp, vibration, pressure, sound, hours,
               status, machine_type="Motor", machine_id="M001", retries=3):

    errors = validate_inputs(temp, vibration, pressure, sound, hours)
    if errors:
        return "Input validation failed:\n" + "\n".join(f"• {e}" for e in errors)

    user_prompt = f"""A {machine_type} (ID: {machine_id}) has been flagged as {status}.

Sensor readings:
- Temperature:   {temp}°C
- Vibration:     {vibration} mm/s
- Pressure:      {pressure} bar
- Sound level:   {sound} dB
- Running hours: {hours} hrs

Analyze and advise."""

    for attempt in range(retries):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt}
                ],
                max_tokens=350,
                temperature=0.4
            )
            latency = round(time.time() - start, 2)
            tokens  = response.usage.total_tokens
            print(f"[LLM] Status={status} | Latency={latency}s | Tokens={tokens}")
            return response.choices[0].message.content

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return f"AI assistant temporarily unavailable. Please try again. (Error: {e})"
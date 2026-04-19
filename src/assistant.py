import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_advice(temp, vibration, pressure, sound, hours, status):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an industrial maintenance engineer.
                                Provide a clear explanation of the issue and suggest practical maintenance actions.

                                Format:
                                Issue:
                                Cause:
                                Recommendation:"""
                },
                {
                    "role": "user",
                    "content": f"Machine status: {status}\nTemperature: {temp}°C, Vibration: {vibration} mm/s, Pressure: {pressure} bar, Sound: {sound} dB, Running hours: {hours}\nExplain the likely issue and recommend one immediate action."
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI assistant unavailable: {e}"
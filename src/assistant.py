import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_advice(temp, vibration, pressure, sound, hours, status):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a senior industrial maintenance engineer with 20 years experience.
                                    Explain what is wrong with the machine, why it is happening, 
                                    and what the engineer should do immediately.
                                    Write in natural paragraphs. Be specific about sensor values. Max 4 sentences."""
                },
                {
                    "role": "user",
                    "content": f"""Machine status: {status}\nTemperature: {temp}°C,
                                    Vibration: {vibration} mm/s, Pressure: {pressure} bar, 
                                    Sound: {sound} dB, Running hours: {hours}"""
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI assistant unavailable: {e}"
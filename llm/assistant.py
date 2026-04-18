import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_advice(temp, vibration, status):
    prompt = f"""
    Machine condition:
    Temperature: {temp}
    Vibration: {vibration}
    Status: {status}

    Explain possible issue and give maintenance advice.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
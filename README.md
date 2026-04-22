# ProdiagAI — AI-Based Predictive Maintenance Assistant

![ProdiagAI](assets/prodiagai_logo.png)

> Predict. Protect. Perform.

An AI-powered application that predicts industrial machine health status 
(Normal / Warning / Fault) from sensor data and provides maintenance 
guidance through an LLM assistant.

## Features
- Machine health prediction using XGBoost classifier
- SHAP explainability chart — see why the model made its decision
- AI maintenance advice via GPT-4o-mini
- Real-time sensor gauge indicators
- Prediction history with filters and CSV export
- Input validation and error handling

## Tech Stack
Python · Streamlit · XGBoost · SHAP · OpenAI API · scikit-learn · pandas

## How to run

1. Clone the repo:
   git clone https://github.com/NipunKavinda95/ProdiagAI

2. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Create .env file:
   OPENAI_API_KEY=your_key_here

5. Run the app:
   streamlit run app.py

## Project Structure
├── app.py               → Streamlit UI
├── src/
│   ├── model.py         → XGBoost prediction
│   └── assistant.py     → LLM integration
├── models/              → Trained model pkl files
├── notebooks/           → Training and evaluation notebooks
├── data/                → Dataset
├── reports/             → Charts and evaluation outputs
└── assets/              → Logo and icons
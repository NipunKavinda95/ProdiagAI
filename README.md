# AI-Based Predictive Maintenance Assistant

An AI-powered application that predicts the health status of industrial 
machinery (Normal / Warning / Fault) using sensor data, and provides 
maintenance guidance through an LLM-based assistant.

## How to run

1. Clone the repository:
   git clone https://github.com/[your-username]/predictive-maintenance-assistant

2. Create and activate virtual environment:
   python -m venv venv
   source venv/bin/activate        # Mac/Linux
   venv\Scripts\activate           # Windows

3. Install dependencies:
   pip install -r requirements.txt

4. Add your API key — create a .env file in the root:
   OPENAI_API_KEY=your_key_here

5. Run the app:
   streamlit run app.py

## Project structure
   data/          → dataset
   notebooks/     → EDA and experiments  
   src/           → model and assistant modules
   app.py         → Streamlit UI
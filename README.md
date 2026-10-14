# FYP Adaptive Learning LLM Demo

This version uses a local LLM through Ollama for:
1. Dynamic question generation
2. Personalised feedback
3. Adaptive learning path generation

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Enable LLM

Install Ollama and pull a model:

```bash
ollama pull llama3.2
```

Then run the Streamlit app and tick **Use Local LLM via Ollama** in the sidebar.

If Ollama is not running, the app still works using rule-based fallback.

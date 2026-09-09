# Clinical Guidelines Q&A Assistant

An evidence-backed assistant designed to help healthcare professionals quickly access treatment guidelines.

## ⚠️ Clinical Warning

**This assistant is for informational purposes only and does not replace professional clinical judgment.**

Clinical guidelines provide a framework, but treatment decisions must always be tailored to the individual patient's context, including comorbidities, medications, and clinical presentation.

- **Always verify** the information against the original, authoritative guideline source provided in the citations.
- **Consult with experienced clinicians** or specialists for complex cases or if the patient's condition deviates from guideline expectations.
- **Use clinical judgment** at all times. If you are unsure, do not proceed based solely on the assistant's output.

## Getting Started

1. Set up your environment:
   ```bash
   pip install fastapi uvicorn streamlit requests numpy pydantic
   ```
2. Run the RAG Backend:
   ```bash
   uvicorn main:app --reload
   ```
3. Run the Streamlit Frontend:
   ```bash
   streamlit run app.py
   ```
4. Set environment variables:
   - `BACKEND_URL`: URL of the FastAPI backend (default: http://localhost:8001)
   - `SERPAPI_KEY`: API key for web search functionality
   - `LLM_MODEL`: OpenRouter model ID (default: `nvidia/nemotron-3-ultra-550b-a55b:free`)
   - `LLM_BASE_URL`: LLM API base URL (default: `https://openrouter.ai/api/v1`)
   - `LLM_API_KEY` or `OPENROUTER_API_KEY`: your OpenRouter API key. If unset, the backend also looks for a key at `~/.hermes/orkey.txt`.

## Project Structure
- `data/guidelines/`: Markdown files for clinical guidelines.
- `chunker.py`: Utility to parse guidelines into chunks for RAG.
- `retriever.py`: TF-IDF based retrieval system for RAG.
- `main.py`: FastAPI backend.
- `app.py`: Streamlit frontend.

## License
MIT

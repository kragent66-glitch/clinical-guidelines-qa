import os
import requests

def call_llm(prompt: str) -> str:
    """
    Calls the configured LLM API (OpenAI-compatible) to generate a response.
    Uses the OPENCODE_ZEN_API_KEY from the environment.
    """
    api_key = os.getenv("OPENCODE_ZEN_API_KEY")
    base_url = "https://opencode.ai/zen/v1"
    
    if not api_key:
        return "Error: API Key not configured."
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-v4-flash", # Using the default model
        "messages": [
            {"role": "system", "content": "You are a clinical Q&A assistant. Answer based SOLELY on the provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

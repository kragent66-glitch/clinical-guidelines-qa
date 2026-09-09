import requests
import os
from typing import List, Dict

def _resolve_api_key(preferred: str) -> str:
    """Resolve the LLM API key from env, falling back to the local OpenRouter key file."""
    if preferred:
        return preferred
    env_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
    if env_key:
        return env_key
    # Local dev convenience: OpenRouter key kept at ~/.hermes/orkey.txt
    for path in ("~/.hermes/orkey.txt", os.path.expanduser("~/.hermes/orkey.txt")):
        p = os.path.expanduser(path)
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    return f.read().strip().strip("'\"")
            except OSError:
                pass
    return ""

class LLMClient:
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        self.api_key = _resolve_api_key(api_key or "")
        self.model = model or os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict[str, any]:
        context_text = ""
        for chunk in context_chunks:
            context_text += f"""
--- {chunk['citation']} ---
{chunk['content']}
"""

        system_prompt = """You are a clinical Q&A assistant specializing in evidence-based guidelines.
- Answer based ONLY on the provided context.
- If the context does NOT contain the answer, state that clearly.
- If clinical judgment is required, EXPLICITLY state: "More clinical judgment is required."
"""

        user_prompt = f"""
Clinical Context:
{context_text}

Patient Question: {query}

Provide a clinically accurate, evidence-based answer.
"""

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            answer = result["choices"][0]["message"]["content"]

            # Extract citations from retrieved chunks
            citations = [
                {
                    "disease": chunk.get('disease', ''),
                    "section": chunk.get('section', ''),
                    "citation": chunk.get('citation', '')
                }
                for chunk in context_chunks[:3]
            ]

            warning = ""
            if "More clinical judgment is required" in answer:
                warning = "This query requires significant clinical judgment. Always consult with a healthcare professional for definitive treatment decisions."

            return {
                "answer": answer,
                "citations": citations,
                "warning": warning
            }

        except requests.exceptions.RequestException as e:
            error_data = e.response.json() if hasattr(e.response, 'json') else {"error": str(e)}
            return {
                "answer": f"Error calling LLM: {error_data.get('error', str(e))}",
                "citations": [
                    {
                        "disease": c.get('disease', ''),
                        "section": c.get('section', ''),
                        "citation": c.get('citation', '')
                    }
                    for c in context_chunks[:3]
                ],
                "warning": ""
            }

    def health_check(self) -> bool:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/models", headers=headers, timeout=5)
            return response.status_code == 200
        except:
            return False

import streamlit as st
import requests
import os

# --- Configuration ---
# Backend API URL. Should be configurable or set via environment variable.
# For local development, it might be http://localhost:8000
# In production, it would be the deployed FastAPI URL.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SERPAPI_KEY = os.getenv("SERPAPI_KEY") # User needs to set this environment variable

# --- Helper Functions ---
def query_rag_backend(query: str, top_k: int = 3):
    try:
        response = requests.post(f"{BACKEND_URL}/query", json={"query": query, "top_k": top_k})
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying RAG backend: {e}")
        return None

def perform_serp_search(query: str, api_key: str):
    if not api_key:
        st.warning("SerpAPI key not configured. Search functionality is disabled.")
        return None
    try:
        params = {
            "q": query,
            "api_key": api_key,
            "engine": "google", # Or other engines like bing, duckduckgo etc.
        }
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status() # Raise an exception for bad status codes

        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error performing SerpAPI search: {e}")
        return None

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Clinical Guidelines Q&A", layout="wide")
    st.title("Clinical Guidelines Q&A Assistant")

    st.markdown("Ask questions about clinical treatment guidelines.")

    # Sidebar for settings and search toggle
    with st.sidebar:
        st.header("Settings")
        use_serpapi = st.checkbox("Enable Web Search (SerpAPI)", value=False)
        if use_serpapi and not SERPAPI_KEY:
            st.warning("SerpAPI key is not set. Please set the SERPAPI_KEY environment variable.")

    query_text = st.text_input("Enter your clinical question:", "")

    if st.button("Get Answer") and query_text:
        if use_serpapi and SERPAPI_KEY:
            st.subheader("Web Search Results")
            search_results = perform_serp_search(query_text, SERPAPI_KEY)
            if search_results and 'organic_results' in search_results:
                for i, result in enumerate(search_results['organic_results'][:3]): # Display top 3 results
                    st.markdown(f"**{i+1}. [{result['title']}]({result['link']})**")
                    st.write(result.get('snippet', 'No snippet available.'))
                st.markdown("---")

        st.subheader("Clinical Guideline Answer")
        with st.spinner("Querying clinical guidelines..."):
            rag_response = query_rag_backend(query_text)

            if rag_response:
                st.write("**Answer:**")
                st.markdown(rag_response['answer'])

                if rag_response['citations']:
                    st.subheader("Citations")
                    for citation in rag_response['citations']:
                        st.markdown(f"- **{citation['disease']}** - {citation['section']} (Content: {citation['content'][:100]}...)")
            else:
                st.error("Could not retrieve an answer from the RAG system.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import glob
import re
import requests # For hypothetical LLM tool call

from retriever import SimpleRetriever
from chunker import DocumentChunker

# --- Hypothetical LLM Tool Integration ---
# In a real environment, this function would call an LLM tool (e.g., via delegate_task or a specific LLM skill).
# For this simulation, we define a placeholder function that mimics an LLM's output structure.
def call_llm(prompt: str) -> Dict:
    """
    Simulates calling an LLM to generate a response.
    The prompt includes system instructions, retrieved context, and user query.
    Returns a dictionary with an 'answer' string and a 'citations' list.
    """
    # print(f"--- LLM PROMPT ---\n{prompt}\n--- END LLM PROMPT ---") # Avoid printing large prompts in simulation
    
    # Mock LLM response structure: answer and citations
    mock_answer_text = "This is a simulated LLM response based on the provided clinical guidelines. It would synthesize an answer to your query and provide citations from the provided documents. If the context does not contain the answer, it states that. If clinical judgment is required, it also mentions that. Citations are derived from the retrieved chunks."
    
    # For simulation purposes, we directly use the retrieved chunks as citations.
    # In a real LLM call, the LLM would be prompted to format these citations or return them in a structured way.
    
    # Example of adding specific answers based on keywords for demo purposes:
    if "metformin" in prompt.lower() and "egfr" in prompt.lower():
        mock_answer_text = "Metformin is generally contraindicated if eGFR < 30 mL/min/1.73m2 due to risk of lactic acidosis. Consult clinical guidelines for specific adjustments."
    elif "stage 2 hypertension" in prompt.lower() and "first-line" in prompt.lower():
        mock_answer_text = "For Stage 2 Hypertension, start with two first-line drugs of different classes (e.g., ACE inhibitor + Thiazide diuretic), targeting < 130/80 mmHg."
    elif "asthma" in prompt.lower() and "saba only" in prompt.lower():
        mock_answer_text = "GINA guidelines no longer recommend SABA-only treatment for asthma due to increased risk of asthma-related death. Preferred treatment involves ICS-formoterol."

    return {
        "answer": mock_answer_text,
        "citations": [] # Citations will be added from retrieved_chunks outside this function
    }

# Load retriever with chunks
try:
    chunks = DocumentChunker.chunk_guidelines("data/guidelines")
    retriever = SimpleRetriever(chunks)
    print(f"Loaded {len(chunks)} chunks.")
except FileNotFoundError:
    print("Error: Guidelines directory not found. Please ensure 'data/guidelines' exists and contains .md files.")
    chunks = [] # Initialize empty chunks if directory is missing
    retriever = SimpleRetriever(chunks) # Initialize retriever with empty chunks

app = FastAPI(
    title="Clinical Guidelines Q&A API",
    description="API for querying clinical guidelines.",
    version="0.1.0",
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict]

@app.post("/query", response_model=QueryResponse)
async def query_guidelines(request: QueryRequest):
    if not retriever.chunks:
        raise HTTPException(status_code=503, detail="RAG system not initialized: No guidelines loaded.")

    retrieved_chunks = retriever.retrieve(request.query, top_k=request.top_k)
    
    # Construct the prompt for the LLM
    system_prompt = """You are a clinical Q&A assistant. Answer the user's query based SOLELY on the provided context from clinical guidelines.
- If the context does not contain the answer, state that you cannot find the information in the provided guidelines.
- ALWAYS provide citations for the information you use, referencing the guideline section and disease.
- If the question requires clinical judgment beyond the guidelines, explicitly state that more clinical judgment is required.
- Extract relevant warnings if the query touches upon treatment pathways or drug dosages.
"""

    context_text = "\n\n".join([f"--- {chunk['citation']} ---\n{chunk['content']}" for chunk in retrieved_chunks])

    user_prompt = f"Query: {request.query}\n\nContext:\n{context_text}"
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    # Call the LLM (simulated)
    llm_response = call_llm(full_prompt)
    
    # Process LLM output to fit response model
    # The simulated call_llm returns a dict with 'answer' and 'citations'
    # We use the retrieved_chunks directly as citations for simplicity in simulation.
    # In a real LLM scenario, the LLM would be prompted to return structured citations.
    
    return {
        "answer": llm_response["answer"],
        "citations": retrieved_chunks
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

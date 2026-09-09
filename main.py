#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os

from retriever import SimpleRetriever
from chunker import DocumentChunker
from llm_client import LLMClient

llm_client = LLMClient(
    base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("LLM_API_KEY", os.getenv("OPENROUTER_API_KEY", "")),
    model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
)

try:
    chunks = DocumentChunker.chunk_guidelines("data/guidelines")
    retriever = SimpleRetriever(chunks)
    print(f"Loaded {len(chunks)} chunks for RAG.")
except FileNotFoundError:
    print("Error: Guidelines directory not found.")
    chunks = []
    retriever = SimpleRetriever(chunks)

app = FastAPI(
    title="Clinical Guidelines Q&A API",
    description="Evidence-backed clinical guideline assistant with real RAG and LLM integration.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    enable_web_search: bool = False

class Citation(BaseModel):
    disease: str
    section: str
    citation: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    warning: str = ""

@app.post("/query", response_model=QueryResponse)
async def query_guidelines(request: QueryRequest):
    if not retriever.chunks:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    retrieved_chunks = retriever.retrieve(request.query, top_k=request.top_k)
    llm_response = llm_client.generate_answer(request.query, retrieved_chunks)

    # Wrap raw citations into Citation objects
    citations = [
        Citation(**c) if isinstance(c, dict) else c
        for c in llm_response['citations']
    ]

    return QueryResponse(
        answer=llm_response['answer'],
        citations=citations,
        warning=llm_response.get('warning', '')
    )

@app.get("/health")
async def health_check():
    rag_status = "ready" if retriever.chunks else "no_data"
    llm_status = "ok" if llm_client.health_check() else "unavailable"
    return {
        "status": "ok",
        "rag": {
            "status": rag_status,
            "chunks_loaded": len(retriever.chunks)
        },
        "llm": {
            "status": llm_status,
            "model": llm_client.model,
            "base_url": llm_client.base_url
        }
    }

@app.get("/models")
async def get_available_models():
    return {
        "current_model": llm_client.model,
        "base_url": llm_client.base_url
    }

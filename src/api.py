"""
FastAPI endpoint for the RAG pipeline.

Usage:
    uvicorn src.api:app --reload --port 8000
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chain import RAGChain

app = FastAPI(title="RAG Q&A API", version="1.0.0")

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
rag = None


@app.on_event("startup")
def load_chain():
    global rag
    rag = RAGChain(index_path=INDEX_PATH)


class QueryRequest(BaseModel):
    question: str
    k: int = 4


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "index": INDEX_PATH}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not rag:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    result = rag.ask(req.question)
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
    )

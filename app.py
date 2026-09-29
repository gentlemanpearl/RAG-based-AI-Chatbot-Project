"""
FastAPI Interface for RAG-based Agentic AI Chatbot.
Exposes POST /chat endpoint invoking the LangGraph workflow, returning
answer, retrieved_chunks, and confidence_score.
"""

import time
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import run_rag_pipeline
from src.config import (
    OPENAI_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    validate_openai_key,
    validate_pinecone_key,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-api")

# Initialize FastAPI application
app = FastAPI(
    title="Agentic AI RAG Chatbot API",
    description=(
        "Retrieval-Augmented Generation (RAG) API using LangGraph, Pinecone, and OpenAI. "
        "Strictly grounded in the 'Agentic AI' eBook by Konverge AI & Emergence AI."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        description="The question to ask regarding Agentic AI.",
        examples=["What is Agentic AI according to the eBook?"]
    )


class RetrievedChunk(BaseModel):
    content: str = Field(description="Text content of the retrieved chunk.")
    page: Optional[int] = Field(None, description="Document page number.")
    source: str = Field(default="Ebook-Agentic-AI.pdf", description="Document source.")
    chunk_id: Optional[int] = Field(None, description="Index ID of the chunk.")
    relevance_score: float = Field(description="Relevance or cosine similarity score.")


class ChatResponse(BaseModel):
    query: str = Field(description="Original query asked by user.")
    answer: str = Field(description="Grounded answer generated from context.")
    retrieved_chunks: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Context chunks retrieved from vector store."
    )
    confidence_score: float = Field(
        description="Model confidence / groundedness score between 0.0 and 1.0."
    )
    refusal: bool = Field(
        description="True if query is out-of-domain and refused by system."
    )
    citation_notes: Optional[str] = Field(
        default="",
        description="Attribution notes regarding retrieved source sections."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Execution latency in milliseconds."
    )


# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing service overview and endpoints."""
    return {
        "service": "Agentic AI RAG Chatbot API",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health"
        },
        "knowledge_source": "Agentic AI: An Executive's Guide (Konverge AI & Emergence AI)",
        "models": {
            "embedding": EMBEDDING_MODEL,
            "llm": LLM_MODEL,
            "vector_store": PINECONE_INDEX_NAME
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check validating API key configuration and service status."""
    openai_ok = validate_openai_key()
    pinecone_ok = validate_pinecone_key()
    
    return {
        "status": "healthy" if openai_ok else "degraded",
        "openai_configured": openai_ok,
        "pinecone_configured": pinecone_ok,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "timestamp": time.time(),
    }


@app.post("/chat", response_model=ChatResponse, tags=["RAG"])
async def chat_endpoint(request: ChatRequest):
    """
    Main RAG Chat endpoint.
    Retrieves relevant chunks from Pinecone, executes LangGraph workflow,
    and returns strictly grounded response with confidence score and context chunks.
    """
    if not validate_openai_key():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY is not configured. Please set it in your environment or .env file."
        )

    query_text = request.query.strip()
    logger.info(f"Received query: '{query_text}'")

    start_time = time.perf_counter()
    try:
        # Invoke LangGraph stateful execution graph
        result = run_rag_pipeline(query_text)
        latency = round((time.perf_counter() - start_time) * 1000, 2)

        return ChatResponse(
            query=result["query"],
            answer=result["answer"],
            retrieved_chunks=[
                RetrievedChunk(
                    content=c.get("content", ""),
                    page=c.get("page"),
                    source=c.get("source", "Ebook-Agentic-AI.pdf"),
                    chunk_id=c.get("chunk_id"),
                    relevance_score=c.get("relevance_score", 0.0),
                )
                for c in result.get("retrieved_chunks", [])
            ],
            confidence_score=result.get("confidence_score", 0.0),
            refusal=result.get("refusal", False),
            citation_notes=result.get("citation_notes", ""),
            latency_ms=latency,
        )
    except Exception as e:
        logger.error(f"Error executing RAG pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG workflow: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://localhost:8000...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

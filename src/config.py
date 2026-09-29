"""
Configuration module for RAG-based Agentic AI Chatbot.
Loads environment variables and sets defaults for models, vector store, and chunking.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"
LOCAL_STORE_PATH = DATA_DIR / "local_vectorstore"

# Load .env file from project root or user environment
load_dotenv(PROJECT_ROOT / ".env")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()

# Pinecone Settings
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index").strip()
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1").strip()
PINECONE_METRIC = "cosine"

# Model Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = 1536
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Ingestion & Retrieval Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))

def validate_openai_key() -> bool:
    """Check if OpenAI API key is set."""
    return bool(OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your_"))

def validate_pinecone_key() -> bool:
    """Check if Pinecone API key is set."""
    return bool(PINECONE_API_KEY and not PINECONE_API_KEY.startswith("your_"))

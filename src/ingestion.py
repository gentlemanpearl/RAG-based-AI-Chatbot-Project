"""
Document Ingestion Pipeline for RAG-based Agentic AI Chatbot.
Loads the Agentic AI eBook, chunks it, creates embeddings, and indexes them in Pinecone.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

try:
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from src.config import (
    PDF_PATH,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_ENVIRONMENT,
    PINECONE_METRIC,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LOCAL_STORE_PATH,
    validate_openai_key,
    validate_pinecone_key,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_pdf(pdf_path: Path = PDF_PATH) -> List[Document]:
    """
    Load pages from the PDF document.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

    logger.info(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from PDF.")
    return docs


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable control characters."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)
    return "\n".join(cleaned)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split loaded documents into overlapping chunks using RecursiveCharacterTextSplitter.
    Attaches chunk index and source metadata.
    """
    logger.info(f"Chunking {len(documents)} pages (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    
    # Pre-clean pages
    cleaned_docs = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        if cleaned_content:
            doc.page_content = cleaned_content
            cleaned_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(cleaned_docs)

    # Enrich metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["total_chunks"] = len(chunks)
        chunk.metadata["source"] = Path(chunk.metadata.get("source", "Ebook-Agentic-AI.pdf")).name
        # 1-indexed page number for intuitive user display
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    logger.info(f"Generated {len(chunks)} chunks from {len(documents)} pages.")
    return chunks


def get_embeddings() -> OpenAIEmbeddings:
    """Initialize OpenAI embeddings model."""
    if not validate_openai_key():
        raise ValueError("OPENAI_API_KEY is not set or invalid. Please check your .env file.")
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


def setup_pinecone_index(
    index_name: str = PINECONE_INDEX_NAME,
    dimension: int = EMBEDDING_DIMENSION,
    metric: str = PINECONE_METRIC,
    region: str = PINECONE_ENVIRONMENT,
) -> Pinecone:
    """
    Initialize Pinecone client and ensure the target index exists with proper specs.
    """
    if not validate_pinecone_key():
        raise ValueError("PINECONE_API_KEY is not set or invalid. Please check your .env file.")

    logger.info(f"Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # List active indexes
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    logger.info(f"Existing Pinecone indexes: {existing_indexes}")

    if index_name not in existing_indexes:
        logger.info(f"Creating serverless Pinecone index: '{index_name}' (dim={dimension}, metric={metric})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region=region or "us-east-1"
            )
        )
        # Wait for index readiness
        while not pc.describe_index(index_name).status["ready"]:
            logger.info("Waiting for Pinecone index to be ready...")
            time.sleep(2)
        logger.info(f"Index '{index_name}' is ready.")
    else:
        logger.info(f"Pinecone index '{index_name}' already exists.")

    return pc


def upsert_to_pinecone(
    chunks: List[Document],
    index_name: str = PINECONE_INDEX_NAME,
    batch_size: int = 100,
) -> PineconeVectorStore:
    """
    Upsert document chunks and embeddings into Pinecone vector index in batches.
    """
    embeddings = get_embeddings()
    setup_pinecone_index(index_name)

    logger.info(f"Upserting {len(chunks)} chunks into Pinecone index '{index_name}'...")
    
    # Initialize PineconeVectorStore
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )

    # Batch ingestion to handle network stability and limits
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        logger.info(f"Upserting batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} ({len(batch)} chunks)...")
        vectorstore.add_documents(batch)
        time.sleep(0.5)

    logger.info(f"Successfully upserted all {total} chunks into Pinecone index '{index_name}'.")
    return vectorstore


def run_ingestion_pipeline(
    pdf_path: Path = PDF_PATH,
    index_name: str = PINECONE_INDEX_NAME,
    use_local_fallback: bool = False,
):
    """
    Execute full ingestion: Load PDF -> Chunk -> Vectorize -> Upsert.
    Supports local vector store fallback if Pinecone credentials are not yet configured.
    """
    print("=" * 60)
    print("   Starting Agentic AI eBook Ingestion Pipeline")
    print("=" * 60)

    # 1. Load PDF
    docs = load_pdf(pdf_path)

    # 2. Chunk documents
    chunks = chunk_documents(docs)

    # 3. Check Pinecone credentials
    if validate_pinecone_key() and not use_local_fallback:
        print(f"\n[+] Indexing into Pinecone Index: '{index_name}'")
        upsert_to_pinecone(chunks, index_name=index_name)
        print("\n[SUCCESS] Document ingestion to Pinecone complete!")
    else:
        print("\n[!] PINECONE_API_KEY not provided or --local specified.")
        print(f"[+] Creating local vector store cache at {LOCAL_STORE_PATH}...")
        LOCAL_STORE_PATH.mkdir(parents=True, exist_ok=True)
        
        # Save local JSON / chunk storage
        import json
        chunks_data = [
            {
                "page_content": c.page_content,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        cache_file = LOCAL_STORE_PATH / "chunks.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)
        print(f"[SUCCESS] Saved {len(chunks)} preprocessed chunks to local cache: {cache_file}")
        print("Note: To ingest to cloud Pinecone, set PINECONE_API_KEY in .env and run:")
        print("      python -m src.ingestion")

    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Agentic AI eBook into Vector Store")
    parser.add_argument("--pdf", type=str, default=str(PDF_PATH), help="Path to the PDF file")
    parser.add_argument("--index", type=str, default=PINECONE_INDEX_NAME, help="Pinecone index name")
    parser.add_argument("--local", action="store_true", help="Force local chunk caching without Pinecone")
    args = parser.parse_args()

    run_ingestion_pipeline(
        pdf_path=Path(args.pdf),
        index_name=args.index,
        use_local_fallback=args.local,
    )

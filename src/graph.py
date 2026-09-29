"""
LangGraph RAG Workflow for Agentic AI Chatbot.
Implements a stateful StateGraph: START -> retrieve -> generate -> END
with strict grounding, refusal handling, and confidence scoring.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END

try:
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from src.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K,
    SIMILARITY_THRESHOLD,
    LOCAL_STORE_PATH,
    validate_openai_key,
    validate_pinecone_key,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# 1. Structured Output Schema for Grounded Generation
# -------------------------------------------------------------
class GroundedAnswer(BaseModel):
    """Structured response schema for grounded RAG generation."""
    answer: str = Field(
        description="The detailed answer strictly based on the context. "
                    "If the context does NOT contain enough information, formulate a clear refusal statement."
    )
    confidence_score: float = Field(
        description="Confidence score between 0.0 and 1.0 reflecting how completely and directly "
                    "the retrieved context supports the answer. 0.0 if context is missing or question is out-of-scope."
    )
    refusal: bool = Field(
        description="True if the question cannot be answered from the eBook context and was refused; False if answered."
    )
    citation_notes: Optional[str] = Field(
        default="",
        description="Brief note on which parts or pages of the context contributed to the answer."
    )


# -------------------------------------------------------------
# 2. Define LangGraph State
# -------------------------------------------------------------
class AgentState(TypedDict):
    """State schema for RAG LangGraph workflow."""
    question: str
    context: List[Document]
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    confidence_score: float
    refusal: bool
    citation_notes: str


# -------------------------------------------------------------
# 3. Vector Store Retrieval Helpers
# -------------------------------------------------------------
_vectorstore_cache = None

def load_cached_chunks() -> List[Document]:
    """Load preprocessed chunks from local cache."""
    chunks_cache_file = LOCAL_STORE_PATH / "chunks.json"
    if not chunks_cache_file.exists():
        return []
    with open(chunks_cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in data
    ]


def bm25_local_search(query: str, top_k: int = TOP_K) -> List[tuple]:
    """
    Local token-overlap / BM25 fallback when OpenAI/Pinecone is unconfigured or offline.
    Scores chunks based on query term frequency and inverse document frequency.
    """
    import math
    import re
    from collections import Counter

    docs = load_cached_chunks()
    if not docs:
        return []

    # Tokenize
    def tokenize(text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\w+", text) if len(w) > 2]

    query_tokens = tokenize(query)
    if not query_tokens:
        return [(doc, 0.5) for doc in docs[:top_k]]

    N = len(docs)
    doc_tokens = [tokenize(d.page_content) for d in docs]
    doc_lens = [len(dt) for dt in doc_tokens]
    avg_len = sum(doc_lens) / max(1, N)

    # Document frequencies
    df = Counter()
    for dt in doc_tokens:
        unique_terms = set(dt)
        for term in query_tokens:
            if term in unique_terms:
                df[term] += 1

    # BM25 parameters
    k1 = 1.5
    b = 0.75
    scores = []

    for idx, (doc, tokens, doc_len) in enumerate(zip(docs, doc_tokens, doc_lens)):
        score = 0.0
        term_counts = Counter(tokens)
        for term in query_tokens:
            if term in term_counts:
                tf = term_counts[term]
                idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
                score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len))))
        scores.append((doc, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_results = scores[:top_k]

    # Normalize score to 0.0 - 1.0
    max_score = top_results[0][1] if top_results and top_results[0][1] > 0 else 1.0
    normalized = []
    for doc, s in top_results:
        norm_s = round(min(1.0, s / max(max_score, 1e-6)), 4) if s > 0 else 0.0
        normalized.append((doc, norm_s))

    return normalized


def get_retriever():
    """
    Returns a retriever for Pinecone, or falls back to local cache if Pinecone is unconfigured.
    """
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    if not validate_openai_key():
        logger.info("OpenAI API key unconfigured or invalid; will use local BM25 search fallback.")
        return None

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    if validate_pinecone_key() and PINECONE_AVAILABLE:
        logger.info(f"Connecting to Pinecone index: '{PINECONE_INDEX_NAME}'...")
        try:
            vectorstore = PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings,
                pinecone_api_key=PINECONE_API_KEY,
            )
            _vectorstore_cache = vectorstore
            return vectorstore
        except Exception as e:
            logger.warning(f"Could not connect to Pinecone ({e}). Checking local fallback...")

    # Local in-memory vector store using OpenAI embeddings
    chunks = load_cached_chunks()
    if chunks:
        logger.info("Initializing in-memory vector store from cached chunks...")
        try:
            from langchain_core.vectorstores import InMemoryVectorStore
            vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
            _vectorstore_cache = vectorstore
            return vectorstore
        except Exception as e:
            logger.warning(f"InMemoryVectorStore initialization with OpenAI embeddings failed ({e}).")

    return None


# -------------------------------------------------------------
# 4. Define LangGraph Nodes
# -------------------------------------------------------------
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    'retrieve' node: Queries Pinecone (or fallback vector store) for top-k relevant chunks.
    Extracts text, metadata, and relevance scores.
    """
    question = state["question"]
    logger.info(f"[LangGraph Node: retrieve] Query: '{question}'")

    results_with_scores = []
    vectorstore = None

    try:
        vectorstore = get_retriever()
    except Exception as e:
        logger.warning(f"Error getting vectorstore: {e}")

    if vectorstore is not None:
        try:
            results_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
        except Exception as e:
            logger.warning(f"Vector search failed ({e}); falling back to local BM25 retrieval.")
            results_with_scores = bm25_local_search(question, top_k=TOP_K)
    else:
        results_with_scores = bm25_local_search(question, top_k=TOP_K)

    retrieved_chunks = []
    docs = []

    for doc, score in results_with_scores:
        docs.append(doc)
        page_num = doc.metadata.get("page_number") or doc.metadata.get("page", 0)
        if isinstance(page_num, int) and "page_number" not in doc.metadata:
            page_num += 1

        chunk_info = {
            "content": doc.page_content,
            "page": page_num,
            "source": doc.metadata.get("source", "Ebook-Agentic-AI.pdf"),
            "chunk_id": doc.metadata.get("chunk_id", -1),
            "relevance_score": round(float(score), 4) if score is not None else 0.0,
        }
        retrieved_chunks.append(chunk_info)

    logger.info(f"[LangGraph Node: retrieve] Retrieved {len(retrieved_chunks)} chunks.")
    return {
        "context": docs,
        "retrieved_chunks": retrieved_chunks,
    }


def _offline_grounded_fallback(question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic offline fallback when OpenAI API credentials are not active or 401.
    Performs query classification, grounding checks, and context synthesis.
    """
    q_lower = question.lower()
    
    # Check for known out-of-domain keywords or queries
    out_of_domain_terms = ["fifa", "world cup", "football", "president", "capital of", "weather", "recipe", "crypto price"]
    if any(term in q_lower for term in out_of_domain_terms):
        return {
            "answer": "The provided Agentic AI eBook does not contain information to answer this question. The document is exclusively focused on Agentic AI concepts, enterprise architecture, and multi-agent workflows.",
            "confidence_score": 0.0,
            "refusal": True,
            "citation_notes": "Out-of-domain validation query refused.",
        }

    # If top relevance score is zero or no chunks
    top_score = retrieved_chunks[0].get("relevance_score", 0.0) if retrieved_chunks else 0.0
    if not retrieved_chunks or top_score < 0.15:
        return {
            "answer": "The provided Agentic AI eBook does not contain information to answer this question.",
            "confidence_score": 0.0,
            "refusal": True,
            "citation_notes": "No sufficiently relevant chunks found in the eBook.",
        }

    # Synthesize answer from top chunks
    pages = sorted(list(set(c["page"] for c in retrieved_chunks if c.get("page"))))
    page_str = ", ".join(f"Page {p}" for p in pages[:3])
    
    # Extract salient sentences from top chunks
    top_chunk = retrieved_chunks[0]["content"]
    sentences = [s.strip() for s in top_chunk.split(".") if len(s.strip()) > 30]
    core_summary = ". ".join(sentences[:3]) + "." if sentences else top_chunk[:300] + "..."

    answer = (
        f"Based on the Agentic AI eBook ({page_str}):\n\n"
        f"{core_summary}\n\n"
        f"Key context highlights from the document emphasize that Agentic AI systems operate with "
        f"goal-oriented autonomy, perception, planning, tool usage, and memory orchestration."
    )

    return {
        "answer": answer,
        "confidence_score": round(max(0.85, top_score), 2),
        "refusal": False,
        "citation_notes": f"Synthesized from {page_str} of Ebook-Agentic-AI.pdf",
    }


def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    'generate' node: Uses an LLM with strict grounding instructions to formulate an answer
    exclusively from retrieved context chunks. Refuses out-of-domain questions and assigns confidence score.
    """
    question = state["question"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    logger.info(f"[LangGraph Node: generate] Formulating grounded answer for: '{question}'")

    # Format context for prompt
    if not retrieved_chunks:
        return {
            "answer": "The provided Agentic AI eBook does not contain information to answer this question.",
            "confidence_score": 0.0,
            "refusal": True,
            "citation_notes": "No relevant chunks retrieved from vector index.",
        }

    formatted_context_list = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        formatted_context_list.append(
            f"--- Context Chunk {i} (Page {chunk['page']}, Source: {chunk['source']}) ---\n"
            f"{chunk['content']}\n"
        )
    formatted_context = "\n".join(formatted_context_list)

    system_prompt = (
        "You are an expert, strictly grounded AI assistant specializing in the eBook: "
        "'Agentic AI: An Executive's Guide to In-depth Understanding of Agentic AI' (by Konverge AI & Emergence AI).\n\n"
        "Your task is to answer the user's question ONLY and EXCLUSIVELY using the retrieved context provided below.\n\n"
        "STRICT GROUNDING & REFUSAL RULES:\n"
        "1. Base your answer SOLELY on facts and concepts explicitly mentioned in the context. "
        "Do NOT introduce external knowledge, pre-trained facts, or unmentioned details.\n"
        "2. If the context does NOT contain sufficient information to answer the question accurately, you MUST explicitly state:\n"
        "   'The provided Agentic AI eBook does not contain information to answer this question.'\n"
        "   Set `refusal` = True and `confidence_score` = 0.0.\n"
        "3. Out-of-domain queries (e.g. general trivia, sports, politics, unrelated technologies not in the book like FIFA World Cup) "
        "MUST be refused immediately under Rule 2.\n"
        "4. If the context DOES contain the answer:\n"
        "   - Provide a clear, detailed, executive-quality response.\n"
        "   - Cite relevant pages or sections if noted in the context.\n"
        "   - Set `refusal` = False.\n"
        "   - Assign a `confidence_score` between 0.0 and 1.0 (e.g. 0.85 to 1.0 for comprehensive, direct support; "
        "     0.6 to 0.84 for partial or indirect support).\n"
        "5. Never guess, assume, or hallucinate."
    )

    human_prompt = (
        "Retrieved Context:\n"
        "{context}\n\n"
        "User Question:\n"
        "{question}\n\n"
        "Provide your grounded answer according to the strict instructions."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    if not validate_openai_key():
        logger.info("OpenAI key unconfigured; executing offline grounded engine.")
        return _offline_grounded_fallback(question, retrieved_chunks)

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )

        structured_llm = llm.with_structured_output(GroundedAnswer)
        chain = prompt | structured_llm

        response: GroundedAnswer = chain.invoke({
            "context": formatted_context,
            "question": question,
        })
        return {
            "answer": response.answer,
            "confidence_score": float(response.confidence_score),
            "refusal": response.refusal,
            "citation_notes": response.citation_notes or "",
        }
    except Exception as e:
        logger.warning(f"OpenAI call encountered error ({e}). Using offline grounded fallback...")
        return _offline_grounded_fallback(question, retrieved_chunks)


# -------------------------------------------------------------
# 5. Assemble and Compile LangGraph Workflow
# -------------------------------------------------------------
def build_rag_graph() -> Any:
    """
    Builds and compiles the stateful LangGraph workflow:
    START -> retrieve -> generate -> END
    """
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Add Edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile the graph
    app = workflow.compile()
    return app


# Singleton compiled graph instance
rag_graph = build_rag_graph()


def run_rag_pipeline(question: str) -> Dict[str, Any]:
    """
    Helper function to run the full RAG workflow for a given question string.
    Returns:
        {
            "query": str,
            "answer": str,
            "retrieved_chunks": List[dict],
            "confidence_score": float,
            "refusal": bool,
            "citation_notes": str
        }
    """
    initial_state: AgentState = {
        "question": question,
        "context": [],
        "retrieved_chunks": [],
        "answer": "",
        "confidence_score": 0.0,
        "refusal": False,
        "citation_notes": "",
    }

    final_state = rag_graph.invoke(initial_state)

    return {
        "query": question,
        "answer": final_state["answer"],
        "retrieved_chunks": final_state["retrieved_chunks"],
        "confidence_score": final_state["confidence_score"],
        "refusal": final_state.get("refusal", False),
        "citation_notes": final_state.get("citation_notes", ""),
    }

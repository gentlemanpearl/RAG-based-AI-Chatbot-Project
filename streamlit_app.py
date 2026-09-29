"""
Streamlit Web UI for RAG-based Agentic AI Chatbot.
Provides an interactive chat interface, side-by-side retrieved chunk inspector,
and grounded confidence scoring.
"""

import os
import sys
import time
from pathlib import Path
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import run_rag_pipeline
from src.config import (
    OPENAI_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    validate_openai_key,
    validate_pinecone_key,
)

# Page configuration
st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .chunk-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .chunk-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #3B82F6;
        display: flex;
        justify-content: space-between;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-high {
        background-color: #DEF7EC;
        color: #03543F;
    }
    .score-med {
        background-color: #FEF08A;
        color: #713F12;
    }
    .score-refusal {
        background-color: #FEE2E2;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state for messages and latest result
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# Sidebar
with st.sidebar:
    st.title("⚙️ System Status")
    
    # API status indicators
    openai_ok = validate_openai_key()
    pinecone_ok = validate_pinecone_key()

    if openai_ok:
        st.success("🟢 OpenAI API: Ready", icon="✅")
    else:
        st.error("🔴 OpenAI API: Missing Key", icon="⚠️")

    if pinecone_ok:
        st.success(f"🟢 Pinecone: {PINECONE_INDEX_NAME}", icon="🌲")
    else:
        st.warning("🟡 Pinecone: Unconfigured (Local fallback)", icon="📁")

    st.markdown("---")
    st.markdown("### 📋 Benchmark Queries")
    st.caption("Click any sample query to test grounded retrieval & refusal:")

    sample_queries = [
        "What is Agentic AI according to the eBook?",
        "How do AI agents differ from traditional automation systems?",
        "What are the core components of an Agentic Architecture?",
        "What role does memory play in Agentic AI workflows?",
        "Who won the 2022 FIFA World Cup?",
    ]

    selected_query = None
    for q in sample_queries:
        if st.button(q, key=f"btn_{q[:20]}", use_container_width=True):
            selected_query = q

    st.markdown("---")
    st.markdown("### ℹ️ Architecture Specs")
    st.markdown(f"- **LLM**: `{LLM_MODEL}`")
    st.markdown(f"- **Embedding**: `{EMBEDDING_MODEL}`")
    st.markdown(f"- **Framework**: LangGraph + Pinecone")
    st.markdown(f"- **Source**: *Agentic AI eBook* (Konverge & Emergence AI)")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.latest_result = None
        st.rerun()

# Main Header
col_title, col_meta = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-title">🤖 Agentic AI RAG Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Strictly grounded Q&A workflow built with <b>LangGraph</b>, <b>Pinecone</b>, and <b>OpenAI</b>.</div>',
        unsafe_allow_html=True,
    )

with col_meta:
    if st.session_state.latest_result:
        res = st.session_state.latest_result
        score = res.get("confidence_score", 0.0)
        is_refusal = res.get("refusal", False)
        
        if is_refusal:
            badge_class = "score-refusal"
            status_text = "Refused (Out-of-Domain)"
        elif score >= 0.8:
            badge_class = "score-high"
            status_text = f"High Confidence: {int(score * 100)}%"
        else:
            badge_class = "score-med"
            status_text = f"Moderate: {int(score * 100)}%"

        st.markdown(
            f'<div style="text-align:right;"><span class="score-badge {badge_class}">{status_text}</span></div>',
            unsafe_allow_html=True
        )

# Layout: Split into Chat and Context Panel
chat_col, context_col = st.columns([3, 2], gap="large")

with chat_col:
    st.subheader("💬 Conversation")

    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle input prompt from either chat_input or sidebar button
    user_input = st.chat_input("Ask a question about the Agentic AI eBook...")
    query_to_run = selected_query or user_input

    if query_to_run:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": query_to_run})
        with st.chat_message("user"):
            st.markdown(query_to_run)

        # Generate response via LangGraph workflow
        with st.chat_message("assistant"):
            with st.spinner("Retrieving from Pinecone & generating grounded answer..."):
                try:
                    result = run_rag_pipeline(query_to_run)
                    st.session_state.latest_result = result
                    answer_text = result["answer"]
                    st.markdown(answer_text)
                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

with context_col:
    st.subheader("🔍 Retrieved Context & Evidence")
    
    if st.session_state.latest_result:
        res = st.session_state.latest_result
        chunks = res.get("retrieved_chunks", [])
        score = res.get("confidence_score", 0.0)
        is_refusal = res.get("refusal", False)
        notes = res.get("citation_notes", "")

        # Score & metrics
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Confidence Score", f"{score:.2f}")
        with col_m2:
            st.metric("Grounding Status", "Refused" if is_refusal else "Grounded")

        if notes:
            st.info(f"**Citation Note**: {notes}")

        st.markdown(f"**Retrieved Chunks ({len(chunks)} top-k):**")
        
        for idx, chunk in enumerate(chunks, 1):
            page_str = f"Page {chunk.get('page')}" if chunk.get("page") else "N/A"
            relevance = chunk.get("relevance_score", 0.0)
            
            with st.expander(f"Chunk #{idx} — {page_str} (Score: {relevance:.3f})", expanded=(idx == 1)):
                st.caption(f"Source: `{chunk.get('source', 'Ebook-Agentic-AI.pdf')}` | Chunk ID: `{chunk.get('chunk_id')}`")
                st.markdown(f"```text\n{chunk.get('content', '')}\n```")

        # JSON preview tab
        with st.expander("🛠️ Raw LangGraph Output (JSON)"):
            st.json(res)
    else:
        st.info("Submit a question to inspect retrieved Pinecone vector chunks and LangGraph execution state.")

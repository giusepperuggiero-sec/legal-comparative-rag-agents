"""Streamlit dashboard entrypoint for the Multilingual Comparative Legal RAG system."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

st.set_page_config(
    page_title="Legal RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<h1 style='text-align: center;'>Legal Comparative RAG System</h1>", unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>Comparative Legal Assistant (Italy · Estonia · Slovenia)</h4>",
    unsafe_allow_html=True,
)
st.divider()

col_config, col_db, col_chat = st.columns(3)

with col_config:
    with st.container(border=True):
        st.subheader("Configuration")
        st.markdown("Customize generative model and retrieval hyperparameters.")
        st.markdown(
            """
            - **LLM**: Inference model, judge model, temperature
            - **RAG**: Chunk size, overlap, and Top-K retrieval
            - **Embeddings**: Dense encoder model ID
            """
        )
        st.warning("Manage runtime parameters in **Configuration**")

with col_db:
    with st.container(border=True):
        st.subheader("Knowledge Base")
        st.markdown("Ingest JSON legal corpora and build the FAISS index.")
        st.markdown(
            """
            - **Ingestion**: Recursive statute & case parsing
            - **Indexing**: FAISS Flat L2 with metadata injection
            - **Corpus Stats**: Laws vs jurisprudence distribution
            """
        )
        st.success("Build or refresh index in **Vector DB Builder**")

with col_chat:
    with st.container(border=True):
        st.subheader("Legal Chatbot")
        st.markdown("Query comparative family & succession law naturally.")
        st.markdown(
            """
            - **Single-Agent ReAct**: Metadata router + article reranker
            - **Multi-Agent Supervisor**: Divorce, Inheritance, Case Law workers
            - **White-Box Trace**: Transparent reasoning & PII redaction
            """
        )
        st.info("Open **Chatbot** to query the legal corpus")

st.divider()
st.subheader("System Status")

if os.getenv("HF_TOKEN"):
    st.success("**HuggingFace Inference Token (`HF_TOKEN`)**: Configured and detected")
else:
    st.error("**HuggingFace Inference Token (`HF_TOKEN`)**: Missing. Add `HF_TOKEN` in `.env`.")

with st.expander("Architecture Overview"):
    st.markdown(
        """
        - **Retrieval Backend**: Python (LangChain + FAISS Flat L2 + `all-MiniLM-L6-v2`)
        - **Generation & Judge Models**: Qwen 2.5-7B-Instruct & Qwen 2.5-72B-Instruct via HuggingFace Inference API
        - **Security Controls**: Regex PII redaction, XML anti-prompt-injection delimiters, atomic log persistence
        - **Corpus**: 1,472 statutory articles and court cases across Italy, Estonia, and Slovenia
        """
    )

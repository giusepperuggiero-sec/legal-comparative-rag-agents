"""Streamlit Knowledge Base Builder for JSON ingestion and FAISS Flat L2 indexing."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.configuration import RagConfig
from backend.load_data import DataLoader
from backend.vector_store import VectorStoreManager

st.set_page_config(page_title="Vector DB Builder", page_icon="🏗️", layout="wide")
st.title("Knowledge Base & FAISS Index Builder")
st.markdown("Ingest multilingual legal JSON files and build the FAISS Flat L2 vector store.")

with st.container(border=True):
    col_src, col_dst = st.columns(2)
    col_src.markdown(f"**Corpus Source Directory:**\n`{RagConfig.DATA_PATH}`")
    col_dst.markdown(f"**FAISS Output Directory:**\n`{RagConfig.VECTOR_STORE_PATH}`")

st.divider()

col_action, _ = st.columns([1, 2])
with col_action:
    start_btn = st.button("Build FAISS Index", type="primary", use_container_width=True)

if start_btn:
    status_container = st.status("Ingestion pipeline started...", expanded=True)
    try:
        status_container.write("1) Scanning corpus files and injecting metadata headers...")
        docs = DataLoader().load_documents()
        if not docs:
            status_container.update(label="Error: no valid JSON legal documents found.", state="error")
            st.stop()

        laws_count = sum(1 for d in docs if d.metadata.get("doc_type") == "legislation")
        cases_count = sum(1 for d in docs if d.metadata.get("doc_type") == "jurisprudence")

        status_container.write("Corpus analysis completed:")
        m1, m2, m3 = status_container.columns(3)
        m1.metric("Total Documents", len(docs))
        m2.metric("Statutory Articles", laws_count)
        m3.metric("Court Cases", cases_count)

        status_container.write("2) Computing embeddings and persisting FAISS index...")
        progress_bar = status_container.progress(0, text="Initializing embedding model...")
        VectorStoreManager().create_database(docs)
        st.cache_resource.clear()
        progress_bar.progress(100, text="Index persisted successfully.")

        status_container.update(label="FAISS index built successfully.", state="complete", expanded=False)
        st.success(f"Vector store ready. Total documents indexed: {len(docs)}")
    except Exception as exc:
        status_container.update(label="Indexing pipeline failed", state="error")
        st.error(f"Error details: {exc}")

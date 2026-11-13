"""Runtime configuration dashboard reading and persisting settings via RagConfig."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.configuration import RagConfig

st.set_page_config(page_title="Configuration", page_icon="⚙️", layout="wide")
st.title("Configuration Control Panel")
st.markdown("Customize generative LLM, RAGAS judge, and vector retrieval hyperparameters.")
st.info(f"Active configuration path: `{RagConfig.CONFIG_FILE}`")

current_cfg = RagConfig.load()

with st.form("config_form"):
    tab_llm, tab_rag = st.tabs(["LLM Parameters", "Retrieval Parameters (RAG)"])

    with tab_llm:
        st.subheader("Generative & Judge Models")
        col_left, col_right = st.columns(2)
        with col_left:
            new_llm = st.text_input(
                "Chat Model Repository ID (HuggingFace)",
                value=str(current_cfg["LLM_REPO_ID"]),
                help="Example: Qwen/Qwen2.5-7B-Instruct",
            )
            new_eval_llm = st.text_input(
                "RAGAS Judge Repository ID (HuggingFace)",
                value=str(current_cfg["EVAL_LLM_REPO_ID"]),
                help="Example: Qwen/Qwen2.5-72B-Instruct",
            )
            new_temp = st.slider(
                "Sampling Temperature",
                0.0,
                1.0,
                value=float(current_cfg["TEMPERATURE"]),
                step=0.1,
                help="0.0 = Deterministic, 1.0 = Creative",
            )

        with col_right:
            new_tokens = st.number_input(
                "Max New Tokens",
                min_value=128,
                max_value=8192,
                value=int(current_cfg["MAX_NEW_TOKENS"]),
                step=256,
            )
            new_embed = st.text_input(
                "Embedding Model ID",
                value=str(current_cfg["EMBEDDING_MODEL_NAME"]),
                help="Sentence-Transformers model used to vectorize corpus chunks",
            )

    with tab_rag:
        st.subheader("Indexing & Search Strategy")
        col_paths, col_params = st.columns(2)
        with col_paths:
            st.text_input("Raw Corpus Path", str(RagConfig.DATA_PATH), disabled=True)
            st.text_input("FAISS Vector Store Path", str(RagConfig.VECTOR_STORE_PATH), disabled=True)

        with col_params:
            new_chunk = st.slider(
                "Chunk Size (Characters)",
                min_value=500,
                max_value=10000,
                value=int(current_cfg["CHUNK_SIZE"]),
                step=100,
            )
            new_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=1000,
                value=int(current_cfg["CHUNK_OVERLAP"]),
                step=50,
            )
            new_top_k = st.number_input(
                "Top-K Base Multiplier",
                min_value=1,
                max_value=50,
                value=int(current_cfg.get("TOP_K", 5)),
                step=1,
            )

    st.markdown("---")
    submitted = st.form_submit_button("Save & Apply Configuration", use_container_width=True, type="primary")

    if submitted:
        new_config_data = {
            "LLM_REPO_ID": new_llm.strip(),
            "EVAL_LLM_REPO_ID": new_eval_llm.strip(),
            "EMBEDDING_MODEL_NAME": new_embed.strip(),
            "TEMPERATURE": float(new_temp),
            "MAX_NEW_TOKENS": int(new_tokens),
            "CHUNK_SIZE": int(new_chunk),
            "CHUNK_OVERLAP": int(new_overlap),
            "TOP_K": int(new_top_k),
        }
        if RagConfig.save(new_config_data):
            st.cache_resource.clear()
            st.success("Configuration saved and applied to active session.")
        else:
            st.error("Failed persisting configuration file.")

if RagConfig.CONFIG_FILE.exists():
    with st.expander("View Raw Configuration JSON"):
        st.code(RagConfig.CONFIG_FILE.read_text(encoding="utf-8"), language="json")

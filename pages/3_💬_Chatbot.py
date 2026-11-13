"""Interactive Legal Chatbot supporting Single-Agent ReAct and Multi-Agent Supervisor."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.configuration import RagConfig
from backend.guardrails import PIIGuardrail
from backend.llm_provider import LLMProvider
from backend.logger import ConversationLogger
from backend.multi_agent.supervisor import SupervisorAgent
from backend.single_agent.single_agent import SingleAgent
from backend.vector_store import VectorStoreManager

st.set_page_config(page_title="Legal Chatbot", page_icon="💬", layout="wide")
st.title("Comparative Legal RAG Assistant")
st.markdown("Cross-jurisdictional civil-law assistant (**Italy · Estonia · Slovenia**).")


@st.cache_resource
def _load_cached_agent(mode: str, config_sig: str, index_mtime: float):
    vector_store = VectorStoreManager().load_local()
    llm = LLMProvider().get_chat_model()
    if mode == "Single Agent":
        return SingleAgent(vector_store=vector_store, llm=llm)
    return SupervisorAgent(vector_store=vector_store, llm=llm)


if "history" not in st.session_state:
    st.session_state.history = {"Single Agent": [], "Multi-Agent": []}

with st.sidebar:
    st.header("Agent Controls")
    mode_selection = st.radio(
        "Select Agent Architecture:",
        ("Single Agent", "Multi-Agent"),
        index=0,
        key="agent_selector",
    )
    st.divider()
    if st.button("Reset Conversation Memory", type="primary", use_container_width=True):
        st.session_state.history = {"Single Agent": [], "Multi-Agent": []}
        st.rerun()
    st.info("Sanitized conversations are saved atomically to `logs/chat_history.json`.")

current_chat = st.session_state.history[mode_selection]

for message in current_chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "trace" in message:
            with st.expander("White-Box Reasoning Trace"):
                st.markdown(message["trace"])
            if message.get("sources_preview"):
                with st.expander("Retrieved Sources"):
                    for src in message["sources_preview"]:
                        st.text(src)

prompt = st.chat_input(f"Ask {mode_selection} a legal question...", key=f"input_{mode_selection}")

if prompt:
    safe_prompt_display = PIIGuardrail.mask_pii(prompt)
    current_chat.append({"role": "user", "content": safe_prompt_display})
    with st.chat_message("user"):
        st.markdown(safe_prompt_display)

    with st.chat_message("assistant"):
        with st.spinner(f"{mode_selection} is analyzing legal sources..."):
            try:
                cfg = RagConfig.load()
                cfg_sig = f"{cfg['LLM_REPO_ID']}|{cfg['TEMPERATURE']}|{cfg['MAX_NEW_TOKENS']}|{cfg['TOP_K']}|{cfg['EMBEDDING_MODEL_NAME']}"
                index_file = RagConfig.VECTOR_STORE_PATH / "index.faiss"
                index_mtime = index_file.stat().st_mtime if index_file.exists() else 0.0

                agent_engine = _load_cached_agent(mode_selection, cfg_sig, index_mtime)
                mode_tag = "Single-Agent" if mode_selection == "Single Agent" else "Multi-Agent"

                result = agent_engine.run(prompt)
                answer = PIIGuardrail.mask_pii(str(result.get("answer", "Generation failed.")))
                trace = PIIGuardrail.mask_pii(str(result.get("reasoning_trace", "")))
                sources = result.get("source_documents", [])

                st.markdown(answer)

                sources_preview = []
                if sources:
                    with st.expander("Source Passages"):
                        for idx, doc in enumerate(sources, start=1):
                            meta = doc.metadata
                            src_str = f"[{meta.get('country', '?')}] {meta.get('source_file', 'Doc')}"
                            sources_preview.append(src_str)
                            st.markdown(f"**{idx}. {src_str}**")
                            st.caption(doc.page_content.replace("\n", " ")[:250] + "...")

                with st.expander("White-Box Reasoning Trace"):
                    st.markdown(trace)

                current_chat.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "trace": trace,
                        "sources_preview": sources_preview,
                    }
                )
                ConversationLogger().log_interaction(
                    user_query=prompt,
                    system_response=answer,
                    sources=sources,
                    agent_type=mode_tag,
                )
            except Exception as exc:
                st.error(f"Execution error: {exc}")

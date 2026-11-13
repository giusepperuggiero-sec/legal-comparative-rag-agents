"""RAGAS Automated Benchmark Dashboard evaluating Faithfulness, Relevancy, Precision, Recall & Correctness."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.evaluator import RagasEvaluator

st.set_page_config(page_title="RAGAS Evaluation", page_icon="📊", layout="wide")
st.title("LLMOps Evaluation Dashboard (RAGAS)")
st.markdown(
    "Evaluate logged chatbot responses in `logs/chat_history.json` across five RAGAS metrics "
    "using HuggingFace judge models (`Qwen/Qwen2.5-72B-Instruct`)."
)

with st.expander("RAGAS Metrics Legend"):
    st.markdown(
        """
        - **Faithfulness**: Measures factual adherence to retrieved contexts (`0.0` to `1.0`).
        - **Answer Relevancy**: Measures alignment between the generated answer and user query.
        - **Context Precision**: Evaluates ranking quality of retrieved documents *(requires Ground Truth)*.
        - **Context Recall**: Measures whether evidence covering the reference answer was retrieved *(requires Ground Truth)*.
        - **Answer Correctness**: Compares generated output against reference legal answers *(requires Ground Truth)*.
        """
    )

col_ctrl, col_info = st.columns([1, 3])
with col_ctrl:
    btn_eval = st.button("Run RAGAS Evaluation", type="primary", use_container_width=True)
with col_info:
    st.info(
        "Rate-limited judge execution serializes Inference API calls automatically; "
        "ensure `logs/evaluation_references.json` is populated for supervised metrics."
    )

if btn_eval:
    evaluator = RagasEvaluator()
    with st.spinner("Running RAGAS evaluation over logged interactions..."):
        try:
            res_df, diag = evaluator.run_evaluation()
            if res_df.empty:
                st.warning(f"Evaluation skipped: {diag.get('message', 'Empty dataset')}")
                st.info("Verify `logs/chat_history.json` contains valid queries, answers, and retrieved contexts.")
            else:
                st.success("RAGAS evaluation completed.")
                if not diag.get("embeddings_available", True):
                    st.warning("Embeddings unavailable: supervised & relevancy metrics omitted.")
                st.caption(
                    f"Rows evaluated: {diag.get('rows', 0)} | "
                    f"Matched references: {diag.get('with_reference', 0)} | "
                    f"Supervised rows scored: {diag.get('rows_with_reference_used', 0)}"
                )

                def _mean_str(col_name: str) -> str:
                    if col_name in res_df.columns:
                        val = pd.to_numeric(res_df[col_name], errors="coerce").mean()
                        return f"{val:.4f}" if pd.notna(val) else "N/A"
                    return "N/A"

                st.subheader("Mean RAGAS Metrics")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Faithfulness", _mean_str("faithfulness"))
                m2.metric("Answer Relevancy", _mean_str("answer_relevancy"))
                m3.metric("Context Precision", _mean_str("context_precision"))
                m4.metric("Context Recall", _mean_str("context_recall"))
                m5.metric("Answer Correctness", _mean_str("answer_correctness"))

                st.subheader("Per-Query Breakdown")
                cols_to_show = [
                    "user_input",
                    "response",
                    "reference",
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                    "answer_correctness",
                ]
                st.dataframe(res_df[[c for c in cols_to_show if c in res_df.columns]], use_container_width=True)

                st.subheader("Hallucination Risk Audit")
                if "faithfulness" in res_df.columns:
                    res_df["faithfulness_numeric"] = pd.to_numeric(res_df["faithfulness"], errors="coerce")
                    low_faith = res_df[res_df["faithfulness_numeric"] < 0.5]
                    if not low_faith.empty:
                        st.error(f"Detected {len(low_faith)} low-faithfulness responses (< 0.50).")
                        with st.expander("Inspect flagged queries"):
                            st.table(low_faith[["user_input", "faithfulness"]])
                    else:
                        st.success("All evaluated responses adhere strongly to retrieved contexts (>= 0.50).")
        except Exception as exc:
            st.error(f"Evaluation error: {exc}")

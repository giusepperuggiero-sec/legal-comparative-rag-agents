"""Automated 5-metric RAGAS benchmark evaluator over JSON conversation logs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import unicodedata

from datasets import Dataset
from huggingface_hub import InferenceClient
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from backend.configuration import RagConfig
from backend.llm_provider import HuggingFaceEvalChatWrapper, LLMProvider
from backend.vector_store import VectorStoreManager

try:
    from ragas.llms import LangchainLLM
except ImportError:
    LangchainLLM = None
try:
    from ragas.llms import LangchainLLMWrapper
except ImportError:
    LangchainLLMWrapper = None
try:
    from ragas.embeddings import LangchainEmbeddings
except ImportError:
    LangchainEmbeddings = None
try:
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError:
    LangchainEmbeddingsWrapper = None


class RagasEvaluator:
    """Run supervised & reference-free RAGAS evaluation against stored conversation turns."""

    def __init__(
        self,
        log_file: Path | str | None = None,
        reference_file: Path | str | None = None,
    ) -> None:
        logs_dir = RagConfig.BASE_DIR / "logs"
        self.log_file = Path(log_file) if log_file is not None else logs_dir / "chat_history.json"
        self.reference_file = (
            Path(reference_file) if reference_file is not None else logs_dir / "evaluation_references.json"
        )
        self.embedding_init_error: str | None = None
        self.llm_provider = LLMProvider()

        self.chat_model = self._build_eval_chat_model()
        self.ragas_llm = self._build_ragas_llm(self.chat_model)

        self.vector_manager = None
        self.base_embeddings = None
        self.ragas_embeddings = None
        try:
            self.vector_manager = VectorStoreManager()
            self.base_embeddings = self.vector_manager.embeddings
            self.ragas_embeddings = self._build_ragas_embeddings(self.base_embeddings)
        except Exception as exc:
            self.embedding_init_error = str(exc)
            self.ragas_embeddings = None

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
        normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
        return normalized.strip()

    def _build_eval_chat_model(self) -> HuggingFaceEvalChatWrapper:
        api_key = os.getenv("HF_TOKEN") or ""
        config = RagConfig.load()
        eval_model = str(config.get("EVAL_LLM_REPO_ID", "Qwen/Qwen2.5-72B-Instruct"))
        print(f"[RAGAS] Using judge model: {eval_model}")
        client = InferenceClient(token=api_key or None)
        return HuggingFaceEvalChatWrapper(
            client=client,
            model_name=eval_model,
            temperature=0.0,
            max_tokens=4096,
        )

    @staticmethod
    def _ensure_set_run_config(obj: Any) -> Any:
        if obj is not None and not hasattr(obj, "set_run_config"):
            obj.set_run_config = lambda run_config: None
        return obj

    def _build_ragas_llm(self, chat_model: Any) -> Any:
        for wrapper_cls in (LangchainLLMWrapper, LangchainLLM):
            if not wrapper_cls:
                continue
            for kwargs in ({"langchain_llm": chat_model}, {"llm": chat_model}):
                try:
                    return self._ensure_set_run_config(wrapper_cls(**kwargs))
                except Exception:
                    continue
        return self._ensure_set_run_config(chat_model)

    def _build_ragas_embeddings(self, embeddings: Any) -> Any:
        for wrapper_cls in (LangchainEmbeddingsWrapper, LangchainEmbeddings):
            if not wrapper_cls:
                continue
            for kwargs in ({"langchain_embeddings": embeddings}, {"embeddings": embeddings}):
                try:
                    return self._ensure_set_run_config(wrapper_cls(**kwargs))
                except Exception:
                    continue
        return self._ensure_set_run_config(embeddings)

    @staticmethod
    def _extract_contexts(ctx_raw: Any) -> list[str]:
        if not isinstance(ctx_raw, list):
            return []
        contexts = []
        for item in ctx_raw:
            if isinstance(item, dict):
                snippet = item.get("content") or item.get("content_snippet") or ""
                text = str(snippet).strip()
            else:
                text = str(item).strip()
            if text:
                contexts.append(text)
        return contexts

    def _load_reference_map(self) -> dict[str, str]:
        if not self.reference_file.exists():
            return {}
        try:
            with self.reference_file.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}

        reference_map: dict[str, str] = {}
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    q = str(row.get("question", "")).strip()
                    r = str(row.get("reference", "")).strip()
                    if q and r:
                        reference_map[q] = r
        elif isinstance(payload, dict):
            for q, r in payload.items():
                q_clean, r_clean = str(q).strip(), str(r).strip()
                if q_clean and r_clean:
                    reference_map[q_clean] = r_clean
        return reference_map

    def load_data_from_logs(self) -> tuple[Dataset | None, dict[str, Any]]:
        print(f"[RAGAS] Loading logs from: {self.log_file}")
        if not self.log_file.exists():
            return None, {
                "rows": 0,
                "with_reference": 0,
                "all_reference_missing": True,
                "message": "Conversation log file not found.",
            }

        try:
            with self.log_file.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None, {
                "rows": 0,
                "with_reference": 0,
                "all_reference_missing": True,
                "message": "Invalid JSON format in conversation logs.",
            }

        if not data:
            return None, {
                "rows": 0,
                "with_reference": 0,
                "all_reference_missing": True,
                "message": "No interactions recorded in logs.",
            }

        df = pd.DataFrame(data).rename(
            columns={
                "user_query": "user_input",
                "query": "user_input",
                "question": "user_input",
                "system_answer": "response",
                "answer": "response",
                "retrieved_contexts": "retrieved_contexts",
                "contexts": "retrieved_contexts",
                "ground_truth": "reference",
                "reference": "reference",
            }
        )

        for col, default in (
            ("user_input", ""),
            ("response", ""),
            ("timestamp", None),
            ("reference", None),
        ):
            if col not in df.columns:
                df[col] = default
        if "retrieved_contexts" not in df.columns:
            df["retrieved_contexts"] = [[] for _ in range(len(df))]

        df["retrieved_contexts"] = df["retrieved_contexts"].apply(self._extract_contexts)
        df["user_input"] = df["user_input"].astype(str)
        df["response"] = df["response"].astype(str)

        # Filter provider API error placeholders so judge metrics evaluate genuine generations only
        df = df[
            ~df["response"].str.contains(
                r"Errore LLM|LLM Error|Payment Required|402 Client Error",
                case=False,
                regex=True,
                na=False,
            )
        ]

        reference_map = self._load_reference_map()
        if reference_map:
            normalized_map = {self._normalize_text(k): v for k, v in reference_map.items()}
            df["reference"] = df.apply(
                lambda row: row["reference"]
                if isinstance(row["reference"], str) and row["reference"].strip()
                else normalized_map.get(self._normalize_text(str(row["user_input"]))),
                axis=1,
            )

        df["reference"] = df["reference"].apply(
            lambda x: str(x).strip() if isinstance(x, str) and str(x).strip() else None
        )

        final_df = df[["user_input", "response", "retrieved_contexts", "reference", "timestamp"]].copy()
        final_df = final_df[
            (final_df["user_input"].str.len() > 0)
            & (final_df["response"].str.len() > 0)
            & (final_df["retrieved_contexts"].apply(lambda x: isinstance(x, list) and len(x) > 0))
        ]
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce")
        final_df = final_df.sort_values("timestamp").drop_duplicates(subset=["user_input"], keep="last")
        final_df = final_df.drop(columns=["timestamp"])

        final_df["ground_truth"] = final_df["reference"]
        final_df["question"] = final_df["user_input"]
        final_df["answer"] = final_df["response"]
        final_df["contexts"] = final_df["retrieved_contexts"]

        diagnostics: dict[str, Any] = {
            "rows": int(len(final_df)),
            "with_reference": int(final_df["reference"].notna().sum()),
            "all_reference_missing": bool(final_df["reference"].notna().sum() == 0),
            "message": "ok",
        }
        if final_df.empty:
            diagnostics["message"] = "No valid rows with question, answer, and retrieved contexts."
            return None, diagnostics

        return Dataset.from_pandas(final_df, preserve_index=False), diagnostics

    def run_evaluation(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        dataset, diagnostics = self.load_data_from_logs()

        metrics = [faithfulness]
        if self.ragas_embeddings is not None:
            metrics.append(answer_relevancy)
        has_supervised = not diagnostics["all_reference_missing"]
        if has_supervised and self.ragas_embeddings is not None:
            metrics.extend([context_precision, context_recall, answer_correctness])

        diagnostics["embeddings_available"] = self.ragas_embeddings is not None
        if self.embedding_init_error:
            diagnostics["embedding_init_error"] = self.embedding_init_error
        if self.ragas_embeddings is None:
            diagnostics["message"] = "Embeddings unavailable: running LLM-only metrics only."

        for metric in metrics:
            if hasattr(metric, "llm"):
                metric.llm = self.ragas_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = self.ragas_embeddings

        if dataset is None or len(dataset) == 0:
            empty = pd.DataFrame(
                columns=["user_input", "response", "reference", "faithfulness", "answer_relevancy"]
            )
            return empty, diagnostics

        eval_dataset = dataset
        if has_supervised:
            df_eval = dataset.to_pandas()
            df_supervised = df_eval[
                df_eval["reference"].notna() & (df_eval["reference"].astype(str).str.len() > 0)
            ].copy()
            diagnostics["rows_with_reference_used"] = int(len(df_supervised))
            if df_supervised.empty:
                metrics = [faithfulness] if self.ragas_embeddings is None else [faithfulness, answer_relevancy]
            else:
                keep_cols = [
                    "user_input",
                    "response",
                    "retrieved_contexts",
                    "reference",
                    "question",
                    "answer",
                    "contexts",
                    "ground_truth",
                ]
                eval_dataset = Dataset.from_pandas(df_supervised[keep_cols], preserve_index=False)
        else:
            diagnostics["rows_with_reference_used"] = 0

        print("[RAGAS] Evaluation started")
        try:
            results = evaluate(
                dataset=eval_dataset,
                metrics=metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                raise_exceptions=False,
                run_config=RunConfig(max_workers=2, timeout=600, max_retries=2),
            )
            res_df = results.to_pandas().rename(
                columns={
                    "question": "user_input",
                    "answer": "response",
                    "ground_truth": "reference",
                    "contexts": "retrieved_contexts",
                }
            )
            orig_df = eval_dataset.to_pandas()
            for col in ("user_input", "response", "retrieved_contexts", "reference"):
                if col not in res_df.columns and col in orig_df.columns:
                    res_df[col] = orig_df[col]
            print("[RAGAS] Evaluation completed")
            return res_df, diagnostics
        except Exception as exc:
            print(f"[RAGAS ERROR] Evaluation failed: {exc}")
            diagnostics["message"] = f"RAGAS execution error: {exc}"
            empty = pd.DataFrame(
                columns=["user_input", "response", "reference", "faithfulness", "answer_relevancy"]
            )
            return empty, diagnostics

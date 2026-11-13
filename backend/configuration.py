"""Centralized JSON configuration manager with atomic persistence and dynamic reload."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


class RagConfig:
    """Manage paths, model identifiers, and retrieval hyperparameters."""

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data"
    VECTOR_STORE_PATH = BASE_DIR / "vector_store"
    CONFIG_FILE = BASE_DIR / "config.json"

    DEFAULT_CONFIG: dict[str, Any] = {
        "LLM_REPO_ID": "Qwen/Qwen2.5-7B-Instruct",
        "EVAL_LLM_REPO_ID": "Qwen/Qwen2.5-72B-Instruct",
        "EMBEDDING_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
        "TEMPERATURE": 0.1,
        "MAX_NEW_TOKENS": 2048,
        "CHUNK_SIZE": 4000,
        "CHUNK_OVERLAP": 200,
        "TOP_K": 5,
    }

    LLM_REPO_ID: str = DEFAULT_CONFIG["LLM_REPO_ID"]
    EVAL_LLM_REPO_ID: str = DEFAULT_CONFIG["EVAL_LLM_REPO_ID"]
    EMBEDDING_MODEL_NAME: str = DEFAULT_CONFIG["EMBEDDING_MODEL_NAME"]
    TEMPERATURE: float = float(DEFAULT_CONFIG["TEMPERATURE"])
    MAX_NEW_TOKENS: int = int(DEFAULT_CONFIG["MAX_NEW_TOKENS"])
    CHUNK_SIZE: int = int(DEFAULT_CONFIG["CHUNK_SIZE"])
    CHUNK_OVERLAP: int = int(DEFAULT_CONFIG["CHUNK_OVERLAP"])
    TOP_K: int = int(DEFAULT_CONFIG["TOP_K"])

    @classmethod
    def load(cls) -> dict[str, Any]:
        """Load config merged onto DEFAULT_CONFIG so newly added keys are never lost."""
        merged = cls.DEFAULT_CONFIG.copy()
        if not cls.CONFIG_FILE.exists():
            return merged
        try:
            with cls.CONFIG_FILE.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                merged.update(loaded)
        except (OSError, json.JSONDecodeError, ValueError):
            return cls.DEFAULT_CONFIG.copy()
        return merged

    @classmethod
    def refresh(cls) -> dict[str, Any]:
        """Synchronize class attributes with the active persisted configuration."""
        cfg = cls.load()
        cls.LLM_REPO_ID = str(cfg["LLM_REPO_ID"])
        cls.EVAL_LLM_REPO_ID = str(cfg["EVAL_LLM_REPO_ID"])
        cls.EMBEDDING_MODEL_NAME = str(cfg["EMBEDDING_MODEL_NAME"])
        cls.TEMPERATURE = float(cfg["TEMPERATURE"])
        cls.MAX_NEW_TOKENS = int(cfg["MAX_NEW_TOKENS"])
        cls.CHUNK_SIZE = int(cfg["CHUNK_SIZE"])
        cls.CHUNK_OVERLAP = int(cfg["CHUNK_OVERLAP"])
        cls.TOP_K = int(cfg["TOP_K"])
        return cfg

    @classmethod
    def save(cls, new_config: dict[str, Any]) -> bool:
        """Merge updates with active config, write atomically, and refresh attributes."""
        merged = cls.load()
        merged.update(new_config)
        cls.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=cls.CONFIG_FILE.parent,
                delete=False,
                suffix=".tmp",
            ) as tmp:
                json.dump(merged, tmp, indent=4)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, cls.CONFIG_FILE)
            cls.refresh()
            return True
        except (OSError, TypeError, ValueError):
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return False


RagConfig.refresh()

"""Thread-safe atomic conversation logger with PII redaction for RAGAS evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

from backend.configuration import RagConfig
from backend.guardrails import PIIGuardrail


class ConversationLogger:
    """Persist chat interactions atomically to `logs/chat_history.json` with PII masking."""

    _write_lock = threading.Lock()

    def __init__(self, log_file: Path | str | None = None) -> None:
        if log_file is not None:
            self.log_file = Path(log_file)
            self.log_dir = self.log_file.parent
        else:
            self.log_dir = RagConfig.BASE_DIR / "logs"
            self.log_file = self.log_dir / "chat_history.json"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self._atomic_write([])

    def _atomic_write(self, records: list[dict[str, Any]]) -> None:
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.log_dir,
                delete=False,
                suffix=".tmp",
            ) as tmp:
                json.dump(records, tmp, indent=4, ensure_ascii=False)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, self.log_file)
        except Exception:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def log_interaction(
        self,
        user_query: str,
        system_response: str,
        sources: list[Any],
        agent_type: str,
    ) -> bool:
        """Append a PII-sanitized conversation turn atomically without truncation corruption."""
        formatted_sources = [
            {
                "content_snippet": doc.page_content[:200],
                "content": doc.page_content,
                "metadata": getattr(doc, "metadata", {}),
            }
            for doc in (sources or [])
        ]

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_mode": agent_type,
            "user_query": PIIGuardrail.mask_pii(user_query),
            "system_answer": PIIGuardrail.mask_pii(system_response),
            "retrieved_contexts": formatted_sources,
        }

        with self._write_lock:
            try:
                if self.log_file.exists():
                    raw_text = self.log_file.read_text(encoding="utf-8")
                    data = json.loads(raw_text) if raw_text.strip() else []
                    if not isinstance(data, list):
                        data = []
                else:
                    data = []
            except (OSError, json.JSONDecodeError):
                data = []

            data.append(entry)
            try:
                self._atomic_write(data)
                print(f"[LOGGER] Interaction saved to {self.log_file}")
                return True
            except OSError as exc:
                print(f"[LOGGER ERROR] Failed saving conversation log: {exc}")
                return False

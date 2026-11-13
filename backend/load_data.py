"""Legal JSON corpus loader with normalized metadata extraction and header injection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from backend.configuration import RagConfig


class DataLoader:
    """Ingest multilingual statutory and case-law JSON documents into LangChain Documents."""

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.data_path = Path(data_path) if data_path is not None else RagConfig.DATA_PATH

    @staticmethod
    def _clean_text(text: Any) -> str:
        if not text:
            return ""
        return str(text).replace("\n", " ").replace("\u2028", " ").strip()

    @staticmethod
    def _normalize_metadata(raw_meta: dict[str, Any], file_path: Path) -> dict[str, str]:
        """Normalize jurisdiction, legal domain, doc_type, and source filename metadata."""
        normalized: dict[str, str] = {}
        path_upper = str(file_path).upper()
        filename = file_path.name

        country = "UNKNOWN"
        if isinstance(raw_meta.get("state"), str):
            country = raw_meta["state"].upper()
        elif raw_meta.get("type") in ("ITALY", "ESTONIA", "SLOVENIA"):
            country = str(raw_meta["type"]).upper()

        if country == "UNKNOWN":
            for candidate in ("ITALY", "ESTONIA", "SLOVENIA"):
                if candidate in path_upper:
                    country = candidate
                    break
        normalized["country"] = country

        if isinstance(raw_meta.get("law"), str) and raw_meta["law"].strip():
            normalized["topic"] = raw_meta["law"].strip()
        elif "DIVORCE" in path_upper:
            normalized["topic"] = "Divorce"
        elif "INHERITANCE" in path_upper:
            normalized["topic"] = "Inheritance"
        else:
            normalized["topic"] = "General"

        if "CASE_ID" in raw_meta or "PROCESSED" in path_upper:
            normalized["doc_type"] = "jurisprudence"
        elif "Article" in filename or "civil_codes_used" in raw_meta:
            normalized["doc_type"] = "legislation"
        else:
            normalized["doc_type"] = "general"

        normalized["source_file"] = filename
        return normalized

    def load_documents(self) -> list[Document]:
        """Recursively read JSON corpus files and inject normalized metadata headers."""
        if not self.data_path.exists():
            print(f"[DATA LOADER WARNING] Corpus directory not found: {self.data_path}")
            return []

        files = sorted(self.data_path.rglob("*.json"))
        print(f"[DATA LOADER] Found {len(files)} JSON corpus files.")

        documents: list[Document] = []
        laws_count = 0
        cases_count = 0

        for f_path in files:
            try:
                with f_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if not isinstance(data, dict):
                    continue

                content = data.get("content") or data.get("text") or data.get("body")
                clean_content = self._clean_text(content)
                if len(clean_content) < 5:
                    continue

                raw_meta = data.get("metadata", {})
                if not isinstance(raw_meta, dict):
                    raw_meta = {}

                meta = self._normalize_metadata(raw_meta, f_path)
                header_info = [
                    f"FILENAME: {meta['source_file']}",
                    f"COUNTRY: {meta['country']}",
                    f"TYPE: {meta['doc_type']}",
                ]
                if "CASE_ID" in raw_meta:
                    header_info.append(f"CASE ID: {raw_meta['CASE_ID']}")
                if "civil_codes_used" in raw_meta:
                    header_info.append(f"RELATED ARTICLES: {raw_meta['civil_codes_used']}")

                contextualized_content = f"{' | '.join(header_info)}\n\nCONTENT:\n{clean_content}"
                if meta["doc_type"] == "legislation":
                    laws_count += 1
                elif meta["doc_type"] == "jurisprudence":
                    cases_count += 1

                documents.append(Document(page_content=contextualized_content, metadata=meta))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                print(f"[DATA LOADER WARNING] Skipped {f_path.name}: {exc}")

        print(f"[DATA LOADER] Loaded {laws_count} statutory articles and {cases_count} court cases.")
        return documents


if __name__ == "__main__":
    DataLoader().load_documents()

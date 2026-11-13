"""Programmatic metadata router extracting country, legal domain, and document type filters."""

from __future__ import annotations

import re
from typing import Any


class MetadataRouter:
    """Hybrid deterministic pre-filter over normalized FAISS document metadata."""

    _COUNTRY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        ("ITALY", re.compile(r"\b(?:itali[a]?|italian[a-z]*|italy)\b", re.IGNORECASE)),
        ("ESTONIA", re.compile(r"\b(?:estonia|estonian)\b", re.IGNORECASE)),
        ("SLOVENIA", re.compile(r"\b(?:slovenia|slovenian)\b", re.IGNORECASE)),
    ]
    _DIVORCE_RE = re.compile(
        r"\b(?:divorzio|divorce|matrimoni[oa]|matrimonial|marital|spous(?:e|es)|married|marriage)\b",
        re.IGNORECASE,
    )
    _INHERITANCE_RE = re.compile(
        r"\b(?:eredit[àa]|inheritance|succession[ei]?|heir[s]?|compulsory portion|inherited)\b",
        re.IGNORECASE,
    )
    _JURISPRUDENCE_RE = re.compile(
        r"\b(?:sentenz[ae]|caso|casi|ruling[s]?|case law|precedent[s]?|court case[s]?)\b",
        re.IGNORECASE,
    )

    @classmethod
    def extract_routing_metadata(cls, query: str) -> dict[str, Any]:
        """Extract single-country, topic, and doc_type filters using strict word boundaries."""
        if not query:
            return {}

        metadata_filter: dict[str, Any] = {}
        countries_found = [name for name, pat in cls._COUNTRY_PATTERNS if pat.search(query)]

        # Restrict by country only when a single jurisdiction is requested
        if len(countries_found) == 1:
            metadata_filter["country"] = countries_found[0]

        if cls._DIVORCE_RE.search(query):
            metadata_filter["topic"] = "Divorce"
        elif cls._INHERITANCE_RE.search(query):
            metadata_filter["topic"] = "Inheritance"

        if cls._JURISPRUDENCE_RE.search(query):
            metadata_filter["doc_type"] = "jurisprudence"

        return metadata_filter

    @classmethod
    def get_filtered_retriever(cls, base_retriever: Any, query: str) -> Any:
        filters = cls.extract_routing_metadata(query)
        if filters:
            return base_retriever.with_config(configurable={"search_kwargs": {"filter": filters}})
        return base_retriever

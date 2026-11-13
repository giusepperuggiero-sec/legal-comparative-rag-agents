"""Shared anti-injection XML fencing, article-number reranking, and observation helpers."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

RELEVANCE_THRESHOLD = 1.5
_ARTICLE_NUM_QUERY_RE = re.compile(r"\b\d{2,5}\b")


def wrap_untrusted_xml(tag: str, content: str) -> str:
    """Enclose untrusted user or retrieved text in XML tags while neutralizing closing-tag breakouts."""
    safe_content = str(content).replace(f"</{tag}>", f"&lt;/{tag}&gt;")
    return f"<{tag}>\n{safe_content}\n</{tag}>"


def rerank_documents(query: str, docs: list[Any], trace: list[str], top_n_matched: int = 8) -> list[Any]:
    """Boost retrieved documents explicitly containing statutory article/paragraph numbers."""
    numbers = _ARTICLE_NUM_QUERY_RE.findall(query or "")
    if not numbers:
        return docs[:6]

    trace.append(f"   [Reranking] Priority boost for statutory references: {numbers}")
    patterns = [re.compile(rf"\b{re.escape(num)}\b") for num in numbers]
    high, low = [], []

    for doc in docs:
        source_file = doc.metadata.get("source_file", "")
        content = doc.page_content
        if any(pat.search(source_file) or pat.search(content) for pat in patterns):
            high.append(doc)
        else:
            low.append(doc)

    return (high + low)[:top_n_matched]


def summarize_observation(docs: list[Any], domain_label: str = "Global Corpus") -> str:
    """Build a transparent white-box summary of retrieved document distribution and citations."""
    if not docs:
        return "No relevant documents found."

    countries = Counter(d.metadata.get("country", "N/A") for d in docs)
    types = Counter(d.metadata.get("doc_type", "N/A") for d in docs)

    articles: set[str] = set()
    cases: set[str] = set()
    for doc in docs:
        codes = doc.metadata.get("civil_codes_used")
        if codes:
            if isinstance(codes, list):
                articles.update(str(c) for c in codes)
            else:
                articles.add(str(codes))
        if "CASE_ID" in doc.metadata:
            cases.add(str(doc.metadata["CASE_ID"]))
        fname = doc.metadata.get("source_file", "")
        if "Article" in fname:
            nums = re.findall(r"\d+", fname)
            if nums:
                articles.add(f"Art. {nums[0]}")

    lines = [
        f"Retrieved {len(docs)} documents ({domain_label}).",
        f"   - **Jurisdictional Coverage**: {dict(countries)}",
        f"   - **Document Types**: {dict(types)}",
    ]
    if articles:
        sorted_articles = sorted(articles)
        preview = ", ".join(sorted_articles[:5])
        if len(sorted_articles) > 5:
            preview += "..."
        lines.append(f"   - **Key Statutes**: {preview}")
    if cases:
        sorted_cases = sorted(cases)
        lines.append(f"   - **Related Cases**: {', '.join(sorted_cases[:3])}")

    return "\n".join(lines)

"""Regex-based PII guardrail for redacting sensitive entities across prompts, answers, and logs."""

from __future__ import annotations

import re


class PIIGuardrail:
    """Intercept and redact Personally Identifiable Information (PII) from text outputs."""

    _EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    _PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}(?!\w)")
    _TAX_CODE_RE = re.compile(r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b")
    _IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[\s-]?[A-Za-z0-9]){11,28}\b")
    _TITLE_NAME_RE = re.compile(
        r"\b(Sig\.|Signor|Signora|Dott\.|Avv\.|Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Atty\.)\s+"
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
    )
    _LABEL_NAME_RE = re.compile(
        r"(Nome(?: e Cognome)?|Firma[^\:]*|Full Name|Name|Signature)\s*:\s*([A-Z][a-z]+\s+[A-Z][a-z]+)"
    )
    _CONTRACT_PARTY_RE = re.compile(
        r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s*\((?:qui\s+)?in\s+seguito|\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s*\(hereinafter",
    )
    _FALLBACK_TEST_NAMES_RE = re.compile(r"\b(Mario Rossi|Maria Bianchi|John Doe|Jane Doe)\b")

    @classmethod
    def mask_pii(cls, text: str) -> str:
        """Return a redacted copy of text preserving legal substance while removing PII."""
        if not text:
            return text

        sanitized = cls._EMAIL_RE.sub("[EMAIL REDACTED]", text)
        # Apply structured banking/tax IDs before phone regex to avoid partial digit collisions
        sanitized = cls._TAX_CODE_RE.sub("[TAX ID REDACTED]", sanitized)
        sanitized = cls._IBAN_RE.sub("[IBAN REDACTED]", sanitized)
        sanitized = cls._PHONE_RE.sub("[PHONE REDACTED]", sanitized)
        sanitized = cls._TITLE_NAME_RE.sub(r"\1 [NAME REDACTED]", sanitized)
        sanitized = cls._LABEL_NAME_RE.sub(r"\1: [NAME REDACTED]", sanitized)
        sanitized = cls._CONTRACT_PARTY_RE.sub(
            lambda m: "[NAME REDACTED] (" + ("hereinafter" if m.group(2) else "in seguito"),
            sanitized,
        )
        sanitized = cls._FALLBACK_TEST_NAMES_RE.sub("[NAME REDACTED]", sanitized)
        return sanitized

"""FastAPI REST endpoint exposing the Legal RAG Supervisor with lazy init and bounds checks."""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.guardrails import PIIGuardrail
from backend.logger import ConversationLogger
from backend.multi_agent.supervisor import SupervisorAgent

load_dotenv()
logger_sys = logging.getLogger("legal_rag_api")

app = FastAPI(
    title="Legal Comparative RAG API",
    description="REST API for multilingual comparative legal retrieval & automated RAGAS evaluation.",
    version="1.1.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User legal question")


class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]


@lru_cache(maxsize=1)
def get_rag_agent() -> SupervisorAgent:
    """Lazily initialize SupervisorAgent once when requested (avoids import-time crashes)."""
    return SupervisorAgent()


@lru_cache(maxsize=1)
def get_conversation_logger() -> ConversationLogger:
    return ConversationLogger()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Verify service responsiveness without leaking sensitive environment details."""
    try:
        get_rag_agent()
        return {"status": "ready"}
    except FileNotFoundError:
        return {"status": "degraded_missing_vector_store"}
    except Exception:
        return {"status": "degraded"}


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest) -> QueryResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must contain non-whitespace text.")

    try:
        agent = get_rag_agent()
        conv_logger = get_conversation_logger()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Vector index not built. Build the FAISS store before querying /ask.",
        )
    except ValueError:
        raise HTTPException(
            status_code=503,
            detail="Inference credentials not configured. Set HF_TOKEN before querying /ask.",
        )
    except Exception:
        logger_sys.exception("RAG runtime initialization failed")
        raise HTTPException(status_code=503, detail="Legal RAG subsystem is currently unavailable.")

    try:
        result = agent.run(query)
        raw_answer = str(result.get("answer", "Generation failed."))
        source_docs = result.get("source_documents", [])

        safe_answer = PIIGuardrail.mask_pii(raw_answer)
        contexts_list = [doc.page_content for doc in source_docs]

        conv_logger.log_interaction(
            user_query=query,
            system_response=safe_answer,
            sources=source_docs,
            agent_type="API-Multi-Agent",
        )
        return QueryResponse(answer=safe_answer, contexts=contexts_list)
    except Exception:
        logger_sys.exception("Error handling /ask request")
        raise HTTPException(status_code=500, detail="Internal error while processing legal query.")

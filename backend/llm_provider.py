"""HuggingFace Inference API wrappers for LangChain agents and rate-limited RAGAS judging."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

from huggingface_hub import InferenceClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from backend.configuration import RagConfig

# Global serializer preventing concurrent bursts against HuggingFace Inference endpoints during eval
_eval_lock = threading.Lock()
_eval_last_call = 0.0
_EVAL_MIN_DELAY = 5.0


class HuggingFaceChatWrapper(BaseChatModel):
    """LangChain adapter over `huggingface_hub.InferenceClient` with System-as-User framing."""

    client: InferenceClient
    model_name: str
    temperature: float
    max_tokens: int

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        hf_messages: list[dict[str, str]] = []
        system_instruction = ""

        for message in messages:
            if isinstance(message, SystemMessage):
                system_instruction += f"MANDATORY SYSTEM INSTRUCTIONS:\n{message.content}\n\n"
            elif isinstance(message, HumanMessage):
                full_content = f"{system_instruction}USER REQUEST:\n{message.content}"
                hf_messages.append({"role": "user", "content": full_content})
                system_instruction = ""
            elif isinstance(message, AIMessage):
                hf_messages.append({"role": "assistant", "content": str(message.content)})

        if system_instruction:
            hf_messages.append({"role": "user", "content": system_instruction})

        try:
            response = self.client.chat_completion(
                messages=hf_messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:
            content = f"LLM Error: {exc}"

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "huggingface-hub-wrapper"

    def set_run_config(self, run_config: Any) -> "HuggingFaceChatWrapper":
        self._ragas_run_config = run_config
        return self


class HuggingFaceEvalChatWrapper(BaseChatModel):
    """Rate-limited LangChain chat wrapper tailored for structured JSON judging in RAGAS."""

    client: InferenceClient
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 4096

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        hf_messages: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                hf_messages.append({"role": "system", "content": str(message.content)})
            elif isinstance(message, HumanMessage):
                hf_messages.append({"role": "user", "content": str(message.content)})
            elif isinstance(message, AIMessage):
                hf_messages.append({"role": "assistant", "content": str(message.content)})

        raw = self._call_api(hf_messages)
        content = self._strip_fences(raw)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _call_api(self, hf_messages: list[dict[str, str]]) -> str:
        global _eval_last_call
        max_retries = 8
        with _eval_lock:
            for attempt in range(max_retries):
                wait = _EVAL_MIN_DELAY - (time.time() - _eval_last_call)
                if wait > 0:
                    time.sleep(wait)
                try:
                    resp = self.client.chat_completion(
                        messages=hf_messages,
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=False,
                    )
                    _eval_last_call = time.time()
                    return resp.choices[0].message.content or ""
                except Exception as exc:
                    _eval_last_call = time.time()
                    err_str = str(exc)
                    if any(code in err_str for code in ("402", "429", "503", "504")):
                        backoff = min(5 * (attempt + 1), 30)
                        print(
                            f"[EVAL-WRAPPER] Rate/credit limit hit (attempt {attempt + 1}/{max_retries}); "
                            f"retrying in {backoff}s..."
                        )
                        time.sleep(backoff)
                        continue
                    return json.dumps({"error": err_str})

            return json.dumps({"error": "Max retries exhausted"})

    @staticmethod
    def _strip_fences(text: str) -> str:
        if not text:
            return text
        stripped = text.strip()
        fence = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", stripped)
        return fence.group(1).strip() if fence else stripped

    @property
    def _llm_type(self) -> str:
        return "huggingface-eval-wrapper"

    def set_run_config(self, run_config: Any) -> "HuggingFaceEvalChatWrapper":
        self._ragas_run_config = run_config
        return self


class LLMProvider:
    """Resolve `HF_TOKEN` from environment and instantiate dynamically configured chat models."""

    def __init__(self) -> None:
        self.api_key = os.getenv("HF_TOKEN") or ""
        if not self.api_key:
            print("[LLM PROVIDER WARNING] HF_TOKEN environment variable is not set.")

    def get_chat_model(self) -> HuggingFaceChatWrapper:
        """Build a chat wrapper reflecting live saved configuration parameters."""
        current_config = RagConfig.load()
        token = os.getenv("HF_TOKEN") or self.api_key
        if not token:
            raise ValueError("HF_TOKEN is required in environment variables (.env) to query HuggingFace Inference API.")
        client = InferenceClient(token=token)
        return HuggingFaceChatWrapper(
            client=client,
            model_name=str(current_config["LLM_REPO_ID"]),
            temperature=float(current_config["TEMPERATURE"]),
            max_tokens=int(current_config["MAX_NEW_TOKENS"]),
        )

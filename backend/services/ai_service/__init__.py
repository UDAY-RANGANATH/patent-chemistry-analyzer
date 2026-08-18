"""AI provider abstraction.

Provider chain (auto-detected from configured keys):
    Groq (OpenAI-compatible) > OpenAI > Google Gemini > Ollama (local)

The chemistry layer NEVER calls the LLM directly for chemical facts; it calls
`ChemistryService`. The LLM is used only for *interpretation* (extracting
entities, classifying reactions, drafting explanations) and every output is
validated afterwards by RDKit + chemistry APIs + the QC agent.
"""

from __future__ import annotations

import json
import logging
import re
import time
import threading
from typing import Any, Literal

from openai import OpenAI

from backend.config import settings
from backend.utils import retry

logger = logging.getLogger("pca.ai")

SystemMessage = dict[str, str]
UserMessage = dict[str, str]

# Global rate limiter: spaces LLM calls so we don't burst past Groq's TPM limit.
# Groq free tier: 8000 TPM. With ~1000 tokens/call, we need ~1.5s between calls.
_llm_lock = threading.Lock()
_last_llm_call = 0.0
_MIN_GAP = 1.5


def _throttle() -> None:
    """Wait if needed so we don't burst past the TPM limit."""
    global _last_llm_call
    with _llm_lock:
        now = time.monotonic()
        wait = _MIN_GAP - (now - _last_llm_call)
        if wait > 0:
            time.sleep(wait)
        _last_llm_call = time.monotonic()


class AIService:
    """Uniform chat / JSON interface over multiple providers."""

    def __init__(self) -> None:
        self._clients: dict[str, OpenAI] = {}

    @property
    def provider(self) -> str:
        return settings.active_ai_provider

    def _client(self, provider: str) -> OpenAI:
        if provider not in self._clients:
            if provider == "groq":
                self._clients[provider] = OpenAI(
                    api_key=settings.GROQ_API_KEY,
                    base_url=settings.GROQ_BASE_URL,
                    timeout=90.0,
                    max_retries=0,
                )
            elif provider == "openai":
                self._clients[provider] = OpenAI(
                    api_key=settings.OPENAI_API_KEY, timeout=90.0, max_retries=0
                )
            elif provider == "ollama":
                self._clients[provider] = OpenAI(
                    api_key="ollama",
                    base_url=f"{settings.OLLAMA_BASE_URL}/v1",
                    timeout=900.0,
                    max_retries=0,
                )
            else:
                raise ValueError(f"Unknown AI provider: {provider}")
        return self._clients[provider]

    def _model_for(self, provider: str) -> str:
        if provider == "groq":
            return settings.GROQ_MODEL
        if provider == "openai":
            return settings.OPENAI_MODEL
        if provider == "google":
            return settings.GOOGLE_MODEL
        return settings.OLLAMA_MODEL

    def chat(
        self,
        messages: list[SystemMessage | UserMessage],
        *,
        provider: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion and return the text (or raw JSON string)."""
        prov = provider or self.provider

        if prov == "google":
            return self._chat_google(messages, temperature, max_tokens, json_mode)

        client = self._client(prov)
        model = self._model_for(prov)
        kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            messages = list(messages)
            messages[-1] = {
                "role": messages[-1]["role"],
                "content": messages[-1]["content"] + "\n\nRespond ONLY with a valid JSON object. No markdown fences, no extra text.",
            }

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            _throttle()
            try:
                resp = client.chat.completions.create(messages=messages, **kwargs)  # type: ignore[call-arg]
                return resp.choices[0].message.content or ""
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or "rate_limit" in exc_str:
                    wait = _parse_wait(exc_str)
                    logger.warning("Rate limit (attempt %d/%d), waiting %.1fs — %.200s",
                                   attempt, max_attempts, wait, exc_str)
                    if attempt < max_attempts:
                        time.sleep(wait)
                        continue
                    raise
                if attempt < max_attempts:
                    delay = 1.5 * (2.0 ** (attempt - 1))
                    logger.warning("AI chat failed (attempt %d/%d), retry in %.1fs: %.200s",
                                   attempt, max_attempts, delay, exc_str)
                    time.sleep(delay)
                    continue
                logger.warning("AI chat failed (%s/%s): %.300s", prov, model, exc_str)
                raise

    def _chat_google(self, messages, temperature, max_tokens, json_mode):  # type: ignore[no-untyped-def]
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("langchain-google-genai not installed") from exc
        llm = ChatGoogleGenerativeAI(
            model=settings.GOOGLE_MODEL, google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature, max_output_tokens=max_tokens,
        )
        from langchain_core.messages import HumanMessage, SystemMessage as LCMsg

        lc_messages = [
            (LCMsg(content=m["content"]) if m["role"] == "system" else HumanMessage(content=m["content"]))
            for m in messages
        ]
        if json_mode:
            prompt = messages[-1]["content"]
            lc_messages[-1] = HumanMessage(content=f"{prompt}\n\nRespond ONLY with valid JSON. No markdown fences.")
        resp = llm.invoke(lc_messages)
        return str(resp.content)

    def chat_json(
        self,
        messages: list[SystemMessage | UserMessage],
        *,
        provider: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Chat then parse the reply as JSON. Raises ValueError if unparseable."""
        raw = self.chat(messages, provider=provider, temperature=temperature,
                        max_tokens=max_tokens, json_mode=True)
        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning("Unparseable JSON from AI: %.200s", raw)
            raise ValueError("AI returned non-JSON output")
        return parsed

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Text embeddings for RAG. Returns None when no embedding provider is available."""
        # Prefer local Ollama embeddings (privacy) — no key required.
        try:
            if settings.OLLAMA_BASE_URL:
                client = self._client("ollama")
                resp = client.embeddings.create(model="nomic-embed-text", input=texts)
                return [e.embedding for e in resp.data]
        except Exception as exc:  # noqa: BLE001
            logger.info("Ollama embeddings unavailable: %s", exc)
        return None


def _parse_wait(exc_str: str) -> float:
    """Extract 'try again in X.Xs' from a Groq 429 error message."""
    m = re.search(r"try again in (\d+\.?\d*)s", exc_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 0.5  # small buffer
    return 3.0  # default fallback


def _extract_json(raw: str) -> dict | None:
    """Best-effort JSON extraction (handles markdown fences and stray text)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def system_message(content: str) -> SystemMessage:
    return {"role": "system", "content": content}


def user_message(content: str) -> UserMessage:
    return {"role": "user", "content": content}

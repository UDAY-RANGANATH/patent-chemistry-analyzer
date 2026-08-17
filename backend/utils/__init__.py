"""Shared utilities: logging, retries, HTTP client, timestamps."""

from __future__ import annotations

import logging
import sys
import time
from functools import wraps
from typing import Callable, TypeVar

import httpx

logger = logging.getLogger("pca")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)

T = TypeVar("T")


def coerce_str(value: object, default: str = "") -> str:
    """Coerce an LLM field to a string, tolerating lists/dicts the model may emit."""
    if value is None:
        return default
    if isinstance(value, list):
        return "; ".join(str(x) for x in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    retryable_excs: tuple[type[Exception], ...] = (httpx.TransportError, httpx.TimeoutException),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a function with exponential backoff on transient errors."""

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except retryable_excs:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    delay = base_delay * (backoff ** (attempt - 1))
                    logger.info("%s transient error, retry %d in %.1fs", fn.__name__, attempt, delay)
                    time.sleep(delay)

        return wrapper

    return deco


def rate_limited(max_calls_per_sec: float = 2.0):
    """Minimal async-free rate limiter for chemistry APIs (thread-safe enough here)."""
    import threading

    lock = threading.Lock()
    last = [0.0]

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            with lock:
                now = time.monotonic()
                wait = (1.0 / max_calls_per_sec) - (now - last[0])
                if wait > 0:
                    time.sleep(wait)
                last[0] = time.monotonic()
            return fn(*args, **kwargs)

        return wrapper

    return deco


class HttpClient:
    """Thin httpx wrapper with timeouts and json handling."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        kwargs = {"timeout": timeout, "follow_redirects": True}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = httpx.Client(**kwargs)

    def get_json(self, path: str, params: dict | None = None) -> dict:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def get_text(self, path: str, params: dict | None = None) -> str:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.text

    def get_bytes(self, path: str, params: dict | None = None) -> bytes:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.content

    def close(self) -> None:
        self._client.close()


def chunk_text(text: str, size: int = 1200, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks on paragraph/sentence boundaries."""
    if len(text) <= size:
        return [text] if text.strip() else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start + size // 2, end),
                           text.rfind(". ", start + size // 2, end),
                           text.rfind("; ", start + size // 2, end))
            if boundary != -1:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]

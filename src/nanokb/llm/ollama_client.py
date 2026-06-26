"""Ollama LLM 客户端（方案 §3.4.1）。

complete 走 Ollama 原生 /api/chat（format="json" 原生 JSON mode）；
embed 走 /api/embed；
count_tokens 用 tiktoken cl100k_base 近似（离线可用，Ollama 模型 tokenizer 与 tiktoken 不同）。

通过 httpx 直接调用 Ollama REST API（httpx 经 openai/anthropic 传递依赖可用，无需额外 SDK）。
"""

from __future__ import annotations

import logging

import httpx
import tiktoken

from nanokb.llm.base import ResponseFormat
from nanokb.llm.throttle import RateLimiter

logger = logging.getLogger("nanokb")

_DEFAULT_TIMEOUT = 60.0


class OllamaClient:
    """Ollama provider 实现（原生 JSON mode）。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        embedding_model: str,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._embedding_model = embedding_model
        self._client = httpx.Client(timeout=_DEFAULT_TIMEOUT)
        # 线程安全节流器（方案 §3.2）：本地服务通常 interval=0（RateLimiter 无开销）；
        # 由 make_llm_client / make_embedding_client 注入；None 时无限流。
        self._rate_limiter = rate_limiter
        # 懒加载：避免构造期触发 tiktoken BPE 文件下载
        self._encoding: tiktoken.Encoding | None = None

    def _get_encoding(self) -> tiktoken.Encoding:
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def complete(
        self,
        system: str,
        user: str,
        response_format: ResponseFormat = "json",
        temperature: float = 0.0,
    ) -> str:
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if response_format == "json":
            payload["format"] = "json"
        r = self._client.post(f"{self._base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        message = data.get("message") or {}
        return message.get("content", "") or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for text in texts:
            r = self._client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._embedding_model, "input": text},
            )
            r.raise_for_status()
            data = r.json()
            embeddings = data.get("embeddings") or []
            if embeddings:
                results.append([float(v) for v in embeddings[0]])
        return results

    def count_tokens(self, text: str) -> int:
        # Ollama 模型 tokenizer 与 tiktoken 不同；离线场景用 cl100k_base 近似
        return len(self._get_encoding().encode(text))


__all__ = ["OllamaClient"]

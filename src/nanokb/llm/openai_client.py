"""OpenAI LLM 客户端（方案 §3.4.1）。

complete 走原生 response_format={"type":"json_object"} JSON mode；
embed 走 OpenAI embeddings API；
count_tokens 用 tiktoken.encoding_for_model 精确计数（未知模型降级 cl100k_base）。

速率限制（三重防护）：
1. SDK 内置 ``max_retries`` 指数退避（429/5xx 自动重试）；
2. 应用层 ``RateLimitError`` 补充重试（SDK 重试耗尽后再退避重试）；
3. 请求间最小间隔节流（注入的线程安全 ``RateLimiter``，方案 §3.2），控制 RPM
   避免触发限额。由 ``make_llm_client`` / ``make_embedding_client`` 创建进程级
   共享 / 独立实例注入；``rate_limiter=None`` 时无限流（向后兼容旧默认行为）。
"""

from __future__ import annotations

import logging
import time

import tiktoken
from openai import OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam

from nanokb.llm.base import ResponseFormat
from nanokb.llm.throttle import RateLimiter

logger = logging.getLogger("nanokb")

#: 单次 embeddings 请求的 input 数组上限。OpenAI 兼容端点（如智谱 GLM
#: embedding-3）限制 input 不得超过 64 条（HTTP 400 错误码 1214：
#: input数组最大不得超过64条）；取 64 兼容所有当前支持的 provider。
_EMBED_INPUT_MAX = 64


class OpenAIClient:
    """OpenAI provider 实现。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        embedding_model: str,
        base_url: str | None = None,
        max_retries: int = 6,
        rate_limiter: RateLimiter | None = None,
        rate_limit_retries: int = 3,
    ) -> None:
        # 仅在显式指定 base_url 时透传，未指定时由 SDK 读 OPENAI_BASE_URL 环境变量或用默认端点
        self._client = (
            OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)
            if base_url
            else OpenAI(api_key=api_key, max_retries=max_retries)
        )
        self._model = model
        self._embedding_model = embedding_model
        # 线程安全节流器（方案 §3.2）：由工厂注入进程级共享实例；None 时无限流，
        # 等价于原 request_interval=0.0 默认行为（保留直接构造的向后兼容）。
        self._rate_limiter = rate_limiter
        self._rate_limit_retries = rate_limit_retries
        # 懒加载：避免构造期触发 tiktoken BPE 文件下载（离线/无缓存环境下构造仍可用）
        self._encoding: tiktoken.Encoding | None = None

    def _get_encoding(self) -> tiktoken.Encoding:
        if self._encoding is None:
            try:
                self._encoding = tiktoken.encoding_for_model(self._model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    @staticmethod
    def _compute_backoff(attempt: int) -> float:
        """指数退避：10s, 20s, 40s ...（上限 120s）。

        SDK 重试耗尽说明限流窗口未恢复，需要更长等待。
        """
        return float(min(10.0 * (2**attempt), 120.0))

    def complete(
        self,
        system: str,
        user: str,
        response_format: ResponseFormat = "json",
        temperature: float = 0.0,
    ) -> str:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        use_json = response_format == "json"

        total_attempts = self._rate_limit_retries + 1
        for attempt in range(total_attempts):
            # 节流保留在 for attempt 循环内部（与原 _throttle 位置一致）：首次请求
            # 与每次 RateLimitError 退避后的重试请求都先 acquire（方案 §3.2.3）。
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            try:
                if use_json:
                    resp = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                else:
                    resp = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=temperature,
                    )
            except RateLimitError:
                if attempt >= self._rate_limit_retries:
                    logger.error(
                        "rate limit exhausted after %d app-level retries",
                        self._rate_limit_retries,
                    )
                    raise
                backoff = self._compute_backoff(attempt)
                logger.warning(
                    "rate limited (attempt %d/%d), backing off %.1fs",
                    attempt + 1,
                    total_attempts,
                    backoff,
                )
                time.sleep(backoff)
                continue
            return resp.choices[0].message.content or ""

        raise RuntimeError("unreachable")  # pragma: no cover

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_INPUT_MAX):
            chunk = texts[start : start + _EMBED_INPUT_MAX]
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            resp = self._client.embeddings.create(
                model=self._embedding_model,
                input=chunk,
            )
            out.extend(item.embedding for item in resp.data)
        return out

    def count_tokens(self, text: str) -> int:
        return len(self._get_encoding().encode(text))


__all__ = ["OpenAIClient"]

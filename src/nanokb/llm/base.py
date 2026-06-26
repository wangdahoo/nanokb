"""LLM 客户端抽象层（方案 §3.4.1）。

定义 LLMClient Protocol（complete/embed/count_tokens）、EmbeddingClient Protocol
（embed-only，生文与向量解耦）、make_llm_client 工厂（按 settings.llm_provider 分发，
启动校验 API key，缺失则 exit code 2）、make_embedding_client 工厂（按
settings.embedding_provider 分发独立 embedding 客户端）以及 parse_json_loose
容错解析（提取首个平衡 {...} 子串，兼容 markdown 围栏/前言噪声）。

provider 实现差异：
- OpenAIClient：response_format={"type":"json_object"} 原生 JSON mode；tiktoken 精确计数。
- AnthropicClient：tool-use（emit_json 工具强制 JSON 通道，Opt #4 v3：仅保证通道非 schema 校验），
  降级 prompt + parse_json_loose。
- OllamaClient：format="json" 原生 JSON mode。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Protocol, runtime_checkable

from nanokb.config import Settings
from nanokb.llm.throttle import RateLimiter

logger = logging.getLogger("nanokb")

ResponseFormat = Literal["text", "json"]


@runtime_checkable
class LLMClient(Protocol):
    """LLM 客户端统一契约。

    complete/embed/count_tokens 三方法。response_format='json' 时各 provider 走
    原生 JSON 通道或 tool-use 通道，返回可被 json.loads/parse_json_loose 解析的字符串。
    """

    def complete(
        self,
        system: str,
        user: str,
        response_format: ResponseFormat = "json",
        temperature: float = 0.0,
    ) -> str:
        """生成补全。"""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """生成向量嵌入。"""
        ...

    def count_tokens(self, text: str) -> int:
        """精确 token 计数（OpenAI 用 tiktoken；其余 provider 用近似 tokenizer）。"""
        ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """向量嵌入客户端契约（embed-only）。

    生文（chat）与向量（embedding）解耦后的独立协议：仅要求 ``embed`` 方法。
    :class:`LLMClient` 是其结构超集（实现了 embed），因此任何 ``LLMClient``
    实例都满足 ``EmbeddingClient``，保证向后兼容（未配置独立 embedding 端点时，
    流水线可直接复用 chat client 做 embedding）。

    实际的 embedding-only 客户端由 :func:`make_embedding_client` 按
    ``embedding_provider`` 构造，与生文端点完全分离（如生文走 DeepSeek、
    embedding 走智谱 GLM embedding-3 或本地 Ollama）。
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """生成向量嵌入。"""
        ...


def parse_json_loose(raw: str) -> dict[str, Any] | None:
    """容错 JSON 解析：先尝试直接解析，失败则提取首个平衡 {...} 子串。

    兼容 LLM 常见噪声：markdown 围栏（```json ... ```）、前言文本、尾部多余字符。
    扫描为引号感知（字符串内部的花括号不会被误计），确保嵌套结构正确闭合。
    返回首个解析成功的 dict；非 dict（如数组）或无法解析时返回 None。
    """
    text = raw.strip()
    # 直接解析
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        return obj

    # 降级：定位首个平衡 {...}（引号感知，避免误判字符串内花括号）
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def make_llm_client(settings: Settings) -> LLMClient:
    """按 settings.llm_provider 构造 LLMClient。

    启动期校验 API key：openai/anthropic 缺 key 则记 error 并 exit code 2
    （不进入半工作状态，方案 §3.6）。ollama 无需 key（本地服务）。
    """
    provider = settings.llm_provider
    # chat 端进程级共享 RateLimiter（方案 §3.2.4）：三 provider 共享同一实例，
    # 保证全局 RPM 语义。interval<=0 时无锁快速返回（ollama / 无限流场景零开销）。
    chat_rate_limiter = RateLimiter(settings.llm_request_interval)
    if provider == "openai":
        key = settings.openai_api_key
        if key is None or not key.get_secret_value():
            logger.error("openai_api_key is required when llm_provider='openai'; refusing to start")
            raise SystemExit(2)
        from nanokb.llm.openai_client import OpenAIClient

        return OpenAIClient(
            api_key=key.get_secret_value(),
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
            base_url=settings.openai_base_url,
            max_retries=settings.llm_max_retries,
            rate_limiter=chat_rate_limiter,
            rate_limit_retries=settings.llm_rate_limit_retries,
        )

    if provider == "anthropic":
        key = settings.anthropic_api_key
        if key is None or not key.get_secret_value():
            logger.error(
                "anthropic_api_key is required when llm_provider='anthropic'; refusing to start"
            )
            raise SystemExit(2)
        from nanokb.llm.anthropic_client import AnthropicClient

        return AnthropicClient(
            api_key=key.get_secret_value(),
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
            openai_api_key=(
                settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
            ),
            openai_base_url=settings.openai_base_url,
            rate_limiter=chat_rate_limiter,
        )

    if provider == "ollama":
        from nanokb.llm.ollama_client import OllamaClient

        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
            rate_limiter=chat_rate_limiter,
        )

    # Literal 类型保证不会到达；防御性 exit（未知 provider）
    logger.error("unknown llm_provider: %r", provider)
    raise SystemExit(2)


def make_embedding_client(settings: Settings) -> EmbeddingClient:
    """按 settings.embedding_provider 构造独立的 embedding-only 客户端。

    与生文（``make_llm_client``）解耦：embedding 可走不同厂商 / 端点 / key。
    场景示例：生文用 DeepSeek、embedding 用智谱 GLM ``embedding-3`` 或本地 Ollama。

    provider 分发：
    - ``ollama``：走 ``ollama_base_url`` 的 ``/api/embed``（无需 key）。
    - ``openai``（兼容，含智谱 GLM / DeepSeek 等）：key 取
      ``embedding_api_key``，缺失回退 ``openai_api_key``；base_url 取
      ``embedding_base_url``，缺失回退 ``openai_base_url``。两者均缺则 exit 2。

    向后兼容：未显式配置 embedding 专用字段时，pipeline 层的 ``_resolve_embedder``
    会直接复用 chat client，无需调用本工厂；仅在用户显式声明解耦配置时触发。
    """
    provider = settings.embedding_provider
    # embedding 端独立 RateLimiter（方案 §3.2.4 Medium #1）：与 chat 解耦——embedding
    # 与 chat 多为不同端点/厂商，限额不同，应独立节流。注意：``_resolve_embedder``
    # 复用 chat_llm 做 embedding 的情形不经过本工厂，天然共享 chat 的 RateLimiter
    # （embedding 在阶段 B，阶段 A 已 join，无并发竞争）。
    embed_rate_limiter = RateLimiter(settings.llm_request_interval)
    if provider == "ollama":
        from nanokb.llm.ollama_client import OllamaClient

        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
            rate_limiter=embed_rate_limiter,
        )

    if provider == "openai":
        key = settings.embedding_api_key or settings.openai_api_key
        if key is None or not key.get_secret_value():
            logger.error(
                "embedding requires an api key (embedding_api_key or openai_api_key) "
                "when embedding_provider='openai'; refusing to start"
            )
            raise SystemExit(2)
        base_url = settings.embedding_base_url or settings.openai_base_url
        from nanokb.llm.openai_client import OpenAIClient

        return OpenAIClient(
            api_key=key.get_secret_value(),
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
            base_url=base_url,
            max_retries=settings.llm_max_retries,
            rate_limiter=embed_rate_limiter,
            rate_limit_retries=settings.llm_rate_limit_retries,
        )

    # Literal 类型保证不会到达；防御性 exit（未知 provider）
    logger.error("unknown embedding_provider: %r", provider)
    raise SystemExit(2)


__all__ = [
    "EmbeddingClient",
    "LLMClient",
    "ResponseFormat",
    "make_embedding_client",
    "make_llm_client",
    "parse_json_loose",
]

"""LLM 客户端抽象层（方案 §3.4.1）。

定义 LLMClient Protocol（complete/embed/count_tokens）、make_llm_client 工厂
（按 settings.llm_provider 分发，启动校验 API key，缺失则 exit code 2）以及
parse_json_loose 容错解析（提取首个平衡 {...} 子串，兼容 markdown 围栏/前言噪声）。

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
    if provider == "openai":
        key = settings.openai_api_key
        if key is None or not key.get_secret_value():
            logger.error(
                "openai_api_key is required when llm_provider='openai'; refusing to start"
            )
            raise SystemExit(2)
        from nanokb.llm.openai_client import OpenAIClient

        return OpenAIClient(
            api_key=key.get_secret_value(),
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
            base_url=settings.openai_base_url,
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
                settings.openai_api_key.get_secret_value()
                if settings.openai_api_key
                else None
            ),
            openai_base_url=settings.openai_base_url,
        )

    if provider == "ollama":
        from nanokb.llm.ollama_client import OllamaClient

        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
        )

    # Literal 类型保证不会到达；防御性 exit（未知 provider）
    logger.error("unknown llm_provider: %r", provider)
    raise SystemExit(2)


__all__ = ["LLMClient", "ResponseFormat", "make_llm_client", "parse_json_loose"]

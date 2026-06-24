"""LLM 客户端抽象层。

方案 §3.4.1：Protocol + make_llm_client / make_embedding_client 工厂 + parse_json_loose 容错。
具体 provider 实现（OpenAI/Anthropic/Ollama）在各自子模块，按需导入以保持本包轻量。
"""

from __future__ import annotations

from nanokb.llm.base import (
    EmbeddingClient,
    LLMClient,
    ResponseFormat,
    make_embedding_client,
    make_llm_client,
    parse_json_loose,
)

__all__ = [
    "EmbeddingClient",
    "LLMClient",
    "ResponseFormat",
    "make_embedding_client",
    "make_llm_client",
    "parse_json_loose",
]

"""Anthropic LLM 客户端（方案 §3.4.1，Opt #4 v3）。

JSON 通道：tool-use emit_json 工具强制返回合法 JSON。
- tools=[{"name":"emit_json","input_schema":{"type":"object","properties":{}}}]，
  tool_choice={"type":"tool","name":"emit_json"} 强制走工具通道。
- Opt #4 v3：空 input_schema 仅保证通道为合法 JSON（避免 markdown 包裹），
  不构成 schema 级结构校验；内部结构由 prompt 指令 + parse_json_loose + pydantic 校验共同保证。
- 从 tool_use.input（dict）取 JSON 并 json.dumps 为字符串，与 Protocol 字符串契约一致。
- 未出现 tool_use 块时降级为文本输出（上游配合 parse_json_loose 容错）。

embed：Anthropic 无 embeddings API，委派给 OpenAI（当 openai_api_key 可用时）。
count_tokens：Anthropic 未公开离线 tokenizer，用 tiktoken cl100k_base 近似。
"""

from __future__ import annotations

import json
import logging

import tiktoken
from anthropic import Anthropic
from anthropic.types import Message
from openai import OpenAI

from nanokb.llm.base import ResponseFormat
from nanokb.llm.throttle import RateLimiter

logger = logging.getLogger("nanokb")

_MAX_TOKENS = 4096

#: 单次 embeddings 请求的 input 数组上限（与 OpenAIClient 一致；详见
#: openai_client._EMBED_INPUT_MAX 的说明）。
_EMBED_INPUT_MAX = 64


class AnthropicClient:
    """Anthropic provider 实现（tool-use JSON 通道）。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        embedding_model: str,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._embedding_model = embedding_model
        self._openai_api_key = openai_api_key
        self._openai_base_url = openai_base_url
        # 线程安全节流器（方案 §3.2）：由 make_llm_client 注入共享实例；None 时无限流。
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
        if response_format == "json":
            return self._complete_json(system, user, temperature)
        resp = self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=_MAX_TOKENS,
        )
        return self._extract_text(resp)

    def _complete_json(self, system: str, user: str, temperature: float) -> str:
        """tool-use 通道：emit_json 强制 JSON 返回。

        Opt #4 v3：空 input_schema 仅保证通道为合法 JSON，不校验内部结构。
        """
        resp = self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": "emit_json",
                    "description": "Emit the structured JSON result.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice={"type": "tool", "name": "emit_json"},
            temperature=temperature,
            max_tokens=_MAX_TOKENS,
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == "emit_json":
                # input 已是 dict；json.dumps 为字符串以统一契约
                return json.dumps(block.input)
        # 未出现 tool_use 块 → 退回文本（上游配合 parse_json_loose 容错）
        logger.warning(
            "anthropic tool-use channel returned no tool_use block; falling back to text"
        )
        return self._extract_text(resp)

    def _extract_text(self, resp: Message) -> str:
        parts: list[str] = []
        for block in resp.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._openai_api_key:
            logger.error(
                "anthropic has no embeddings API; configure embedding_provider with openai_api_key"
            )
            raise RuntimeError(
                "AnthropicClient.embed requires openai_api_key (Anthropic has no embeddings API)"
            )
        client = (
            OpenAI(api_key=self._openai_api_key, base_url=self._openai_base_url)
            if self._openai_base_url
            else OpenAI(api_key=self._openai_api_key)
        )
        out: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_INPUT_MAX):
            chunk = texts[start : start + _EMBED_INPUT_MAX]
            resp = client.embeddings.create(model=self._embedding_model, input=chunk)
            out.extend(item.embedding for item in resp.data)
        return out

    def count_tokens(self, text: str) -> int:
        # Anthropic 未公开离线 tokenizer；cl100k_base 为最佳可用近似
        return len(self._get_encoding().encode(text))


__all__ = ["AnthropicClient"]

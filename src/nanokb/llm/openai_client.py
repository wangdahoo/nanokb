"""OpenAI LLM 客户端（方案 §3.4.1）。

complete 走原生 response_format={"type":"json_object"} JSON mode；
embed 走 OpenAI embeddings API；
count_tokens 用 tiktoken.encoding_for_model 精确计数（未知模型降级 cl100k_base）。
"""

from __future__ import annotations

import tiktoken
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from nanokb.llm.base import ResponseFormat


class OpenAIClient:
    """OpenAI provider 实现。"""

    def __init__(self, api_key: str, model: str, embedding_model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._embedding_model = embedding_model
        # 懒加载：避免构造期触发 tiktoken BPE 文件下载（离线/无缓存环境下构造仍可用）
        self._encoding: tiktoken.Encoding | None = None

    def _get_encoding(self) -> tiktoken.Encoding:
        if self._encoding is None:
            try:
                self._encoding = tiktoken.encoding_for_model(self._model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

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
        if response_format == "json":
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
        return resp.choices[0].message.content or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(
            model=self._embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def count_tokens(self, text: str) -> int:
        return len(self._get_encoding().encode(text))


__all__ = ["OpenAIClient"]

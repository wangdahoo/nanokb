"""embed() 输入分块回归测试。

某些 OpenAI 兼容 embedding 端点（如智谱 GLM embedding-3）限制单次 input 数组
不得超过 64 条（HTTP 400 错误码 1214：input数组最大不得超过64条）。
embed() 须自行按 provider 上限分块，避免上层调用方（如 vector_store.index_nodes
按 EMBED_BATCH_SIZE 攒批）传入大数组时触发 400 BadRequest。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from nanokb.llm.openai_client import OpenAIClient


def _fake_embeddings_response(texts: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=[0.0, 0.0]) for _ in texts]
    return resp


def test_embed_chunks_input_within_provider_limit() -> None:
    """embed(70 texts) 应拆成多次 create 调用，每次 input ≤ 64，结果数量守恒。"""
    client = OpenAIClient(api_key="sk-fake", model="m", embedding_model="embedding-3")

    call_sizes: list[int] = []

    def _create(**kwargs):  # type: ignore[no-untyped-def]
        inputs = kwargs["input"]
        call_sizes.append(len(inputs))
        return _fake_embeddings_response(inputs)

    client._client.embeddings.create = MagicMock(side_effect=_create)

    texts = [f"text-{i}" for i in range(70)]
    embeddings = client.embed(texts)

    assert len(embeddings) == 70
    assert max(call_sizes) <= 64
    assert call_sizes == [64, 6]


def test_embed_small_input_single_call() -> None:
    """≤ 上限的输入仍走单次调用，无多余分块。"""
    client = OpenAIClient(api_key="sk-fake", model="m", embedding_model="embedding-3")
    call_sizes: list[int] = []

    def _create(**kwargs):  # type: ignore[no-untyped-def]
        inputs = kwargs["input"]
        call_sizes.append(len(inputs))
        return _fake_embeddings_response(inputs)

    client._client.embeddings.create = MagicMock(side_effect=_create)

    embeddings = client.embed([f"t{i}" for i in range(10)])

    assert len(embeddings) == 10
    assert call_sizes == [10]


def test_embed_empty_returns_empty() -> None:
    """空输入不触发任何 API 调用。"""
    client = OpenAIClient(api_key="sk-fake", model="m", embedding_model="embedding-3")
    client._client.embeddings.create = MagicMock()

    assert client.embed([]) == []
    client._client.embeddings.create.assert_not_called()

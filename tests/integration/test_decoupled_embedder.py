"""生文与向量解耦集成测试。

覆盖 ``compile()`` 与 ``answer_query()`` 端到端使用独立 embedder（与 chat llm
分离）的完整链路：compile 时用 embedder 做 index_nodes 维度探测 + 向量索引，
answer_query 时用同一 embedder 做 query embedding 召回。

验证核心不变量：
- chat llm 的 ``embed`` 不被调用（生文与向量真正解耦）。
- embedder 的 ``embed`` 被调用（探测维度 + 索引 + 查询 embedding）。
- 端到端 compile → answer_query 产出来源含 vector 的命中。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanokb import pipeline
from nanokb.config import Settings


class _CallTrackingEmbedder:
    """仅实现 ``embed`` 的 embedding-only 客户端（满足 EmbeddingClient 协议）。

    记录所有 embed 调用，用于断言解耦后 chat llm 的 embed 不被触发。
    """

    def __init__(self, embedding_dim: int = 8) -> None:
        self._dim = embedding_dim
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[float((hash(t) >> i) & 0xFF) / 255.0 for i in range(self._dim)] for t in texts]


class FakeChatLLM:
    """模拟 chat LLM：complete 按序消费；embed 抛错（生文不应做 embedding）。"""

    def __init__(self, responses: list[str] | None = None, embedding_dim: int = 8) -> None:
        self._responses = list(responses) if responses else []
        self._dim = embedding_dim
        self.complete_calls: list[dict[str, Any]] = []
        self.embed_calls: list[list[str]] = []

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        self.complete_calls.append({"user": user, "response_format": response_format})
        if self._responses:
            return self._responses.pop(0)
        return ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[float((hash(t) >> i) & 0xFF) / 255.0 for i in range(self._dim)] for t in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _extract_response() -> str:
    return json.dumps(
        {
            "triples": [
                {
                    "head": "Transformer",
                    "relation": "uses",
                    "tail": "Attention",
                    "confidence": "EXTRACTED",
                },
            ],
            "concepts": [
                {"name": "Transformer", "description": "A sequence model.", "node_type": "model"},
                {
                    "name": "Attention",
                    "description": "A focus mechanism.",
                    "node_type": "mechanism",
                },
            ],
        }
    )


def _ner(entities: list[str]) -> str:
    return json.dumps({"entities": entities})


# ══════════════════════════════════════════════════════════════════════
# compile: decoupled embedder indexes vectors, chat llm does extraction
# ══════════════════════════════════════════════════════════════════════


def test_compile_uses_decoupled_embedder_for_indexing(tmp_path: Path) -> None:
    """compile 时 index_nodes 用 embedder 做向量索引，chat llm 不调用 embed。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    settings = Settings(
        raw_dir=raw_dir,
        out_dir=out_dir,
        enable_vector_recall=True,
    )
    chat_llm = FakeChatLLM(responses=[_extract_response()])
    embedder = _CallTrackingEmbedder(embedding_dim=8)

    result = pipeline.compile(settings, llm=chat_llm, embedding_client=embedder)

    assert result.extracted_count == 1

    # chat llm 只做了抽取（complete），不应被调用 embed
    assert chat_llm.embed_calls == []

    # embedder 至少被调用：1) 维度探测 + 2) index_nodes
    assert len(embedder.embed_calls) >= 2

    # 向量库已生成
    assert (out_dir / "chroma").exists()


# ══════════════════════════════════════════════════════════════════════
# answer_query: decoupled embedder used for query embedding
# ══════════════════════════════════════════════════════════════════════


def test_answer_query_uses_decoupled_embedder_for_recall(tmp_path: Path) -> None:
    """answer_query(ask) 用 embedder 做 query embedding，chat llm 仅做 generate。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    settings = Settings(
        raw_dir=raw_dir,
        out_dir=out_dir,
        enable_vector_recall=True,
    )

    chat_llm = FakeChatLLM(responses=[_extract_response()])
    embedder = _CallTrackingEmbedder(embedding_dim=8)
    pipeline.compile(settings, llm=chat_llm, embedding_client=embedder)

    # 重置调用记录
    chat_llm.embed_calls.clear()
    embedder.embed_calls.clear()

    # ask 模式仅走向量路：1 个 query embed + 1 个 generate complete
    gen = "Transformer uses attention mechanism^[doc.md]."
    qa_llm = FakeChatLLM(responses=[gen])
    qa_embedder = _CallTrackingEmbedder(embedding_dim=8)

    result = pipeline.answer_query(
        settings,
        "Transformer 如何工作？",
        mode="ask",
        llm=qa_llm,
        embedding_client=qa_embedder,
    )

    # 命中含 vector 来源
    sources = {h.source for h in result.hits}
    assert "vector" in sources, f"vector route missing; sources={sources}"

    # chat llm 的 embed 不被调用（仅 generate 的 complete）
    assert qa_llm.embed_calls == []

    # embedder 至少被调用 1 次（query embedding）
    assert len(qa_embedder.embed_calls) >= 1


# ══════════════════════════════════════════════════════════════════════
# Full decoupled round-trip: separate provider config end-to-end
# ══════════════════════════════════════════════════════════════════════


def test_decoupled_embedder_roundtrip_different_dim(tmp_path: Path) -> None:
    """compile + answer_query 使用不同维度的独立 embedder，端到端一致。

    验证 embedder 维度（如 16）正确透传到 VectorStore 元数据，且 answer_query
    时用同一维度 embedder 召回，不产生维度不匹配。
    """
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    embedding_dim = 16  # 故意用非默认维度
    settings = Settings(
        raw_dir=raw_dir,
        out_dir=out_dir,
        enable_vector_recall=True,
    )

    chat_llm = FakeChatLLM(responses=[_extract_response()])
    embedder = _CallTrackingEmbedder(embedding_dim=embedding_dim)
    pipeline.compile(settings, llm=chat_llm, embedding_client=embedder)

    # 向量库元数据维度应与 embedder 一致
    from nanokb.index.vector_store import EMBEDDING_DIM_KEY, VectorStore

    vs = VectorStore(out_dir / "chroma", settings.embedding_model, embedding_dim)
    actual_dim = vs._collection.metadata.get(EMBEDDING_DIM_KEY)
    assert actual_dim == embedding_dim

    # answer_query 用同一 embedder
    ner = _ner(["Transformer"])
    gen = "answer^[doc.md]"
    qa_llm = FakeChatLLM(responses=[ner, gen])
    qa_embedder = _CallTrackingEmbedder(embedding_dim=embedding_dim)

    result = pipeline.answer_query(
        settings,
        "Transformer",
        mode="query",
        llm=qa_llm,
        embedding_client=qa_embedder,
    )

    sources = {h.source for h in result.hits}
    assert "vector" in sources

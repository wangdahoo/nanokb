"""VectorStore 单测（方案 §3.4.4 + §3.5.1 step 7，Feature s1-feat-011）。

覆盖 3 条验收标准：
- AC #1：图谱节点含 description → index_nodes → ChromaDB 写入对应向量，
  id=f"{source_file}::{node}"。
- AC #2：ChromaDB collection metadata embedding_dim=1536，settings 换 768 维模型 →
  _ensure_collection drop 重建（Medium #7）。
- AC #3：同文件二次 index_nodes → 向量不重复累积（upsert 主键幂等，Medium #9）。

另覆盖：
- delete_by_source（Severe #1 + Medium #2 先清后建）。
- search 向量召回。
- 空描述节点跳过 + WARNING（Opt #2 v3）。
- VectorStore 满足 pipeline.VectorStoreBackend 协议。

全部离线，用 tmp_path 隔离 ChromaDB，FakeLLMClient 返回可控维度 embedding。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from nanokb import pipeline
from nanokb.index.vector_store import (
    COLLECTION_NAME,
    EMBEDDING_DIM_KEY,
    EMBEDDING_MODEL_KEY,
    VectorStore,
)

# ── 测试 doubles ─────────────────────────────────────────────────────


class FakeLLMClient:
    """模拟 LLMClient，返回固定维度 embedding。"""

    def __init__(self, embedding_dim: int = 8) -> None:
        self._dim = embedding_dim
        self.embed_calls: int = 0
        self.embedded_texts: list[str] = []

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        return json.dumps({"triples": [], "concepts": []})

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        self.embedded_texts.extend(texts)
        return [[float(i) / 10.0 for i in range(self._dim)] for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _make_graph(
    nodes: dict[str, dict[str, Any]] | None = None,
) -> nx.MultiDiGraph:
    """构造小型图谱（节点带 description + source_file）。"""
    g = nx.MultiDiGraph()
    if nodes is None:
        nodes = {
            "Transformer": {
                "description": "A neural network architecture.",
                "source_file": "doc.md",
                "node_type": "concept",
            },
            "Attention": {
                "description": "A mechanism for focusing on relevant input.",
                "source_file": "doc.md",
                "node_type": "concept",
            },
        }
    for node, data in nodes.items():
        g.add_node(node, **data)
    return g


# ── AC #1：index_nodes 写入向量，id=f"{source_file}::{node}" ──────────


def test_index_nodes_writes_vectors_with_correct_ids(tmp_path: Path) -> None:
    """AC #1：index_nodes 为节点 description 生成 embedding，id=f"{source_file}::{node}"。"""
    llm = FakeLLMClient(embedding_dim=8)
    vs = VectorStore(tmp_path / "chroma", "text-embedding-3-small", 8)
    graph = _make_graph()

    vs.index_nodes(graph, llm)

    # ChromaDB collection 含 2 条向量
    assert vs.count() == 2
    ids = set(vs.list_ids())
    assert ids == {"doc.md::Transformer", "doc.md::Attention"}

    # embed 被调用（为 description 生成向量）
    assert llm.embed_calls >= 1


def test_index_nodes_stores_source_file_metadata(tmp_path: Path) -> None:
    """index_nodes 写入的 metadata 含 source_file + node（供 delete_by_source 使用）。"""
    llm = FakeLLMClient(embedding_dim=4)
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    graph = _make_graph()

    vs.index_nodes(graph, llm)

    # 通过 chromadb API 验证 metadata
    col = vs._client.get_collection(COLLECTION_NAME)
    result = col.get(include=["metadatas"])
    metadatas = result.get("metadatas", [])
    assert len(metadatas) == 2
    source_files = {m["source_file"] for m in metadatas}
    assert source_files == {"doc.md"}


# ── AC #2：_ensure_collection 维度不匹配 drop 重建（Medium #7） ───────


def test_ensure_collection_dim_mismatch_drops_and_rebuilds(tmp_path: Path) -> None:
    """AC #2：collection metadata embedding_dim=8，换 4 维模型 → drop 重建。"""
    chroma_path = tmp_path / "chroma"

    # 第一次：dim=8，写入向量
    llm8 = FakeLLMClient(embedding_dim=8)
    vs1 = VectorStore(chroma_path, "model-8d", 8)
    vs1.index_nodes(_make_graph(), llm8)
    assert vs1.count() == 2

    # 验证 collection metadata 记录 dim=8
    col_meta = vs1._client.get_collection(COLLECTION_NAME).metadata
    assert col_meta is not None
    assert col_meta[EMBEDDING_DIM_KEY] == 8
    assert col_meta[EMBEDDING_MODEL_KEY] == "model-8d"

    # 第二次：dim=4（模型切换），_ensure_collection 检测不匹配 → drop 重建
    vs2 = VectorStore(chroma_path, "model-4d", 4)

    # 旧向量已被 drop 重建（count 归零）
    assert vs2.count() == 0

    # 新 collection metadata 记录 dim=4
    col_meta2 = vs2._client.get_collection(COLLECTION_NAME).metadata
    assert col_meta2 is not None
    assert col_meta2[EMBEDDING_DIM_KEY] == 4
    assert col_meta2[EMBEDDING_MODEL_KEY] == "model-4d"


def test_ensure_collection_dim_match_keeps_data(tmp_path: Path) -> None:
    """维度匹配时 collection 保留（不 drop）。"""
    chroma_path = tmp_path / "chroma"

    llm = FakeLLMClient(embedding_dim=8)
    vs1 = VectorStore(chroma_path, "model", 8)
    vs1.index_nodes(_make_graph(), llm)
    assert vs1.count() == 2

    # 第二次：同维度 → 保留旧数据
    vs2 = VectorStore(chroma_path, "model", 8)
    assert vs2.count() == 2  # 旧向量仍在


def test_ensure_collection_creates_when_missing(tmp_path: Path) -> None:
    """collection 不存在时自动创建。"""
    chroma_path = tmp_path / "chroma"
    vs = VectorStore(chroma_path, "test-model", 8)
    # 构造期已 _ensure_collection，collection 存在且为空
    assert vs.count() == 0
    # collection metadata 正确
    col = vs._client.get_collection(COLLECTION_NAME)
    assert col.metadata is not None
    assert col.metadata[EMBEDDING_DIM_KEY] == 8


# ── AC #3：二次 index_nodes 幂等不累积（Medium #9） ──────────────────


def test_index_nodes_idempotent_no_duplication(tmp_path: Path) -> None:
    """AC #3：同文件二次 index_nodes → 向量不重复累积（upsert 主键幂等）。"""
    llm = FakeLLMClient(embedding_dim=8)
    vs = VectorStore(tmp_path / "chroma", "test-model", 8)
    graph = _make_graph()

    vs.index_nodes(graph, llm)
    assert vs.count() == 2

    # 二次 index 同图 → upsert 幂等，count 不变
    vs.index_nodes(graph, llm)
    assert vs.count() == 2

    # id 列表不变（无重复 id）
    ids = vs.list_ids()
    assert len(ids) == 2
    assert set(ids) == {"doc.md::Transformer", "doc.md::Attention"}


def test_index_nodes_upsert_updates_description(tmp_path: Path) -> None:
    """upsert 主键幂等：同 id 的 description 变更被覆盖（不新增条目）。"""
    llm = FakeLLMClient(embedding_dim=4)
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    graph = _make_graph()

    vs.index_nodes(graph, llm)
    assert vs.count() == 2

    # 修改 description 后重新 index
    graph.nodes["Transformer"]["description"] = "Updated description v2."
    vs.index_nodes(graph, llm)

    # count 不变（upsert 覆盖）
    assert vs.count() == 2


# ── 空描述节点跳过（Opt #2 v3）──────────────────────────────────────


def test_index_nodes_skips_empty_description(tmp_path: Path) -> None:
    """空描述节点被跳过（Opt #2 v3；正常流程下 synthesize_fallback 先于此填充描述）。"""
    llm = FakeLLMClient(embedding_dim=4)
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    graph = nx.MultiDiGraph()
    graph.add_node("HasDesc", description="A node with description.", source_file="doc.md")
    graph.add_node("NoDesc", source_file="doc.md")  # 无 description
    graph.add_node("EmptyDesc", description="", source_file="doc.md")  # 空描述

    vs.index_nodes(graph, llm)

    # 只有 HasDesc 被索引
    assert vs.count() == 1
    assert vs.list_ids() == ["doc.md::HasDesc"]


def test_index_nodes_all_empty_returns_without_error(tmp_path: Path) -> None:
    """全部节点无 description → 不报错，count 为零。"""
    llm = FakeLLMClient(embedding_dim=4)
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    graph = nx.MultiDiGraph()
    graph.add_node("A", source_file="doc.md")
    graph.add_node("B", source_file="doc.md")

    vs.index_nodes(graph, llm)
    assert vs.count() == 0


# ── delete_by_source（Severe #1 + Medium #2）────────────────────────


def test_delete_by_source_removes_vectors(tmp_path: Path) -> None:
    """delete_by_source 删除指定 source_file 的全部向量（Severe #1）。"""
    llm = FakeLLMClient(embedding_dim=4)
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    graph = nx.MultiDiGraph()
    graph.add_node("A", description="Node A.", source_file="doc1.md")
    graph.add_node("B", description="Node B.", source_file="doc1.md")
    graph.add_node("C", description="Node C.", source_file="doc2.md")

    vs.index_nodes(graph, llm)
    assert vs.count() == 3

    vs.delete_by_source("doc1.md")
    assert vs.count() == 1
    assert vs.list_ids() == ["doc2.md::C"]


def test_delete_by_source_nonexistent_is_noop(tmp_path: Path) -> None:
    """删除不存在的 source_file 是 no-op（不报错）。"""
    llm = FakeLLMClient(embedding_dim=4)
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    graph = _make_graph()
    vs.index_nodes(graph, llm)
    assert vs.count() == 2

    vs.delete_by_source("nonexistent.md")
    assert vs.count() == 2


# ── search 向量召回 ─────────────────────────────────────────────────


def test_search_returns_retrieval_hits(tmp_path: Path) -> None:
    """search 向量召回返回 RetrievalHit 列表（score = 1 - distance）。"""
    llm = FakeLLMClient(embedding_dim=4)
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    graph = _make_graph()
    vs.index_nodes(graph, llm)

    hits = vs.search("query text", k=2, embedder=llm)

    assert len(hits) <= 2
    assert len(hits) > 0
    for hit in hits:
        assert hit.source == "vector"
        assert hit.concept is not None
        assert 0.0 <= hit.score <= 1.0


def test_search_empty_collection_returns_empty(tmp_path: Path) -> None:
    """空 collection 的 search 返回空列表。"""
    llm = FakeLLMClient(embedding_dim=4)
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    assert vs.search("query", k=5, embedder=llm) == []


def test_search_requires_llm(tmp_path: Path) -> None:
    """search 未提供 embedder → ValueError。"""
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    with pytest.raises(ValueError, match="embedder"):
        vs.search("query", k=5)


# ── 协议满足验证 ────────────────────────────────────────────────────


def test_vector_store_satisfies_backend_protocol(tmp_path: Path) -> None:
    """VectorStore 满足 pipeline.VectorStoreBackend Protocol（@runtime_checkable）。"""
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    assert isinstance(vs, pipeline.VectorStoreBackend)


# ── 多文件 / 多 source 索引 ─────────────────────────────────────────


def test_index_nodes_multiple_source_files(tmp_path: Path) -> None:
    """多文件节点混合索引，id 前缀正确区分 source_file。"""
    llm = FakeLLMClient(embedding_dim=4)
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    graph = nx.MultiDiGraph()
    graph.add_node("A", description="Desc A.", source_file="doc1.md")
    graph.add_node("B", description="Desc B.", source_file="doc2.md")
    graph.add_node("C", description="Desc C.", source_file="doc1.md")

    vs.index_nodes(graph, llm)
    assert vs.count() == 3

    vs.delete_by_source("doc1.md")
    assert vs.count() == 1
    assert vs.list_ids() == ["doc2.md::B"]

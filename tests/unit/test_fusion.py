"""``MultiRetriever`` + ``fuse`` + ``VectorRetriever`` + ``CommunityRetriever`` 单测
（方案 §3.5.3 step 3-4，Feature s1-feat-012 AC #5）。

覆盖：
- AC #5：fuse 按 ``confidence 权重 × 相似度`` 排序，EXTRACTED 排在 AMBIGUOUS 前；
  INFERRED 介于中间。
- 去重：triple / concept / community 三类 hit 按各自键去重。
- tiktoken 裁剪到 ``max_context_tokens``（至少保留 1 条）。
- 空入参、稳定排序（相同加权分保留入序）。
- ``VectorRetriever`` 包装 ``VectorStore.search``，source 标 "vector"。
- ``CommunityRetriever`` NER → 社区成员匹配，score = 实体覆盖率。
- ``MultiRetriever`` 三路合并 + 容错（单路抛错降级为空）。

全部用 FakeLLMClient / 内存社区结构，``tmp_path`` 隔离 ChromaDB。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from nanokb.config import Settings
from nanokb.index.community import Community, CommunityResult
from nanokb.index.vector_store import VectorStore
from nanokb.models import Concept, Confidence, RetrievalHit, Triple
from nanokb.qa.retriever import (
    CommunityRetriever,
    GraphRetriever,
    MultiRetriever,
    VectorRetriever,
    fuse,
)

# ── 测试 doubles ─────────────────────────────────────────────────────


class FakeLLMClient:
    """模拟 LLM：complete 按序消费响应；embed 返回可控向量。"""

    def __init__(
        self,
        responses: list[str] | None = None,
        default: str = "",
        embedding_dim: int = 8,
    ) -> None:
        self._responses = list(responses) if responses else []
        self._default = default
        self._dim = embedding_dim
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        self.calls.append({"user": user, "response_format": response_format})
        if self._responses:
            return self._responses.pop(0)
        return self._default

    def embed(self, texts: list[str]) -> list[list[float]]:
        # 不同文本返回不同向量（按文本首字符 hash），让向量召回可区分
        return [[float((hash(t) >> i) & 0xFF) / 255.0 for i in range(self._dim)] for t in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class StubRetriever:
    """测试 stub：可预设召回结果，满足 Retriever Protocol。"""

    SOURCE: str = "stub"

    def __init__(self, hits: list[RetrievalHit] | None = None, *, fail: bool = False) -> None:
        self._hits = list(hits) if hits else []
        self._fail = fail

    def recall(self, question: str) -> list[RetrievalHit]:
        if self._fail:
            raise RuntimeError("simulated failure")
        return list(self._hits)


def _ner(entities: list[str]) -> str:
    return json.dumps({"entities": entities})


def _triple_hit(
    head: str,
    relation: str,
    tail: str,
    confidence: Confidence,
    *,
    source: str = "graph",
    score: float = 1.0,
    source_file: str = "doc.md",
) -> RetrievalHit:
    """构造 triple RetrievalHit（score 为原始相似度）。"""
    return RetrievalHit(
        triple=Triple(
            head=head,
            relation=relation,
            tail=tail,
            confidence=confidence,
            source_file=source_file,
        ),
        score=score,
        source=source,
    )


# ══════════════════════════════════════════════════════════════════════
# AC #5：fuse 按 confidence 权重 × 相似度排序
# ══════════════════════════════════════════════════════════════════════


def test_fuse_weights_extracted_above_inferred_above_ambiguous() -> None:
    """AC #5：相同原始相似度下，EXTRACTED > INFERRED > AMBIGUOUS（按权重 1.0/0.6/0.3）。"""
    settings = Settings()
    llm = FakeLLMClient()
    hits = [
        _triple_hit("X", "amb", "Y", Confidence.AMBIGUOUS, score=0.5),
        _triple_hit("X", "ext", "Y", Confidence.EXTRACTED, score=0.5),
        _triple_hit("X", "inf", "Y", Confidence.INFERRED, score=0.5),
    ]

    fused = fuse(hits, settings, llm)

    assert len(fused) == 3
    assert fused[0].triple.relation == "ext"  # EXTRACTED, weighted 0.5
    assert fused[1].triple.relation == "inf"  # INFERRED, weighted 0.3
    assert fused[2].triple.relation == "amb"  # AMBIGUOUS, weighted 0.15


def test_fuse_extracted_ranks_before_ambiguous_even_with_lower_similarity() -> None:
    """AC #5 核心断言：EXTRACTED 即使相似度较低也能排在 AMBIGUOUS 前（当权重差足以弥补）。

    EXTRACTED sim=0.4 → weighted 0.4；AMBIGUOUS sim=1.0 → weighted 0.3。
    """
    settings = Settings()
    llm = FakeLLMClient()
    hits = [
        _triple_hit("X", "amb", "Y", Confidence.AMBIGUOUS, score=1.0),
        _triple_hit("X", "ext", "Y", Confidence.EXTRACTED, score=0.4),
    ]

    fused = fuse(hits, settings, llm)

    assert fused[0].triple.confidence == Confidence.EXTRACTED
    assert fused[1].triple.confidence == Confidence.AMBIGUOUS


def test_fuse_high_similarity_ambiguous_beats_low_similarity_extracted() -> None:
    """权重 × 相似度：AMBIGUOUS 高相似度可超过 EXTRACTED 极低相似度。

    EXTRACTED sim=0.2 → weighted 0.2；AMBIGUOUS sim=1.0 → weighted 0.3。
    """
    settings = Settings()
    llm = FakeLLMClient()
    hits = [
        _triple_hit("X", "ext", "Y", Confidence.EXTRACTED, score=0.2),
        _triple_hit("X", "amb", "Y", Confidence.AMBIGUOUS, score=1.0),
    ]

    fused = fuse(hits, settings, llm)

    assert fused[0].triple.confidence == Confidence.AMBIGUOUS  # 0.3 > 0.2
    assert fused[1].triple.confidence == Confidence.EXTRACTED


def test_fuse_stable_sort_preserves_insertion_order_for_ties() -> None:
    """相同加权分的 hit 保留入序（稳定排序，便于测试可复现）。"""
    settings = Settings()
    llm = FakeLLMClient()
    # 三个 EXTRACTED 同相似度，加权分均为 1.0
    hits = [
        _triple_hit("A", "r1", "B", Confidence.EXTRACTED, score=1.0),
        _triple_hit("A", "r2", "B", Confidence.EXTRACTED, score=1.0),
        _triple_hit("A", "r3", "B", Confidence.EXTRACTED, score=1.0),
    ]

    fused = fuse(hits, settings, llm)

    assert [h.triple.relation for h in fused] == ["r1", "r2", "r3"]


# ══════════════════════════════════════════════════════════════════════
# fuse 去重
# ══════════════════════════════════════════════════════════════════════


def test_fuse_dedups_identical_triple_hits() -> None:
    """triple hit 按 (head, relation, tail, source_file, source) 去重，保留首个。"""
    settings = Settings()
    llm = FakeLLMClient()
    hit1 = _triple_hit("A", "r", "B", Confidence.EXTRACTED, score=0.9)
    hit2 = _triple_hit("A", "r", "B", Confidence.EXTRACTED, score=0.5)  # 同键

    fused = fuse([hit1, hit2], settings, llm)

    assert len(fused) == 1
    assert fused[0].score == 0.9  # 保留首个


def test_fuse_keeps_same_triple_from_different_retrievers() -> None:
    """不同 retriever 对同一三元组的召回都保留（source 不同视为不同证据）。"""
    settings = Settings()
    llm = FakeLLMClient()
    hit_g = _triple_hit("A", "r", "B", Confidence.EXTRACTED, source="graph", score=1.0)
    hit_v = _triple_hit("A", "r", "B", Confidence.EXTRACTED, source="vector", score=0.8)

    fused = fuse([hit_g, hit_v], settings, llm)

    assert len(fused) == 2


def test_fuse_dedups_community_summary_hits() -> None:
    """community hit 按 community_summary 文本去重。"""
    settings = Settings()
    llm = FakeLLMClient()
    hit1 = RetrievalHit(
        community_summary="DL community.",
        concept=Concept(name="c1", description="DL community.", source_file="d.md"),
        score=0.5,
        source="community",
    )
    hit2 = RetrievalHit(
        community_summary="DL community.",  # 同摘要
        concept=Concept(name="c2", description="DL community.", source_file="d.md"),
        score=0.3,
        source="community",
    )

    fused = fuse([hit1, hit2], settings, llm)

    assert len(fused) == 1
    assert fused[0].score == 0.5


# ══════════════════════════════════════════════════════════════════════
# fuse 裁剪
# ══════════════════════════════════════════════════════════════════════


def test_fuse_truncates_to_max_context_tokens() -> None:
    """tiktoken 裁剪到 max_context_tokens（保留高分的，至少 1 条）。"""
    # max_context_tokens=4 + count_tokens=len(text)//4：每条 hit 渲染后约 5+ token
    settings = Settings(max_context_tokens=10)
    llm = FakeLLMClient()
    hits = [
        _triple_hit("A", "r1", "B", Confidence.EXTRACTED, score=1.0),
        _triple_hit("C", "r2", "D", Confidence.EXTRACTED, score=0.9),
        _triple_hit("E", "r3", "F", Confidence.EXTRACTED, score=0.8),
    ]

    fused = fuse(hits, settings, llm)

    # 至少保留 1 条，可能因 token 上限截断
    assert len(fused) >= 1
    assert len(fused) <= len(hits)
    # 第一条是最高分
    assert fused[0].score == 1.0


def test_fuse_keeps_at_least_one_even_when_over_limit() -> None:
    """max_context_tokens 极小时仍至少保留首条（避免完全空）。"""
    settings = Settings(max_context_tokens=1)
    llm = FakeLLMClient()
    hits = [
        _triple_hit("Verylonghead", "rel", "Verylongtail", Confidence.EXTRACTED, score=1.0),
    ]

    fused = fuse(hits, settings, llm)

    assert len(fused) == 1


def test_fuse_empty_hits_returns_empty() -> None:
    """空入参 → 空出参。"""
    settings = Settings()
    llm = FakeLLMClient()
    assert fuse([], settings, llm) == []


# ══════════════════════════════════════════════════════════════════════
# VectorRetriever
# ══════════════════════════════════════════════════════════════════════


def test_vector_retriever_returns_vector_sourced_hits(tmp_path: Path) -> None:
    """VectorRetriever 包装 VectorStore.search，source="vector"，score=1-distance。"""
    llm = FakeLLMClient(embedding_dim=4)
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    # 索引两个节点
    g = nx.MultiDiGraph()
    g.add_node("Alpha", description="Alpha desc.", source_file="doc.md")
    g.add_node("Beta", description="Beta desc.", source_file="doc.md")
    vs.index_nodes(g, llm)

    retriever = VectorRetriever(vs, llm, Settings())
    hits = retriever.recall("anything")

    assert len(hits) > 0
    assert all(h.source == "vector" for h in hits)
    # score 为原始相似度（0..1）
    for h in hits:
        assert 0.0 <= h.score <= 1.0


def test_vector_retriever_empty_collection_returns_empty(tmp_path: Path) -> None:
    """空 ChromaDB → VectorRetriever.recall 返回空。"""
    llm = FakeLLMClient(embedding_dim=4)
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    retriever = VectorRetriever(vs, llm, Settings())

    assert retriever.recall("q") == []


# ══════════════════════════════════════════════════════════════════════
# CommunityRetriever
# ══════════════════════════════════════════════════════════════════════


def _community_result() -> CommunityResult:
    """构造 {Transformer, Attention, Model} / {Python, Java} 两社区。"""
    return CommunityResult(
        communities=[
            Community(
                id=0,
                members=["Transformer", "Attention", "Model"],
                size=3,
                summary="Deep learning models community.",
                source_files=["doc.md"],
            ),
            Community(
                id=1,
                members=["Python", "Java"],
                size=2,
                summary="Programming languages community.",
                source_files=["code.py"],
            ),
        ],
        method="leiden",
        symmetrize="sum",
        total_nodes=5,
    )


def test_community_retriever_matches_member_entity() -> None:
    """NER 抽出 Transformer → 命中社区 0（深度学习）。"""
    g = nx.MultiDiGraph()
    llm = FakeLLMClient(responses=[_ner(["Transformer"])])
    retriever = CommunityRetriever(_community_result(), g, llm, Settings())

    hits = retriever.recall("Transformer 是什么？")

    assert len(hits) == 1
    assert hits[0].source == "community"
    assert hits[0].community_summary == "Deep learning models community."
    # score = 1/1 实体覆盖
    assert hits[0].score == 1.0


def test_community_retriever_partial_overlap_score() -> None:
    """多实体部分覆盖社区 → score = overlap/total_entities。"""
    g = nx.MultiDiGraph()
    # 2 实体，1 个命中社区 0
    llm = FakeLLMClient(responses=[_ner(["Transformer", "Kotlin"])])
    retriever = CommunityRetriever(_community_result(), g, llm, Settings())

    hits = retriever.recall("Transformer 与 Kotlin")

    assert len(hits) == 1
    assert hits[0].score == 0.5  # 1/2 实体覆盖


def test_community_retriever_no_match_returns_empty() -> None:
    """NER 实体不在任何社区 → 空召回。"""
    g = nx.MultiDiGraph()
    llm = FakeLLMClient(responses=[_ner(["Quantum"])])
    retriever = CommunityRetriever(_community_result(), g, llm, Settings())

    assert retriever.recall("q") == []


def test_community_retriever_empty_communities_returns_empty() -> None:
    """无社区结构 → 空召回。"""
    g = nx.MultiDiGraph()
    llm = FakeLLMClient(responses=[_ner(["X"])])
    empty = CommunityResult(communities=[], total_nodes=0)
    retriever = CommunityRetriever(empty, g, llm, Settings())

    assert retriever.recall("q") == []


def test_community_retriever_ner_empty_returns_empty() -> None:
    """NER 抽不出实体 → 空召回。"""
    g = nx.MultiDiGraph()
    llm = FakeLLMClient(responses=[_ner([])])
    retriever = CommunityRetriever(_community_result(), g, llm, Settings())

    assert retriever.recall("q") == []


# ══════════════════════════════════════════════════════════════════════
# MultiRetriever：多路合并 + 容错
# ══════════════════════════════════════════════════════════════════════


def test_multi_retriever_combines_multiple_sources_and_ranks() -> None:
    """MultiRetriever 合并多路召回后 fuse 排序：高分 graph EXTRACTED 排首。"""
    settings = Settings()
    llm = FakeLLMClient()

    graph_stub = StubRetriever(
        [_triple_hit("A", "r", "B", Confidence.EXTRACTED, source="graph", score=1.0)]
    )
    graph_stub.SOURCE = "graph"
    vector_stub = StubRetriever(
        [_triple_hit("A", "r", "B", Confidence.EXTRACTED, source="vector", score=0.6)]
    )
    vector_stub.SOURCE = "vector"

    multi = MultiRetriever([graph_stub, vector_stub], settings, llm)
    fused = multi.recall("q")

    # graph (1.0) 排在 vector (0.6) 前
    assert fused[0].source == "graph"
    assert fused[1].source == "vector"


def test_multi_retriever_empty_retrievers_returns_empty() -> None:
    """无 retriever → fuse([]) → 空列表。"""
    settings = Settings()
    llm = FakeLLMClient()
    multi = MultiRetriever([], settings, llm)
    assert multi.recall("q") == []


def test_multi_retriever_swallows_single_retriever_failure() -> None:
    """单路抛错被捕获并降级为空（不阻塞其他路）。"""
    settings = Settings()
    llm = FakeLLMClient()

    failing = StubRetriever(fail=True)
    failing.SOURCE = "failing"
    good = StubRetriever(
        [_triple_hit("A", "r", "B", Confidence.EXTRACTED, source="good", score=0.9)]
    )
    good.SOURCE = "good"

    multi = MultiRetriever([failing, good], settings, llm)
    fused = multi.recall("q")

    # 失败路降级为空，好路结果仍出现
    assert len(fused) == 1
    assert fused[0].source == "good"


# ── 三路融合端到端：GraphRetriever + VectorRetriever + CommunityRetriever ──


def test_multi_retriever_three_routes_integration(tmp_path: Path) -> None:
    """三路融合端到端：graph + vector + community 合并后 fuse 排序。

    覆盖 AC #1（三路召回融合）的单元层验证：三路各自的 hit 出现在融合结果中。
    """
    settings = Settings()
    llm = FakeLLMClient(
        responses=[
            _ner(["Transformer"]),  # graph NER
            _ner(["Transformer"]),  # community NER
        ],
        embedding_dim=4,
    )

    # graph 路：Transformer 节点 + 邻居边（NER 抽 Transformer 命中）
    g = nx.MultiDiGraph()
    g.add_node(
        "Transformer",
        description="A sequence model.",
        source_file="doc.md",
        confidence="EXTRACTED",
    )
    g.add_node(
        "Attention",
        description="A focus mechanism.",
        source_file="doc.md",
        confidence="EXTRACTED",
    )
    g.add_edge(
        "Transformer",
        "Attention",
        relation="uses",
        source_file="doc.md",
        confidence="EXTRACTED",
    )

    # vector 路
    vs = VectorStore(tmp_path / "chroma", "test-model", 4)
    vs.index_nodes(g, llm)

    # community 路
    communities = _community_result()

    multi = MultiRetriever(
        [
            GraphRetriever(g, llm, settings),
            VectorRetriever(vs, llm, settings),
            CommunityRetriever(communities, g, llm, settings),
        ],
        settings,
        llm,
    )

    fused = multi.recall("Transformer")

    # 三路都贡献了 hit（graph 边 + vector 节点 + community 摘要）
    sources = {h.source for h in fused}
    assert "graph" in sources
    assert "vector" in sources
    assert "community" in sources
    # 融合后非空
    assert len(fused) >= 3

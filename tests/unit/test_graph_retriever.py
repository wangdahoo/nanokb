"""``qa.retriever.GraphRetriever`` 单测（方案 §3.5.3 step 3-4，Feature s1-feat-009）。

覆盖：
- AC #3：实体大小写不一致（``transformer`` vs 图节点 ``Transformer``）经 normalize 命中。
- fuzzy 兜底：实体拼写差异（``Transfomer`` 缺字母）经 difflib.get_close_matches 命中。
- 真正不相关实体 → 空召回。
- NER 抽出多个实体 → 全部尝试匹配。
- N 跳子图扩展收集邻近边。
- LLM NER 失败容错（返回非 JSON → 空实体 → 空召回）。
- 边 hit 携带 source_file + confidence 权重 score。
- 孤立节点（无边）→ concept hit 兜底。
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx

from nanokb.config import Settings
from nanokb.qa.retriever import GraphRetriever


class FakeLLMClient:
    """模拟 LLM：complete 按顺序返回预设响应。"""

    def __init__(self, responses: list[str] | None = None, default: str = "") -> None:
        self._responses = list(responses) if responses else []
        self._default = default
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
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _ner_response(entities: list[str]) -> str:
    return json.dumps({"entities": entities})


def _build_graph() -> nx.MultiDiGraph:
    """构造测试图：Transformer--uses-->Attention, Transformer--is_a-->Model。"""
    g = nx.MultiDiGraph()
    g.add_node("Transformer", description="A model.", source_file="doc.md", confidence="EXTRACTED")
    g.add_node(
        "Attention", description="A mechanism.", source_file="doc.md", confidence="EXTRACTED"
    )
    g.add_node(
        "Model", description="A model category.", source_file="doc.md", confidence="EXTRACTED"
    )
    g.add_edge(
        "Transformer",
        "Attention",
        relation="uses",
        source_file="doc.md",
        confidence="EXTRACTED",
    )
    g.add_edge(
        "Transformer",
        "Model",
        relation="is_a",
        source_file="doc.md",
        confidence="INFERRED",
    )
    return g


# ── AC #3：大小写不一致经 normalize 命中 ─────────────────────────────


def test_lowercase_entity_normalized_matches_titlecase_node() -> None:
    graph = _build_graph()
    llm = FakeLLMClient(responses=[_ner_response(["transformer"])])
    retriever = GraphRetriever(graph, llm, Settings())

    hits = retriever.recall("Transformer 如何依赖 Attention？")

    assert len(hits) >= 1
    # 应包含 Transformer 出发的边
    heads = {h.triple.head for h in hits if h.triple is not None}
    assert "Transformer" in heads
    assert all(h.source == "graph" for h in hits)


def test_entity_with_extra_spaces_normalized() -> None:
    graph = _build_graph()
    llm = FakeLLMClient(responses=[_ner_response(["  transformer  "])])
    retriever = GraphRetriever(graph, llm, Settings())

    hits = retriever.recall("question")
    assert len(hits) >= 1


# ── fuzzy 兜底（AC #3：不漏召回）──────────────────────────────────────


def test_fuzzy_match_catches_typo() -> None:
    graph = _build_graph()
    # "Transfomer" 缺一个 r，与 "transformer" 相似度 ~0.9 > 0.8 cutoff
    llm = FakeLLMClient(responses=[_ner_response(["Transfomer"])])
    retriever = GraphRetriever(graph, llm, Settings(fuzzy_match_cutoff=0.8))

    hits = retriever.recall("question")
    assert len(hits) >= 1
    heads = {h.triple.head for h in hits if h.triple is not None}
    assert "Transformer" in heads


def test_fuzzy_below_cutoff_returns_empty() -> None:
    graph = _build_graph()
    # 完全不相似的实体
    llm = FakeLLMClient(responses=[_ner_response(["QuantumComputing"])])

    # 默认 cutoff=0.8，"quantumcomputing" vs "transformer"/"attention"/"model" 均极低
    retriever = GraphRetriever(graph, llm, Settings())
    hits = retriever.recall("question")
    assert hits == []


def test_fuzzy_cutoff_zero_disables_fuzzy_matching() -> None:
    graph = _build_graph()
    llm = FakeLLMClient(responses=[_ner_response(["Transfomer"])])
    retriever = GraphRetriever(graph, llm, Settings(fuzzy_match_cutoff=0.0))

    hits = retriever.recall("question")
    # cutoff=0 禁用 fuzzy，且 normalize("Transfomer") != 任何节点 → 空召回
    assert hits == []


# ── N 跳扩展 ─────────────────────────────────────────────────────────


def test_n_hop_expansion_collects_nearby_edges() -> None:
    # 构造 A→B→C 三链路，hops=2 从 A 应能到 C
    g = nx.MultiDiGraph()
    for n in ("A", "B", "C"):
        g.add_node(n, description=n, source_file="f.md", confidence="EXTRACTED")
    g.add_edge("A", "B", relation="r1", source_file="f.md", confidence="EXTRACTED")
    g.add_edge("B", "C", relation="r2", source_file="f.md", confidence="EXTRACTED")

    llm = FakeLLMClient(responses=[_ner_response(["A"])])
    retriever = GraphRetriever(g, llm, Settings(retrieval_hops=2))

    hits = retriever.recall("q")
    tails = {h.triple.tail for h in hits if h.triple is not None}
    assert "B" in tails
    assert "C" in tails  # 2 跳扩展到 C


def test_one_hop_does_not_reach_two_hop_neighbor() -> None:
    g = nx.MultiDiGraph()
    for n in ("A", "B", "C"):
        g.add_node(n, description=n, source_file="f.md", confidence="EXTRACTED")
    g.add_edge("A", "B", relation="r1", source_file="f.md", confidence="EXTRACTED")
    g.add_edge("B", "C", relation="r2", source_file="f.md", confidence="EXTRACTED")

    llm = FakeLLMClient(responses=[_ner_response(["A"])])
    retriever = GraphRetriever(g, llm, Settings(retrieval_hops=1))

    hits = retriever.recall("q")
    tails = {h.triple.tail for h in hits if h.triple is not None}
    assert "B" in tails
    assert "C" not in tails  # 1 跳到不了 C


def test_zero_hops_only_seeds_no_edges() -> None:
    g = _build_graph()
    llm = FakeLLMClient(responses=[_ner_response(["Attention"])])
    retriever = GraphRetriever(g, llm, Settings(retrieval_hops=0))

    hits = retriever.recall("q")
    # hops=0 → 仅 Attention 节点本身（无边）；走 concept hit 兜底
    assert len(hits) == 1
    assert hits[0].triple is None
    assert hits[0].concept is not None
    assert hits[0].concept.name == "Attention"


# ── 空图 / 无实体 ────────────────────────────────────────────────────


def test_empty_graph_returns_empty_hits() -> None:
    g = nx.MultiDiGraph()
    llm = FakeLLMClient(responses=[_ner_response(["A"])])
    retriever = GraphRetriever(g, llm, Settings())

    assert retriever.recall("q") == []


def test_ner_empty_entities_returns_empty_hits() -> None:
    graph = _build_graph()
    llm = FakeLLMClient(responses=[_ner_response([])])
    retriever = GraphRetriever(graph, llm, Settings())

    assert retriever.recall("q") == []


def test_ner_malformed_json_returns_empty_entities() -> None:
    graph = _build_graph()
    llm = FakeLLMClient(default="not json at all")
    retriever = GraphRetriever(graph, llm, Settings())

    assert retriever.recall("q") == []


# ── score / confidence ──────────────────────────────────────────────


def test_inferred_edge_score_raw_similarity_confidence_in_triple() -> None:
    """GraphRetriever 返回原始相似度（精确匹配=1.0）；confidence 由 fuse 统一加权。

    s1-feat-012 重构：``hit.score`` 为原始相似度（不再乘 confidence 权重），
    权重乘入由 ``MultiRetriever.fuse`` 统一执行（保证 graph/vector/community
    三路口径一致）。本测验证 score 字段为原始相似度，confidence 仍正确标注在
    triple 上供 fuse 读取。
    """
    graph = _build_graph()
    # Transformer-uses->Attention EXTRACTED；Transformer-is_a->Model INFERRED
    llm = FakeLLMClient(responses=[_ner_response(["Transformer"])])
    retriever = GraphRetriever(graph, llm, Settings())

    hits = retriever.recall("q")
    by_rel_score = {h.triple.relation: h.score for h in hits if h.triple is not None}
    by_rel_conf = {h.triple.relation: h.triple.confidence for h in hits if h.triple is not None}
    # score 一律原始相似度 1.0（精确图节点匹配）
    assert by_rel_score["uses"] == 1.0
    assert by_rel_score["is_a"] == 1.0
    # confidence 差异保留在 triple 上（fuse 据此加权排序）
    from nanokb.models import Confidence

    assert by_rel_conf["uses"] == Confidence.EXTRACTED
    assert by_rel_conf["is_a"] == Confidence.INFERRED


def test_hits_carry_source_file_from_edge_data() -> None:
    g = nx.MultiDiGraph()
    g.add_node("X", description="X", source_file="x.md", confidence="EXTRACTED")
    g.add_node("Y", description="Y", source_file="x.md", confidence="EXTRACTED")
    g.add_edge("X", "Y", relation="rel", source_file="custom.md", confidence="EXTRACTED")
    llm = FakeLLMClient(responses=[_ner_response(["X"])])
    retriever = GraphRetriever(g, llm, Settings())

    hits = retriever.recall("q")
    assert len(hits) == 1
    assert hits[0].triple.source_file == "custom.md"


# ── 多实体 ───────────────────────────────────────────────────────────


def test_multiple_entities_all_seeds_collected() -> None:
    g = nx.MultiDiGraph()
    for n in ("A", "B"):
        g.add_node(n, description=n, source_file="f.md", confidence="EXTRACTED")
    g.add_edge("A", "X", relation="r", source_file="f.md", confidence="EXTRACTED")
    g.add_edge("B", "Y", relation="r", source_file="f.md", confidence="EXTRACTED")
    g.add_node("X", description="X", source_file="f.md", confidence="EXTRACTED")
    g.add_node("Y", description="Y", source_file="f.md", confidence="EXTRACTED")

    llm = FakeLLMClient(responses=[_ner_response(["A", "B"])])
    retriever = GraphRetriever(g, llm, Settings())

    heads = {h.triple.head for h in retriever.recall("q") if h.triple is not None}
    assert heads == {"A", "B"}


# ── 孤立节点 ─────────────────────────────────────────────────────────


def test_isolated_seed_node_falls_back_to_concept_hit() -> None:
    g = nx.MultiDiGraph()
    g.add_node(
        "Lonely",
        description="An isolated node.",
        source_file="lonely.md",
        confidence="EXTRACTED",
    )
    llm = FakeLLMClient(responses=[_ner_response(["Lonely"])])
    retriever = GraphRetriever(g, llm, Settings())

    hits = retriever.recall("q")
    assert len(hits) == 1
    assert hits[0].triple is None
    assert hits[0].concept is not None
    assert hits[0].concept.name == "Lonely"
    assert hits[0].concept.description == "An isolated node."


# ── 并行边去重 ──────────────────────────────────────────────────────


def test_parallel_edges_same_relation_deduplicated() -> None:
    """MultiDiGraph 同 (head,relation,tail,source_file) 多 key 边只产 1 个 hit。"""
    g = nx.MultiDiGraph()
    g.add_node("A", description="A", source_file="f.md", confidence="EXTRACTED")
    g.add_node("B", description="B", source_file="f.md", confidence="EXTRACTED")
    g.add_edge("A", "B", key=0, relation="r", source_file="f.md", confidence="EXTRACTED")
    g.add_edge("A", "B", key=1, relation="r", source_file="f.md", confidence="EXTRACTED")

    llm = FakeLLMClient(responses=[_ner_response(["A"])])
    retriever = GraphRetriever(g, llm, Settings())

    hits = retriever.recall("q")
    # 两条 parallel edge 去重为 1
    assert len(hits) == 1


# ── 归一化索引懒缓存（s2-feat-001） ──────────────────────────────────


def test_norm_index_cached_across_recalls() -> None:
    """s2-feat-001：连续两次 recall 复用同一归一化索引，构建计数 == 1。"""
    graph = _build_graph()
    llm = FakeLLMClient(responses=[_ner_response(["Transformer"]), _ner_response(["Transformer"])])
    retriever = GraphRetriever(graph, llm, Settings())

    assert retriever._norm_index_builds == 0
    retriever.recall("q")
    assert retriever._norm_index_builds == 1
    retriever.recall("q")
    # 第二次 recall 命中缓存，不再重建
    assert retriever._norm_index_builds == 1


# ── fuzzy 长度桶安全预筛（s2-feat-002） ──────────────────────────────


def test_norm_buckets_cached_across_recalls() -> None:
    """s2-feat-002：长度桶与 norm_index 同生命周期，连续 recall 构建计数 == 1。"""
    graph = _build_graph()
    # 两次 fuzzy 兜底（实体拼写有误 → 走 fuzzy 分支触发桶构建）
    llm = FakeLLMClient(responses=[_ner_response(["Transfomer"]), _ner_response(["Atention"])])
    retriever = GraphRetriever(graph, llm, Settings())

    assert retriever._norm_buckets_builds == 0
    retriever.recall("q")
    assert retriever._norm_buckets_builds == 1
    retriever.recall("q")
    assert retriever._norm_buckets_builds == 1


def test_fuzzy_candidates_matches_naive_difflib() -> None:
    """s2-feat-002 等价性：长度桶预筛返回的候选是 difflib 全量的安全超集，
    且在该候选集上跑 get_close_matches 的最终命中 == 全量 difflib 命中。"""
    import difflib as _difflib

    g = nx.MultiDiGraph()
    # 长度差异大的节点名：短词 + 长词混合
    names = ["Transformer", "Attention", "Model", "AI", "ML", "ConvolutionalNeuralNetwork"]
    for n in names:
        g.add_node(n, description=n, source_file="f.md", confidence="EXTRACTED")

    retriever = GraphRetriever(g, FakeLLMClient(), Settings(fuzzy_match_cutoff=0.8))
    norm_index = retriever._ensure_norm_index()
    all_norms = list(norm_index.keys())

    for query in ["Transfomer", "Atention", "model", "AI", "ConvolutionalNetwork", "Quantum"]:
        norm = query.lower()
        candidates = retriever._fuzzy_candidates(norm, 0.8)
        # 候选是全量的子集
        assert set(candidates).issubset(set(all_norms))
        # 在候选上跑 get_close_matches == 全量结果（安全超集，不丢命中）
        naive = _difflib.get_close_matches(norm, all_norms, n=3, cutoff=0.8)
        bucketed = _difflib.get_close_matches(norm, candidates, n=3, cutoff=0.8)
        assert bucketed == naive, f"mismatch for {query!r}: bucketed={bucketed} naive={naive}"


def test_fuzzy_candidates_empty_for_disjoint_length() -> None:
    """s2-feat-002：查询长度与所有节点名长度比 < cutoff/(2-cutoff) 时无候选。"""
    g = nx.MultiDiGraph()
    # 只有长名（22 字符），查询很短（如 "ai" 2 字符）：2/22 < 2/3 → 被剪
    g.add_node("ConvolutionalNeuralNetwork", description="long", source_file="f.md", confidence="EXTRACTED")
    retriever = GraphRetriever(g, FakeLLMClient(), Settings(fuzzy_match_cutoff=0.8))

    assert retriever._fuzzy_candidates("ai", 0.8) == []
    # 反向：长查询对短名也剪
    retriever2 = GraphRetriever(
        nx.MultiDiGraph(),  # placeholder, rebuild below
        FakeLLMClient(),
        Settings(fuzzy_match_cutoff=0.8),
    )
    g2 = nx.MultiDiGraph()
    g2.add_node("AI", description="short", source_file="f.md", confidence="EXTRACTED")
    retriever2 = GraphRetriever(g2, FakeLLMClient(), Settings(fuzzy_match_cutoff=0.8))
    assert retriever2._fuzzy_candidates("convolutionalneuralnetwork", 0.8) == []



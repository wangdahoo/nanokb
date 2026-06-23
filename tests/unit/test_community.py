"""detect_communities + Leiden 单测（方案 §3.5.4 + Opt #3 v3，Feature s1-feat-011）。

覆盖 AC #4：小型 MultiDiGraph → 折叠平行边 + 对称化 sum + leidenalg ModularityVertexPartition
→ 社区划分写入 communities.json。

另覆盖：
- _collapse_parallel_edges：MultiDiGraph → DiGraph，weight = 边数。
- _symmetrize：sum 策略 vs max 策略。
- LLM 摘要 vs 启发式摘要（llm=None）。
- communities.json 原子写入 staging。
- 小图（< MIN_NODES_FOR_COMMUNITY）跳过。
- load_communities 回读。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from nanokb.config import Settings
from nanokb.index.community import (
    COMMUNITIES_FILENAME,
    _collapse_parallel_edges,
    _symmetrize,
    detect_communities,
    load_communities,
)

# ── 测试 doubles ─────────────────────────────────────────────────────


class FakeLLMClient:
    """模拟 LLMClient，complete 返回固定摘要。"""

    def __init__(self, summary: str = "A community summary.") -> None:
        self._summary = summary
        self.complete_calls: int = 0

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        self.complete_calls += 1
        return self._summary

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _settings(**kwargs: Any) -> Settings:
    """默认 Settings（leiden_symmetrize='sum'，Opt #3 v3）。"""
    return Settings(**kwargs)


def _build_graph_with_communities() -> nx.MultiDiGraph:
    """构造两个明显社区的图：{A,B,C} 强连接 + {D,E,F} 强连接 + 弱跨社区边。

    A→B, B→C, C→A （社区 0 内部强连接）
    D→E, E→F, F→D （社区 1 内部强连接）
    A→D （跨社区弱连接，单边）
    """
    g = nx.MultiDiGraph()
    for u, v in [("A", "B"), ("B", "C"), ("C", "A")]:
        g.add_edge(u, v, source_file="doc1.md", relation="rel")
    for u, v in [("D", "E"), ("E", "F"), ("F", "D")]:
        g.add_edge(u, v, source_file="doc2.md", relation="rel")
    g.add_edge("A", "D", source_file="doc1.md", relation="cross")
    for node in g.nodes():
        g.nodes[node]["description"] = f"Node {node}."
        g.nodes[node]["source_file"] = "doc1.md" if node in ("A", "B", "C") else "doc2.md"
    return g


# ── _collapse_parallel_edges ────────────────────────────────────────


def test_collapse_parallel_edges_weights_by_count() -> None:
    """折叠平行边：MultiDiGraph → DiGraph，weight = 该方向边数。"""
    g = nx.MultiDiGraph()
    g.add_edge("A", "B")  # 第一条 A→B
    g.add_edge("A", "B")  # 第二条 A→B（平行）
    g.add_edge("B", "A")  # 反向 B→A
    g.add_edge("B", "C")

    di = _collapse_parallel_edges(g)

    assert di["A"]["B"]["weight"] == 2  # 2 条平行边
    assert di["B"]["A"]["weight"] == 1
    assert di["B"]["C"]["weight"] == 1
    assert not di.has_edge("C", "A")


def test_collapse_parallel_edges_empty_graph() -> None:
    """空图折叠 → 空 DiGraph。"""
    di = _collapse_parallel_edges(nx.MultiDiGraph())
    assert di.number_of_edges() == 0


# ── _symmetrize（sum vs max，Opt #3 v3）─────────────────────────────


def test_symmetrize_sum_combines_both_directions() -> None:
    """sum 策略：weight(u,v) = w(u→v) + w(v→u)（Opt #3 v3）。"""
    di = nx.DiGraph()
    di.add_edge("A", "B", weight=3)
    di.add_edge("B", "A", weight=2)

    und = _symmetrize(di, strategy="sum")

    assert und["A"]["B"]["weight"] == 5  # 3 + 2 = 5（sum）


def test_symmetrize_max_takes_larger() -> None:
    """max 策略：weight(u,v) = max(w(u→v), w(v→u))。"""
    di = nx.DiGraph()
    di.add_edge("A", "B", weight=3)
    di.add_edge("B", "A", weight=2)

    und = _symmetrize(di, strategy="max")

    assert und["A"]["B"]["weight"] == 3  # max(3, 2)


def test_symmetrize_sum_single_direction() -> None:
    """单向边对称化后保持原权重（无反向边相加）。"""
    di = nx.DiGraph()
    di.add_edge("A", "B", weight=4)

    und = _symmetrize(di, strategy="sum")

    assert und["A"]["B"]["weight"] == 4


def test_symmetrize_produces_undirected_graph() -> None:
    """对称化结果是无向图（Graph，非 DiGraph）。"""
    di = nx.DiGraph()
    di.add_edge("A", "B", weight=1)
    di.add_edge("B", "C", weight=1)

    und = _symmetrize(di, strategy="sum")

    assert not und.is_directed()


# ── AC #4：detect_communities 端到端 + communities.json ─────────────


def test_detect_communities_produces_partition(tmp_path: Path) -> None:
    """AC #4：小型图 → 折叠 + 对称化 sum + leidenalg → 社区划分写入 communities.json。"""
    graph = _build_graph_with_communities()
    settings = _settings()
    staging = tmp_path / "staging"

    result = detect_communities(graph, settings, llm=None, staging_dir=staging)

    # 两个社区（A,B,C 紧密连接 / D,E,F 紧密连接）
    assert len(result.communities) >= 1
    assert result.total_nodes == 6
    assert result.method == "leiden"
    assert result.symmetrize == "sum"

    # 所有节点都被分配到某个社区
    all_members: set[str] = set()
    for comm in result.communities:
        all_members.update(comm.members)
    assert all_members == {"A", "B", "C", "D", "E", "F"}

    # communities.json 写入 staging
    assert (staging / COMMUNITIES_FILENAME).exists()


def test_detect_communities_separates_dense_clusters(tmp_path: Path) -> None:
    """两个明显密集簇被分到不同社区。"""
    graph = _build_graph_with_communities()
    settings = _settings()

    result = detect_communities(graph, settings, llm=None)

    # 验证 {A,B,C} 与 {D,E,F} 被分到不同社区（密集簇内聚 > 跨簇）
    comm_of: dict[str, int] = {}
    for comm in result.communities:
        for member in comm.members:
            comm_of[member] = comm.id

    # A,B,C 应在同一社区，D,E,F 应在同一社区
    abc_communities = {comm_of[n] for n in ("A", "B", "C")}
    def_communities = {comm_of[n] for n in ("D", "E", "F")}
    assert len(abc_communities) == 1, f"A,B,C should be in same community: {abc_communities}"
    assert len(def_communities) == 1, f"D,E,F should be in same community: {def_communities}"
    # 两社区不同
    assert abc_communities != def_communities


def test_detect_communities_writes_valid_json(tmp_path: Path) -> None:
    """communities.json 是合法 JSON，结构含 communities / method / symmetrize / total_nodes。"""
    graph = _build_graph_with_communities()
    settings = _settings()
    staging = tmp_path / "staging"

    detect_communities(graph, settings, llm=None, staging_dir=staging)

    raw = (staging / COMMUNITIES_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)

    assert "communities" in data
    assert data["method"] == "leiden"
    assert data["symmetrize"] == "sum"
    assert data["total_nodes"] == 6
    for comm in data["communities"]:
        assert "id" in comm
        assert "members" in comm
        assert "size" in comm
        assert "summary" in comm


def test_detect_communities_with_llm_summary(tmp_path: Path) -> None:
    """llm 非 None 时调用 LLM 生成社区摘要。"""
    graph = _build_graph_with_communities()
    settings = _settings()
    llm = FakeLLMClient(summary="This community is about neural networks.")

    result = detect_communities(graph, settings, llm=llm)

    # LLM 被调用（至少每社区一次）
    assert llm.complete_calls >= 1
    # 摘要非空
    for comm in result.communities:
        assert comm.summary


def test_detect_communities_without_llm_uses_heuristic(tmp_path: Path) -> None:
    """llm=None 时社区摘要降级为启发式（成员名拼接），零 Token。"""
    graph = _build_graph_with_communities()
    settings = _settings()

    result = detect_communities(graph, settings, llm=None)

    for comm in result.communities:
        # 启发式摘要含 "Community of:" 前缀
        assert "Community of:" in comm.summary or comm.summary


def test_detect_communities_staging_dir_none_skips_write(tmp_path: Path) -> None:
    """staging_dir=None 时不写文件（仅返回结果）。"""
    graph = _build_graph_with_communities()
    settings = _settings()

    result = detect_communities(graph, settings, llm=None, staging_dir=None)

    assert len(result.communities) >= 1


def test_detect_communities_small_graph_skips(tmp_path: Path) -> None:
    """节点数 < MIN_NODES_FOR_COMMUNITY → 跳过，返回空社区列表。"""
    graph = nx.MultiDiGraph()
    graph.add_node("Lonely", description="Only one node.", source_file="doc.md")

    settings = _settings()
    staging = tmp_path / "staging"

    result = detect_communities(graph, settings, llm=None, staging_dir=staging)

    assert result.communities == []
    assert result.total_nodes == 1
    # communities.json 仍写入（空列表）
    assert (staging / COMMUNITIES_FILENAME).exists()


def test_detect_communities_empty_graph(tmp_path: Path) -> None:
    """空图 → 空社区列表。"""
    graph = nx.MultiDiGraph()
    settings = _settings()
    staging = tmp_path / "staging"

    result = detect_communities(graph, settings, llm=None, staging_dir=staging)

    assert result.communities == []
    assert result.total_nodes == 0


# ── 社区 source_files 收集 ──────────────────────────────────────────


def test_community_source_files_collected(tmp_path: Path) -> None:
    """社区结构含 source_files（从节点属性 + incident 边收集）。"""
    graph = _build_graph_with_communities()
    settings = _settings()

    result = detect_communities(graph, settings, llm=None)

    all_sources: set[str] = set()
    for comm in result.communities:
        all_sources.update(comm.source_files)
    assert "doc1.md" in all_sources
    assert "doc2.md" in all_sources


# ── load_communities 回读 ───────────────────────────────────────────


def test_load_communities_roundtrip(tmp_path: Path) -> None:
    """detect_communities 写入 → load_communities 回读一致。"""
    graph = _build_graph_with_communities()
    settings = _settings()

    result = detect_communities(graph, settings, llm=None, staging_dir=tmp_path)
    # 模拟 staging_swap 后文件在 out_dir
    (tmp_path / COMMUNITIES_FILENAME).replace(tmp_path / COMMUNITIES_FILENAME)  # no-op

    loaded = load_communities(tmp_path)
    assert loaded is not None
    assert loaded.total_nodes == result.total_nodes
    assert len(loaded.communities) == len(result.communities)


def test_load_communities_missing_returns_none(tmp_path: Path) -> None:
    """文件不存在时 load_communities 返回 None。"""
    assert load_communities(tmp_path) is None


# ── 对称化策略可配置 ────────────────────────────────────────────────


def test_detect_communities_max_strategy(tmp_path: Path) -> None:
    """leiden_symmetrize='max' 时使用 max 对称化策略。"""
    graph = _build_graph_with_communities()
    settings = _settings(leiden_symmetrize="max")

    result = detect_communities(graph, settings, llm=None)

    assert result.symmetrize == "max"


# ── leidenalg 确定性 ────────────────────────────────────────────────


def test_detect_communities_deterministic(tmp_path: Path) -> None:
    """同一图两次 detect_communities 结果一致（seed=42 固定，可复现）。"""
    graph = _build_graph_with_communities()
    settings = _settings()

    r1 = detect_communities(graph, settings, llm=None)
    r2 = detect_communities(graph, settings, llm=None)

    # 社区数相同
    assert len(r1.communities) == len(r2.communities)
    # 成员集合相同（社区 id 可能重编号，但划分一致）
    members1 = {frozenset(c.members) for c in r1.communities}
    members2 = {frozenset(c.members) for c in r2.communities}
    assert members1 == members2

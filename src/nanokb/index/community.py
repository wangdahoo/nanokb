"""社区发现（方案 §3.5.4 + Opt #3 v3，Feature s1-feat-011）。

``detect_communities`` 实现基于 Leiden 算法的社区发现：

1. **折叠平行边**：``MultiDiGraph`` → ``DiGraph``，每对 (u,v) 的 weight = 该方向边数
   之和（反映连接强度）。
2. **对称化 sum**（Opt #3 v3 拍定）：无向加权图，weight(u,v) = w(u→v) + w(v→u)，
   反映双向总连接强度，消除方向歧义保证可复现。
3. **NetworkX → igraph**：节点顺序映射保留，边带 weight。
4. **leidenalg.find_partition** + ``ModularityVertexPartition`` + ``weights="weight"``：
   模块度最大化的社区划分。
5. **社区摘要**：每社区 LLM 生成一句话摘要（可选，llm 为 None 时用启发式——成员名拼接），
   写入 ``staging_dir/communities.json``（v4 Opt #1：纳入 staging 原子切换五件套）。

依据：Leiden 经典实现作用于无向图；折叠 + 对称化是标准预处理。选 sum 而非 max
反映总连接强度，消除歧义保证可复现（Opt #3 v3）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from nanokb.config import Settings
from nanokb.utils.io import atomic_write_json

if TYPE_CHECKING:
    from nanokb.llm.base import LLMClient

logger = logging.getLogger("nanokb")

#: communities.json 输出文件名（纳入 staging 原子切换，v4 Opt #1）
COMMUNITIES_FILENAME = "communities.json"

#: Leiden 社区发现的最小节点数（少于则不执行社区发现，输出空列表）
MIN_NODES_FOR_COMMUNITY = 2


class Community(BaseModel):
    """单个社区的序列化结构。"""

    id: int
    members: list[str] = Field(default_factory=list)
    size: int = 0
    summary: str = ""
    source_files: list[str] = Field(default_factory=list)


class CommunityResult(BaseModel):
    """detect_communities 返回值 + communities.json 的序列化根结构。"""

    communities: list[Community] = Field(default_factory=list)
    method: str = "leiden"
    symmetrize: str = "sum"
    total_nodes: int = 0


def detect_communities(
    graph: nx.MultiDiGraph,
    settings: Settings,
    llm: LLMClient | None = None,
    *,
    staging_dir: Path | None = None,
) -> CommunityResult:
    """执行 Leiden 社区发现并（可选）写入 ``staging_dir/communities.json``。

    流程（方案 §3.5.4）：
    1. 折叠平行边 → ``DiGraph`` 带 weight = 边数。
    2. 对称化 sum → 无向加权图。
    3. NetworkX → igraph + leidenalg ``ModularityVertexPartition``。
    4. 每社区 LLM 摘要（或启发式）→ ``CommunityResult``。
    5. 若提供 ``staging_dir``，原子写入 ``communities.json``。

    Args:
        graph: 知识图谱（``MultiDiGraph``）。
        settings: 全局配置（读 ``leiden_symmetrize``）。
        llm: LLM 客户端；``None`` 时用启发式摘要（成员名拼接），不消耗 Token。
        staging_dir: staging 目录；``None`` 时不写文件（仅返回结果）。

    Returns:
        ``CommunityResult`` —— 社区列表 + 方法 / 对称化策略 / 总节点数。
    """
    nodes = list(graph.nodes())
    result = CommunityResult(
        symmetrize=settings.leiden_symmetrize,
        total_nodes=len(nodes),
    )

    if len(nodes) < MIN_NODES_FOR_COMMUNITY:
        logger.debug(
            "detect_communities: graph has %d nodes (< %d); skipping",
            len(nodes),
            MIN_NODES_FOR_COMMUNITY,
            extra={"stage": "community"},
        )
        if staging_dir is not None:
            _write_communities(staging_dir, result)
        return result

    # step 1: 折叠平行边 → DiGraph 带 weight
    collapsed = _collapse_parallel_edges(graph)

    # step 2: 对称化 → 无向加权图（sum 策略，Opt #3 v3）
    undirected = _symmetrize(collapsed, strategy=settings.leiden_symmetrize)

    # step 3-4: NetworkX → igraph + leidenalg
    membership = _run_leiden(undirected, nodes)

    # step 5: 组装社区结构 + 摘要
    communities = _build_communities(graph, nodes, membership, llm)
    result.communities = communities

    if staging_dir is not None:
        _write_communities(staging_dir, result)

    logger.info(
        "detect_communities: %d communities from %d nodes (symmetrize=%s)",
        len(communities),
        len(nodes),
        settings.leiden_symmetrize,
        extra={"stage": "community"},
    )

    return result


# ── 内部辅助 ─────────────────────────────────────────────────────────


def _collapse_parallel_edges(graph: nx.MultiDiGraph) -> nx.DiGraph:
    """折叠平行边：MultiDiGraph → DiGraph，weight = 该方向边数之和。

    对每对 (u, v)，weight = MultiDiGraph 中 u→v 的边数。反映单方向连接强度。
    """
    di = nx.DiGraph()
    for u, v in graph.edges():
        if di.has_edge(u, v):
            di[u][v]["weight"] += 1
        else:
            di.add_edge(u, v, weight=1)
    return di


def _symmetrize(di_graph: nx.DiGraph, *, strategy: str = "sum") -> nx.Graph:
    """对称化：DiGraph → 无向加权图。

    - ``sum``（Opt #3 v3，默认）：weight(u,v) = w(u→v) + w(v→u)。
    - ``max``（备用）：weight(u,v) = max(w(u→v), w(v→u))。
    """
    und = nx.Graph()
    edge_weight_map: dict[tuple[str, str], float] = {}

    for u, v, data in di_graph.edges(data=True):
        key = tuple(sorted((u, v)))
        weight = float(data.get("weight", 1))
        if strategy == "max":
            edge_weight_map[key] = max(edge_weight_map.get(key, 0.0), weight)
        else:  # "sum" (default)
            edge_weight_map[key] = edge_weight_map.get(key, 0.0) + weight

    for (a, b), weight in edge_weight_map.items():
        und.add_edge(a, b, weight=weight)

    return und


def _run_leiden(undirected: nx.Graph, original_nodes: list[str]) -> list[int]:
    """NetworkX → igraph + leidenalg ModularityVertexPartition。

    节点顺序映射保留：``original_nodes[i]`` 对应 igraph 节点 ``i``，membership[i]
    为其社区 id。

    Returns:
        membership 列表（与 original_nodes 同序），社区 id 从 0 开始。图无边时
        所有节点归入社区 0。
    """
    import igraph as ig  # type: ignore[import-untyped]
    import leidenalg  # type: ignore[import-untyped]

    node_to_idx = {node: i for i, node in enumerate(original_nodes)}
    ig_edges: list[tuple[int, int]] = []
    ig_weights: list[float] = []

    for u, v, data in undirected.edges(data=True):
        if u in node_to_idx and v in node_to_idx:
            ig_edges.append((node_to_idx[u], node_to_idx[v]))
            ig_weights.append(float(data.get("weight", 1.0)))

    n = len(original_nodes)
    ig_graph = ig.Graph(n=n, edges=ig_edges, directed=False)

    if not ig_edges:
        # 无边图：每个节点自成社区（或全归 0）。此处全归 0 保证社区摘要有意义。
        return [0] * n

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights=ig_weights,
        seed=42,
    )
    return list(partition.membership)


def _build_communities(
    graph: nx.MultiDiGraph,
    nodes: list[str],
    membership: list[int],
    llm: LLMClient | None,
) -> list[Community]:
    """按 membership 组装社区结构，为每社区生成摘要。

    LLM 摘要：对每社区发送成员名列表，请求一句话概述。llm 为 None 或调用失败时
    降级为启发式（成员名拼接）。
    """
    groups: dict[int, list[str]] = {}
    for node, comm_id in zip(nodes, membership, strict=True):
        groups.setdefault(int(comm_id), []).append(node)

    communities: list[Community] = []
    for comm_id in sorted(groups):
        members = sorted(groups[comm_id])
        source_files = sorted(_collect_source_files(graph, members))
        summary = _summarize_community(members, llm)
        communities.append(
            Community(
                id=comm_id,
                members=members,
                size=len(members),
                summary=summary,
                source_files=source_files,
            )
        )

    return communities


def _collect_source_files(graph: nx.MultiDiGraph, members: list[str]) -> set[str]:
    """收集社区成员关联的 source_file 集合（来自节点属性 + incident 边）。"""
    sources: set[str] = set()
    for member in members:
        if not graph.has_node(member):
            continue
        data = graph.nodes[member]
        sf = data.get("source_file")
        if isinstance(sf, str) and sf:
            sources.add(sf)
        for _, _, edge_data in graph.in_edges(member, data=True):
            esf = edge_data.get("source_file")
            if isinstance(esf, str) and esf:
                sources.add(esf)
        for _, _, edge_data in graph.out_edges(member, data=True):
            esf = edge_data.get("source_file")
            if isinstance(esf, str) and esf:
                sources.add(esf)
    return sources


def _summarize_community(members: list[str], llm: LLMClient | None) -> str:
    """为社区生成摘要。

    LLM 可用时：请求一句话概述成员间的关系主题。
    LLM 不可用 / 调用失败时：降级为启发式（成员名拼接，取前 10 个避免过长）。
    """
    if llm is None:
        return _heuristic_summary(members)

    try:
        member_list = ", ".join(members[:20])
        system = "You are a knowledge graph community summarizer."
        user = (
            f"Summarize the shared theme of the following knowledge graph nodes in one "
            f"concise sentence: {member_list}. Reply in the same language as the nodes."
        )
        text = llm.complete(system, user, response_format="text", temperature=0.0)
        summary = text.strip()
        if summary:
            return summary
    except Exception:
        logger.debug("community LLM summary failed; falling back to heuristic", exc_info=True)

    return _heuristic_summary(members)


def _heuristic_summary(members: list[str]) -> str:
    """启发式社区摘要：成员名拼接（取前 10 个避免过长）。"""
    preview = ", ".join(members[:10])
    suffix = f" 等 {len(members)} 个节点" if len(members) > 10 else ""
    return f"Community of: {preview}{suffix}"


def _write_communities(staging_dir: Path, result: CommunityResult) -> None:
    """原子写入 ``staging_dir/communities.json``。"""
    staging_dir.mkdir(parents=True, exist_ok=True)
    path = staging_dir / COMMUNITIES_FILENAME
    atomic_write_json(path, result.model_dump(mode="json"))


def load_communities(out_dir: Path) -> CommunityResult | None:
    """从 ``out/communities.json`` 加载社区结果；文件不存在返回 None。

    供 s1-feat-012 ``CommunityRetriever`` 消费。
    """
    path = out_dir / COMMUNITIES_FILENAME
    if not path.exists():
        return None
    import json

    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return CommunityResult.model_validate(data)


__all__ = [
    "COMMUNITIES_FILENAME",
    "Community",
    "CommunityResult",
    "detect_communities",
    "load_communities",
]

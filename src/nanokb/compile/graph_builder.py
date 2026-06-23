"""图谱编译器（方案 §3.4.4 + §3.5.1 step 5/6，Feature s1-feat-007）。

``GraphBuilder`` 在 ``networkx.MultiDiGraph`` 上提供四项核心操作：

- ``upsert``：按 ``(source_file, head, relation, tail)`` 主键先删旧边再插新边（Medium #9 幂等）；
  concept 节点描述按 source_file 覆盖合并（last-write-wins）。
- ``delete_by_source``：删 ``source_file`` 的全部边 + 清理 degree==0 孤立节点
  （Severe #1 删除传播；跨文件共享节点保留）。Medium #2 modified 路径复用此方法做"先清后建"。
- ``synthesize_fallback_descriptions``：对仍无 description 的节点用 incident edges
  合成 ``"{node}: {relation} {neighbor}; ..."`` 兜底描述（Opt #2 v3，Severe #2 向量侧防御层）。
  **v4 Medium #1**：pipeline 中必须在 ``VectorStore.index_nodes`` 之前调用，否则合成的
  兜底描述不会进入向量库，防御层失效。
- ``save_graph``：序列化 JSON 主（``node_link_data`` 保类型）+ GraphML 副（可视化导出）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx  # type: ignore[import-untyped]  # networkx 3.x 缺 py.typed 标记

from nanokb.config import Settings
from nanokb.models import ExtractionResult
from nanokb.utils.io import atomic_write_json

logger = logging.getLogger("nanokb")


class GraphBuilder:
    """MultiDiGraph 图谱编译器。

    构造期注入 ``graph``（流水线加载或新建的 ``MultiDiGraph``）与 ``settings``
    （控制 ``fallback_description_max_edges`` 等行为）。所有方法就地修改 ``graph``，
    不返回新图（``synthesize_fallback_descriptions`` 仅返回合成计数）。
    """

    def __init__(self, graph: nx.MultiDiGraph, settings: Settings) -> None:
        self._graph = graph
        self._settings = settings

    def upsert(self, result: ExtractionResult, source_file: str) -> None:
        """插入 triples 为边 + 合并 concepts 为节点描述（Medium #9 幂等）。

        幂等保证：每条三元组在插入前，先删除图中同 ``(head, tail, relation, source_file)``
        的既有边，确保同主键二次 upsert 不累积重复边。

        concept 节点描述：同名节点按 source_file 覆盖合并（last-write-wins）；
        新增节点缺省属性由 ``add_node`` 自动初始化。

        Args:
            result: 抽取结果（triples + concepts）。
            source_file: 本次 upsert 的来源文件标识，与 ``result`` 内 triple/concept 的
                ``source_file`` 一致（pipeline 保证）。
        """
        graph = self._graph

        # 1. 插入 triples：确保端点节点存在 → 按主键删旧边 → 插新边
        for triple in result.triples:
            head = triple.head
            tail = triple.tail
            graph.add_node(head)
            graph.add_node(tail)

            self._remove_edges_by_key(
                head=head,
                tail=tail,
                relation=triple.relation,
                source_file=triple.source_file,
            )

            graph.add_edge(
                head,
                tail,
                relation=triple.relation,
                source_file=triple.source_file,
                confidence=triple.confidence.value,
                track=triple.track.value,
                chunk_index=triple.chunk_index,
            )

        # 2. 合并 concepts：覆盖节点描述（last-write-wins）
        for concept in result.concepts:
            node = concept.name
            if not graph.has_node(node):
                graph.add_node(node)
            data = graph.nodes[node]
            data["description"] = concept.description or concept.name
            data["node_type"] = concept.node_type or "concept"
            data["source_file"] = concept.source_file
            data["confidence"] = concept.confidence.value
            if concept.extra:
                data["extra"] = dict(concept.extra)

    def delete_by_source(self, source_file: str) -> None:
        """删除 ``source_file`` 的全部边 + 清理 degree==0 孤立节点（Severe #1）。

        仅删除端点节点中 ``degree==0`` 者（删边后无任何残留边的孤立节点）；
        跨文件共享节点（仍被其他 source_file 的边引用，degree>0）保留。

        Medium #2：modified 路径在 ``upsert`` 新边前先调用此方法清旧边 + 孤立节点，
        保证"实体减少"无残留。
        """
        graph = self._graph

        edges_to_remove: list[tuple[str, str, Any]] = []
        affected_nodes: set[str] = set()
        for u, v, key, data in graph.edges(keys=True, data=True):
            if data.get("source_file") == source_file:
                edges_to_remove.append((u, v, key))
                affected_nodes.add(u)
                affected_nodes.add(v)

        if not edges_to_remove:
            return

        graph.remove_edges_from(edges_to_remove)

        # 仅检查受影响节点：degree==0 的孤立节点删除，degree>0 的跨文件共享节点保留
        isolated = [n for n in affected_nodes if graph.degree(n) == 0]
        if isolated:
            graph.remove_nodes_from(isolated)
            logger.debug(
                "delete_by_source(%s): removed %d isolated nodes", source_file, len(isolated)
            )

    def synthesize_fallback_descriptions(self) -> int:
        """对无 description 的节点用 incident edges 合成兜底描述（Opt #2 v3）。

        遍历图中所有节点，对 ``description`` 为空/缺失的节点，收集其 incident edges
        （先出边 node→tail 再入边 head→node，确定可复现），取前
        ``settings.fallback_description_max_edges`` 条，组装为
        ``"{node}: {relation} {neighbor}; ..."`` 形式描述。

        作为 Severe #2 的向量侧防御层：LLM 漏抽 Concept 的节点经此合成后拥有描述，
        确保 ``VectorStore.index_nodes`` 不会因空描述跳过。

        **v4 Medium #1**：pipeline 中必须在 ``index_nodes`` 之前调用（方案 §3.5.1
        step 6 先于 step 7）。

        Returns:
            本次合成描述的节点数。
        """
        graph = self._graph
        max_edges = self._settings.fallback_description_max_edges
        synthesized = 0

        for node, data in graph.nodes(data=True):
            existing = data.get("description")
            if isinstance(existing, str) and existing.strip():
                continue

            fragments = self._collect_fallback_fragments(node, max_edges)
            if not fragments:
                continue

            fallback = f"{node}: " + "; ".join(fragments)
            data["description"] = fallback
            data.setdefault("node_type", "entity")
            data.setdefault("confidence", "INFERRED")
            synthesized += 1

        if synthesized:
            logger.info("synthesized fallback descriptions for %d nodes", synthesized)
        return synthesized

    def save_graph(self, staging_dir: Path) -> None:
        """序列化图谱到 ``staging_dir``：JSON 主 + GraphML 副。

        - ``graph.json``：``networkx.node_link_data`` 序列化，保留全部属性类型
          （dict/None/int 等），作为权威图谱文件（供 pipeline 加载与 retriever 查询）。
        - ``graph.graphml``：GraphML 副本，仅供可视化工具消费；dict 属性序列化为 JSON
          字符串、None 丢弃、list/tuple 转 JSON 字符串（GraphML 属性类型有限，完整类型
          保真由 graph.json sidecar 承担）。

        两文件均先写同目录临时文件再 ``os.replace`` 原子切换，避免半写。``staging_dir``
        不存在时自动创建。
        """
        staging_dir.mkdir(parents=True, exist_ok=True)
        graph_json_path = staging_dir / "graph.json"
        graph_graphml_path = staging_dir / "graph.graphml"

        # JSON 主：node_link_data 保类型（原子写）
        data = nx.node_link_data(self._graph)
        atomic_write_json(graph_json_path, data)

        # GraphML 副：清洗 dict/None/list 后写（lxml writer 直写文件，先写 .tmp 再 replace 原子切换）
        sanitized = self._sanitize_for_graphml(self._graph)
        tmp_graphml = staging_dir / ".graph.graphml.tmp"
        try:
            nx.write_graphml(sanitized, tmp_graphml)
            tmp_graphml.replace(graph_graphml_path)
        except BaseException:
            if tmp_graphml.exists():
                tmp_graphml.unlink()
            raise

    # ── 内部辅助 ─────────────────────────────────────────────────────

    def _remove_edges_by_key(
        self, *, head: str, tail: str, relation: str, source_file: str
    ) -> None:
        """删除匹配 ``(head, tail, relation, source_file)`` 的全部 multi-edge（含各 key）。

        MultiDiGraph 允许同一对端点存在多条边（按 key 区分）；本方法扫描 ``head→tail``
        的所有 key，移除 ``relation`` 与 ``source_file`` 均匹配的边，保证主键幂等。
        """
        graph = self._graph
        if not graph.has_edge(head, tail):
            return
        keys_to_remove = [
            key
            for key, data in graph[head][tail].items()
            if data.get("relation") == relation
            and data.get("source_file") == source_file
        ]
        for key in keys_to_remove:
            graph.remove_edge(head, tail, key=key)

    def _collect_fallback_fragments(self, node: str, max_edges: int) -> list[str]:
        """收集节点 incident edges 的 ``"{relation} {neighbor}"`` 片段。

        顺序：先出边（node 是 head，neighbor 是 tail）再入边（node 是 tail，neighbor 是
        head），保证确定可复现。每条片段格式为 ``"{relation} {neighbor}"``。达到
        ``max_edges`` 条立即返回。
        """
        graph = self._graph
        fragments: list[str] = []

        for _, tail, data in graph.out_edges(node, data=True):
            relation = data.get("relation")
            if relation is None:
                continue
            fragments.append(f"{relation} {tail}")
            if len(fragments) >= max_edges:
                return fragments

        for head, _, data in graph.in_edges(node, data=True):
            relation = data.get("relation")
            if relation is None:
                continue
            fragments.append(f"{relation} {head}")
            if len(fragments) >= max_edges:
                return fragments

        return fragments

    @staticmethod
    def _sanitize_for_graphml(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """构造 GraphML 友好的图副本：dict/list → JSON 字符串，None → 丢弃。

        GraphML 属性类型仅支持 str/int/float/bool 等标量；dict 类型（如 concept.extra）
        会触发 ``nx.write_graphml`` 的 ``TypeError``。本方法不修改原始图，返回清洗后的副本，
        完整类型保真由 graph.json sidecar 承担。
        """
        sanitized = nx.MultiDiGraph()
        for node, data in graph.nodes(data=True):
            sanitized.add_node(node, **_sanitize_attrs(data))
        for u, v, key, data in graph.edges(keys=True, data=True):
            sanitized.add_edge(u, v, key=key, **_sanitize_attrs(data))
        return sanitized


def _sanitize_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """清洗单个属性 dict 以兼容 GraphML 标量约束。"""
    cleaned: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, dict):
            cleaned[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (list, tuple)):
            cleaned[key] = json.dumps(list(value), ensure_ascii=False)
        else:
            cleaned[key] = value
    return cleaned


__all__ = ["GraphBuilder"]

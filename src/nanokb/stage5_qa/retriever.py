"""图谱召回器（方案 §3.5.3 step 3-4 + Medium #10，Feature s1-feat-009）。

``GraphRetriever.recall`` 实现图路召回的完整链路（Opt #5 v3 降级：阶段 3 仅此路）：

1. **LLM NER**：从自然语言问题抽取实体提及（``parse_json_loose`` 容错）。
2. **normalize_entity 预处理**（Medium #10）：大小写/空白归一化，保证抽取端与查询端
   实体名比对一致（``Transformer`` / ``  transformer `` / ``TRANSFORMER`` 视作同一）。
3. **精确匹配**：归一化形式在图节点归一化索引中查找。
4. **fuzzy 兜底**：未命中走 ``difflib.get_close_matches``（cutoff=fuzzy_match_cutoff），
   避免大小写/前后缀不一致漏召回。
5. **N 跳 BFS 子图扩展**：从种子节点向外扩展 ``retrieval_hops`` 跳，收集子图。
6. **hit 构建**：子图边转 ``RetrievalHit``（score = confidence 权重 × 图路相似度 1.0）。

``VectorRetriever`` / ``CommunityRetriever`` / ``MultiRetriever`` 三路融合在
s1-feat-012 补全。本类的 ``SOURCE = "graph"`` 用于 ``RetrievalHit.source`` 字段。
"""

from __future__ import annotations

import difflib
import logging

import networkx as nx  # type: ignore[import-untyped]  # networkx 3.x 缺 py.typed 标记

from nanokb.config import Settings
from nanokb.llm.base import LLMClient, parse_json_loose
from nanokb.models import (
    Concept,
    Confidence,
    RetrievalHit,
    Triple,
)
from nanokb.stage3_compile.normalize import normalize_entity

logger = logging.getLogger("nanokb")

#: confidence 权重（与方案 §3.5.3 step 4 fuse 权重一致）
_CONFIDENCE_WEIGHT: dict[Confidence, float] = {
    Confidence.EXTRACTED: 1.0,
    Confidence.INFERRED: 0.6,
    Confidence.AMBIGUOUS: 0.3,
}

#: 图路相似度（精确图节点匹配视为满相似度）
_GRAPH_SIMILARITY: float = 1.0

_NER_SYSTEM_PROMPT = (
    "You are an entity recognizer. Read the user's question and extract all "
    "entity mentions as STRICT JSON.\n\n"
    'Output schema (return ONLY this object, no markdown fences, no prose):\n'
    '{"entities": ["Entity One", "Entity Two"]}\n'
    "\n"
    "Rules:\n"
    "- Each entity must be a short noun phrase as it appears (or could appear) "
    "in a knowledge graph.\n"
    "- Include both proper nouns and technical terms.\n"
    '- If no entities are present, return {"entities": []}.'
)


class GraphRetriever:
    """图谱召回器：NER → normalize → 查图 → fuzzy 兜底 → N 跳子图扩展。"""

    #: ``RetrievalHit.source`` 标识（s1-feat-012 的多路融合按此字段区分）。
    SOURCE = "graph"

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self._graph = graph
        self._llm = llm
        self._settings = settings

    def recall(self, question: str) -> list[RetrievalHit]:
        """从问题召回图谱 hit 列表。

        空图或 NER 抽不出任何可命中实体时返回空列表（上游据此返回
        ``未找到相关知识点``）。
        """
        if self._graph.number_of_nodes() == 0:
            return []

        entities = self._extract_entities(question)
        if not entities:
            logger.debug("recall: NER returned no entities for question: %s", question)
            return []

        seeds = self._collect_seed_nodes(entities)
        if not seeds:
            logger.debug("recall: no seed nodes matched; entities=%s", entities)
            return []

        subgraph = self._expand_subgraph(seeds)
        return self._build_hits(subgraph)

    # ── NER ──────────────────────────────────────────────────────────

    def _extract_entities(self, question: str) -> list[str]:
        """调用 LLM 抽取问题中的实体提及（parse_json_loose 容错）。"""
        raw = self._llm.complete(
            _NER_SYSTEM_PROMPT,
            question,
            response_format="json",
            temperature=0.0,
        )
        parsed = parse_json_loose(raw)
        if not isinstance(parsed, dict):
            logger.warning("recall: NER returned non-dict, falling back to empty")
            return []
        entities = parsed.get("entities")
        if not isinstance(entities, list):
            return []
        result: list[str] = []
        for ent in entities:
            if isinstance(ent, (str, int, float)):
                text = str(ent).strip()
                if text:
                    result.append(text)
        return result

    # ── 实体匹配（精确 + fuzzy） ─────────────────────────────────────

    def _collect_seed_nodes(self, entities: list[str]) -> set[str]:
        """对每个实体先做归一化精确匹配，未命中走 fuzzy 兜底。"""
        norm_index = self._build_normalized_node_index()
        all_norms = list(norm_index.keys())
        cutoff = self._settings.fuzzy_match_cutoff

        seeds: set[str] = set()
        for ent in entities:
            norm = normalize_entity(ent)
            if not norm:
                continue
            if norm in norm_index:
                seeds.update(norm_index[norm])
                continue
            if cutoff > 0.0 and all_norms:
                matches = difflib.get_close_matches(
                    norm, all_norms, n=3, cutoff=cutoff
                )
                for m in matches:
                    seeds.update(norm_index[m])
        return seeds

    def _build_normalized_node_index(self) -> dict[str, list[str]]:
        """构建 ``{normalize(node): [original_node, ...]}`` 索引。

        同一归一化形式可能对应多个原始节点（极少见但理论可能），全部保留。
        """
        index: dict[str, list[str]] = {}
        for node in self._graph.nodes():
            index.setdefault(normalize_entity(node), []).append(node)
        return index

    # ── N 跳子图扩展 ─────────────────────────────────────────────────

    def _expand_subgraph(self, seeds: set[str]) -> nx.MultiDiGraph:
        """从 seeds 做 BFS，收集 ``retrieval_hops`` 跳内的全部边与端点节点。

        种子节点本身无条件入图（即便无任何边）；BFS 每轮把当前层节点的出/入边
        端点纳入下一层。``retrieval_hops <= 0`` 时仅含种子节点本身。
        """
        sub = nx.MultiDiGraph()
        for seed in seeds:
            if seed in self._graph:
                sub.add_node(seed, **dict(self._graph.nodes[seed]))
            else:
                sub.add_node(seed)

        hops = self._settings.retrieval_hops
        if hops <= 0:
            return sub

        visited: set[str] = set()
        current: set[str] = {s for s in seeds if s in self._graph}

        for _ in range(hops):
            if not current:
                break
            next_level: set[str] = set()
            for node in current:
                if node in visited:
                    continue
                visited.add(node)

                for _, tail, key, data in self._graph.out_edges(node, keys=True, data=True):
                    sub.add_edge(node, tail, key=key, **dict(data))
                    if tail in self._graph:
                        sub.nodes[tail].update(dict(self._graph.nodes[tail]))
                        if tail not in visited:
                            next_level.add(tail)

                for head, _, key, data in self._graph.in_edges(node, keys=True, data=True):
                    sub.add_edge(head, node, key=key, **dict(data))
                    if head in self._graph:
                        sub.nodes[head].update(dict(self._graph.nodes[head]))
                        if head not in visited:
                            next_level.add(head)

            current = next_level

        return sub

    # ── hit 构建 ─────────────────────────────────────────────────────

    def _build_hits(self, subgraph: nx.MultiDiGraph) -> list[RetrievalHit]:
        """子图边转 RetrievalHit；无边时退化用种子节点描述构造 concept hit。"""
        hits: list[RetrievalHit] = []
        seen_keys: set[tuple[str, str, str, str]] = set()

        for u, v, _key, data in subgraph.edges(keys=True, data=True):
            relation = str(data.get("relation", ""))
            source_file = str(data.get("source_file", ""))
            dedup_key = (u, relation, v, source_file)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            confidence = _coerce_confidence(str(data.get("confidence", "")).upper())
            triple = Triple(
                head=u,
                relation=relation,
                tail=v,
                confidence=confidence,
                source_file=source_file,
            )
            hits.append(
                RetrievalHit(
                    triple=triple,
                    score=_CONFIDENCE_WEIGHT.get(confidence, 1.0) * _GRAPH_SIMILARITY,
                    source=self.SOURCE,
                )
            )

        if hits:
            return hits

        # 无边但有种子节点：退化用节点描述构造 concept hit（避免空召回漏报）
        for node, _data in subgraph.nodes(data=True):
            if node not in self._graph:
                continue
            raw_data = dict(self._graph.nodes[node])
            description = str(raw_data.get("description") or node)
            source_file = str(raw_data.get("source_file", ""))
            confidence = _coerce_confidence(str(raw_data.get("confidence", "")).upper())
            hits.append(
                RetrievalHit(
                    concept=Concept(
                        name=node,
                        description=description,
                        source_file=source_file,
                        confidence=confidence,
                    ),
                    score=_CONFIDENCE_WEIGHT.get(confidence, 1.0) * _GRAPH_SIMILARITY,
                    source=self.SOURCE,
                )
            )
        return hits


def _coerce_confidence(raw: str) -> Confidence:
    """安全转换 confidence 字符串为枚举；非法值降级 EXTRACTED（与 SemanticTrack 一致）。"""
    if not raw:
        return Confidence.EXTRACTED
    try:
        return Confidence(raw)
    except ValueError:
        logger.warning("unknown confidence %r, defaulting to EXTRACTED", raw)
        return Confidence.EXTRACTED


__all__ = ["GraphRetriever"]

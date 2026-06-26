"""三路召回融合（方案 §3.5.3 step 3-4 + §3.5.4，Feature s1-feat-009 + s1-feat-012）。

四块职责（s1-feat-012 完成三路融合）：

1. **GraphRetriever**（s1-feat-009）：NER → normalize → 查图 → fuzzy 兜底 → N 跳子图扩展。
   ``SOURCE = "graph"``。
2. **VectorRetriever**（s1-feat-012）：embed query → ChromaDB 近邻查询。``SOURCE = "vector"``。
3. **CommunityRetriever**（s1-feat-012）：NER → 社区成员匹配 → 社区摘要 hit。
   ``SOURCE = "community"``。
4. **MultiRetriever + fuse**（s1-feat-012）：按命令映射启用 retriever 子集
   （query=三路 / ask=仅向量 / search=仅社区），融合去重重排（confidence 权重 × 相似度），
   tiktoken 裁剪到 ``max_context_tokens``。

**score 语义统一约定**（s1-feat-012 拍定）：各 retriever 返回的 ``RetrievalHit.score``
一律为**原始相似度**（graph 精确匹配=1.0；vector=1.0−distance；community=NER 实体对社区
成员的覆盖率），confidence 权重不在此处乘入。``fuse`` 统一按
``_CONFIDENCE_WEIGHT[confidence] × score`` 计算最终融合分并排序——这样三路相似度口径
一致，融合排序唯一可复现（方案 §3.5.3 step 4）。
"""

from __future__ import annotations

import difflib
import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import networkx as nx  # type: ignore[import-untyped]

from nanokb.compile.normalize import normalize_entity
from nanokb.config import Settings
from nanokb.llm.base import EmbeddingClient, LLMClient, parse_json_loose
from nanokb.models import (
    Concept,
    Confidence,
    RetrievalHit,
    Triple,
)
from nanokb.qa.progress import (
    _SOURCE_LABELS,
    NullProgressReporter,
    ProgressReporter,
)
from nanokb.qa.prompt import render_hit

# 可选 fuzzy 加速后端（s2-feat-003）：rapidfuzz 的 fuzz.ratio 与 difflib ratio 同为
# Indel 相似度（数值相等，仅刻度 0-100 vs 0-1）；可用则 C 加速，不可用降级 difflib。
try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_process

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - optional dependency absent
    _HAS_RAPIDFUZZ = False
    _rf_fuzz = None  # type: ignore[assignment]
    _rf_process = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from nanokb.index.community import CommunityResult
    from nanokb.index.vector_store import VectorStore

logger = logging.getLogger("nanokb")

#: confidence 权重（与方案 §3.5.3 step 4 fuse 权重一致；fuse 据此 × 相似度排序）。
_CONFIDENCE_WEIGHT: dict[Confidence, float] = {
    Confidence.EXTRACTED: 1.0,
    Confidence.INFERRED: 0.6,
    Confidence.AMBIGUOUS: 0.3,
}

#: 图路相似度（精确图节点匹配视为满相似度，graph 路 recall 的原始相似度）。
_GRAPH_SIMILARITY: float = 1.0

#: VectorRetriever 默认召回数（提供给 ChromaDB search 的 k 值）。
_VECTOR_SEARCH_K: int = 10

_NER_SYSTEM_PROMPT = (
    "You are an entity recognizer. Read the user's question and extract all "
    "entity mentions as STRICT JSON.\n\n"
    "Output schema (return ONLY this object, no markdown fences, no prose):\n"
    '{"entities": ["Entity One", "Entity Two"]}\n'
    "\n"
    "Rules:\n"
    "- Each entity must be a short noun phrase as it appears (or could appear) "
    "in a knowledge graph.\n"
    "- Include both proper nouns and technical terms.\n"
    '- If no entities are present, return {"entities": []}.'
)


# ── Retriever 协议 ────────────────────────────────────────────────────


class Retriever(Protocol):
    """单路召回器协议：``recall(question)`` 返回 ``RetrievalHit`` 列表。

    实现方：``GraphRetriever`` / ``VectorRetriever`` / ``CommunityRetriever``。
    所有 retriever 返回的 ``hit.score`` 必须为**原始相似度**（不含 confidence 权重），
    权重乘入由 ``fuse`` 统一执行（s1-feat-012 拍定，保证三路口径一致）。
    """

    SOURCE: str

    def recall(self, question: str) -> list[RetrievalHit]:
        """从问题召回图谱/向量/社区 hit 列表。"""
        ...


# ── 共用 NER 辅助 ─────────────────────────────────────────────────────


def _ner_entities(llm: LLMClient, question: str) -> list[str]:
    """调用 LLM 抽取问题中的实体提及（``parse_json_loose`` 容错）。

    graph / community 两路都需要 NER，抽出来共享避免重复实现。
    """
    raw = llm.complete(
        _NER_SYSTEM_PROMPT,
        question,
        response_format="json",
        temperature=0.0,
    )
    parsed = parse_json_loose(raw)
    if not isinstance(parsed, dict):
        logger.warning("NER returned non-dict, falling back to empty")
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


# ── GraphRetriever（s1-feat-009） ─────────────────────────────────────


class GraphRetriever:
    """图谱召回器：NER → normalize → 查图 → fuzzy 兜底 → N 跳子图扩展。"""

    #: ``RetrievalHit.source`` 标识（``MultiRetriever`` 按此字段区分来源）。
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
        # 归一化索引懒缓存（s2-feat-001）：graph 实例生命周期内只读，
        # 首次 recall 时构建一次并复用，避免每次 recall 重建 O(N)。
        self._norm_index: dict[str, list[str]] | None = None
        # 构建计数（测试/诊断用，验证缓存命中）。
        self._norm_index_builds: int = 0
        # 长度桶懒缓存（s2-feat-002）：{len(name): [name,...]}，fuzzy 预筛用，
        # 与 _norm_index 同生命周期构建一次。
        self._norm_buckets: dict[int, list[str]] | None = None
        self._norm_buckets_builds: int = 0

    def recall(self, question: str) -> list[RetrievalHit]:
        """从问题召回图谱 hit 列表。

        空图或 NER 抽不出任何可命中实体时返回空列表（上游据此返回
        ``未找到相关知识点``）。
        """
        if self._graph.number_of_nodes() == 0:
            return []

        entities = _ner_entities(self._llm, question)
        if not entities:
            logger.debug("recall: NER returned no entities for question: %s", question)
            return []

        seeds = self._collect_seed_nodes(entities)
        if not seeds:
            logger.debug("recall: no seed nodes matched; entities=%s", entities)
            return []

        subgraph = self._expand_subgraph(seeds)
        return self._build_hits(subgraph)

    # ── 实体匹配（精确 + fuzzy） ─────────────────────────────────────

    def _collect_seed_nodes(self, entities: list[str]) -> set[str]:
        """对每个实体先做归一化精确匹配，未命中走 fuzzy 兜底。

        fuzzy 兜底（s2-feat-002）先用长度桶预筛候选——``difflib`` 的 ratio
        ``r = 2M/(la+lb)`` 要 ``>= cutoff`` 必须 ``min_len/max_len >= cutoff/(2-cutoff)``，
        故仅对长度落在 ``[ceil(L·cutoff/(2-cutoff)), floor(L·(2-cutoff)/cutoff)]`` 的
        归一化名跑 ``difflib.get_close_matches``。这是 ``difflib`` 自身
        ``real_quick_ratio`` 的同判据安全超集，命中集合不变，仅削减 Python 层循环开销。
        """
        norm_index = self._ensure_norm_index()
        cutoff = self._settings.fuzzy_match_cutoff

        seeds: set[str] = set()
        for ent in entities:
            norm = normalize_entity(ent)
            if not norm:
                continue
            if norm in norm_index:
                seeds.update(norm_index[norm])
                continue
            if cutoff > 0.0:
                matches = self._fuzzy_match(
                    norm, self._fuzzy_candidates(norm, cutoff), cutoff=cutoff, n=3
                )
                for m in matches:
                    seeds.update(norm_index[m])
        return seeds

    def _fuzzy_match(
        self, norm: str, candidates: list[str], *, cutoff: float, n: int = 3
    ) -> list[str]:
        """返回 ``cutoff`` 以上的近似匹配归一化名（s2-feat-003）。

        优先用 rapidfuzz C 后端（``fuzz.ratio`` 与 difflib ratio 数值相等，刻度 0-100，
        故 ``score_cutoff = cutoff * 100``）；rapidfuzz 不可用时降级回
        ``difflib.get_close_matches``。唯一差异是 ``n`` 边界并列项 tie-breaking 顺序
        可能不同——属可接受的 fuzzy 兜底容差（下游 ``_build_hits`` 有去重）。
        """
        if _HAS_RAPIDFUZZ:
            results = _rf_process.extract(
                norm,
                candidates,
                scorer=_rf_fuzz.ratio,
                score_cutoff=cutoff * 100.0,
                limit=n,
            )
            return [r[0] for r in results]
        return difflib.get_close_matches(norm, candidates, n=n, cutoff=cutoff)

    def _ensure_norm_index(self) -> dict[str, list[str]]:
        """懒构建并缓存 ``{normalize(node): [original_node, ...]}`` 索引（s2-feat-001）。

        graph 在 retriever 实例生命周期内视为只读，故索引构建一次后复用，
        避免每次 recall 重建 O(N)。同一归一化形式可能对应多个原始节点
        （极少见但理论可能），全部保留。
        """
        if self._norm_index is None:
            index: dict[str, list[str]] = {}
            for node in self._graph.nodes():
                index.setdefault(normalize_entity(node), []).append(node)
            self._norm_index = index
            self._norm_index_builds += 1
        return self._norm_index

    def _ensure_norm_buckets(self) -> dict[int, list[str]]:
        """懒构建并缓存 ``{len(name): [normalized_name, ...]}`` 长度桶（s2-feat-002）。

        与 ``_norm_index`` 同生命周期，构建一次复用。供 ``_fuzzy_candidates`` 按长度
        区间取候选，避免对全部归一化名进入 difflib 循环。
        """
        if self._norm_buckets is None:
            buckets: dict[int, list[str]] = {}
            for name in self._ensure_norm_index().keys():
                buckets.setdefault(len(name), []).append(name)
            self._norm_buckets = buckets
            self._norm_buckets_builds += 1
        return self._norm_buckets

    def _fuzzy_candidates(self, norm: str, cutoff: float) -> list[str]:
        """按长度安全超集返回 fuzzy 候选归一化名（s2-feat-002）。

        ``difflib`` ratio ``>= cutoff`` 的必要条件是
        ``min(L, M) / max(L, M) >= cutoff / (2 - cutoff)``，故候选长度 M 必落在
        ``[ceil(L·cutoff/(2-cutoff)), floor(L·(2-cutoff)/cutoff)]``。被排除的候选其
        ratio 必然 ``< cutoff``，``difflib.get_close_matches`` 本也不会命中——故此过滤
        是安全超集，不改变最终命中集合。
        """
        if cutoff <= 0.0:
            return []
        length = len(norm)
        if length == 0:
            return []
        buckets = self._ensure_norm_buckets()
        lo = max(1, math.ceil(length * cutoff / (2.0 - cutoff)))
        hi = math.floor(length * (2.0 - cutoff) / cutoff)
        candidates: list[str] = []
        for bucket_len in range(lo, hi + 1):
            candidates.extend(buckets.get(bucket_len, ()))
        return candidates

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
        """子图边转 RetrievalHit；score 为原始相似度（confidence 权重由 fuse 统一乘入）。

        无边时退化用种子节点描述构造 concept hit（避免空召回漏报）。
        """
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
                    score=_GRAPH_SIMILARITY,
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
                    score=_GRAPH_SIMILARITY,
                    source=self.SOURCE,
                )
            )
        return hits


# ── VectorRetriever（s1-feat-012） ────────────────────────────────────


class VectorRetriever:
    """向量召回器：embed query → ChromaDB 近邻查询。

    包装 ``index.vector_store.VectorStore.search``，返回 ``source="vector"``
    的 ``RetrievalHit`` 列表。``hit.score`` 为原始相似度（``1.0 - distance``），
    confidence 统一为 ``EXTRACTED``（向量召回无三元组置信度概念，由 fuse 权重为 1.0）。
    """

    SOURCE = "vector"

    def __init__(
        self,
        vector_store: VectorStore,
        llm: EmbeddingClient,
        settings: Settings,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._settings = settings

    def recall(self, question: str) -> list[RetrievalHit]:
        """向量语义召回：embed query → ChromaDB 近邻查询 → RetrievalHit 列表。

        ``VectorStore.search`` 失败（如 collection 损坏）时降级为空召回，不阻塞融合。
        """
        try:
            hits = self._vector_store.search(question, k=_VECTOR_SEARCH_K, embedder=self._llm)
        except Exception:
            logger.warning(
                "vector recall failed; degrading to empty",
                exc_info=True,
                extra={"stage": "qa-retriever"},
            )
            return []
        # VectorStore.search 已写入 source="vector"；统一显式标 SOURCE 防漂移。
        return [h.model_copy(update={"source": self.SOURCE}) for h in hits]


# ── CommunityRetriever（s1-feat-012） ─────────────────────────────────


class CommunityRetriever:
    """社区召回器：NER → 社区成员匹配 → 社区摘要 hit。

    对问题做 NER，把实体归一化后与各社区成员集合做交集；命中社区返回携带
    ``community_summary`` 的 ``RetrievalHit``。``hit.score`` 为实体覆盖率
    （命中社区成员数 / NER 实体数），反映问题与社区主题的相关度。
    """

    SOURCE = "community"

    def __init__(
        self,
        communities: CommunityResult,
        graph: nx.MultiDiGraph,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self._communities = communities
        self._graph = graph
        self._llm = llm
        self._settings = settings

    def recall(self, question: str) -> list[RetrievalHit]:
        """社区宏观召回：NER → 社区成员匹配 → 命中社区的摘要 hit 列表。"""
        if not self._communities.communities:
            return []

        entities = _ner_entities(self._llm, question)
        if not entities:
            return []

        norm_entities = {normalize_entity(e) for e in entities if normalize_entity(e)}
        if not norm_entities:
            return []

        hits: list[RetrievalHit] = []
        for comm in self._communities.communities:
            norm_members = {normalize_entity(m) for m in comm.members}
            overlap = norm_entities & norm_members
            if not overlap:
                continue
            similarity = len(overlap) / len(norm_entities)
            source_file = comm.source_files[0] if comm.source_files else ""
            hits.append(
                RetrievalHit(
                    community_summary=comm.summary,
                    concept=Concept(
                        name=f"community-{comm.id}",
                        description=comm.summary,
                        source_file=source_file,
                        confidence=Confidence.EXTRACTED,
                        node_type="community",
                    ),
                    score=similarity,
                    source=self.SOURCE,
                )
            )
        return hits


# ── MultiRetriever + fuse（s1-feat-012） ──────────────────────────────


class MultiRetriever:
    """多路召回编排器：按命令映射启用 retriever 子集，融合 fuse 重排。

    用法：``MultiRetriever([g, v, c], settings, llm).recall(question)`` —— 依次调用
    各 retriever.recall，合并结果后经 ``fuse`` 去重重排 + tiktoken 裁剪。

    任一 retriever 抛错被捕获并降级为空召回（不阻塞其他路），符合"三路可并行召回
    后 fuse"的容错语义（方案 technical_notes）。
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        settings: Settings,
        llm: LLMClient,
        *,
        progress: ProgressReporter | None = None,
    ) -> None:
        self._retrievers = list(retrievers)
        self._settings = settings
        self._llm = llm
        self._progress: ProgressReporter = progress or NullProgressReporter()

    def recall(self, question: str) -> list[RetrievalHit]:
        """依次调用各 retriever，合并 → fuse（去重 + confidence 权重排序 + 裁剪）。

        每路召回与 fuse 各自包进一个 ``progress.stage``，向 CLI 报告检索进度
        （``progress=None`` 时为空实现，行为与无进度反馈一致）。
        """
        all_hits: list[RetrievalHit] = []
        for retriever in self._retrievers:
            label = _SOURCE_LABELS.get(retriever.SOURCE, retriever.SOURCE)
            with self._progress.stage(f"{label}中..."):
                try:
                    hits = retriever.recall(question)
                except Exception:
                    logger.warning(
                        "retriever %s failed; degrading to empty",
                        type(retriever).__name__,
                        exc_info=True,
                        extra={"stage": "qa-retriever"},
                    )
                    hits = []
                all_hits.extend(hits)
        with self._progress.stage("融合重排中..."):
            return fuse(all_hits, self._settings, self._llm)


def fuse(
    hits: list[RetrievalHit],
    settings: Settings,
    llm: LLMClient,
) -> list[RetrievalHit]:
    """三路融合：去重 → confidence 权重 × 相似度排序 → tiktoken 裁剪。

    步骤（方案 §3.5.3 step 4）：
    1. **去重**：triple hit 按 ``(head, relation, tail, source_file, retriever)``，
       concept hit 按 ``(name, retriever)``，community hit 按 ``community_summary``。
       同键保留首个（已按 retriever 调用顺序排列）。
    2. **重排**：按 ``_CONFIDENCE_WEIGHT[confidence] × hit.score`` 降序稳定排序
       —— EXTRACTED(1.0) > INFERRED(0.6) > AMBIGUOUS(0.3)，相同加权分保留入序。
    3. **裁剪**：逐条 ``render_hit`` + ``llm.count_tokens`` 累加到 ``max_context_tokens``
       停止（至少保留 1 条，避免完全空）——复用 ``prompt`` 模块的渲染口径。

    Args:
        hits: 三路召回合并后的原始 hit 列表（``hit.score`` 为原始相似度）。
        settings: 提供 ``max_context_tokens`` 裁剪阈值。
        llm: 提供 ``count_tokens`` 做 tiktoken 精确计数。

    Returns:
        融合后的 hit 列表，按 confidence 权重 × 相似度降序排列，已裁剪到上限内。
    """
    if not hits:
        return []

    deduped = _dedup_hits(hits)
    ranked = sorted(deduped, key=_weighted_score, reverse=True)
    return _truncate_to_context(ranked, settings, llm)


# ── fuse 内部辅助 ─────────────────────────────────────────────────────


def _weighted_score(hit: RetrievalHit) -> float:
    """计算融合分：``_CONFIDENCE_WEIGHT[confidence] × hit.score``。

    confidence 从 ``triple.confidence`` / ``concept.confidence`` 读取；二者均无时
    默认 EXTRACTED（社区摘要、无 triple 的 concept hit）。
    """
    confidence = _hit_confidence(hit)
    weight = _CONFIDENCE_WEIGHT.get(confidence, 1.0)
    return weight * hit.score


def _hit_confidence(hit: RetrievalHit) -> Confidence:
    """从 hit 读取 confidence（triple 优先，其次 concept，默认 EXTRACTED）。"""
    if hit.triple is not None:
        return hit.triple.confidence
    if hit.concept is not None:
        return hit.concept.confidence
    return Confidence.EXTRACTED


def _dedup_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """按 hit 类型去重，同键保留首个。

    - triple hit：``(head, relation, tail, source_file, source)`` —— 不同 retriever
      对同一三元组的召回都保留（source 不同视为不同证据）。
    - concept hit（无 triple）：``(concept.name, source)``。
    - community hit：``community_summary``（摘要文本相同视为同一社区）。
    """
    seen_triples: set[tuple[str, str, str, str, str]] = set()
    seen_concepts: set[tuple[str, str]] = set()
    seen_communities: set[str] = set()
    result: list[RetrievalHit] = []

    for hit in hits:
        if hit.triple is not None:
            t = hit.triple
            triple_key = (t.head, t.relation, t.tail, t.source_file, hit.source)
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)
            result.append(hit)
            continue
        if hit.concept is not None and hit.community_summary is None:
            concept_key = (hit.concept.name, hit.source)
            if concept_key in seen_concepts:
                continue
            seen_concepts.add(concept_key)
            result.append(hit)
            continue
        if hit.community_summary is not None:
            if hit.community_summary in seen_communities:
                continue
            seen_communities.add(hit.community_summary)
            result.append(hit)
    return result


def _truncate_to_context(
    hits: list[RetrievalHit],
    settings: Settings,
    llm: LLMClient,
) -> list[RetrievalHit]:
    """按 tiktoken 计数裁剪 hits 到 ``max_context_tokens``（至少保留 1 条）。

    复用 ``prompt.render_hit`` 的渲染口径，保证裁剪与最终上下文渲染口径一致
    （technical_notes："fuse 的 tiktoken 裁剪复用 prompt 模块"）。
    """
    max_tokens = settings.max_context_tokens
    kept: list[RetrievalHit] = []
    running = 0
    for hit in hits:
        line = render_hit(hit)
        if not line:
            continue
        line_tokens = llm.count_tokens(line)
        if kept and running + line_tokens > max_tokens:
            break
        kept.append(hit)
        running += line_tokens
        if running >= max_tokens:
            break
    return kept


# ── 通用辅助 ──────────────────────────────────────────────────────────


def _coerce_confidence(raw: str) -> Confidence:
    """安全转换 confidence 字符串为枚举；非法值降级 EXTRACTED（与 SemanticTrack 一致）。"""
    if not raw:
        return Confidence.EXTRACTED
    try:
        return Confidence(raw)
    except ValueError:
        logger.warning("unknown confidence %r, defaulting to EXTRACTED", raw)
        return Confidence.EXTRACTED


#: MultiRetriever 工厂签名（按命令映射构造 retriever 子集，pipeline 注入）
RetrieverFactory = Callable[[str, nx.MultiDiGraph, LLMClient, Settings], list[Retriever]]


__all__ = [
    "CommunityRetriever",
    "GraphRetriever",
    "MultiRetriever",
    "Retriever",
    "RetrieverFactory",
    "VectorRetriever",
    "fuse",
]

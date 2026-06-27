"""编译流水线编排（方案 §3.5.1 + §3.5.5，Feature s1-feat-008）+
问答流程编排（方案 §3.5.3，Feature s1-feat-009）。

**两阶段结构**（v3 Medium #1 一致性核心）：

- **阶段 A（抽取，失败安全）**：detect_changes → ingest → SemanticTrack.extract，
  全程不触碰 chroma/triples.jsonl/graph 写操作。任一文件抽取失败仅记日志标 skip，
  不影响已成功抽取的文件。
- **阶段 B（破坏性变更）**：抽取全部完成后统一执行，缩小一致性窗口：
  - step 3 deletion 级联（Severe #1）：graph_builder.delete_by_source +
    vector_store.delete_by_source + triples.jsonl delete 标记 + manifest pop。
  - step 4 modified 先清后建（Medium #2）：graph_builder.delete_by_source +
    vector_store.delete_by_source（在 upsert 之前）。
  - step 5 图构建（无向量，v4 拆分）：triples.jsonl upsert 追加 + graph_builder.upsert。
  - step 6 synthesize_fallback_descriptions（Opt #2 v3 + v4 Medium #1）：
    兜底描述合成，全局一次，必须在 step 7 之前。
  - step 7 向量索引（v4 新增独立步骤）：vector_store.index_nodes（逐 path 子图）。
  - step 8 build_indexes（community + keyword）—— s1-feat-011 实现（staging_swap
    跳过缺失的 communities.json/keywords.json）。
  - step 9 manifest 更新（仅成功抽取的 path）。
  - step 10-11 staging 五件套原子切换（manifest 最后写）。

**replay**（§3.5.5 去重收敛规则）：从 triples.jsonl 重建图谱，跳过 step 7（不重建
ChromaDB），不消耗 LLM chat Token。

**向量侧扩展点**：``VectorStoreBackend`` Protocol 定义流水线所需的最小向量库接口
（delete_by_source / index_nodes）。s1-feat-011 的 ``VectorStore``（ChromaDB）满足此协议。
``vector_store=None`` 时，compile 自动构造真实 ``VectorStore``（探测 embedding 维度后），
使 CLI ``build`` 端到端产出 ChromaDB 向量；测试可注入 FakeVectorStore 替换。

**问答流程**（``answer_query``）：阶段 3 仅 graph 路（Opt #5 降级），冷启动校验
（graph.json 不存在或 raw/ 为空 → ``ColdStartError`` exit 1）。vector/community
三路融合在 s1-feat-012 接入。
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import networkx as nx  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from nanokb.compile import GraphBuilder
from nanokb.config import Settings
from nanokb.config_signature import (
    embedding_config_signature,
    extraction_config_signature,
    index_config_signature,
)
from nanokb.extract import build_default_extractor
from nanokb.extract.base import Extractor
from nanokb.extract.cache import ExtractionCache
from nanokb.index import VectorStore, build_indexes
from nanokb.index.community import CommunityResult, load_communities
from nanokb.llm.base import EmbeddingClient, LLMClient, make_embedding_client, make_llm_client
from nanokb.llm.embed_cache import EmbeddingCache
from nanokb.load.detector import (
    SUPPORTED_SUFFIXES,
    ChangeSet,
    detect_changes,
)
from nanokb.load.ingest import ingest_file
from nanokb.loaders import (
    CodeLoader,
    LoaderRegistry,
    UnstructuredLoader,
    UnsupportedFormatError,
)
from nanokb.models import (
    Answer,
    Concept,
    ExtractionResult,
    FileState,
    Manifest,
    RetrievalHit,
    Triple,
)
from nanokb.qa.generator import generate
from nanokb.qa.progress import (
    NullProgressReporter,
    ProgressReporter,
)
from nanokb.qa.prompt import compile_context
from nanokb.qa.retriever import (
    CommunityRetriever,
    GraphRetriever,
    MultiRetriever,
    Retriever,
    VectorRetriever,
)
from nanokb.qa.review import (
    ReviewQueue,
    collect_entities,
    determine_reason,
)
from nanokb.utils.io import atomic_write_json, staging_swap
from nanokb.utils.progress import BuildProgressWriter, BuildStage

logger = logging.getLogger("nanokb")

# ── 常量 ──────────────────────────────────────────────────────────────

#: triples.jsonl 文件名（best-effort 追加写，非 staging 覆盖范围）
TRIPLES_FILENAME = "triples.jsonl"

#: 当前 schema 版本（写入每条 triples.jsonl 记录 + manifest.version）
SCHEMA_VERSION = "2"

#: staging 目录名
STAGING_DIRNAME = ".staging"


# ── 结果数据结构 ──────────────────────────────────────────────────────


class CompileResult(BaseModel):
    """compile() 返回值：变更摘要。"""

    changes: ChangeSet = Field(default_factory=ChangeSet)
    extracted_count: int = 0
    skipped: list[str] = Field(default_factory=list)
    synthesized_fallback_count: int = 0
    cached_count: int = 0


class ReplayResult(BaseModel):
    """replay() 返回值：重放摘要。"""

    rebuilt_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    synthesized_fallback_count: int = 0


class AnswerQueryResult(BaseModel):
    """answer_query() 返回值：答案 + 召回命中（供 CLI 展示引用/调试）。"""

    answer: Answer
    hits: list[RetrievalHit] = Field(default_factory=list)


class ColdStartError(RuntimeError):
    """图谱未编译（冷启动，Opt #8）——CLI 应提示并 exit 1。"""


#: 问答模式（方案 §3.5.3 命令语义映射）：
#: - ``query``：graph + vector + community 三路融合（图谱推理问答）。
#: - ``ask``：仅向量路（语义模糊问答）。
#: - ``search``：仅社区路（社区宏观检索，``--community``）。
AnswerMode = Literal["query", "ask", "search"]


# ── 向量库后端协议（s1-feat-011 VectorStore 满足） ────────────────────


@runtime_checkable
class VectorStoreBackend(Protocol):
    """流水线所需的最小向量库后端接口。

    s1-feat-011 的 ``VectorStore`` 实现将满足此协议。当前 stage4 尚未实现，
    ``compile(vector_store=None)`` 时向量操作被跳过（仅影响 ChromaDB 一致性，
    graph/triples.jsonl/manifest 主线不受影响）。

    round 3（Opt#4）：``index_nodes`` 新增可选关键字参数 ``embed_fn`` / ``on_progress``，
    默认 None 时走原 ``llm.embed`` 路径（零回归），保证测试 mock VectorStore 仍兼容。
    """

    def delete_by_source(self, source_file: str) -> None:
        """删除 ``source_file`` 的全部向量（where={"source_file":source_file}）。"""
        ...

    def index_nodes(
        self,
        graph: nx.MultiDiGraph,
        llm: EmbeddingClient,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """为图中每个节点的 description 生成 embedding 并 upsert。

        v4 Medium #1：调用前须确保 fallback 描述已合成（pipeline 保证
        synthesize_fallback_descriptions 先于 index_nodes 执行）。

        round 3 Opt#4：``embed_fn`` 提供时一次性接收全部 description 文本，
        通常为 ``EmbeddingCache.embed_batch``；为 None 时走 ``llm.embed``。
        """
        ...


# ── 编译 ──────────────────────────────────────────────────────────────


def compile(  # noqa: A001  — 故意与内建同名，方案 §3.5.1 指定
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    registry: LoaderRegistry | None = None,
    extractor_factory: Callable[[LLMClient, Settings], Extractor] | None = None,
    vector_store: VectorStoreBackend | None = None,
    embedding_client: EmbeddingClient | None = None,
    force: bool = False,
) -> CompileResult:
    """执行编译流水线（两阶段结构 + v4 step 5/6/7 三段时机）。

    Args:
        settings: 全局配置（raw_dir / out_dir / 模型身份等）。
        llm: LLM 客户端（生文抽取）；``None`` 时通过 ``make_llm_client(settings)`` 创建
            （缺 API key 会 exit 2）。
        registry: 文档加载注册表；``None`` 时用默认（UnstructuredLoader）。
        extractor_factory: 自定义抽取器工厂；``None`` 时用默认分发抽取器
            （代码文件 → CodeTrack，其余 → SemanticTrack）。
        vector_store: 向量库后端；``None`` 时跳过向量操作（stage4 未实现阶段）。
        embedding_client: 向量嵌入客户端；``None`` 时经 ``_resolve_embedder`` 解析
            （未配置独立 embedding 端点时复用 ``llm``，否则 ``make_embedding_client``）。
        force: ``True`` 时忽略 manifest 全量重编译（空 manifest + 空图起步）。

    Returns:
        ``CompileResult`` —— 变更 / 抽取 / 跳过 / 兜底合成计数摘要。
    """
    out_dir = settings.out_dir
    raw_dir = settings.raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if registry is None:
        registry = build_default_registry()

    # 加载或初始化状态
    if force:
        manifest = Manifest()
        graph: nx.MultiDiGraph = nx.MultiDiGraph()
    else:
        manifest = _load_manifest(out_dir)
        graph = _load_graph(out_dir)

    graph_builder = GraphBuilder(graph, settings)

    # ── 运行时进度文件（Feature s3-feat-004）──────────────────────────
    # 跨进程可见：build 周期性原子写 out/.build_progress.json，status 只读。
    # enabled=settings.enable_build_progress（默认 True）；False 时 writer 全方法
    # no-op、不起 heartbeat 线程（零回归）。except 在 compile 内部覆盖
    # KeyboardInterrupt / Exception（os._exit 兼容，不依赖 finally/atexit）。
    writer = BuildProgressWriter(out_dir, force, enabled=settings.enable_build_progress)

    try:
        # ── 阶段 A：抽取（失败安全，不触碰 graph/chroma/triples.jsonl 写） ──
        writer.set_stage(BuildStage.DETECT)
        changes = detect_changes(raw_dir, manifest, settings)

        if not force and not changes.has_changes:
            logger.info("no changes detected; skipping compilation")
            writer.done()
            return CompileResult(changes=changes)

        to_process_preview = sorted(set(changes.added) | set(changes.modified))
        logger.info(
            "changes detected: %d added, %d modified, %d deleted — %d file(s) to extract",
            len(changes.added),
            len(changes.modified),
            len(changes.deleted),
            len(to_process_preview),
            extra={"stage": "compile-detect"},
        )

        if llm is None:
            llm = make_llm_client(settings)

        # 生文与向量解耦：embedder 可独立于 chat llm（生文 DeepSeek + embedding GLM/ollama）。
        # _resolve_embedder 在未配置独立 embedding 端点时复用 llm（向后兼容）。
        embedder = _resolve_embedder(settings, embedding_client, llm)

        logger.info("probing embedding dimensions...", extra={"stage": "compile-probe"})
        # s1-feat-011: 向量库后端。vector_store=None 时自动构造真实 ChromaDB VectorStore，
        # 使 CLI build 端到端产出向量。探测 embedding 维度保证 VectorStore 元数据与 embedder.embed
        # 实际输出维度一致（Medium #7 维度校验依赖此一致性）。
        actual_embedding_dim = _probe_embedding_dim(embedder)
        if vector_store is None:
            vector_store = VectorStore(
                settings.out_dir / "chroma",
                settings.embedding_model,
                actual_embedding_dim,
            )

        extractor: Extractor
        if extractor_factory is not None:
            extractor = extractor_factory(llm, settings)
        else:
            # 默认按扩展名分发：代码文件（.py/.js/.java）走 CodeTrack（零 Token），
            # 其余走 SemanticTrack（LLM 抽取）。s1-feat-010。
            extractor = build_default_extractor(llm, settings)

        results_map: dict[str, ExtractionResult] = {}
        sha_map: dict[str, str] = {}
        skipped: list[str] = []

        cache = ExtractionCache(settings.out_dir / "extract_cache")
        extraction_sig = extraction_config_signature(settings)
        cached_count = 0

        to_process = sorted(set(changes.added) | set(changes.modified))

        def _process_one_file(
            path: str,
        ) -> tuple[str, ExtractionResult | None, str | None, bool, bool]:
            """单文件处理：ingest → cache.get → extract → cache.put（方案 §3.4，Feature s1-feat-004）。

            返回 ``(path, result_or_None, sha256_or_None, is_skipped, was_cached)``。
            线程安全：``ingest_file`` / ``cache`` 无共享状态；``extractor`` 共享单例
            （``DefaultExtractor`` DCL + ``SemanticTrack`` 无状态），可跨文档并发调用。
            worker 只返回不可变元组，``results_map``/``sha_map``/``skipped``/``cached_count``
            由主线程在归并时写（无锁、无竞态）。
            """
            abs_path = raw_dir / path
            logger.info("ingesting %s ...", path, extra={"stage": "compile-ingest", "file": path})
            try:
                doc = ingest_file(abs_path, raw_dir, registry, settings)
            except UnsupportedFormatError as exc:
                logger.warning(
                    "skip unsupported file: %s (%s)",
                    path,
                    exc,
                    extra={"stage": "compile-ingest", "file": path},
                )
                return (path, None, None, True, False)
            except Exception:
                logger.exception(
                    "ingest failed for %s",
                    path,
                    extra={"stage": "compile-ingest", "file": path},
                )
                return (path, None, None, True, False)

            cached = cache.get(doc.sha256, extraction_sig, settings.llm_model)
            if cached is not None:
                logger.info(
                    "cache hit %s → %d triples, %d concepts",
                    path,
                    len(cached.triples),
                    len(cached.concepts),
                    extra={"stage": "compile-extract", "file": path},
                )
                return (path, _normalize_result_source(cached, path), doc.sha256, False, True)

            try:
                result = extractor.extract(doc)
            except Exception:
                logger.exception(
                    "extraction failed for %s",
                    path,
                    extra={"stage": "compile-extract", "file": path},
                )
                return (path, None, None, True, False)

            try:
                cache.put(doc.sha256, extraction_sig, settings.llm_model, result)
            except Exception:
                logger.warning(
                    "cache put failed for %s; result kept in memory",
                    path,
                    extra={"stage": "compile-extract", "file": path},
                )

            logger.info(
                "extracted %s → %d triples, %d concepts",
                path,
                len(result.triples),
                len(result.concepts),
                extra={"stage": "compile-extract", "file": path},
            )
            return (path, _normalize_result_source(result, path), doc.sha256, False, False)

        def _merge_outcome(
            path: str,
            result: ExtractionResult | None,
            sha: str | None,
            is_skipped: bool,
            was_cached: bool,
        ) -> None:
            """主线程归并单个文件结果到 results_map/sha_map/skipped/cached_count。

            Feature s3-feat-004：同步推进 BuildProgressWriter.extract 计数（completed /
            cached / skipped delta），使 status 跨进程可见抽取进度。``_merge_outcome``
            始终在主线程执行（串行循环或 ``as_completed`` 归并），writer 内部自带锁。
            """
            nonlocal cached_count
            if is_skipped:
                skipped.append(path)
                writer.update_extract(skipped_delta=1)
                return
            assert result is not None and sha is not None
            results_map[path] = result
            sha_map[path] = sha
            if was_cached:
                cached_count += 1
            writer.update_extract(
                completed_delta=1,
                cached_delta=1 if was_cached else 0,
            )

        # ── 阶段 A：抽取（可配置文档级并发；失败安全——任一文件失败仅记日志标 skip） ──
        # concurrency<=1 串行回退（默认，零回归）；>1 用 ThreadPoolExecutor 并发，
        # 主线程 as_completed 归并。阶段 B 在 with 块退出（全部 join）后才执行。
        writer.set_stage(BuildStage.EXTRACT)
        writer.update_extract(total=len(to_process), force_flush=True)
        doc_concurrency = max(1, settings.extract_doc_concurrency)
        if doc_concurrency == 1:
            for path in to_process:
                _merge_outcome(*_process_one_file(path))
        else:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=doc_concurrency)
            futures = {pool.submit(_process_one_file, path): path for path in to_process}
            try:
                for future in concurrent.futures.as_completed(futures):
                    path = futures[future]
                    try:
                        outcome = future.result()
                    except Exception:
                        # 兜底：_process_one_file 内部异常未捕获时的最终隔离
                        logger.exception(
                            "unexpected failure processing %s",
                            path,
                            extra={"stage": "compile-extract", "file": path},
                        )
                        skipped.append(path)
                        continue
                    _merge_outcome(*outcome)
                pool.shutdown(wait=True)
            except KeyboardInterrupt:
                # Ctrl-C：取消排队任务、不阻塞等待在途 worker（与 semantic_track 一致）；
                # 失败安全——阶段 A 抽取不触碰 graph/triples/vector 写入。
                pool.shutdown(wait=False, cancel_futures=True)
                raise

        # ── 阶段 B：破坏性变更（抽取全部成功后统一执行） ──────────────────
        writer.set_stage(BuildStage.GRAPH)

        # step 3: deletion 级联（Severe #1）
        for path in changes.deleted:
            graph_builder.delete_by_source(path)
            if vector_store is not None:
                vector_store.delete_by_source(path)
            _append_triples_log(
                out_dir,
                {
                    "schema_version": manifest.version,
                    "op": "delete",
                    "source_file": path,
                    "ts": _now_iso(),
                },
            )
            manifest.files.pop(path, None)

        # step 4: modified 先清后建（Medium #2）——在 upsert 之前清旧边/旧向量
        for path in changes.modified:
            if path not in results_map:
                continue
            graph_builder.delete_by_source(path)
            if vector_store is not None:
                vector_store.delete_by_source(path)

        # step 5: added/modified 图构建（无向量，v4 拆分独立小阶段）
        logger.info(
            "building graph from %d file(s)...",
            len(results_map),
            extra={"stage": "compile-graph"},
        )
        for path in to_process:
            if path not in results_map:
                continue
            result = results_map[path]
            _append_triples_log(
                out_dir,
                {
                    "schema_version": manifest.version,
                    "op": "upsert",
                    "source_file": path,
                    "triples": [t.model_dump(mode="json") for t in result.triples],
                    "concepts": [c.model_dump(mode="json") for c in result.concepts],
                    "ts": _now_iso(),
                },
            )
            graph_builder.upsert(result, path)

        # step 6: synthesize_fallback_descriptions（Opt #2 v3 + v4 Medium #1）
        # 必须在 step 7（index_nodes）之前——漏抽 Concept 的节点经此合成后才有描述，
        # 否则 index_nodes 会因空描述跳过这些节点。
        fallback_count = graph_builder.synthesize_fallback_descriptions()

        # step 7: 向量索引（v4 新增独立步骤）——逐 path 子图，描述已就绪
        # round 2 Severe #1 + round 3 Opt#4：构造 EmbeddingCache（cache 与并发正交，
        # Medium #4——enable_embed_cache=False 时仍构造，get/put 为 no-op，embed_batch
        # 仍提供串行/并发 embed），把 cache.embed_batch 作为 embed_fn 注入 index_nodes。
        # Feature s3-feat-005：total_vector_nodes 在 if 块外初始化，确保下方 manifest
        # 字段写入时变量必定存在（vector_store 为 None 时为 0）。
        total_vector_nodes = 0
        if vector_store is not None:
            writer.set_stage(BuildStage.VECTOR)
            logger.info(
                "indexing vectors (embedding model: %s)...",
                settings.embedding_model,
                extra={"stage": "compile-vector"},
            )
            embed_cache = EmbeddingCache(
                out_dir / "embed_cache",
                embedding_model=settings.embedding_model,
                embedding_dim=actual_embedding_dim,
                embedder=embedder,
                embed_concurrency=settings.embed_concurrency,
                enable_cache=settings.enable_embed_cache,
            )
            # Feature s3-feat-004：预计算待索引节点总数（跨子图汇总）写入 vector.total_nodes，
            # 每子图索引后 update_vector(indexed_delta=...) 推进进度。
            # 一次性构建 source_file → nodes 倒排（单次 O(E)），避免逐 path
            # 全图遍历（旧 _subgraph_for_source 对每个 path 扫全图边，N paths × E
            # 边 = 百亿次操作，是 vector 阶段长时间无输出的根因）。
            source_to_nodes: dict[str, set[str]] = {}
            for u, v, edata in graph.edges(data=True):
                sf = edata.get("source_file")
                if sf:
                    source_to_nodes.setdefault(sf, set()).update((u, v))

            subgraphs_to_index: list[tuple[str, nx.MultiDiGraph]] = []
            for path in to_process:
                if path not in results_map:
                    continue
                nodes = source_to_nodes.get(path)
                if not nodes:
                    continue
                sub = nx.MultiDiGraph()
                for node in nodes:
                    if not graph.has_node(node):
                        continue
                    ndata = dict(graph.nodes[node])
                    ndata.setdefault("source_file", path)
                    sub.add_node(node, **ndata)
                if sub.number_of_nodes() > 0:
                    subgraphs_to_index.append((path, sub))
                    total_vector_nodes += sub.number_of_nodes()
            writer.update_vector(total=total_vector_nodes, force_flush=True)
            for _path, subgraph in subgraphs_to_index:
                vector_store.index_nodes(
                    subgraph, embedder, embed_fn=embed_cache.embed_batch
                )
                writer.update_vector(indexed_delta=subgraph.number_of_nodes())

        # step 8: build_indexes（community + keyword）——s1-feat-011
        # 由 _finalize_staging 内部调用 build_indexes 写入 staging 目录。
        writer.set_stage(BuildStage.INDEX)

        # step 9: manifest 更新（仅 results_map 中成功抽取的 path，Opt #2 v4）
        now = _now_iso()
        for path in sorted(results_map):
            manifest.files[path] = FileState(
                path=path,
                sha256=sha_map.get(path, ""),
                processed_at=now,
                extractor_version=settings.extractor_version,
                llm_model=settings.llm_model,
                embedding_model=settings.embedding_model,
                embedding_dim=actual_embedding_dim,
                extraction_config=extraction_config_signature(settings),
                index_config=index_config_signature(settings),
                embedding_config=embedding_config_signature(settings),
            )

        # Feature s3-feat-005：Manifest 顶层 2.x 增量字段（status 静态展示用）。
        # total_vectors 取 step 7 跨子图汇总的节点数（vector_store 为 None 时为 0）；
        # last_compiled_at / last_llm_model / last_embedding_model 记录本次编译身份。
        # 写在 staging_swap 之前确保原子切换后 out/manifest.json 携带这些字段。
        manifest.total_vectors = total_vector_nodes
        manifest.last_compiled_at = now
        manifest.last_llm_model = settings.llm_model
        manifest.last_embedding_model = settings.embedding_model

        # step 10-11: 序列化到 staging + 原子切换（manifest 最后写）
        logger.info(
            "building community + keyword indexes, writing to disk...",
            extra={"stage": "compile-finalize"},
        )
        # llm=None: 社区摘要用启发式（成员名拼接，零 Token），避免给已有 LLM 调用计数
        # 断言的集成测试引入额外 complete 调用。社区 LLM 摘要可经 detect_communities
        # 直接调用时启用（llm 非 None），或后续 feature 增设开关接入。
        _finalize_staging(
            graph_builder,
            manifest,
            out_dir,
            graph=graph,
            settings=settings,
            llm=None,
        )

        writer.set_stage(BuildStage.FINALIZE)
        logger.info(
            "compile done: added=%d modified=%d deleted=%d extracted=%d skipped=%d fallback=%d",
            len(changes.added),
            len(changes.modified),
            len(changes.deleted),
            len(results_map),
            len(skipped),
            fallback_count,
            extra={"stage": "compile"},
        )

        writer.done()
        return CompileResult(
            changes=changes,
            extracted_count=len(results_map),
            skipped=skipped,
            synthesized_fallback_count=fallback_count,
            cached_count=cached_count,
        )
    except KeyboardInterrupt:
        # os._exit(130) 兼容：interrupted() 在异常上抛到 cli.build 之前执行，
        # 写 INTERRUPTED 保留文件供 status 展示「上次编译中断」（AC3.3）。
        writer.interrupted()
        raise
    except Exception:
        writer.interrupted()
        raise


# ── 重放 ──────────────────────────────────────────────────────────────


def replay(settings: Settings) -> ReplayResult:
    """从 ``out/triples.jsonl`` 重放重建图谱（§3.5.5 去重收敛规则）。

    不消耗 LLM chat Token（不从 raw/ 重新抽取，仅从回放日志重建）。
    跳过 step 7（vector index_nodes）——replay 仅重建图谱与派生索引，不重建 ChromaDB。

    去重收敛规则（确定性，可复现）：
    a. 按 source_file 分组；
    b. 组内按 ts 升序；
    c. 同 (source_file, ts) 去重保留 schema_version 最高者；
    d. 取该 source_file 组内 ts 最大的一条 op 为最终态：
       delete → 该文件不参与重建；upsert → 用其 triples/concepts 重建。

    schema_version 迁移：jsonl > manifest（降级异常）或无迁移函数 → exit 3。
    """
    out_dir = settings.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _read_triples_log(out_dir)
    if not records:
        logger.warning("replay: no triples.jsonl records found; nothing to replay")
        return ReplayResult()

    manifest = _load_manifest(out_dir)

    # step 3: schema_version 校验 / 迁移
    _validate_and_migrate_schema(records, manifest.version)

    # step 2: 去重收敛
    converged = _dedup_converge(records)

    # step 4: 从空图谱重建
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    graph_builder = GraphBuilder(graph, settings)

    rebuilt_files: list[str] = []
    deleted_files: list[str] = []

    for source_file in sorted(converged):
        record = converged[source_file]
        op = str(record.get("op", ""))
        if op == "delete":
            deleted_files.append(source_file)
            continue
        if op == "upsert":
            result = _parse_upsert_record(record)
            graph_builder.upsert(result, source_file)
            rebuilt_files.append(source_file)

    # step 6: synthesize_fallback_descriptions
    fallback_count = graph_builder.synthesize_fallback_descriptions()

    # step 7: 跳过（replay 不重建向量库）

    # step 8: build_indexes —— s1-feat-011（由 _finalize_staging 内部调用，
    # llm=None 时社区摘要降级为启发式，零 Token）

    # step 9: manifest 更新
    now = _now_iso()
    for path in rebuilt_files:
        existing = manifest.files.get(path)
        sha = existing.sha256 if existing else ""
        manifest.files[path] = FileState(
            path=path,
            sha256=sha,
            processed_at=now,
            extractor_version=settings.extractor_version,
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            embedding_dim=existing.embedding_dim if existing else 0,
            extraction_config=extraction_config_signature(settings),
            index_config=index_config_signature(settings),
            embedding_config=embedding_config_signature(settings),
        )
    for path in deleted_files:
        manifest.files.pop(path, None)

    # step 10-11: 序列化 + 原子切换
    _finalize_staging(
        graph_builder,
        manifest,
        out_dir,
        graph=graph,
        settings=settings,
        llm=None,
    )

    logger.info(
        "replay done: rebuilt=%d deleted=%d fallback=%d",
        len(rebuilt_files),
        len(deleted_files),
        fallback_count,
        extra={"stage": "replay"},
    )

    return ReplayResult(
        rebuilt_files=rebuilt_files,
        deleted_files=deleted_files,
        synthesized_fallback_count=fallback_count,
    )


# ── 问答 ──────────────────────────────────────────────────────────────


def answer_query(
    settings: Settings,
    question: str,
    *,
    mode: AnswerMode = "query",
    llm: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    graph: nx.MultiDiGraph | None = None,
    vector_store: VectorStoreBackend | None = None,
    communities: CommunityResult | None = None,
    progress: ProgressReporter | None = None,
) -> AnswerQueryResult:
    """执行问答流程（方案 §3.5.3，按 ``mode`` 启用 retriever 子集）。

    命令语义映射（Opt #5 + s1-feat-012 三路融合）：
    - ``mode="query"``：graph + vector + community 三路融合（升级 s1-feat-009 的仅 graph 路）。
    - ``mode="ask"``：仅向量路。
    - ``mode="search"``：仅社区路（配合 ``search --community``）。

    流程：
    1. 冷启动校验（Opt #8）：``out/graph.json`` 不存在或 ``raw/`` 为空 →
       抛 ``ColdStartError``，CLI 据此 exit 1。
    2. 加载 graph / ChromaDB / communities（按 mode 懒加载所需资源）。
    3. ``MultiRetriever.recall``：按 mode 启用 retriever 子集 → fuse 融合重排
       （confidence 权重 × 相似度）→ tiktoken 裁剪。
    4. ``compile_context`` 渲染 hits 为纯文本上下文。
    5. ``generate`` 生成带 ``^[source_file]`` 引用的 ``Answer``。
    6. ``should_flag`` 判定是否入 review_queue（``generate`` 内部设置 ``review_flagged``）；
       命中则 ``ReviewQueue.append`` 写入 ``out/review_queue.md``（s1-feat-013）。

    Args:
        settings: 全局配置。
        question: 用户自然语言问题。
        mode: 问答模式（见 ``AnswerMode``）。
        llm: LLM 客户端；``None`` 时经 ``make_llm_client`` 创建（缺 key exit 2）。
        graph: 已加载的图谱；``None`` 时从 ``out/graph.json`` 加载。
        vector_store: 向量库后端；``None`` 时按 mode 决定是否从 ``out/chroma`` 加载。
        communities: 社区结果；``None`` 时按 mode 决定是否从 ``out/communities.json`` 加载。
        progress: 检索进度报告器；``None`` 时用空实现（无进度反馈）。CLI 注入基于
            ``rich`` 的实现，在各阶段展示 spinner + 持久日志。

    Returns:
        ``AnswerQueryResult`` —— 答案 + 召回命中。

    Raises:
        ColdStartError: 图谱未编译（Opt #8）。
    """
    if _is_cold_start(settings):
        logger.warning(
            "cold start: graph.json exists=%s, raw_empty=%s",
            (settings.out_dir / "graph.json").exists(),
            _is_raw_empty(settings.raw_dir),
        )
        raise ColdStartError("知识库未编译，请先运行 nanokb build")

    progress_reporter: ProgressReporter = progress or NullProgressReporter()

    with progress_reporter.stage("加载知识库..."):
        if graph is None:
            graph = _load_graph(settings.out_dir)
        if llm is None:
            llm = make_llm_client(settings)
        # 生文与向量解耦：未配置独立 embedding 端点时复用 llm（向后兼容）
        embedder = _resolve_embedder(settings, embedding_client, llm)

        retrievers = _build_retrievers_for_mode(
            mode,
            graph,
            llm,
            settings,
            embedder=embedder,
            vector_store=vector_store,
            communities=communities,
        )
    multi = MultiRetriever(retrievers, settings, llm, progress=progress_reporter)
    hits = multi.recall(question)
    with progress_reporter.stage("构建上下文..."):
        context = compile_context(hits, settings, llm)
    with progress_reporter.stage("生成答案中..."):
        answer = generate(question, context, hits, llm, settings)
    # 方案 §阶段 5：generate 调用 should_flag 设置 review_flagged；命中则 append 到
    # review_queue.md（s1-feat-013 主动学习闭环）。
    if answer.review_flagged:
        ReviewQueue(settings.out_dir).append(
            question=question,
            reason=determine_reason(hits, settings),
            entities=collect_entities(hits),
        )
    return AnswerQueryResult(answer=answer, hits=hits)


def search_communities(
    settings: Settings,
    keyword: str,
    *,
    llm: LLMClient | None = None,
    graph: nx.MultiDiGraph | None = None,
    communities: CommunityResult | None = None,
    progress: ProgressReporter | None = None,
) -> list[RetrievalHit]:
    """``search --community`` 社区宏观检索（方案 §3.5.3，s1-feat-012 AC #3）。

    流程：
    1. 冷启动校验。
    2. 加载 communities.json（缺失时抛 ``ColdStartError`` 并提示先 build）。
    3. ``CommunityRetriever.recall(keyword)``：NER → 社区成员匹配 → 命中社区摘要 hit。

    Args:
        settings: 全局配置。
        keyword: 检索关键词（如 ``'深度学习'``）。
        llm: LLM 客户端；``None`` 时经 ``make_llm_client`` 创建。
        graph: 已加载图谱；``None`` 时从 ``out/graph.json`` 加载。
        communities: 社区结果；``None`` 时从 ``out/communities.json`` 加载。
        progress: 检索进度报告器；``None`` 时用空实现（无进度反馈）。

    Returns:
        命中的 ``RetrievalHit`` 列表（携带 ``community_summary``）；无命中返回空。

    Raises:
        ColdStartError: 图谱未编译或社区索引缺失。
    """
    if _is_cold_start(settings):
        raise ColdStartError("知识库未编译，请先运行 nanokb build")

    progress_reporter: ProgressReporter = progress or NullProgressReporter()

    with progress_reporter.stage("加载社区索引..."):
        if communities is None:
            communities = load_communities(settings.out_dir)
        if communities is None or not communities.communities:
            raise ColdStartError("社区索引未编译，请先运行 nanokb build 完成高级索引")

        if graph is None:
            graph = _load_graph(settings.out_dir)
        if llm is None:
            llm = make_llm_client(settings)

    with progress_reporter.stage("社区召回中..."):
        retriever = CommunityRetriever(communities, graph, llm, settings)
        return retriever.recall(keyword)


def _build_retrievers_for_mode(
    mode: AnswerMode,
    graph: nx.MultiDiGraph,
    llm: LLMClient,
    settings: Settings,
    *,
    embedder: EmbeddingClient,
    vector_store: VectorStoreBackend | object | None = None,
    communities: CommunityResult | None = None,
) -> list[Retriever]:
    """按命令映射启用 retriever 子集（Opt #5 + s1-feat-012）。

    - ``ask``：仅 ``VectorRetriever``（缺失向量库时返回空，上游据此提示用户）。
    - ``search``：仅 ``CommunityRetriever``（缺失社区时返回空）。
    - ``query``：``GraphRetriever`` 始终启用；``enable_vector_recall`` 且向量库就绪
      时加 ``VectorRetriever``；``enable_community_recall`` 且社区就绪时加
      ``CommunityRetriever``（升级 s1-feat-009 的仅 graph 路为三路融合）。

    ``llm`` 用于 graph/community 两路的 NER（``complete``）；``embedder`` 用于
    ``VectorRetriever`` 的 query embedding（生文与向量解耦）。
    """
    if mode == "ask":
        vs = _ensure_vector_store(settings, embedder, vector_store)
        if vs is None:
            logger.warning("ask: vector store unavailable; returning empty retrievers")
            return []
        return [VectorRetriever(vs, embedder, settings)]

    if mode == "search":
        comm = communities if communities is not None else load_communities(settings.out_dir)
        if comm is None or not comm.communities:
            logger.warning("search: communities unavailable; returning empty retrievers")
            return []
        return [CommunityRetriever(comm, graph, llm, settings)]

    # mode == "query"：三路融合
    retrievers: list[Retriever] = [GraphRetriever(graph, llm, settings)]
    if settings.enable_vector_recall:
        vs = _ensure_vector_store(settings, embedder, vector_store)
        if vs is not None:
            retrievers.append(VectorRetriever(vs, embedder, settings))
    if settings.enable_community_recall:
        comm = communities if communities is not None else load_communities(settings.out_dir)
        if comm is not None and comm.communities:
            retrievers.append(CommunityRetriever(comm, graph, llm, settings))
    return retrievers


def _ensure_vector_store(
    settings: Settings,
    embedder: EmbeddingClient,
    explicit: VectorStoreBackend | object | None,
) -> VectorStore | None:
    """加载或复用 ``VectorStore``；不可用（chroma 目录不存在）返回 None。

    优先复用调用方传入的 ``explicit``（测试注入），否则从 ``out/chroma`` 加载真实
    ``VectorStore``。embedding_dim 从 manifest FileState 推断（取已记录维度的众数），
    缺失时降级用 ``_probe_embedding_dim`` 探测。

    ``embedder`` 用于维度探测（``embed`` 探针）；生文与向量解耦后独立于 chat llm。
    """
    if explicit is not None:
        # 测试注入的 FakeVectorStore 无 search 方法，仅 query 模式下真实 VectorStore 才走
        # VectorRetriever；此处保留传参以便未来扩展，但 VectorRetriever 需要真实 search。
        if isinstance(explicit, VectorStore):
            return explicit
        # 非 VectorStore（如测试 FakeVectorStore）—— search 不可用，跳过向量路
        logger.debug("explicit vector_store is not VectorStore; skipping vector recall")
        return None

    chroma_path = settings.out_dir / "chroma"
    if not chroma_path.exists():
        return None
    try:
        manifest = _load_manifest(settings.out_dir)
        dims = [fs.embedding_dim for fs in manifest.files.values() if fs.embedding_dim]
        embedding_dim = dims[0] if dims else _probe_embedding_dim(embedder)
        return VectorStore(chroma_path, settings.embedding_model, embedding_dim)
    except Exception:
        logger.warning(
            "failed to load vector store; vector recall disabled",
            exc_info=True,
            extra={"stage": "qa-retriever"},
        )
        return None


# ── 辅助函数 ──────────────────────────────────────────────────────────


def _resolve_embedder(
    settings: Settings,
    embedding_client: EmbeddingClient | None,
    chat_llm: LLMClient,
) -> EmbeddingClient:
    """解析向量嵌入客户端（生文与向量解耦的核心调度点）。

    优先级：
    1. 调用方显式注入的 ``embedding_client``（测试 / 自定义）。
    2. 用户未配置独立 embedding 端点（``embedding_provider=openai`` 且未设
       ``embedding_api_key`` / ``embedding_base_url``）→ **复用 chat llm**
       （向后兼容：生文与 embedding 共用同一 OpenAI 兼容端点，行为与旧版一致）。
    3. 用户配置了独立 embedding（``embedding_provider=ollama`` 或专用 key/url）
       → ``make_embedding_client(settings)`` 构造独立客户端。

    ``chat_llm`` 是 ``EmbeddingClient`` 的结构超集（实现了 ``embed``），故情形 2
    直接返回 ``chat_llm`` 满足 ``EmbeddingClient`` 协议。
    """
    if embedding_client is not None:
        return embedding_client
    if (
        settings.embedding_provider == "openai"
        and not settings.embedding_api_key
        and not settings.embedding_base_url
    ):
        return chat_llm
    return make_embedding_client(settings)


def build_default_registry() -> LoaderRegistry:
    """构造默认 LoaderRegistry：UnstructuredLoader + CodeLoader。

    二者支持的扩展名集合互斥（md/txt/pdf/docx vs py/js/java），按注册顺序首个
    ``supports`` 胜出，由各 loader 自行判定，无需额外优先级协调（s1-feat-010）。
    """
    registry = LoaderRegistry()
    registry.register(UnstructuredLoader())
    registry.register(CodeLoader())
    return registry


def _now_iso() -> str:
    """当前 UTC 时间的 ISO 8601 字符串（用于 triples.jsonl 时间戳，可字典序排序）。"""
    return datetime.now(timezone.utc).isoformat()


def _is_raw_empty(raw_dir: Path) -> bool:
    """raw_dir 不存在或无任何受支持文档文件时为空。"""
    if not raw_dir.exists():
        return True
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            return False
    return True


def _is_cold_start(settings: Settings) -> bool:
    """冷启动判定（Opt #8）：graph.json 不存在 OR raw/ 为空。"""
    graph_path = settings.out_dir / "graph.json"
    if not graph_path.exists():
        return True
    if _is_raw_empty(settings.raw_dir):
        return True
    return False


def _load_manifest(out_dir: Path) -> Manifest:
    """从 ``out/manifest.json`` 加载 Manifest；不存在则返回空 Manifest。"""
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return Manifest()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Manifest.model_validate(data)


def _load_graph(out_dir: Path) -> nx.MultiDiGraph:
    """从 ``out/graph.json`` 加载 MultiDiGraph；不存在则返回空图。"""
    graph_path = out_dir / "graph.json"
    if not graph_path.exists():
        return nx.MultiDiGraph()
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, directed=True, multigraph=True)


def _normalize_result_source(result: ExtractionResult, source_file: str) -> ExtractionResult:
    """将 ExtractionResult 中所有 triple/concept 的 source_file 统一为相对路径 key。

    SemanticTrack.extract 使用 ``str(doc.path)``（绝对路径）作为 source_file，
    而流水线的 manifest key / 删除逻辑使用相对 raw_dir 的路径。本函数确保二者一致，
    使 ``graph_builder.delete_by_source(rel_key)`` 能正确匹配边/节点。
    """
    new_triples = [t.model_copy(update={"source_file": source_file}) for t in result.triples]
    new_concepts = [c.model_copy(update={"source_file": source_file}) for c in result.concepts]
    return ExtractionResult(triples=new_triples, concepts=new_concepts)


def _append_triples_log(out_dir: Path, record: dict[str, Any]) -> None:
    """向 ``out/triples.jsonl`` 追加一条 JSON 记录（best-effort，非原子写）。

    triples.jsonl 为追加日志，不在 staging 原子切换范围内；一致性靠幂等重跑收敛。
    """
    path = out_dir / TRIPLES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_triples_log(out_dir: Path) -> list[dict[str, Any]]:
    """读取 ``out/triples.jsonl`` 的全部记录；文件不存在返回空列表。"""
    path = out_dir / TRIPLES_FILENAME
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _subgraph_for_source(graph: nx.MultiDiGraph, source_file: str) -> nx.MultiDiGraph:
    """构建 source_file 所属节点的子图（用于 per-path 向量索引）。

    收集所有与 ``source_file`` 边关联的端点节点（head/tail），携带完整节点数据
    （含 step 6 合成的兜底 description）。节点若无显式 source_file 属性（仅出现在
    triples 无 Concept 的节点），则默认填入当前 path 以保证向量 id 前缀正确。
    """
    nodes: set[str] = set()
    for u, v, data in graph.edges(data=True):
        if data.get("source_file") == source_file:
            nodes.add(u)
            nodes.add(v)

    sub = nx.MultiDiGraph()
    for node in nodes:
        if not graph.has_node(node):
            continue
        data = dict(graph.nodes[node])
        data.setdefault("source_file", source_file)
        sub.add_node(node, **data)
    return sub


def _dedup_converge(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """应用 §3.5.5 去重收敛规则，返回 ``{source_file: 最终态 record}``。

    规则：a. 按 source_file 分组；b. 组内按 ts 升序；c. 同 (source_file, ts)
    去重保留 schema_version 最高者；d. 取组内 ts 最大的一条 op 为最终态。
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        sf = str(rec.get("source_file", ""))
        groups.setdefault(sf, []).append(rec)

    converged: dict[str, dict[str, Any]] = {}
    for source_file, group_records in groups.items():
        sorted_records = sorted(group_records, key=lambda r: str(r.get("ts", "")))

        best_by_ts: dict[str, dict[str, Any]] = {}
        for rec in sorted_records:
            ts = str(rec.get("ts", ""))
            existing = best_by_ts.get(ts)
            if existing is None:
                best_by_ts[ts] = rec
            elif _schema_version_key(rec) > _schema_version_key(existing):
                best_by_ts[ts] = rec

        max_ts = max(best_by_ts)
        converged[source_file] = best_by_ts[max_ts]

    return converged


def _validate_and_migrate_schema(records: list[dict[str, Any]], target_version: str) -> None:
    """校验 triples.jsonl 记录的 schema_version 与 manifest 兼容。

    - jsonl < manifest（升级）：按迁移函数链逐级迁移（当前无注册迁移函数）。
    - jsonl > manifest（降级异常）或无迁移函数 → 拒绝回放 exit 3。
    """
    target = _schema_version_key_raw(target_version)
    for rec in records:
        rec_version = _schema_version_key(rec)
        if rec_version > target:
            logger.error(
                "replay: record schema_version %d > manifest %d; cannot downgrade",
                rec_version,
                target,
            )
            raise SystemExit(3)
        if rec_version < target:
            logger.error(
                "replay: no migration from schema_version %d to %d",
                rec_version,
                target,
            )
            raise SystemExit(3)


def _parse_upsert_record(record: dict[str, Any]) -> ExtractionResult:
    """从序列化的 upsert 记录重建 ExtractionResult。"""
    triples = [Triple.model_validate(t) for t in record.get("triples", [])]
    concepts = [Concept.model_validate(c) for c in record.get("concepts", [])]
    return ExtractionResult(triples=triples, concepts=concepts)


def _schema_version_key(record: dict[str, Any]) -> int:
    """从 record 提取 schema_version 并转为可比较的 int。"""
    return _schema_version_key_raw(str(record.get("schema_version", "")))


def _schema_version_key_raw(version: str) -> int:
    """将 schema_version 字符串转为 int；非数字返回 0（最低）。"""
    try:
        return int(version)
    except (ValueError, TypeError):
        return 0


def _finalize_staging(
    graph_builder: GraphBuilder,
    manifest: Manifest,
    out_dir: Path,
    *,
    graph: nx.MultiDiGraph | None = None,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> None:
    """序列化图谱与 manifest 到 staging 目录，再原子切换到 out/。

    s1-feat-011：当提供 ``graph`` 与 ``settings`` 时，执行 step 8 ``build_indexes``
    （community + keyword），将 ``communities.json`` 与 ``keywords.json`` 写入 staging，
    随 ``staging_swap`` 原子切换（v4 Opt #1 五件套）。``llm`` 为 None 时社区摘要
    降级为启发式（成员名拼接，零 Token）。
    """
    staging = out_dir / STAGING_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)

    graph_builder.save_graph(staging)

    # step 8: build_indexes（community + keyword）——s1-feat-011
    if graph is not None and settings is not None:
        build_indexes(graph, settings, llm, staging)

    atomic_write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
    staging_swap(staging, out_dir)


def _probe_embedding_dim(embedder: EmbeddingClient) -> int:
    """探测 ``embedder.embed`` 实际输出维度（用于 VectorStore 元数据 + FileState）。

    发送单条探针文本，取返回向量的维度。探测失败（网络 / API 异常）时返回 0，
    VectorStore 以 dim=0 构造（metadata 记 0，ChromaDB 实际维度由首次 upsert 推断）。
    """
    try:
        probe = embedder.embed(["nanokb-embedding-dim-probe"])
        if probe and probe[0]:
            return len(probe[0])
    except Exception:
        logger.debug(
            "embedding dim probe failed; defaulting to 0",
            exc_info=True,
            extra={"stage": "compile"},
        )
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "TRIPLES_FILENAME",
    "AnswerMode",
    "AnswerQueryResult",
    "ColdStartError",
    "CompileResult",
    "ReplayResult",
    "VectorStoreBackend",
    "answer_query",
    "build_default_registry",
    "compile",
    "replay",
    "search_communities",
]

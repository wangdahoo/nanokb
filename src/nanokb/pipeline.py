"""编译流水线编排（方案 §3.5.1 + §3.5.5，Feature s1-feat-008）。

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
（delete_by_source / index_nodes）。``vector_store=None`` 时向量操作被跳过（stage4
尚未实现的阶段）。s1-feat-011 的 ``VectorStore`` 将满足此协议。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import networkx as nx  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from nanokb.config import Settings
from nanokb.llm.base import LLMClient, make_llm_client
from nanokb.loaders import LoaderRegistry, UnstructuredLoader, UnsupportedFormatError
from nanokb.models import (
    Concept,
    ExtractionResult,
    FileState,
    Manifest,
    Triple,
)
from nanokb.stage1_load.detector import ChangeSet, detect_changes
from nanokb.stage1_load.ingest import ingest_file
from nanokb.stage2_extract.base import Extractor
from nanokb.stage2_extract.semantic_track import SemanticTrack
from nanokb.stage3_compile import GraphBuilder
from nanokb.utils.io import atomic_write_json, staging_swap

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


class ReplayResult(BaseModel):
    """replay() 返回值：重放摘要。"""

    rebuilt_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    synthesized_fallback_count: int = 0


# ── 向量库后端协议（s1-feat-011 VectorStore 满足） ────────────────────


@runtime_checkable
class VectorStoreBackend(Protocol):
    """流水线所需的最小向量库后端接口。

    s1-feat-011 的 ``VectorStore`` 实现将满足此协议。当前 stage4 尚未实现，
    ``compile(vector_store=None)`` 时向量操作被跳过（仅影响 ChromaDB 一致性，
    graph/triples.jsonl/manifest 主线不受影响）。
    """

    def delete_by_source(self, source_file: str) -> None:
        """删除 ``source_file`` 的全部向量（where={"source_file":source_file}）。"""
        ...

    def index_nodes(self, graph: nx.MultiDiGraph, llm: LLMClient) -> None:
        """为图中每个节点的 description 生成 embedding 并 upsert。

        v4 Medium #1：调用前须确保 fallback 描述已合成（pipeline 保证
        synthesize_fallback_descriptions 先于 index_nodes 执行）。
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
    force: bool = False,
) -> CompileResult:
    """执行编译流水线（两阶段结构 + v4 step 5/6/7 三段时机）。

    Args:
        settings: 全局配置（raw_dir / out_dir / 模型身份等）。
        llm: LLM 客户端；``None`` 时通过 ``make_llm_client(settings)`` 创建
            （缺 API key 会 exit 2）。
        registry: 文档加载注册表；``None`` 时用默认（UnstructuredLoader）。
        extractor_factory: 自定义抽取器工厂；``None`` 时用 ``SemanticTrack``。
        vector_store: 向量库后端；``None`` 时跳过向量操作（stage4 未实现阶段）。
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

    # ── 阶段 A：抽取（失败安全，不触碰 graph/chroma/triples.jsonl 写） ──
    changes = detect_changes(raw_dir, manifest, settings)

    if not force and not changes.has_changes:
        logger.info("no changes detected; skipping compilation")
        return CompileResult(changes=changes)

    if llm is None:
        llm = make_llm_client(settings)

    extractor: Extractor
    if extractor_factory is not None:
        extractor = extractor_factory(llm, settings)
    else:
        extractor = SemanticTrack(llm, settings)

    results_map: dict[str, ExtractionResult] = {}
    sha_map: dict[str, str] = {}
    skipped: list[str] = []

    to_process = sorted(set(changes.added) | set(changes.modified))
    for path in to_process:
        abs_path = raw_dir / path
        try:
            doc = ingest_file(abs_path, raw_dir, registry, settings)
        except UnsupportedFormatError as exc:
            logger.warning(
                "skip unsupported file: %s (%s)", path, exc,
                extra={"stage": "compile-ingest", "file": path},
            )
            skipped.append(path)
            continue
        except Exception:
            logger.exception(
                "ingest failed for %s", path,
                extra={"stage": "compile-ingest", "file": path},
            )
            skipped.append(path)
            continue

        try:
            result = extractor.extract(doc)
        except Exception:
            logger.exception(
                "extraction failed for %s", path,
                extra={"stage": "compile-extract", "file": path},
            )
            skipped.append(path)
            continue

        results_map[path] = _normalize_result_source(result, path)
        sha_map[path] = doc.sha256

    # ── 阶段 B：破坏性变更（抽取全部成功后统一执行） ──────────────────

    # step 3: deletion 级联（Severe #1）
    for path in changes.deleted:
        graph_builder.delete_by_source(path)
        if vector_store is not None:
            vector_store.delete_by_source(path)
        _append_triples_log(out_dir, {
            "schema_version": manifest.version,
            "op": "delete",
            "source_file": path,
            "ts": _now_iso(),
        })
        manifest.files.pop(path, None)

    # step 4: modified 先清后建（Medium #2）——在 upsert 之前清旧边/旧向量
    for path in changes.modified:
        if path not in results_map:
            continue
        graph_builder.delete_by_source(path)
        if vector_store is not None:
            vector_store.delete_by_source(path)

    # step 5: added/modified 图构建（无向量，v4 拆分独立小阶段）
    for path in to_process:
        if path not in results_map:
            continue
        result = results_map[path]
        _append_triples_log(out_dir, {
            "schema_version": manifest.version,
            "op": "upsert",
            "source_file": path,
            "triples": [t.model_dump(mode="json") for t in result.triples],
            "concepts": [c.model_dump(mode="json") for c in result.concepts],
            "ts": _now_iso(),
        })
        graph_builder.upsert(result, path)

    # step 6: synthesize_fallback_descriptions（Opt #2 v3 + v4 Medium #1）
    # 必须在 step 7（index_nodes）之前——漏抽 Concept 的节点经此合成后才有描述，
    # 否则 index_nodes 会因空描述跳过这些节点。
    fallback_count = graph_builder.synthesize_fallback_descriptions()

    # step 7: 向量索引（v4 新增独立步骤）——逐 path 子图，描述已就绪
    if vector_store is not None:
        for path in to_process:
            if path not in results_map:
                continue
            subgraph = _subgraph_for_source(graph, path)
            if subgraph.number_of_nodes() > 0:
                vector_store.index_nodes(subgraph, llm)

    # step 8: build_indexes（community + keyword）——s1-feat-011 实现
    # staging_swap 自动跳过 staging 中缺失的 communities.json/keywords.json

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
            embedding_dim=0,
        )

    # step 10-11: 序列化到 staging + 原子切换（manifest 最后写）
    _finalize_staging(graph_builder, manifest, out_dir)

    logger.info(
        "compile done: added=%d modified=%d deleted=%d extracted=%d skipped=%d fallback=%d",
        len(changes.added), len(changes.modified), len(changes.deleted),
        len(results_map), len(skipped), fallback_count,
        extra={"stage": "compile"},
    )

    return CompileResult(
        changes=changes,
        extracted_count=len(results_map),
        skipped=skipped,
        synthesized_fallback_count=fallback_count,
    )


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

    # step 8: build_indexes —— s1-feat-011 实现

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
            embedding_dim=0,
        )
    for path in deleted_files:
        manifest.files.pop(path, None)

    # step 10-11: 序列化 + 原子切换
    _finalize_staging(graph_builder, manifest, out_dir)

    logger.info(
        "replay done: rebuilt=%d deleted=%d fallback=%d",
        len(rebuilt_files), len(deleted_files), fallback_count,
        extra={"stage": "replay"},
    )

    return ReplayResult(
        rebuilt_files=rebuilt_files,
        deleted_files=deleted_files,
        synthesized_fallback_count=fallback_count,
    )


# ── 辅助函数 ──────────────────────────────────────────────────────────


def build_default_registry() -> LoaderRegistry:
    """构造默认 LoaderRegistry（注册 UnstructuredLoader）。

    CodeLoader（.py/.js/.java）在 s1-feat-010 通过同一接口接入。
    """
    registry = LoaderRegistry()
    registry.register(UnstructuredLoader())
    return registry


def _now_iso() -> str:
    """当前 UTC 时间的 ISO 8601 字符串（用于 triples.jsonl 时间戳，可字典序排序）。"""
    return datetime.now(timezone.utc).isoformat()


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
    new_triples = [
        t.model_copy(update={"source_file": source_file})
        for t in result.triples
    ]
    new_concepts = [
        c.model_copy(update={"source_file": source_file})
        for c in result.concepts
    ]
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


def _validate_and_migrate_schema(
    records: list[dict[str, Any]], target_version: str
) -> None:
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
                rec_version, target,
            )
            raise SystemExit(3)
        if rec_version < target:
            logger.error(
                "replay: no migration from schema_version %d to %d",
                rec_version, target,
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
    graph_builder: GraphBuilder, manifest: Manifest, out_dir: Path
) -> None:
    """序列化图谱与 manifest 到 staging 目录，再原子切换到 out/。"""
    staging = out_dir / STAGING_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)

    graph_builder.save_graph(staging)
    atomic_write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
    staging_swap(staging, out_dir)


__all__ = [
    "SCHEMA_VERSION",
    "TRIPLES_FILENAME",
    "CompileResult",
    "ReplayResult",
    "VectorStoreBackend",
    "build_default_registry",
    "compile",
    "replay",
]

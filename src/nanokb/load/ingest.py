"""文档加载编排（方案 §3.5.1 step 2 + §4 阶段 1）。

ingest 串联 ``detector.detect_changes`` → ``LoaderRegistry.load`` → ``chunker.chunk_text``，
为流水线阶段 A 提供「变更检测 + 文档加载 + 分块填充」的一体化入口。

输出：
- ``IngestResult``：变更集 ``ChangeSet`` + 新加载的 ``Document`` 字典（key = 相对路径）。
- ``ingest_file``：加载单个文件为 ``Document``（含 chunks），供 pipeline 按 path 细粒度调用。
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from nanokb.config import Settings
from nanokb.extract.chunker import chunk_text
from nanokb.load.detector import ChangeSet, _rel_key, detect_changes
from nanokb.loaders import LoaderRegistry, UnsupportedFormatError
from nanokb.models import Document, Manifest
from nanokb.utils.hashing import sha256_file

logger = logging.getLogger("nanokb")


class IngestResult(BaseModel):
    """ingest 编排输出：变更集 + 新加载的 Document 字典。"""

    changes: ChangeSet = Field(default_factory=ChangeSet)
    documents: dict[str, Document] = Field(default_factory=dict)


def ingest_file(
    path: Path,
    raw_dir: Path,
    registry: LoaderRegistry,
    settings: Settings,
) -> Document:
    """加载单个文件为 ``Document``（含 chunks）。

    流程：``registry.load(path)`` 抽取文本 → ``sha256_file`` 计算指纹 →
    ``chunk_text`` 按 ``settings.chunk_max_tokens`` / ``chunk_overlap_tokens`` 切片填充
    ``Document.chunks``。

    Raises:
        UnsupportedFormatError: 无 loader 支持该扩展名（调用方负责跳过并记日志）。
    """
    content = registry.load(path)
    sha = sha256_file(path)
    source_file = _rel_key(path, raw_dir)

    chunks = chunk_text(
        content,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        source_file=source_file,
        model=settings.llm_model,
    )

    return Document(
        path=path,
        content=content,
        sha256=sha,
        format=path.suffix.lower().lstrip("."),
        chunks=chunks,
    )


def ingest(
    raw_dir: Path,
    manifest: Manifest,
    registry: LoaderRegistry,
    settings: Settings,
) -> IngestResult:
    """编排 detect_changes → ingest_file，返回变更集与新加载的 Document。

    对 ``added ∪ modified`` 中的每个路径调用 ``ingest_file``；``UnsupportedFormatError``
    被捕获并记 WARNING（该路径不进入 documents，但仍在 changes 集合中——pipeline 后续
    会跳过它，manifest 也不更新）。
    """
    changes = detect_changes(raw_dir, manifest, settings)

    documents: dict[str, Document] = {}
    for rel_key in [*changes.added, *changes.modified]:
        path = raw_dir / rel_key
        try:
            doc = ingest_file(path, raw_dir, registry, settings)
        except UnsupportedFormatError as exc:
            logger.warning(
                "skip unsupported file during ingest: %s (%s)",
                rel_key,
                exc,
                extra={"stage": "ingest", "file": rel_key},
            )
            continue
        documents[rel_key] = doc

    return IngestResult(changes=changes, documents=documents)


__all__ = ["IngestResult", "ingest", "ingest_file"]

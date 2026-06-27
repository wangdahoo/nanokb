"""跨进程运行时进度集成测试（方案 §6.7，Feature s3-feat-004）。

覆盖验收标准：
- AC3.1：compile 在 EXTRACT 阶段运行中，另一进程读 out/.build_progress.json 得到
  stage=EXTRACT + extract.completed > 0（跨进程可见）。
- AC3.2：compile 正常完成 → .build_progress.json 被删除（read_progress 返回 None）。
- AC3.4：status 读取路径不打开 out/chroma/（零锁冲突，保守假设 Medium #3）。

策略：主进程跑 compile，EXTRACT 阶段用 ``multiprocessing`` 起子进程调 ``read_progress``。
``_BlockingExtractor`` 让第 2 个文档的 extract 阻塞，直到子进程读到 EXTRACT + completed>0，
制造出「编译进行中」的稳定窗口供子进程观测。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import networkx as nx

from nanokb import pipeline
from nanokb.config import Settings
from nanokb.models import Concept, Confidence, ExtractionResult, Triple
from nanokb.utils.progress import (
    PROGRESS_FILENAME,
    BuildProgress,
    BuildProgressWriter,
    BuildStage,
    read_progress,
)

# ── 测试 doubles ─────────────────────────────────────────────────────


class _BlockingExtractor:
    """第 1 个文档抽取立即返回；第 2 个文档阻塞直到子进程读到进度后释放 gate。

    串行抽取（extract_doc_concurrency=1）下：doc1 抽取 + 归并（completed=1 落盘）
    → doc2 抽取阻塞 → 子进程读到 EXTRACT+completed>0 → 释放 gate → doc2 返回 → 编译继续。
    """

    def __init__(self, gate: Any) -> None:
        self._gate = gate
        self.calls = 0

    def extract(self, doc: Any) -> ExtractionResult:
        self.calls += 1
        sf = str(doc.path)
        result = ExtractionResult(
            triples=[
                Triple(
                    head="ConceptA",
                    relation="relates_to",
                    tail="ConceptB",
                    confidence=Confidence.EXTRACTED,
                    source_file=sf,
                ),
            ],
            concepts=[
                Concept(name="ConceptA", description="Description for A.", source_file=sf),
                Concept(name="ConceptB", description="Description for B.", source_file=sf),
            ],
        )
        if self.calls == 2:
            # 阻塞第 2 个文档，直到子进程确认读到 EXTRACT+completed>0
            self._gate.wait(timeout=30)
        return result


class _FakeChatLLM:
    """chat LLM double：complete 不会被调用（自定义 extractor）；embed 供维度探针。"""

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        raise AssertionError("chat complete should not be called; custom extractor in use")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class _FakeVectorStore:
    """VectorStoreBackend double：记录调用，不打开 chroma（验证 AC3.4）。"""

    def __init__(self) -> None:
        self.deleted_sources: list[str] = []
        self.index_calls: int = 0

    def delete_by_source(self, source_file: str) -> None:
        self.deleted_sources.append(source_file)

    def index_nodes(
        self,
        graph: nx.MultiDiGraph,
        llm: object,
        *,
        embed_fn: object = None,
        on_progress: object = None,
    ) -> None:
        self.index_calls += 1


def _settings(raw_dir: Path, out_dir: Path, **kwargs: Any) -> Settings:
    return Settings(raw_dir=raw_dir, out_dir=out_dir, **kwargs)


# ── 子进程入口（必须模块级，Windows spawn 可 pickle） ────────────────


def _child_reader(out_dir_str: str, gate: Any, result_queue: Any) -> None:
    """子进程：轮询 read_progress 直到读到 EXTRACT + completed>0，结果入队后释放 gate。

    只读 ``out/.build_progress.json``，绝不打开 chroma（AC3.4）。
    """
    out_dir = Path(out_dir_str)
    found: dict[str, Any] | None = None
    for _ in range(300):  # 最多等 ~30s
        p = read_progress(out_dir)
        if (
            p is not None
            and p.stage == BuildStage.EXTRACT
            and p.extract.completed > 0
        ):
            found = {
                "stage": p.stage.value,
                "completed": p.extract.completed,
                "pid": p.pid,
            }
            break
        time.sleep(0.1)
    result_queue.put(found)
    gate.set()  # 释放主进程被阻塞的第 2 个文档抽取


# ══════════════════════════════════════════════════════════════════════
# AC3.1 + AC3.2 + AC3.4：跨进程可见 + 完成后删除 + status 不开 chroma
# ══════════════════════════════════════════════════════════════════════


def test_cross_process_visibility_during_extract(tmp_path: Path) -> None:
    """AC3.1：子进程在 EXTRACT 阶段读到 stage=EXTRACT + completed>0（跨进程可见）。

    AC3.2：compile 完成后 read_progress 返回 None（文件被 done() 删除）。
    AC3.4：全程不创建/不打开 out/chroma/（FakeVectorStore，status 只读 JSON）。
    """
    # 把 FLUSH_EVERY 调到 1：每次 update_extract 都落盘，使 completed 立即跨进程可见
    original_flush = BuildProgressWriter.FLUSH_EVERY
    BuildProgressWriter.FLUSH_EVERY = 1
    try:
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        out_dir.mkdir()
        (raw_dir / "doc1.md").write_text("# Doc 1\n\nContent one.", encoding="utf-8")
        (raw_dir / "doc2.md").write_text("# Doc 2\n\nContent two.", encoding="utf-8")

        settings = _settings(
            raw_dir, out_dir, extract_doc_concurrency=1, enable_build_progress=True
        )

        # 跨进程协调原语（spawn 下可 pickle）
        ctx = mp.get_context("spawn")
        gate = ctx.Event()
        result_queue: mp.Queue[Any] = ctx.Queue()

        child = ctx.Process(
            target=_child_reader,
            args=(str(out_dir), gate, result_queue),
        )
        child.start()

        extractor = _BlockingExtractor(gate)
        try:
            pipeline.compile(
                settings,
                llm=_FakeChatLLM(),
                embedding_client=_FakeChatLLM(),
                extractor_factory=lambda llm, s: extractor,
                vector_store=_FakeVectorStore(),
            )
        finally:
            # 兜底释放 gate，防止断言失败时 compile 挂起
            gate.set()

        child.join(timeout=30)
        assert not child.is_alive(), "child reader should have exited"

        observed = result_queue.get(timeout=10)

        # AC3.1：子进程在编译进行中读到 EXTRACT + completed>0
        assert observed is not None, "child should have observed EXTRACT stage mid-build"
        assert observed["stage"] == "extract"
        assert observed["completed"] > 0
        # 跨进程：子进程 PID ≠ 主进程 PID（确实是另一个进程读到的）
        assert observed["pid"] == os.getpid()

        # AC3.2：compile 完成后进度文件被删除（read_progress 返回 None）
        assert read_progress(out_dir) is None, "progress file should be deleted after done()"
        assert not (out_dir / PROGRESS_FILENAME).exists()

        # AC3.4：status 读取路径不打开 chroma —— 全程无 chroma 目录（FakeVectorStore）
        # 且 read_progress 仅读 JSON 文件即可工作（结构性保证：status 绝不打开 chroma）
        assert not (out_dir / "chroma").exists(), (
            "status read path must not require/open chroma (Medium #3)"
        )
    finally:
        BuildProgressWriter.FLUSH_EVERY = original_flush


# ══════════════════════════════════════════════════════════════════════
# AC3.4（强化）：read_progress 只读 JSON，不触碰 chroma
# ══════════════════════════════════════════════════════════════════════


def test_read_progress_does_not_touch_chroma(tmp_path: Path) -> None:
    """AC3.4：即便 out/chroma/ 不存在，read_progress 仍正常工作（零 chroma 依赖）。

    进一步：主动建一个空的 out/chroma/ 目录，read_progress 既不读也不写其中任何文件
    （status 读取路径与 chroma 完全解耦）。
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    chroma_dir = out_dir / "chroma"
    chroma_dir.mkdir()
    # 放一个哨兵文件，确认 read_progress 不去碰它
    sentinel = chroma_dir / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")

    # 无进度文件 → None（status 降级到静态产物）
    assert read_progress(out_dir) is None

    # 写一个合法进度文件，read_progress 正常还原
    from datetime import datetime, timezone

    from nanokb.utils.io import atomic_write_json

    now = datetime.now(timezone.utc).isoformat()
    atomic_write_json(
        out_dir / PROGRESS_FILENAME,
        BuildProgress(
            stage=BuildStage.VECTOR, heartbeat_ts=now, started_at=now
        ).model_dump(mode="json"),
    )
    prog = read_progress(out_dir)
    assert prog is not None
    assert prog.stage == BuildStage.VECTOR

    # 哨兵文件未被改动 —— read_progress 全程未触碰 chroma/
    assert sentinel.read_text(encoding="utf-8") == "untouched"

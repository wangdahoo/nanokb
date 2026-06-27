"""status 命令运行期集成测试（方案 §7.4，Feature s3-feat-005）。

覆盖验收标准：
- AC4.1（端到端）：主进程 compile 运行期间，子进程调 ``nanokb status`` 输出含
  「编译进行中」+ 当前阶段。
- AC4.5（端到端）：子进程 CliRunner（非 TTY）能捕获并断言输出。
- 降级测试：删除 .build_progress.json 后 status 走静态产物分支。

策略：主进程跑 compile，EXTRACT 阶段用 ``_BlockingExtractor`` 让第 2 个文档阻塞，
制造稳定窗口。子进程经 ``multiprocessing`` spawn 调 CliRunner.invoke(app, ["status"])。
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import networkx as nx

from nanokb import pipeline
from nanokb.config import Settings
from nanokb.models import Concept, Confidence, ExtractionResult, Triple
from nanokb.utils.progress import BuildProgressWriter, BuildStage, read_progress

# ── 测试 doubles ─────────────────────────────────────────────────────


class _BlockingExtractor:
    """第 1 个文档立即返回；第 2 个文档阻塞直到子进程读到进度后释放 gate。"""

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
            # 阻塞第 2 个文档，直到子进程完成 status 调用
            self._gate.wait(timeout=60)
        return result


class _FakeChatLLM:
    """chat LLM double：complete 不调用；embed 供维度探针。"""

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
    """VectorStoreBackend double：记录调用，不打开 chroma。"""

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


# ── 子进程入口（必须模块级，Windows spawn 可 pickle） ────────────────


def _child_status_runner(
    raw_dir_str: str,
    out_dir_str: str,
    gate: Any,
    result_queue: Any,
) -> None:
    """子进程：轮询直到读到 EXTRACT+completed>0，调 status 命令，结果入队后释放 gate。"""
    import os

    # 子进程独立解析 Settings：用绝对路径经环境变量注入，禁用 .env 加载
    os.environ["NANOKB_RAW_DIR"] = raw_dir_str
    os.environ["NANOKB_OUT_DIR"] = out_dir_str

    from nanokb.config import Settings as _Settings

    _Settings.model_config["env_file"] = None

    out_dir = Path(out_dir_str)

    # 轮询等待编译窗口（最多 ~30s）
    found = False
    for _ in range(300):
        p = read_progress(out_dir)
        if (
            p is not None
            and p.stage == BuildStage.EXTRACT
            and p.extract.completed > 0
        ):
            found = True
            break
        time.sleep(0.1)

    if not found:
        result_queue.put({"error": "no extract window observed", "stdout": ""})
        gate.set()
        return

    # 调 status 命令（CliRunner 非 TTY）
    from typer.testing import CliRunner

    from nanokb.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["status"])
    result_queue.put(
        {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stage_observed": p.stage.value if p else None,
            "completed_observed": p.extract.completed if p else 0,
        }
    )
    gate.set()  # 释放主进程被阻塞的第 2 个文档抽取


# ══════════════════════════════════════════════════════════════════════
# AC4.1 端到端：编译运行期间子进程调 status → 输出「编译进行中」
# ══════════════════════════════════════════════════════════════════════


def test_status_during_build_shows_running(tmp_path: Path) -> None:
    """AC4.1（端到端）：主进程 compile EXTRACT 阶段，子进程 status 输出「编译进行中」。

    策略：FLUSH_EVERY=1（每次 update_extract 落盘）+ _BlockingExtractor 阻塞第 2 个
    文档制造稳定窗口。子进程轮询到 EXTRACT+completed>0 后调 status 命令。
    """
    original_flush = BuildProgressWriter.FLUSH_EVERY
    BuildProgressWriter.FLUSH_EVERY = 1
    try:
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        out_dir.mkdir()
        (raw_dir / "doc1.md").write_text("# Doc 1\n\nContent one.", encoding="utf-8")
        (raw_dir / "doc2.md").write_text("# Doc 2\n\nContent two.", encoding="utf-8")

        settings = Settings(
            raw_dir=raw_dir,
            out_dir=out_dir,
            extract_doc_concurrency=1,
            enable_build_progress=True,
        )

        ctx = mp.get_context("spawn")
        gate = ctx.Event()
        result_queue: mp.Queue[Any] = ctx.Queue()

        child = ctx.Process(
            target=_child_status_runner,
            args=(str(raw_dir), str(out_dir), gate, result_queue),
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
            gate.set()  # 兜底释放

        child.join(timeout=60)
        assert not child.is_alive(), "child status runner should have exited"

        observed = result_queue.get(timeout=10)
        assert "error" not in observed, f"child reported error: {observed.get('error')}"

        # AC4.1 核心：编译进行中
        assert "编译进行中" in observed["stdout"], observed["stdout"]
        # AC4.5：非 TTY 纯文本可被捕获断言
        assert "extract" in observed["stdout"], observed["stdout"]
        # 观察到的进度阶段
        assert observed["stage_observed"] == "extract"
        assert observed["completed_observed"] > 0
    finally:
        BuildProgressWriter.FLUSH_EVERY = original_flush


# ══════════════════════════════════════════════════════════════════════
# 降级测试：删除 .build_progress.json → status 走静态产物分支
# ══════════════════════════════════════════════════════════════════════


def test_status_falls_back_to_static_when_no_progress(tmp_path: Path) -> None:
    """无 .build_progress.json → status 走静态分支，输出「已编译」（向后兼容 AC4.4）。

    直接在主进程内用 CliRunner 调 status（无并发，无需子进程）。
    """
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    out_dir.mkdir()
    (raw_dir / "doc1.md").write_text("# Doc 1\n\nContent.", encoding="utf-8")

    # 构造静态产物（graph.json + manifest.json），无 .build_progress.json
    from nanokb.models import FileState, Manifest
    from nanokb.utils.io import atomic_write_json

    atomic_write_json(
        out_dir / "graph.json",
        {"directed": True, "multigraph": True, "nodes": [], "links": []},
    )
    manifest = Manifest(
        version="2",
        files={
            "doc1.md": FileState(
                path="doc1.md", sha256="0" * 64, processed_at="2026-01-01T00:00:00Z"
            )
        },
        total_vectors=42,
        last_compiled_at="2026-01-01T00:00:00Z",
        last_llm_model="glm-5.1",
        last_embedding_model="text-embedding-3-small",
    )
    atomic_write_json(out_dir / "manifest.json", manifest.model_dump(mode="json"))

    # 确保没有进度文件
    assert not (out_dir / ".build_progress.json").exists()

    # 用环境变量注入目录给 CliRunner（主进程内 invoke）
    import os

    old_raw = os.environ.get("NANOKB_RAW_DIR")
    old_out = os.environ.get("NANOKB_OUT_DIR")
    os.environ["NANOKB_RAW_DIR"] = str(raw_dir)
    os.environ["NANOKB_OUT_DIR"] = str(out_dir)
    try:
        from typer.testing import CliRunner

        from nanokb.cli import app

        result = CliRunner().invoke(app, ["status"])
    finally:
        if old_raw is not None:
            os.environ["NANOKB_RAW_DIR"] = old_raw
        else:
            os.environ.pop("NANOKB_RAW_DIR", None)
        if old_out is not None:
            os.environ["NANOKB_OUT_DIR"] = old_out
        else:
            os.environ.pop("NANOKB_OUT_DIR", None)

    assert result.exit_code == 0, result.stdout
    # AC4.4 向后兼容：无进度文件走静态分支
    assert "已编译" in result.stdout
    assert "42" in result.stdout  # 向量数（来自 manifest.total_vectors）


# ══════════════════════════════════════════════════════════════════════
# compile 完成后 manifest 携带新字段（Feature s3-feat-005）
# ══════════════════════════════════════════════════════════════════════


def test_compile_writes_manifest_new_fields(tmp_path: Path) -> None:
    """compile 完成后 out/manifest.json 携带 total_vectors / last_compiled_at / 模型字段。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    out_dir.mkdir()
    (raw_dir / "doc1.md").write_text("# Doc 1\n\nContent.", encoding="utf-8")

    settings = Settings(
        raw_dir=raw_dir,
        out_dir=out_dir,
        llm_model="test-llm",
        embedding_model="test-embed",
    )

    pipeline.compile(
        settings,
        llm=_FakeChatLLM(),
        embedding_client=_FakeChatLLM(),
        extractor_factory=lambda llm, s: _BlockingExtractor(_NoopGate()),
        vector_store=_FakeVectorStore(),
    )

    # 读回 manifest.json 验证新字段
    import json

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["version"] == "2"  # version 不变（Opt#3 保守变体）
    assert data["total_vectors"] >= 0
    assert data["last_compiled_at"] != ""
    assert data["last_llm_model"] == "test-llm"
    assert data["last_embedding_model"] == "test-embed"


class _NoopGate:
    """立即返回的 gate 替身（不阻塞，用于不需要跨进程协调的场景）。"""

    def wait(self, timeout: float | None = None) -> bool:
        return True

    def set(self) -> None:
        pass

"""status 命令单测（方案 §7.4，Feature s3-feat-005）。

覆盖验收标准：
- AC4.1：build 运行期输出含「编译进行中」+ 当前阶段 + 进度表。
- AC4.2：build 完成后（无进度文件，静态产物）输出含「已编译」+ 文档数 + 向量数 + 模型。
- AC4.3：上次中断输出含「上次编译中断」+ 中断阶段 + 「重跑零成本」提示。
- AC4.4：旧 out/（无 .build_progress.json）向后兼容（status 仍正常工作）。
- AC4.5：非 TTY（CliRunner）输出纯文本表格可被捕获断言。
- Medium #1：heartbeat 过期但计数增长 → 输出「编译进行中」（不误报僵尸）。
- 场景 4（僵尸）：heartbeat 过期且无增长 → 输出「检测到中断的编译」。
- Opt#3：疑似僵尸场景非 TTY 进入 check_liveliness 不输出 spinner 文案仅 sleep。

策略：``typer.testing.CliRunner`` + ``console.is_terminal==False``（自动），构造
``out/.build_progress.json`` / ``out/manifest.json`` / ``out/graph.json`` 夹具。
被测代码：``src/nanokb/cli.py`` 的 ``status`` 命令（Feature s3-feat-005）。
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typer.testing import CliRunner

from nanokb.cli import app
from nanokb.models import FileState, Manifest
from nanokb.utils.io import atomic_write_json
from nanokb.utils.progress import (
    PROGRESS_FILENAME,
    BuildProgress,
    BuildStage,
    ExtractProgress,
    VectorProgress,
)

runner = CliRunner()


# ── 夹具辅助 ──────────────────────────────────────────────────────────


def _setup_dirs(cwd: Path) -> tuple[Path, Path]:
    """在 cwd 下创建 raw/ 与 out/ 目录，返回 (raw_dir, out_dir)。"""
    raw_dir = cwd / "raw"
    out_dir = cwd / "out"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, out_dir


def _write_doc(raw_dir: Path, name: str = "doc1.md", content: str = "# Doc\n\ncontent") -> None:
    raw_dir.joinpath(name).write_text(content, encoding="utf-8")


def _write_progress(
    out_dir: Path,
    *,
    stage: BuildStage = BuildStage.EXTRACT,
    heartbeat_age_sec: float = 0.0,
    pid: int = 12345,
    started_age_sec: float = 120.0,
    extract_total: int = 0,
    extract_completed: int = 0,
    extract_cached: int = 0,
    vector_total: int = 0,
    vector_indexed: int = 0,
    message: str = "",
    seq: int = 0,
) -> None:
    """直接写一个可控的 .build_progress.json 夹具。"""
    now = datetime.now(timezone.utc)
    hb = now - timedelta(seconds=heartbeat_age_sec)
    started = now - timedelta(seconds=started_age_sec)
    prog = BuildProgress(
        schema_version="1",
        pid=pid,
        stage=stage,
        started_at=started.isoformat(),
        heartbeat_ts=hb.isoformat(),
        force=False,
        seq=seq,
        extract=ExtractProgress(
            total=extract_total, completed=extract_completed, cached=extract_cached, skipped=0
        ),
        vector=VectorProgress(total_nodes=vector_total, indexed_nodes=vector_indexed),
        message=message,
    )
    atomic_write_json(out_dir / PROGRESS_FILENAME, prog.model_dump(mode="json"))


def _write_manifest(
    out_dir: Path,
    *,
    files: dict[str, FileState] | None = None,
    total_vectors: int = 0,
    last_compiled_at: str = "",
    last_llm_model: str = "",
    last_embedding_model: str = "",
) -> None:
    """写一个 manifest.json 夹具。"""
    manifest = Manifest(
        version="2",
        files=files or {},
        total_vectors=total_vectors,
        last_compiled_at=last_compiled_at,
        last_llm_model=last_llm_model,
        last_embedding_model=last_embedding_model,
    )
    atomic_write_json(out_dir / "manifest.json", manifest.model_dump(mode="json"))


def _write_graph(out_dir: Path) -> None:
    """写一个最小合法的 graph.json（node_link_data 格式）。"""
    atomic_write_json(out_dir / "graph.json", {"directed": True, "multigraph": True, "nodes": [], "links": []})


# ══════════════════════════════════════════════════════════════════════
# 空状态（向后兼容：raw/ 无 + out/ 无）
# ══════════════════════════════════════════════════════════════════════


def test_status_empty_state(tmp_path: Path) -> None:
    """空状态：raw/ 无文档且 out/ 未编译 → 旧格式输出。"""
    _setup_dirs(tmp_path)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "raw/" in result.stdout
    assert "0" in result.stdout
    assert "未编译" in result.stdout


# ══════════════════════════════════════════════════════════════════════
# 场景 1：编译进行中（AC4.1）
# ══════════════════════════════════════════════════════════════════════


def test_status_build_running_extract_stage(tmp_path: Path) -> None:
    """AC4.1：heartbeat fresh（EXTRACT 阶段）→ 输出「编译进行中」+ 阶段表 + PID。"""
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)
    _write_progress(
        out_dir,
        stage=BuildStage.EXTRACT,
        heartbeat_age_sec=2.0,  # fresh（< 60s）
        pid=12345,
        started_age_sec=135.0,
        extract_total=100,
        extract_completed=30,
        extract_cached=10,
        message="正在抽取 doc1.md",
    )

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    # AC4.1 关键字
    assert "编译进行中" in result.stdout
    assert "extract" in result.stdout
    assert "12345" in result.stdout  # PID
    # 进度百分比（30/100 = 30%）
    assert "30%" in result.stdout
    assert "30/100" in result.stdout


def test_status_build_running_vector_stage(tmp_path: Path) -> None:
    """AC4.1：VECTOR 阶段运行中 → extract 完成，vector 显示进度。"""
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)
    _write_progress(
        out_dir,
        stage=BuildStage.VECTOR,
        heartbeat_age_sec=1.0,
        pid=99999,
        extract_total=50,
        extract_completed=50,
        extract_cached=20,
        vector_total=5000,
        vector_indexed=2250,
    )

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "编译进行中" in result.stdout
    assert "vector" in result.stdout
    # 2250/5000 = 45%
    assert "45%" in result.stdout
    assert "2250/5000" in result.stdout
    # 之前的 extract 阶段标记为完成
    assert "完成" in result.stdout


def test_status_build_running_pid_zero_shown(tmp_path: Path) -> None:
    """PID=0 时输出仍包含「编译进行中」且不抛异常（边界健壮性）。"""
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)
    _write_progress(out_dir, stage=BuildStage.GRAPH, heartbeat_age_sec=3.0, pid=0)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "编译进行中" in result.stdout


# ══════════════════════════════════════════════════════════════════════
# 场景 2：已编译（静态产物，AC4.2 + AC4.4）
# ══════════════════════════════════════════════════════════════════════


def test_status_compiled_static_full(tmp_path: Path) -> None:
    """AC4.2：无进度文件 + manifest.json 携带新字段 → 输出「已编译」+ 向量数 + 模型。"""
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir, "a.md")
    _write_doc(raw_dir, "b.md")
    _write_graph(out_dir)
    _write_manifest(
        out_dir,
        files={
            "a.md": FileState(path="a.md", sha256="0" * 64, processed_at="2026-01-01T00:00:00Z"),
            "b.md": FileState(path="b.md", sha256="1" * 64, processed_at="2026-01-01T00:00:00Z"),
        },
        total_vectors=1530,
        last_compiled_at="2026-01-01T00:00:00Z",
        last_llm_model="glm-5.1",
        last_embedding_model="text-embedding-3-small",
    )

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    # AC4.2 关键字
    assert "已编译" in result.stdout
    assert "1530" in result.stdout  # 向量数
    assert "glm-5.1" in result.stdout  # LLM 模型
    assert "text-embedding-3-small" in result.stdout  # embedding 模型
    # 已编译文件数
    assert "2" in result.stdout


def test_status_compiled_static_manifest_without_new_fields(tmp_path: Path) -> None:
    """旧 manifest（无新字段）→ 读默认值显示 N/A（Opt#3 向后兼容）。"""
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)
    _write_graph(out_dir)
    # 旧格式 manifest（仅 version + files，无 total_vectors 等）
    atomic_write_json(
        out_dir / "manifest.json",
        {
            "version": "2",
            "files": {
                "doc.md": {
                    "path": "doc.md",
                    "sha256": "0" * 64,
                    "processed_at": "2026-01-01T00:00:00Z",
                }
            },
        },
    )

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "已编译" in result.stdout
    # 缺失字段显示 N/A
    assert "N/A" in result.stdout


def test_status_compiled_only_graph_no_manifest(tmp_path: Path) -> None:
    """AC4.4：仅有 graph.json 无 manifest → 仍能展示「已编译」（向后兼容）。"""
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)
    _write_graph(out_dir)
    # 无 manifest.json

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "已编译" in result.stdout


def test_status_uncompiled_with_docs(tmp_path: Path) -> None:
    """AC4.4：有 raw 文档但无任何 out 产物 → 旧格式「未编译」（向后兼容）。"""
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    # 旧格式：raw/ 下 N 个文档 | out/ 未编译
    assert "raw/" in result.stdout
    assert "未编译" in result.stdout


# ══════════════════════════════════════════════════════════════════════
# 场景 3：上次中断（AC4.3）
# ══════════════════════════════════════════════════════════════════════


def test_status_last_interrupted(tmp_path: Path) -> None:
    """AC4.3：stage=INTERRUPTED → 输出「上次编译中断」+ 中断阶段 + 重跑零成本提示。

    BuildProgressWriter.interrupted() 把 stage 覆写为 INTERRUPTED，但 message 保留
    中断前最后一次 set_stage 的上下文。夹具模拟「正在抽取 doc1.md」中断场景。
    """
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)
    _write_progress(
        out_dir,
        stage=BuildStage.INTERRUPTED,
        heartbeat_age_sec=3600.0,  # 1 小时前（无所谓，INTERRUPTED 短路）
        pid=12345,
        started_age_sec=4000.0,
        message="正在抽取 doc1.md",
    )

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "上次编译中断" in result.stdout
    assert "正在抽取 doc1.md" in result.stdout  # 中断阶段上下文（来自 message）
    assert "重跑零成本" in result.stdout


# ══════════════════════════════════════════════════════════════════════
# 场景 4：僵尸进程（heartbeat 超时且无增长）
# ══════════════════════════════════════════════════════════════════════


def test_status_zombie_heartbeat_stale_no_growth(tmp_path: Path) -> None:
    """heartbeat 过期且计数无增长 → 输出「检测到中断的编译」（僵尸场景）。"""
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)
    _write_progress(
        out_dir,
        stage=BuildStage.VECTOR,
        heartbeat_age_sec=600.0,  # 远超 60s 阈值
        extract_total=10,
        extract_completed=5,
        vector_total=100,
        vector_indexed=20,
    )

    # 非 TTY：check_liveliness 会 sleep progress_liveliness_recheck_sec（默认 1s）
    # 为加速测试，把 recheck_sec 调到 0.05
    result = runner.invoke(
        app,
        ["status"],
        env={"NANOKB_PROGRESS_LIVELINESS_RECHECK_SEC": "0.05"},
    )
    assert result.exit_code == 0, result.stdout
    assert "检测到中断的编译" in result.stdout
    assert "vector" in result.stdout  # 最后阶段
    # 不应误报为「编译进行中」
    assert "编译进行中" not in result.stdout


# ══════════════════════════════════════════════════════════════════════
# Medium #1：heartbeat 过期但计数增长 → 输出「编译进行中」
# ══════════════════════════════════════════════════════════════════════


def test_status_medium1_heartbeat_stale_but_growing(tmp_path: Path) -> None:
    """Medium #1：heartbeat 过期但 extract.completed 增长 → 输出「编译进行中」。

    模拟：夹具初始 completed=3、heartbeat 过期 600s；测试线程在 recheck 窗口内
    把 completed 改写到 4，使 check_liveliness 观察到增长 → 判 alive。
    """
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)
    _write_progress(
        out_dir,
        stage=BuildStage.EXTRACT,
        heartbeat_age_sec=600.0,  # heartbeat 过期
        extract_total=10,
        extract_completed=3,
        seq=1,
    )

    # 后台线程：recheck 窗口内推进 completed → 4（业务 flush 同步 bump seq）
    def _grow() -> None:
        time.sleep(0.1)
        _write_progress(
            out_dir,
            stage=BuildStage.EXTRACT,
            heartbeat_age_sec=600.0,
            extract_total=10,
            extract_completed=4,
            seq=2,
        )

    threading.Thread(target=_grow, daemon=True).start()

    # recheck_sec=0.2：足够让 _grow 触发
    result = runner.invoke(
        app,
        ["status"],
        env={"NANOKB_PROGRESS_LIVELINESS_RECHECK_SEC": "0.2"},
    )
    assert result.exit_code == 0, result.stdout
    # Medium #1 核心：不误报僵尸，仍判「编译进行中」
    assert "编译进行中" in result.stdout
    assert "检测到中断的编译" not in result.stdout


# ══════════════════════════════════════════════════════════════════════
# Opt#3：非 TTY 进入 check_liveliness 不输出 spinner 文案
# ══════════════════════════════════════════════════════════════════════


def test_status_opt3_non_tty_no_spinner_text(tmp_path: Path) -> None:
    """Opt#3：非 TTY 疑似僵尸场景进入 check_liveliness 不输出 spinner 文案。

    CliRunner 默认非 TTY，进入 check_liveliness 时仅 sleep，不输出
    「正在复核进程存活」文案（AC4.5 CliRunner 兼容）。
    """
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)
    _write_progress(
        out_dir,
        stage=BuildStage.VECTOR,
        heartbeat_age_sec=600.0,  # 触发 check_liveliness
    )

    result = runner.invoke(
        app,
        ["status"],
        env={"NANOKB_PROGRESS_LIVELINESS_RECHECK_SEC": "0.05"},
    )
    assert result.exit_code == 0, result.stdout
    # Opt#3：非 TTY 下不应出现 spinner 文案
    assert "正在复核进程存活" not in result.stdout
    # 但仍应输出僵尸判定结果
    assert "检测到中断的编译" in result.stdout


# ══════════════════════════════════════════════════════════════════════
# AC3.4：status 绝不打开 chroma（结构性验证）
# ══════════════════════════════════════════════════════════════════════


def test_status_does_not_open_chroma(tmp_path: Path) -> None:
    """AC3.4：status 读取路径不打开 out/chroma/（哨兵文件未被触碰）。

    构造运行期 + 静态两个场景，均不创建 chroma 目录，确认 status 正常工作。
    进一步：在 out/chroma/ 放哨兵文件，status 运行后哨兵内容不变。
    """
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)
    chroma_dir = out_dir / "chroma"
    chroma_dir.mkdir()
    sentinel = chroma_dir / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")

    # 场景 1：运行期分支
    _write_progress(out_dir, stage=BuildStage.EXTRACT, heartbeat_age_sec=2.0)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "编译进行中" in result.stdout
    assert sentinel.read_text(encoding="utf-8") == "untouched"

    # 场景 2：静态分支（删 progress 文件）
    (out_dir / PROGRESS_FILENAME).unlink()
    _write_graph(out_dir)
    _write_manifest(
        out_dir,
        files={"doc.md": FileState(path="doc.md", sha256="0" * 64, processed_at="2026-01-01T00:00:00Z")},
        total_vectors=100,
        last_llm_model="m1",
        last_embedding_model="m2",
    )
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "已编译" in result.stdout
    assert sentinel.read_text(encoding="utf-8") == "untouched"


# ══════════════════════════════════════════════════════════════════════
# 损坏的进度文件降级（AC3.5）
# ══════════════════════════════════════════════════════════════════════


def test_status_corrupted_progress_falls_back_to_static(tmp_path: Path) -> None:
    """AC3.5：损坏的 .build_progress.json → read_progress 返回 None → 走静态分支。"""
    raw_dir, out_dir = _setup_dirs(tmp_path)
    _write_doc(raw_dir)
    _write_graph(out_dir)
    _write_manifest(
        out_dir,
        files={"doc.md": FileState(path="doc.md", sha256="0" * 64, processed_at="2026-01-01T00:00:00Z")},
        total_vectors=50,
        last_llm_model="llm",
        last_embedding_model="embed",
    )
    # 写损坏的进度文件
    (out_dir / PROGRESS_FILENAME).write_text("{ invalid json @@", encoding="utf-8")

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    # 降级到静态分支
    assert "已编译" in result.stdout

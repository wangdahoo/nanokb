"""Nano KB CLI —— typer 命令（方案 §3.1 + §3.5.3）。

六个子命令：build / query / ask / search / status / review。
build 接入编译流水线（Feature s1-feat-008，``--watch`` 接入 s1-feat-004 queue 模型）；
query/ask/search 接入三路召回问答（Feature s1-feat-012：
query=三路融合 / ask=仅向量 / search=仅社区）；review 接入主动学习闭环
（Feature s1-feat-013：列出 / 清空 out/review_queue.md）。
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nanokb import pipeline
from nanokb.config import Settings
from nanokb.llm.base import make_llm_client
from nanokb.load.detector import start_watch
from nanokb.logging_setup import setup_logging
from nanokb.models import Manifest, RetrievalHit
from nanokb.qa.progress import _SOURCE_LABELS
from nanokb.qa.review import ReviewQueue
from nanokb.utils.progress import (
    PROGRESS_LIVENESS_RECHECK_SEC,
    BuildProgress,
    BuildStage,
    check_liveliness,
    is_alive,
    read_progress,
)

app = typer.Typer(
    name="nanokb",
    help="Nano KB — 基于 LLM-as-Wiki 理念的极简个人知识库工具。",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
logger = logging.getLogger("nanokb")

# 知识库支持的文档扩展名（用于 status 统计）
_SUPPORTED_SUFFIXES = frozenset({".md", ".txt", ".pdf", ".docx", ".py", ".js", ".java"})


def _load_settings() -> Settings:
    """从环境变量 / .env 加载配置。"""
    return Settings()


def _count_documents(raw_dir: Path) -> int:
    """递归统计 raw_dir 下受支持扩展名的文件数。"""
    if not raw_dir.exists():
        return 0
    return sum(
        1 for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )


def _iter_documents(raw_dir: Path) -> Iterable[Path]:
    """遍历 raw_dir 下受支持的文档。"""
    if not raw_dir.exists():
        return ()
    return (
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )


def _print_compile_summary(result: pipeline.CompileResult) -> None:
    """打印编译结果摘要。"""
    ch = result.changes
    parts: list[str] = [
        f"added={len(ch.added)}",
        f"modified={len(ch.modified)}",
        f"deleted={len(ch.deleted)}",
        f"extracted={result.extracted_count}",
    ]
    if result.cached_count:
        parts.append(f"cached={result.cached_count}")
    if result.skipped:
        parts.append(f"skipped={len(result.skipped)}")
    if result.synthesized_fallback_count:
        parts.append(f"fallback={result.synthesized_fallback_count}")
    console.print(f"[green]编译完成：{', '.join(parts)}[/green]")


# ── status 命令渲染辅助（Feature s3-feat-005）─────────────────────────


#: status 渲染的阶段顺序（与 BuildStage 枚举对齐，排除终态 DONE/INTERRUPTED）。
_STAGE_ORDER: tuple[BuildStage, ...] = (
    BuildStage.DETECT,
    BuildStage.EXTRACT,
    BuildStage.GRAPH,
    BuildStage.VECTOR,
    BuildStage.INDEX,
    BuildStage.FINALIZE,
)

#: 各阶段的中文标签（status 表格 / 纯文本共用）。
_STAGE_LABELS: dict[BuildStage, str] = {
    BuildStage.DETECT: "detect (变更检测)",
    BuildStage.EXTRACT: "extract (抽取)",
    BuildStage.GRAPH: "graph (图谱构建)",
    BuildStage.VECTOR: "vector (向量索引)",
    BuildStage.INDEX: "index (社区/关键词索引)",
    BuildStage.FINALIZE: "finalize (落盘切换)",
}


def _format_elapsed(started_at: str) -> str:
    """从 ISO 8601 started_at 计算「已运行时长」的人类可读字符串。

    解析失败 / 时间为空时返回 ``"未知"``。返回形如 ``"2m 15s"`` / ``"45s"`` / ``"1h 3m"``。
    """
    if not started_at:
        return "未知"
    try:
        ts = datetime.fromisoformat(started_at)
    except (ValueError, TypeError):
        return "未知"
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    secs = int((now - ts).total_seconds())
    if secs < 0:
        secs = 0
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    return f"{secs // 3600}h {(secs % 3600) // 60}m"


def _stage_state(progress: BuildProgress, stage: BuildStage) -> tuple[str, str, str]:
    """计算单个阶段的 (状态标记, 进度百分比, 命中/总数) 三元组。

    - 当前阶段（progress.stage）→ ``("run", "<x%>", "<hit>/<total>")``
    - 当前阶段之前的阶段 → ``("ok", "完成", "")`` 或带计数
    - 之后的阶段 → ``("pending", "待开始", "")``

    对 extract / vector 两阶段展示细化进度百分比。
    """
    current = progress.stage
    try:
        current_idx = _STAGE_ORDER.index(current)
    except ValueError:
        # DONE / INTERRUPTED：全部阶段视为完成
        current_idx = len(_STAGE_ORDER)
    try:
        stage_idx = _STAGE_ORDER.index(stage)
    except ValueError:
        return ("ok", "完成", "")

    if stage_idx < current_idx:
        # 已完成阶段：附带最终计数（若有）
        if stage == BuildStage.EXTRACT:
            total = progress.extract.total
            hit = progress.extract.completed
            if total > 0:
                return ("ok", "完成", f"{hit}/{total} (cached {progress.extract.cached})")
        if stage == BuildStage.VECTOR:
            total = progress.vector.total_nodes
            hit = progress.vector.indexed_nodes
            if total > 0:
                return ("ok", "完成", f"{hit}/{total}")
        return ("ok", "完成", "")

    if stage_idx > current_idx:
        return ("pending", "待开始", "")

    # 当前阶段：展示细化进度
    if stage == BuildStage.EXTRACT:
        total = progress.extract.total
        hit = progress.extract.completed
        if total > 0:
            pct = f"{int(hit * 100 / total)}%"
            return ("run", pct, f"{hit}/{total} (cached {progress.extract.cached})")
        return ("run", "进行中", "")
    if stage == BuildStage.VECTOR:
        total = progress.vector.total_nodes
        hit = progress.vector.indexed_nodes
        if total > 0:
            pct = f"{int(hit * 100 / total)}%"
            return ("run", pct, f"{hit}/{total}")
        return ("run", "进行中", "")
    return ("run", "进行中", "")


def _stage_marker(state: str) -> str:
    """状态标记 → 显示符号（ok=✓ run=► pending=·）。"""
    if state == "ok":
        return "✓"
    if state == "run":
        return "►"
    return "·"


def _render_progress_table(progress: BuildProgress, *, terminal: bool) -> Table | list[str]:
    """渲染阶段化进度表（``rich.Table`` 或纯文本多行字符串）。

    ``terminal=True`` 返回 ``rich.table.Table``（console.print 消费）；
    ``terminal=False`` 返回 ``list[str]``（每行一条，便于 ``CliRunner`` 捕获断言）。
    """
    rows: list[tuple[str, str, str, str, str]] = []  # (marker, stage_label, state_text, pct, hit_total)
    for stage in _STAGE_ORDER:
        state, pct, hit_total = _stage_state(progress, stage)
        rows.append((_stage_marker(state), _STAGE_LABELS[stage], state, pct, hit_total))

    if terminal:
        table = Table(title=None, show_header=True, header_style="bold cyan", expand=False)
        table.add_column("状态", style="green", no_wrap=True)
        table.add_column("阶段", style="white")
        table.add_column("进度", style="yellow")
        table.add_column("命中/总数", style="dim")
        for marker, label, _state, pct, hit_total in rows:
            color = "green" if marker == "✓" else ("cyan" if marker == "►" else "dim")
            table.add_row(f"[{color}]{marker}[/{color}]", label, pct, hit_total)
        return table

    # 非 TTY：纯文本表格（固定列宽对齐，CliRunner 友好）
    lines = [
        f"  {marker:<2} {label:<32} {pct:<8} {hit_total}".rstrip()
        for marker, label, _state, pct, hit_total in rows
    ]
    return lines


def _render_running_status(progress: BuildProgress, *, terminal: bool) -> None:
    """场景 1：编译进行中（PID + 已运行时长 + 阶段表 + message）。"""
    elapsed = _format_elapsed(progress.started_at)
    pid_text = f"PID {progress.pid}, 已运行 {elapsed}" if progress.pid else f"已运行 {elapsed}"
    stage_label = _STAGE_LABELS.get(progress.stage, progress.stage.value)
    header_msg = progress.message or ""
    stage_line = f"阶段  {stage_label}"
    if header_msg:
        stage_line += f" — {header_msg}"

    table = _render_progress_table(progress, terminal=terminal)

    if terminal:
        assert isinstance(table, Table)
        panel = Panel(
            Group(Text(f"编译进行中  ({pid_text})", style="bold cyan"), Text(stage_line), table),
            title="nanokb 状态",
            border_style="cyan",
        )
        console.print(panel)
    else:
        console.print("编译进行中  (" + pid_text + ")")
        console.print(stage_line)
        if isinstance(table, list):
            for line in table:
                console.print(line)


def _render_interrupted_status(progress: BuildProgress, *, terminal: bool) -> None:
    """场景 3：上次编译中断（中断阶段 + 时间 + 重跑零成本提示）。

    ``BuildProgressWriter.interrupted()`` 把 ``stage`` 覆写为 ``INTERRUPTED``，但
    ``message`` 字段保留中断前最后一次 ``set_stage`` 的上下文（如「正在抽取 doc.md」）。
    此处优先展示 message；无 message 时降级展示 ``stage.value``。
    """
    if progress.message:
        stage_text = progress.message
    elif progress.stage in _STAGE_LABELS:
        stage_text = _STAGE_LABELS[progress.stage]
    else:
        stage_text = progress.stage.value
    elapsed = _format_elapsed(progress.started_at)
    lines = [
        "上次编译中断",
        f"中断阶段  {stage_text}",
        f"中断时间  {progress.heartbeat_ts or '未知'}  (已运行 {elapsed})",
        "提示  中断后重跑零成本：已抽取的 cache 与已索引的向量不丢失，直接 nanokb build 继续。",
    ]
    if terminal:
        panel = Panel(
            Text("\n".join(lines[1:])),
            title=lines[0],
            border_style="yellow",
        )
        console.print(panel)
    else:
        for line in lines:
            console.print(line)


def _render_zombie_status(progress: BuildProgress, *, terminal: bool) -> None:
    """场景 4：僵尸进程（heartbeat 超时且无增长）。

    展示最后阶段 + 最后心跳 + 提示（中断的编译，可重跑）。
    """
    stage_label = _STAGE_LABELS.get(progress.stage, progress.stage.value)
    lines = [
        "检测到中断的编译",
        f"最后阶段  {stage_label}",
        f"最后心跳  {progress.heartbeat_ts or '未知'}",
        "提示  进程疑似已退出（heartbeat 超时且计数无增长），可 nanokb build 重跑（增量/零成本）。",
    ]
    if terminal:
        panel = Panel(
            Text("\n".join(lines[1:])),
            title=lines[0],
            border_style="red",
        )
        console.print(panel)
    else:
        for line in lines:
            console.print(line)


def _load_manifest_safely(out_dir: Path) -> Manifest | None:
    """读取 out/manifest.json；不存在 / 损坏返回 None（绝不抛异常阻断 status）。"""
    path = out_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        import json

        return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        logger.debug("failed to parse %s; treating as no manifest", path, exc_info=True)
        return None


def _count_index_stats(out_dir: Path) -> tuple[int, int]:
    """读取已编译产物中的 community / keyword 计数（best-effort，失败返回 0）。

    从 out/communities.json 与 out/keywords.json 读取（仅纯 JSON，绝不打开 chroma）。
    """
    import json

    comm_count = 0
    kw_count = 0
    comm_path = out_dir / "communities.json"
    kw_path = out_dir / "keywords.json"
    try:
        if comm_path.exists():
            data = json.loads(comm_path.read_text(encoding="utf-8"))
            communities = data.get("communities") if isinstance(data, dict) else data
            if isinstance(communities, list):
                comm_count = len(communities)
    except Exception:
        logger.debug("failed to parse %s", comm_path, exc_info=True)
    try:
        if kw_path.exists():
            data = json.loads(kw_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # keywords.json schema：{ "keywords": {...} } 或直接 dict
                kws = data.get("keywords", data)
                if isinstance(kws, dict):
                    kw_count = len(kws)
                elif isinstance(kws, list):
                    kw_count = len(kws)
    except Exception:
        logger.debug("failed to parse %s", kw_path, exc_info=True)
    return comm_count, kw_count


def _render_compiled_status(
    *,
    doc_count: int,
    compiled_count: int,
    manifest: Manifest | None,
    out_dir: Path,
    terminal: bool,
) -> None:
    """场景 2：已编译（静态：文档数 + 已编译数 + 向量数 + 模型 + 索引统计）。"""
    if manifest is not None:
        vectors = manifest.total_vectors
        last_llm = manifest.last_llm_model or "N/A"
        last_embed = manifest.last_embedding_model or "N/A"
        compiled_at = manifest.last_compiled_at or "N/A"
    else:
        vectors = 0
        last_llm = "N/A"
        last_embed = "N/A"
        compiled_at = "N/A"

    comm_count, kw_count = _count_index_stats(out_dir)
    vectors_text = str(vectors) if vectors > 0 else "N/A"

    lines = [
        f"已编译  raw/ {doc_count} 个文档 | out/ 已编译 {compiled_count} 个",
        f"向量数  {vectors_text}",
        f"模型    LLM={last_llm} | Embedding={last_embed}",
        f"索引    社区 {comm_count} 个 | 关键词 {kw_count} 个",
        f"编译时间  {compiled_at}",
    ]
    if terminal:
        panel = Panel(
            Text("\n".join(lines[1:])),
            title=lines[0],
            border_style="green",
        )
        console.print(panel)
    else:
        for line in lines:
            console.print(line)


def _check_liveliness_with_spinner(
    out_dir: Path,
    *,
    recheck_sec: float,
    terminal: bool,
) -> bool:
    """包裹 check_liveliness：TTY 下先显示 spinner 再 sleep；非 TTY 直接 sleep。

    round 3 Opt#3：疑似僵尸场景（heartbeat 过期）进入 check_liveliness 时，TTY 显示
    ``「正在复核进程存活…（约 Ns）」`` spinner 改善体感；非 TTY（测试 / 重定向）跳过
    spinner 文案仅 sleep（AC4.5 CliRunner 兼容）。
    """
    if not terminal:
        # 非 TTY：直接调用 check_liveliness（内部 sleep recheck_sec），不输出 spinner 文案
        return check_liveliness(out_dir, recheck_sec=recheck_sec)

    # TTY：spinner 展示「正在复核…」，内部仍走 check_liveliness 的 sleep + 二次采样
    message = f"正在复核进程存活…（约 {int(recheck_sec)}s）"
    result_box: list[bool] = [False]
    with console.status(message, spinner="dots"):
        result_box[0] = check_liveliness(out_dir, recheck_sec=recheck_sec)
    return result_box[0]


class RichProgressReporter:
    """基于 ``rich`` 的检索进度报告器（CLI 表现层）。

    - **TTY**：阶段期间用 ``console.status`` 显示 spinner；退出时打印持久日志行
      ``✓ {msg} ({elapsed}s)``，保留各阶段与耗时记录。
    - **非 TTY**（测试 / 重定向 / 管道）：不转圈，仅打印 ``[dim]{msg}[/dim]`` 状态行，
      保证输出可读且被 ``CliRunner`` 正常捕获。

    实现 ``qa.progress.ProgressReporter`` 协议（结构化匹配，无需显式继承）。
    """

    def __init__(self, console: Console) -> None:
        self._console = console

    def stage(self, message: str) -> AbstractContextManager[None]:
        return _rich_stage(self._console, message)


@contextmanager
def _rich_stage(console: Console, message: str) -> Iterator[None]:
    """``RichProgressReporter.stage`` 的上下文实现（拆出以便用 ``@contextmanager``）。"""
    if not console.is_terminal:
        console.print(f"[dim]{message}[/dim]")
        yield
        return
    start = time.monotonic()
    with console.status(message, spinner="dots"):
        yield
    elapsed = time.monotonic() - start
    console.print(f"[green]✓[/green] [dim]{message}[/dim] [dim]({elapsed:.1f}s)[/dim]")


def _print_recall_summary(hits: list[RetrievalHit]) -> None:
    """打印召回摘要行：``召回 N 条：图谱召回2 / 向量召回2 / 社区召回1``。

    按 ``_SOURCE_LABELS`` 定义的检索顺序（图谱 → 向量 → 社区）排列，与实际召回流程一致。
    """
    if not hits:
        console.print("[dim]召回 0 条[/dim]")
        return
    by_source: dict[str, int] = {}
    for hit in hits:
        by_source[hit.source] = by_source.get(hit.source, 0) + 1
    # 按 _SOURCE_LABELS 定义的检索顺序输出；未知 source 追加在末尾（按出现序）
    ordered: list[str] = []
    for src in _SOURCE_LABELS:
        if src in by_source:
            ordered.append(f"{_SOURCE_LABELS[src]}{by_source.pop(src)}")
    for src in sorted(by_source):
        ordered.append(f"{src}{by_source[src]}")
    console.print(f"[dim]召回 {len(hits)} 条：{' / '.join(ordered)}[/dim]")


def _run_watch(settings: Settings, *, force: bool) -> None:
    """启动 watch 模式：首次编译后监听 raw/ 变更，debounce 后增量编译。

    使用 s1-feat-004 的 watchdog queue 模型（回调入队 + 单 worker 串行消费）。
    """
    raw_dir = settings.raw_dir
    if not raw_dir.exists():
        console.print(f"[red]raw/ 目录不存在：{raw_dir}[/red]")
        raise typer.Exit(code=1)

    llm = make_llm_client(settings)
    registry = pipeline.build_default_registry()

    result = pipeline.compile(settings, llm=llm, registry=registry, force=force)
    _print_compile_summary(result)

    console.print("\n[green]Watch 模式已启动（Ctrl-C 退出）...[/green]")

    def on_change(path: str) -> None:
        console.print(f"[dim]检测到变更：{path}，正在增量编译...[/dim]")
        try:
            res = pipeline.compile(settings, llm=llm, registry=registry)
            _print_compile_summary(res)
        except Exception:
            logger.exception("watch compile failed", extra={"stage": "watch", "file": path})

    ctx = start_watch(raw_dir, on_change=on_change)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        ctx.stop()
        console.print("[green]Watch 已停止。[/green]")


@app.command()
def build(
    watch: bool = typer.Option(False, "--watch", help="监听 raw/ 变更，自动增量编译。"),
    force: bool = typer.Option(False, "--force", help="强制全量重编译。"),
    replay: bool = typer.Option(False, "--replay", help="从 out/triples.jsonl 重放重建图谱。"),
) -> None:
    """编译知识库（增量检测 → 双轨抽取 → 图谱融合 → 索引）。"""
    settings = _load_settings()
    setup_logging(settings.out_dir)

    if replay:
        replay_result = pipeline.replay(settings)
        if not replay_result.rebuilt_files and not replay_result.deleted_files:
            console.print("[yellow]无可重放记录（triples.jsonl 为空或不存在）[/yellow]")
        else:
            console.print(
                f"[green]重放完成：重建 {len(replay_result.rebuilt_files)} 个文件，"
                f"跳过 {len(replay_result.deleted_files)} 个已删除文件[/green]"
            )
        return

    if watch:
        _run_watch(settings, force=force)
        return

    try:
        result = pipeline.compile(settings, force=force)
    except KeyboardInterrupt:
        # Ctrl-C 即时退出。抽取阶段失败安全（staging/原子写），已完成的抽取已落盘
        # cache；但 worker 线程为非守护线程会阻塞解释器 shutdown，故强制退出。
        # 退出码 130 = 128 + SIGINT(2)，符合 POSIX 终止约定。
        console.print("[yellow]\n已中断。[/yellow]")
        os._exit(130)
    _print_compile_summary(result)


@app.command()
def query(
    question: str = typer.Argument(..., help="自然语言问题。"),
) -> None:
    """图谱推理问答（graph + vector + community 三路召回融合）。"""
    settings = _load_settings()
    setup_logging(settings.out_dir)
    progress = RichProgressReporter(console)
    try:
        result = pipeline.answer_query(settings, question, mode="query", progress=progress)
    except pipeline.ColdStartError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    _print_recall_summary(result.hits)
    answer = result.answer
    console.print(answer.text)

    if answer.citations:
        unique = []
        seen: set[str] = set()
        for cite in answer.citations:
            if cite not in seen:
                seen.add(cite)
                unique.append(cite)
        console.print(f"\n[dim]引用来源：{', '.join(unique)}[/dim]")
    elif result.hits:
        sources: set[str] = {h.triple.source_file for h in result.hits if h.triple is not None}
        if sources:
            console.print(f"\n[dim]引用来源：{', '.join(sorted(sources))}[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="自然语言问题。"),
) -> None:
    """向量路语义问答（仅向量召回，适合模糊语义匹配）。"""
    settings = _load_settings()
    setup_logging(settings.out_dir)
    progress = RichProgressReporter(console)
    try:
        result = pipeline.answer_query(settings, question, mode="ask", progress=progress)
    except pipeline.ColdStartError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    _print_recall_summary(result.hits)
    answer = result.answer
    console.print(answer.text)

    if answer.citations:
        unique = []
        seen_cite: set[str] = set()
        for cite in answer.citations:
            if cite not in seen_cite:
                seen_cite.add(cite)
                unique.append(cite)
        console.print(f"\n[dim]引用来源：{', '.join(unique)}[/dim]")


@app.command()
def search(
    keyword: str = typer.Argument(..., help="检索关键词。"),
    community: bool = typer.Option(False, "--community", help="社区宏观检索（返回所属社区摘要）。"),
) -> None:
    """社区路宏观检索（按关键词返回所属社区摘要）。

    ``--community`` 显式启用社区检索（与方案 §3.5.3 命令映射一致：search=仅社区路）。
    缺失社区索引时提示先 ``nanokb build``。
    """
    settings = _load_settings()
    setup_logging(settings.out_dir)
    _ = community  # search 命令始终走社区路（命令映射固定），flag 保留作显式提示
    progress = RichProgressReporter(console)
    try:
        hits = pipeline.search_communities(settings, keyword, progress=progress)
    except pipeline.ColdStartError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not hits:
        console.print(f"[yellow]未找到与 '{keyword}' 相关的社区[/yellow]")
        return

    console.print(f"[green]找到 {len(hits)} 个相关社区：[/green]")
    for hit in hits:
        summary = hit.community_summary or "(无摘要)"
        source = hit.concept.source_file if hit.concept else ""
        line = f"- {summary}"
        if source:
            line += f" (来源:{source})"
        console.print(line)


@app.command()
def status() -> None:
    """显示知识库编译状态（方案 §7，Feature s3-feat-005）。

    数据来源优先级（§7.1）：
    1. 运行期优先：``out/.build_progress.json``（feat-004 产出，跨进程可见）。
    2. 静态兜底：``out/manifest.json`` + ``out/graph.json``（向后兼容，AC4.4）。

    四种场景输出（§7.2）：
    - 编译进行中（heartbeat fresh 或计数增长）→ 阶段化进度表 + PID/已运行时长。
    - 已编译（无进度文件，静态产物）→ 文档数/向量数/模型/索引统计。
    - 上次中断（stage=INTERRUPTED）→ 中断阶段 + 重跑零成本提示。
    - 僵尸进程（heartbeat 超时且无增长）→ 检测到中断的编译 + 静态产物。

    **status 绝不打开 chroma**（AC3.4 / Medium #3）：向量数从 progress 或 manifest 读，
    不从 chroma 读。TTY 彩色 Panel/Table + spinner；非 TTY 纯文本（AC4.5 CliRunner 兼容）。
    """
    settings = _load_settings()
    raw_dir = settings.raw_dir
    out_dir = settings.out_dir

    doc_count = _count_documents(raw_dir)
    graph_compiled = (out_dir / "graph.json").exists()
    terminal = console.is_terminal

    # ── 运行期分支：优先读 .build_progress.json ──────────────────────
    progress = read_progress(out_dir)
    if progress is not None:
        # 场景 3：上次中断（stage=INTERRUPTED，文件保留供诊断）
        if progress.stage == BuildStage.INTERRUPTED:
            _render_interrupted_status(progress, terminal=terminal)
            return

        # 场景 1：编译进行中（is_alive = heartbeat fresh）
        if is_alive(progress):
            _render_running_status(progress, terminal=terminal)
            return

        # heartbeat 过期 → 进入 check_liveliness 次级判据（Medium #1 / Opt#3）。
        # TTY 下先显示 spinner「正在复核进程存活…」再 sleep；非 TTY 直接 sleep。
        recheck_sec = float(settings.progress_liveliness_recheck_sec or PROGRESS_LIVENESS_RECHECK_SEC)
        if _check_liveliness_with_spinner(out_dir, recheck_sec=recheck_sec, terminal=terminal):
            # Medium #1：计数增长 → 仍判「编译进行中」（不误报僵尸）
            refreshed = read_progress(out_dir) or progress
            _render_running_status(refreshed, terminal=terminal)
            return

        # 场景 4：僵尸进程（heartbeat 超时且无增长）
        _render_zombie_status(progress, terminal=terminal)
        return

    # ── 静态分支：无运行期进度文件（向后兼容 AC4.4）──────────────────
    manifest = _load_manifest_safely(out_dir)
    compiled_count = len(manifest.files) if manifest is not None else 0

    # 空状态：raw/ 无文档且 out/ 未编译（与改造前一致）
    if doc_count == 0 and not graph_compiled and compiled_count == 0:
        console.print(f"[yellow]raw/ 下 {doc_count} 个文档，out/ 未编译[/yellow]")
        return

    if graph_compiled or compiled_count > 0:
        # 场景 2：已编译（静态）
        _render_compiled_status(
            doc_count=doc_count,
            compiled_count=compiled_count,
            manifest=manifest,
            out_dir=out_dir,
            terminal=terminal,
        )
        return

    # 有 raw/ 文档但未编译
    state = "已编译" if graph_compiled else "未编译"
    console.print(f"raw/ 下 {doc_count} 个文档 | out/ {state}")


@app.command()
def review(
    clear: bool = typer.Option(False, "--clear", help="清空 review 待审队列。"),
) -> None:
    """列出 / 清空主动学习待审队列（out/review_queue.md）。"""
    settings = _load_settings()
    setup_logging(settings.out_dir)
    queue = ReviewQueue(settings.out_dir)

    if clear:
        queue.clear()
        console.print("[green]已清空 review 待审队列。[/green]")
        return

    entries = queue.list_pending()
    if not entries:
        console.print("[yellow]review 队列为空（无待审条目）。[/yellow]")
        return

    console.print(f"[green]待审条目（{len(entries)} 条）：[/green]")
    for idx, entry in enumerate(entries, 1):
        console.print(f"{idx}. {entry.question}")
        console.print(
            f"   [dim]原因：{entry.reason} | 实体：{entry.entities} | 时间：{entry.timestamp}[/dim]"
        )


__all__ = ["app"]

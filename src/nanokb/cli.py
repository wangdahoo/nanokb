"""Nano KB CLI —— typer 命令（方案 §3.1 + §3.5.3）。

六个子命令：build / query / ask / search / status / review。
build 接入编译流水线（Feature s1-feat-008）；query 接入图路问答
（Feature s1-feat-009，Opt #5 v3 降级：仅 graph 路）；ask/search 为阶段 3 打桩
（Opt #5 v3，阶段 4 s1-feat-012 接入真实 retriever）；review 待 s1-feat-013 接入。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path

import typer
from rich.console import Console

from nanokb import pipeline
from nanokb.config import Settings
from nanokb.llm.base import make_llm_client
from nanokb.logging_setup import setup_logging
from nanokb.stage1_load.detector import start_watch

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
        1
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
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
    if result.skipped:
        parts.append(f"skipped={len(result.skipped)}")
    if result.synthesized_fallback_count:
        parts.append(f"fallback={result.synthesized_fallback_count}")
    console.print(f"[green]编译完成：{', '.join(parts)}[/green]")


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

    result = pipeline.compile(settings, force=force)
    _print_compile_summary(result)


@app.command()
def query(
    question: str = typer.Argument(..., help="自然语言问题。"),
) -> None:
    """图谱推理问答（三路召回融合；阶段 3 仅 graph 路，Opt #5 降级）。"""
    settings = _load_settings()
    setup_logging(settings.out_dir)
    try:
        result = pipeline.answer_query(settings, question)
    except pipeline.ColdStartError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

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
        sources: set[str] = {
            h.triple.source_file for h in result.hits if h.triple is not None
        }
        if sources:
            console.print(f"\n[dim]引用来源：{', '.join(sorted(sources))}[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="自然语言问题。"),
) -> None:
    """向量路语义问答（阶段 3 打桩，Opt #5 v3；阶段 4 s1-feat-012 补全）。"""
    console.print(
        "该命令需先完成阶段4（向量索引）后接入。请先运行 nanokb build 完成高级索引。"
    )


@app.command()
def search(
    keyword: str = typer.Argument(..., help="检索关键词。"),
    community: bool = typer.Option(False, "--community", help="社区宏观检索。"),
) -> None:
    """社区路宏观检索（阶段 3 打桩，Opt #5 v3；阶段 4 s1-feat-012 补全）。"""
    _ = community  # 阶段 4 s1-feat-012 接入社区检索时消费
    console.print(
        "该命令需先完成阶段4（社区索引）后接入。请先运行 nanokb build 完成高级索引。"
    )


@app.command()
def status() -> None:
    """显示知识库编译状态（raw/ 文档数 + out/ 是否已编译）。"""
    settings = _load_settings()
    raw_dir = settings.raw_dir
    out_dir = settings.out_dir

    doc_count = _count_documents(raw_dir)
    graph_path = out_dir / "graph.json"
    compiled = graph_path.exists()

    if doc_count == 0 and not compiled:
        console.print(f"[yellow]raw/ 下 {doc_count} 个文档，out/ 未编译[/yellow]")
        raise typer.Exit(code=0)

    state = "已编译" if compiled else "未编译"
    console.print(f"raw/ 下 {doc_count} 个文档 | out/ {state}")


@app.command()
def review(
    clear: bool = typer.Option(False, "--clear", help="清空 review 待审队列。"),
) -> None:
    """列出 / 清空主动学习待审队列（out/review_queue.md）。"""
    console.print(
        f"[yellow]review 命令需先完成阶段 5（主动学习闭环）后接入。clear={clear}[/yellow]"
    )


__all__ = ["app"]

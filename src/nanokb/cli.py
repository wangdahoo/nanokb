"""Nano KB CLI —— typer 命令骨架。

六个子命令（方案 §3.1）：build / query / ask / search / status / review。
Phase 0 仅 status 完整可用，其余命令打桩（后续 feature 接入真实流水线）。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import typer
from rich.console import Console

from nanokb.config import Settings
from nanokb.logging_setup import setup_logging

app = typer.Typer(
    name="nanokb",
    help="Nano KB — 基于 LLM-as-Wiki 理念的极简个人知识库工具。",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

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


@app.command()
def build(
    watch: bool = typer.Option(False, "--watch", help="监听 raw/ 变更，自动增量编译。"),
    force: bool = typer.Option(False, "--force", help="强制全量重编译。"),
    replay: bool = typer.Option(False, "--replay", help="从 out/triples.jsonl 重放重建图谱。"),
) -> None:
    """编译知识库（增量检测 → 双轨抽取 → 图谱融合 → 索引）。"""
    settings = _load_settings()
    setup_logging(settings.out_dir)
    console.print(
        "[yellow]build 命令尚未接入流水线（Phase 0 骨架）。"
        f"选项：watch={watch}, force={force}, replay={replay}[/yellow]"
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="自然语言问题。"),
) -> None:
    """图谱推理问答（三路召回融合；阶段 3 仅 graph 路）。"""
    console.print(
        f"[yellow]query 命令尚未接入问答流水线（Phase 0 骨架）。问题：{question}[/yellow]"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="自然语言问题。"),
) -> None:
    """向量路语义问答（阶段 3 打桩，阶段 4 补全）。"""
    console.print(
        f"[yellow]ask 命令需先完成阶段 4（向量索引）后接入。问题：{question}[/yellow]"
    )


@app.command()
def search(
    keyword: str = typer.Argument(..., help="检索关键词。"),
    community: bool = typer.Option(False, "--community", help="社区宏观检索。"),
) -> None:
    """社区路宏观检索（阶段 3 打桩，阶段 4 补全）。"""
    console.print(
        f"[yellow]search 命令需先完成阶段 4（社区索引）后接入。"
        f"关键词：{keyword}, community={community}[/yellow]"
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

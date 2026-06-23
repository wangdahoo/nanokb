"""冷启动集成测试（方案 §3.5.3 + Opt #8，Feature s1-feat-009 AC #5）。

覆盖：
- AC #5：未 build 即 query → 抛 ``ColdStartError``，CLI 提示并 exit 1。
- 边界：raw/ 为空但 graph.json 存在 → 仍冷启动。
- 边界：raw/ 有内容、graph.json 存在 → 不冷启动（正常查询流程）。
- ask/search 打桩不依赖 build（不受冷启动影响）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nanokb import pipeline
from nanokb.cli import app
from nanokb.config import Settings

runner = CliRunner()


# ── AC #5：未 build 即 query → ColdStartError + exit 1 ────────────────


def test_query_before_build_raises_cold_start(tmp_path: Path) -> None:
    """raw/ 有内容但未 build（无 graph.json）→ ColdStartError。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    out_dir.mkdir()
    (raw_dir / "doc.md").write_text("content", encoding="utf-8")

    settings = Settings(raw_dir=raw_dir, out_dir=out_dir)

    with pytest.raises(pipeline.ColdStartError) as exc_info:
        pipeline.answer_query(settings, "any question", llm=_StubLLM())

    assert "知识库未编译" in str(exc_info.value)
    assert "nanokb build" in str(exc_info.value)


def test_query_before_build_cli_exits_1(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    out_dir.mkdir()
    (raw_dir / "doc.md").write_text("content", encoding="utf-8")

    # CliRunner 不会自动 cd；通过 env 注入路径
    result = runner.invoke(
        app,
        ["query", "any question"],
        env={
            "NANOKB_RAW_DIR": str(raw_dir),
            "NANOKB_OUT_DIR": str(out_dir),
        },
    )
    assert result.exit_code == 1
    assert "知识库未编译" in result.stdout
    assert "nanokb build" in result.stdout


# ── 边界：raw/ 为空但 graph.json 存在 → 仍冷启动 ─────────────────────


def test_empty_raw_with_graph_json_still_cold_start(tmp_path: Path) -> None:
    """raw/ 为空（用户清空后未重新 build）→ 即便 graph.json 存在也算冷启动。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    out_dir.mkdir()
    # raw/ 为空，但 graph.json 存在（陈旧）
    (out_dir / "graph.json").write_text("{}", encoding="utf-8")

    settings = Settings(raw_dir=raw_dir, out_dir=out_dir)
    with pytest.raises(pipeline.ColdStartError):
        pipeline.answer_query(settings, "q", llm=_StubLLM())


# ── 边界：raw 有内容 + graph.json 不存在 → 冷启动 ────────────────────


def test_raw_present_no_graph_json_cold_start(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    out_dir.mkdir()
    (raw_dir / "doc.md").write_text("# doc", encoding="utf-8")
    # 无 graph.json

    settings = Settings(raw_dir=raw_dir, out_dir=out_dir)
    with pytest.raises(pipeline.ColdStartError):
        pipeline.answer_query(settings, "q", llm=_StubLLM())


# ── 边界：raw 不存在 → 冷启动 ────────────────────────────────────────


def test_missing_raw_dir_cold_start(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "graph.json").write_text("{}", encoding="utf-8")
    # raw/ 完全不存在

    settings = Settings(raw_dir=tmp_path / "does-not-exist", out_dir=out_dir)
    with pytest.raises(pipeline.ColdStartError):
        pipeline.answer_query(settings, "q", llm=_StubLLM())


# ── ask/search 真实接入后同样走冷启动校验（s1-feat-012 移除打桩） ──────


def test_ask_without_build_triggers_cold_start(tmp_path: Path) -> None:
    """s1-feat-012：ask 移除打桩接入真实 retriever，未 build → ColdStartError exit 1。

    旧打桩（s1-feat-009）下 ask/search 不需要 build；三路融合接入后 ask 走向量路，
    受 answer_query 冷启动校验保护，与 query 一致。
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = runner.invoke(
        app,
        ["ask", "any question"],
        env={
            "NANOKB_RAW_DIR": str(tmp_path / "no-raw"),
            "NANOKB_OUT_DIR": str(out_dir),
        },
    )
    assert result.exit_code == 1
    assert "未编译" in result.stdout or "build" in result.stdout


def test_search_without_build_triggers_cold_start(tmp_path: Path) -> None:
    """s1-feat-012：search --community 接入真实社区检索，未 build → ColdStartError exit 1。"""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = runner.invoke(
        app,
        ["search", "kw", "--community"],
        env={
            "NANOKB_RAW_DIR": str(tmp_path / "no-raw"),
            "NANOKB_OUT_DIR": str(out_dir),
        },
    )
    assert result.exit_code == 1
    assert "未编译" in result.stdout or "build" in result.stdout


# ── 辅助 ─────────────────────────────────────────────────────────────


class _StubLLM:
    """冷启动校验在 LLM 调用之前，故 stub 完整 LLMClient 即可。"""

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        return json.dumps({"entities": []})

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

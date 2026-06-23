"""CLI ``ask``/``search`` 打桩测试（方案 Opt #5 v3 降级，Feature s1-feat-009 AC #6）。

阶段 3（s1-feat-009）：``ask`` / ``search`` 为打桩，返回友好提示而非空响应或崩溃。
阶段 4（s1-feat-012）将移除打桩接入真实 retriever。
"""

from __future__ import annotations

from typer.testing import CliRunner

from nanokb.cli import app

runner = CliRunner()


def test_ask_returns_stub_message_exit_0() -> None:
    result = runner.invoke(app, ["ask", "any question"])
    assert result.exit_code == 0
    # AC #6：包含打桩提示（非空、非崩溃）
    assert "该命令需先完成阶段4" in result.stdout


def test_search_returns_stub_message_exit_0() -> None:
    result = runner.invoke(app, ["search", "keyword"])
    assert result.exit_code == 0
    assert "该命令需先完成阶段4" in result.stdout


def test_search_with_community_flag_still_stubbed() -> None:
    # --community 在阶段 4 s1-feat-012 才真正生效，本阶段仍打桩
    result = runner.invoke(app, ["search", "kw", "--community"])
    assert result.exit_code == 0
    assert "该命令需先完成阶段4" in result.stdout


def test_ask_stub_does_not_require_build() -> None:
    """ask/search 打桩不依赖已 build 的图谱（冷启动校验仅作用于 query）。"""
    # 在 tmp_path 隔离环境下运行（conftest _isolate_env 自动 cd）
    result = runner.invoke(app, ["ask", "anything"])
    assert result.exit_code == 0
    assert "该命令需先完成阶段4" in result.stdout

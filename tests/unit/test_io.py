"""atomic_write_* 与 staging_swap 单测（方案 §3.5.1 step 11 + §3.6 + AC #2/#3）。

验证：
- 原子写：写入中途异常时目标保持旧内容，无半写。
- staging_swap：五件套从 .staging/ 原子替换到 out/，manifest 最后写。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nanokb.utils.io import (
    STAGING_FILES,
    atomic_write_json,
    atomic_write_text,
    staging_swap,
)


def test_atomic_write_text_creates_file_with_parent(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "a.txt"
    atomic_write_text(target, "hello\n世界")
    assert target.read_text(encoding="utf-8") == "hello\n世界"


def test_atomic_write_text_overwrites_existing(tmp_path: Path) -> None:
    """覆盖既有文件后内容完整更新。"""
    target = tmp_path / "a.txt"
    target.write_text("OLD", encoding="utf-8")
    atomic_write_text(target, "NEW CONTENT")
    assert target.read_text(encoding="utf-8") == "NEW CONTENT"


def test_atomic_write_text_no_half_written_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模拟写入中途异常：目标文件保持旧内容，无半写、无残留临时文件。"""
    target = tmp_path / "a.txt"
    target.write_text("OLD", encoding="utf-8")

    def boom(src: str, dst: str) -> None:
        raise OSError("simulated mid-write crash")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(OSError):
        atomic_write_text(target, "NEW")

    # 目标保持旧内容（未被半写覆盖）
    assert target.read_text(encoding="utf-8") == "OLD"
    # 不残留临时文件
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".a.txt.")]
    assert leftovers == []


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    payload = {"b": 2, "a": [1, 2, 3], "中文": "ok"}
    atomic_write_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_atomic_write_json_replaces_existing(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_text('{"old": true}', encoding="utf-8")
    atomic_write_json(target, {"new": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": 1}


def test_staging_files_default_is_five_suite_manifest_last() -> None:
    """默认五件套顺序固定，manifest.json 位于末尾（提交点）。"""
    assert STAGING_FILES == (
        "graph.json",
        "graph.graphml",
        "communities.json",
        "keywords.json",
        "manifest.json",
    )
    assert STAGING_FILES[-1] == "manifest.json"


def test_staging_swap_moves_all_five_files(tmp_path: Path) -> None:
    staging = tmp_path / ".staging"
    out = tmp_path / "out"
    staging.mkdir()
    out.mkdir()
    for name in STAGING_FILES:
        (staging / name).write_text(f"<{name}>", encoding="utf-8")

    staging_swap(staging, out)

    for name in STAGING_FILES:
        assert (out / name).read_text(encoding="utf-8") == f"<{name}>"
        assert not (staging / name).exists(), f"{name} 应已从 staging 移走"


def test_staging_swap_manifest_written_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """manifest.json 必须最后 os.replace（提交点）。"""
    staging = tmp_path / ".staging"
    out = tmp_path / "out"
    staging.mkdir()
    out.mkdir()
    for name in STAGING_FILES:
        (staging / name).write_text("x", encoding="utf-8")

    replaced_order: list[str] = []
    original_replace = os.replace

    def spy(src: str, dst: str) -> None:
        replaced_order.append(Path(dst).name)
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)

    staging_swap(staging, out)

    assert replaced_order == list(STAGING_FILES)
    assert replaced_order[-1] == "manifest.json"


def test_staging_swap_missing_staging_file_skipped(tmp_path: Path) -> None:
    """staging 中缺失的文件跳过，不阻断其他文件切换。"""
    staging = tmp_path / ".staging"
    out = tmp_path / "out"
    staging.mkdir()
    out.mkdir()
    # 仅准备 manifest.json
    (staging / "manifest.json").write_text("M", encoding="utf-8")

    staging_swap(staging, out, files=STAGING_FILES)

    assert (out / "manifest.json").read_text(encoding="utf-8") == "M"
    for missing in STAGING_FILES[:-1]:
        assert not (out / missing).exists()

"""pytest 共享 fixtures。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from nanokb.config import Settings

if TYPE_CHECKING:
    pass


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """每个测试自动 cd 到临时目录，清除 NANOKB_ 环境变量，并禁用 .env 文件加载。

    避免本地 .env / 环境变量污染测试断言。``Settings.model_config["env_file"]``
    默认锚定到项目根的 ``.env``（与 CWD 无关），必须在测试期间置 None 才能隔离。
    """
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("NANOKB_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    return tmp_path


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """临时 raw/ 目录。"""
    path = tmp_path / "raw"
    path.mkdir()
    return path


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """临时 out/ 目录。"""
    path = tmp_path / "out"
    path.mkdir()
    return path

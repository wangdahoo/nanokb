"""原子写入与 staging 原子切换（方案 §3.5.1 step 11 + §3.6）。

提供：
- ``atomic_write_text`` / ``atomic_write_json``：写入同目录临时文件后 ``os.replace`` 原子切换，
  避免半写文件污染既有状态（如 manifest/graph.json 解析失败）。
- ``staging_swap``：将 staging 目录下五件套
  （graph.json/graph.graphml/communities.json/keywords.json/manifest.json）
  以 ``os.replace`` 原子切换到 out 目录；manifest 最后写（编译成功的"提交点"）。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# staging 原子切换覆盖的五件套（v4 Opt #1：communities.json/keywords.json 纳入）。
# manifest.json 必须最后写——它是"近似事务"的提交点（方案 §3.5.1 step 11）。
STAGING_FILES: tuple[str, ...] = (
    "graph.json",
    "graph.graphml",
    "communities.json",
    "keywords.json",
    "manifest.json",
)


def atomic_write_text(path: str | Path, data: str, *, encoding: str = "utf-8") -> None:
    """原子写入文本：同目录临时文件 + ``os.replace``。

    临时文件位于目标文件同目录，保证 ``os.replace`` 在同一文件系统（rename 原子）。
    若目标文件存在则被原子替换；写入过程中异常时临时文件被清理，目标文件保持旧内容，
    不会出现半写状态。

    Feature s3-feat-004：``os.replace`` 在 Windows 上若另一进程/线程正打开目标文件
    （典型场景：build 写 ``.build_progress.json`` 时 status 读）会抛 ``PermissionError``
    共享冲突。这是瞬时的——读侧很快释放句柄。此处对 ``PermissionError`` 重试若干次，
    使跨进程 build-write / status-read 在 Windows 上不互相打断（对其它平台无副作用，
    它们极少抛该错误）。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp_name, target)
    except BaseException:
        # 任何异常（含 KeyboardInterrupt）都清理临时文件，保持目标文件旧内容
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _replace_with_retry(src: str, target: Path, *, retries: int = 10, delay: float = 0.005) -> None:
    """``os.replace`` + Windows 共享冲突重试。

    仅 ``PermissionError``（WinError 5 / 32 等）视为瞬时共享冲突重试；其它异常立即上抛。
    总等待上限 ``retries * delay``（默认 50ms），覆盖典型读侧持句柄窗口。
    """
    last_exc: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            os.replace(src, target)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def atomic_write_json(
    path: str | Path,
    obj: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """原子写入 JSON：序列化后委托给 ``atomic_write_text``。

    避免半写 JSON 导致下次解析失败。``default=str`` 让 Path/datetime 等可序列化。
    """
    text = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, default=str)
    atomic_write_text(path, text)


def staging_swap(
    staging_dir: str | Path,
    out_dir: str | Path,
    files: Iterable[str] = STAGING_FILES,
) -> None:
    """将 staging 目录下的产物原子切换到 out 目录。

    默认覆盖五件套（``STAGING_FILES``）；按 ``files`` 给定顺序对每个文件执行
    ``os.replace(staging/<f>, out/<f>)``。manifest.json 位于默认序列末尾，
    作为"近似事务"的提交点（方案 §3.5.1 step 11 + §3.6）。

    staging 中缺失的文件跳过，不阻断其他文件的切换。
    """
    staging = Path(staging_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = staging / name
        if not src.exists():
            continue
        os.replace(src, out / name)


__all__ = ["STAGING_FILES", "atomic_write_json", "atomic_write_text", "staging_swap"]

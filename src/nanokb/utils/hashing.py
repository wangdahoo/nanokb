"""文件哈希工具（SHA256）。

来源：技术实施方案 §4 阶段 1。为增量检测（manifest sha256 比对）与
Document.sha256 计算提供稳定的文件指纹。
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 16) -> str:
    """计算文件的 SHA256 摘要。

    以二进制流式分块读取，避免大文件一次性占用内存。
    返回稳定的 64 位小写 hex 字符串（同内容同结果）。
    """
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ["sha256_file"]

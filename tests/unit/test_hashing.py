"""sha256_file 单测（方案 §4 阶段 1 + AC #1）。

验证：稳定 64 位 hex、同内容同结果、与 stdlib hashlib 一致、流式大文件正确。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from nanokb.utils.hashing import sha256_file


def test_sha256_file_returns_stable_64_hex(tmp_path: Path) -> None:
    """同内容文件两次计算结果一致，且为 64 位小写 hex。"""
    f = tmp_path / "a.txt"
    f.write_text("hello nanokb", encoding="utf-8")
    d1 = sha256_file(f)
    d2 = sha256_file(f)
    assert d1 == d2
    assert len(d1) == 64
    # 全部为 hex 字符（int(_, 16) 不抛错即证明）
    int(d1, 16)


def test_sha256_file_matches_stdlib_hashlib(tmp_path: Path) -> None:
    """与 stdlib hashlib.sha256 在二进制内容上完全一致。"""
    payload = "中英文混合 content\n第二行".encode()
    f = tmp_path / "b.bin"
    f.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_file(f) == expected


def test_sha256_file_different_content_differs(tmp_path: Path) -> None:
    """不同内容产出不同摘要。"""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("content one", encoding="utf-8")
    b.write_text("content two", encoding="utf-8")
    assert sha256_file(a) != sha256_file(b)


def test_sha256_file_accepts_str_and_path(tmp_path: Path) -> None:
    """接受 str 与 Path 入参。"""
    f = tmp_path / "c.txt"
    f.write_text("x", encoding="utf-8")
    assert sha256_file(f) == sha256_file(str(f))


def test_sha256_file_large_file_streaming(tmp_path: Path) -> None:
    """大文件流式读取，不一次性载入内存，结果与一次性哈希一致。"""
    f = tmp_path / "big.bin"
    payload = b"0123456789abcdef" * (1024 * 64)  # 1 MB
    f.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_file(f) == expected

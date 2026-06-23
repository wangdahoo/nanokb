"""chunk_text 分块单测（方案 §3.5.1 step 2b + Medium #1 + AC #1）。

覆盖：
- AC #1：超过 chunk_max_tokens 的长文本 → 切出 ≥2 块且每块 token_count ≤ max_tokens
- 短文本单块保留原文
- overlap 语义（相邻块共享尾部/头部）
- 空文本 / 非法参数边界
- tiktoken 精确计数（非 chars/4 粗估）
"""

from __future__ import annotations

import pytest

from nanokb.extract import chunk_text
from nanokb.models import Chunk
from nanokb.utils.tokenize import count_tokens

# 使用 cl100k_base 兜底的默认 model（离线可用）
MODEL = "glm-5.1"


# --------------------------------------------------------------------------- #
# 辅助：生成已知 token 数的长文本
# --------------------------------------------------------------------------- #


def _make_long_text(target_tokens: int) -> str:
    """生成 token 数 >= target_tokens 的英文重复文本。

    使用 "The quick brown fox jumps over the lazy dog. " 单元（约 13 token），
    重复足够次数覆盖 target_tokens。
    """
    unit = "The quick brown fox jumps over the lazy dog. "
    text = ""
    while count_tokens(text, MODEL) < target_tokens:
        text += unit
    assert count_tokens(text, MODEL) >= target_tokens
    return text


# --------------------------------------------------------------------------- #
# 短文本：单块
# --------------------------------------------------------------------------- #


def test_short_text_returns_single_chunk_preserving_original() -> None:
    """短文本（<= max_tokens）返回单块，内容为原文（不经 decode 重建）。"""
    content = "Hello world. Transformer 使用 Self-Attention。"
    chunks = chunk_text(content, max_tokens=100, overlap_tokens=0, source_file="a.md")

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == content
    assert chunks[0].source_file == "a.md"
    assert chunks[0].token_count == count_tokens(content, MODEL)


def test_single_chunk_token_count_matches_tiktoken() -> None:
    """单块 token_count 与 tiktoken 直接计数一致。"""
    content = "# 标题\n\n段落内容 with English mix.\n"
    chunks = chunk_text(content, max_tokens=500, model=MODEL)

    assert len(chunks) == 1
    assert chunks[0].token_count == count_tokens(content, MODEL)


# --------------------------------------------------------------------------- #
# AC #1：长文本切出 ≥2 块且每块 ≤ max_tokens
# --------------------------------------------------------------------------- #


def test_long_text_produces_multiple_chunks_within_limit() -> None:
    """AC #1：超过 max_tokens 的长文本切出 ≥2 块，每块 token_count ≤ max_tokens。"""
    max_tokens = 50
    content = _make_long_text(target_tokens=200)
    assert count_tokens(content, MODEL) > max_tokens

    chunks = chunk_text(content, max_tokens=max_tokens, overlap_tokens=0)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.token_count <= max_tokens, (
            f"chunk {chunk.index} token_count={chunk.token_count} > max={max_tokens}"
        )


def test_long_text_chunk_indices_are_sequential() -> None:
    """多块的 index 从 0 起递增无跳号。"""
    content = _make_long_text(target_tokens=300)
    chunks = chunk_text(content, max_tokens=30, overlap_tokens=0)

    assert len(chunks) >= 3
    for i, chunk in enumerate(chunks):
        assert chunk.index == i


def test_long_text_chunk_token_count_verified_by_recount() -> None:
    """每块 token_count 与对 chunk.text 重新 tiktoken 计数一致（精确计数，非粗估）。"""
    content = _make_long_text(target_tokens=150)
    chunks = chunk_text(content, max_tokens=40, overlap_tokens=5, model=MODEL)

    assert len(chunks) >= 2
    for chunk in chunks:
        recounted = count_tokens(chunk.text, MODEL)
        assert chunk.token_count == recounted, (
            f"chunk {chunk.index}: stored={chunk.token_count} recounted={recounted}"
        )


# --------------------------------------------------------------------------- #
# overlap 语义
# --------------------------------------------------------------------------- #


def test_overlap_makes_chunks_share_boundary_tokens() -> None:
    """overlap > 0 时，相邻块的 token 序列有 overlap 个 token 的重叠。"""
    content = _make_long_text(target_tokens=200)
    max_tokens = 60
    overlap = 10
    chunks = chunk_text(content, max_tokens=max_tokens, overlap_tokens=overlap)

    assert len(chunks) >= 2
    # stride = max - overlap
    stride = max_tokens - overlap
    # chunk[i].token_count 与 chunk[i+1] 的 token 应有 overlap 个共享
    # 验证方式：chunk[i] 的最后 overlap 个 token 与 chunk[i+1] 的前 overlap 个 token
    # 在原文中对应同一段。这里验证 token_count 总和 > 单纯无重叠切分（即确实有重叠）
    total_with_overlap = sum(c.token_count for c in chunks)
    # 无重叠时的总 token 数 = 原文 token 数
    original_tokens = count_tokens(content, MODEL)
    # 有重叠时总 token 数 > 原文（因为重叠部分被计了两次）
    assert total_with_overlap > original_tokens, (
        f"overlap should make sum ({total_with_overlap}) > original ({original_tokens})"
    )
    # stride 验证：块数应 == ceil((N - max) / stride) + 1
    expected_chunks = max(1, (original_tokens - max_tokens + stride) // stride + 1)
    # 末尾可能有短块，数量大致符合（允许 ±1 的末尾差异）
    assert abs(len(chunks) - expected_chunks) <= 1


def test_zero_overlap_chunks_are_disjoint() -> None:
    """overlap=0 时相邻块不共享 token（总 token 数 = 原文 token 数）。"""
    content = _make_long_text(target_tokens=150)
    chunks = chunk_text(content, max_tokens=40, overlap_tokens=0)

    assert len(chunks) >= 2
    total = sum(c.token_count for c in chunks)
    original = count_tokens(content, MODEL)
    assert total == original, (
        f"zero-overlap sum ({total}) should equal original ({original})"
    )


# --------------------------------------------------------------------------- #
# 边界与非法参数
# --------------------------------------------------------------------------- #


def test_empty_content_returns_empty_list() -> None:
    assert chunk_text("", max_tokens=100) == []


def test_invalid_max_tokens_raises() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_text("hi", max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_text("hi", max_tokens=-1)


def test_negative_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_text("hi", max_tokens=10, overlap_tokens=-1)


def test_overlap_ge_max_tokens_raises() -> None:
    """overlap >= max_tokens 会死循环，必须拒绝。"""
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_text("hi", max_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_text("hi", max_tokens=10, overlap_tokens=11)


# --------------------------------------------------------------------------- #
# Chunk 模型字段
# --------------------------------------------------------------------------- #


def test_chunks_carry_source_file() -> None:
    """每个 Chunk.source_file 被正确填充。"""
    content = _make_long_text(target_tokens=100)
    chunks = chunk_text(content, max_tokens=30, source_file="docs/paper.md")

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.source_file == "docs/paper.md"
        assert isinstance(chunk, Chunk)


def test_exact_fit_returns_single_chunk() -> None:
    """token 数恰好 == max_tokens 时返回单块。"""
    content = "word " * 20  # 约 20 token
    token_count = count_tokens(content, MODEL)
    chunks = chunk_text(content, max_tokens=token_count)
    assert len(chunks) == 1
    assert chunks[0].token_count == token_count

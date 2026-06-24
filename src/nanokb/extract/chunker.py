"""文档分块（方案 §3.5.1 step 2b + Medium #1）。

按 ``max_tokens`` 切片 + ``overlap_tokens`` 重叠，逐块 token 用 tiktoken 精确计数
（非 chars/4 粗估）。``overlap_tokens`` 让相邻块共享尾部/头部上下文，避免切断
跨边界的三元组（head/relation/tail 不被分到两个无法连接的块里）。

实现要点：
- 直接对 tiktoken token 序列切片，``token_count`` 取切片长度（精确）。
- 总 token 数 ≤ ``max_tokens`` 的短文本原样返回单块（保留原文，不经 decode 重建）。
- 长文本按 stride = max_tokens - overlap_tokens 步进，末尾不足 stride 也产出完整块。
"""

from __future__ import annotations

from nanokb.models import Chunk
from nanokb.utils.tokenize import DEFAULT_MODEL, _get_encoding

__all__ = ["chunk_text"]


def chunk_text(
    content: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    *,
    source_file: str = "",
    model: str = DEFAULT_MODEL,
) -> list[Chunk]:
    """将 ``content`` 按 ``max_tokens`` 切片，相邻块共享 ``overlap_tokens`` 个 token。

    Args:
        content: 待分块的纯文本（由 DocumentLoader 抽取）。
        max_tokens: 每块 token 上限（必须 > 0）。
        overlap_tokens: 相邻块共享的 token 数（必须 >= 0 且 < max_tokens）。
        source_file: 来源文件标识，写入每个 Chunk.source_file。
        model: tiktoken tokenizer 选择的 model 名（与 LLM model 对齐）。

    Returns:
        ``Chunk`` 列表，``index`` 从 0 起递增；空文本返回空列表。
        每块 ``token_count`` 为该块 token 序列长度（≤ ``max_tokens``）。
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if overlap_tokens < 0:
        raise ValueError(f"overlap_tokens must be non-negative, got {overlap_tokens}")
    if overlap_tokens >= max_tokens:
        raise ValueError(f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})")

    if not content:
        return []

    encoding = _get_encoding(model)
    tokens = encoding.encode(content)

    # 短文本：单块，保留原文不经 decode（避免 BPE 边界导致的微小文本差异）
    if len(tokens) <= max_tokens:
        return [Chunk(index=0, text=content, token_count=len(tokens), source_file=source_file)]

    # 长文本：按 stride 步进切片
    stride = max_tokens - overlap_tokens
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        text = encoding.decode(chunk_tokens)
        chunks.append(
            Chunk(
                index=index,
                text=text,
                token_count=len(chunk_tokens),
                source_file=source_file,
            )
        )
        if end >= len(tokens):
            break
        start += stride
        index += 1

    return chunks

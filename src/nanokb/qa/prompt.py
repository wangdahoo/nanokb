"""问答上下文编译（方案 §3.5.3 step 5，Feature s1-feat-009）。

把召回的 ``RetrievalHit`` 列表渲染为 LLM 可读的纯文本上下文，按 tiktoken 精确计数
裁剪到 ``settings.max_context_tokens``：

    已知知识点：
    [Transformer]--uses-->[Attention] (来源:doc.md)
    [Transformer]--is_a-->[Model] (来源:doc.md)
    ...

裁剪策略：逐条 hit 渲染并累加 token，达到上限时停止（至少保留 1 条，避免完全空）。
hit 为空时返回空字符串（generator 据此返回 "未找到相关知识点"）。

s1-feat-012 的 fuse 阶段复用本模块对融合后的 hits 做裁剪。
"""

from __future__ import annotations

from nanokb.config import Settings
from nanokb.llm.base import LLMClient
from nanokb.models import RetrievalHit

#: 上下文头部（提供 LLM 一个稳定的锚点识别已知事实区块）。
_CONTEXT_HEADER = "已知知识点："

#: 边 hit 渲染模板（与方案 §3.5.3 描述一致：[head]--[relation]-->[tail] (来源:file)）。
_EDGE_TEMPLATE = "[{head}]--{relation}-->[{tail}] (来源:{file})"


def compile_context(
    hits: list[RetrievalHit],
    settings: Settings,
    llm: LLMClient,
) -> str:
    """把 hits 渲染为纯文本上下文，按 tiktoken 裁剪到 ``max_context_tokens``。

    Args:
        hits: 召回结果（通常已按 score 排序）。
        settings: 提供 ``max_context_tokens``。
        llm: 提供 ``count_tokens`` 做 tiktoken 精确计数。

    Returns:
        渲染后的纯文本上下文；hits 为空或全部超长时返回 ``""``。
    """
    if not hits:
        return ""

    max_tokens = settings.max_context_tokens
    header_tokens = llm.count_tokens(_CONTEXT_HEADER)

    rendered: list[str] = []
    running = header_tokens
    for hit in hits:
        line = render_hit(hit)
        if not line:
            continue
        line_tokens = llm.count_tokens(line)
        if rendered and running + line_tokens > max_tokens:
            break
        rendered.append(line)
        running += line_tokens
        if running >= max_tokens:
            break

    if not rendered:
        return ""
    return _CONTEXT_HEADER + "\n" + "\n".join(rendered)


def render_hit(hit: RetrievalHit) -> str:
    """渲染单条 hit 为文本行；空 hit 返回空字符串。"""
    triple = hit.triple
    if triple is not None:
        return _EDGE_TEMPLATE.format(
            head=triple.head,
            relation=triple.relation,
            tail=triple.tail,
            file=triple.source_file,
        )
    concept = hit.concept
    if concept is not None:
        description = concept.description or concept.name
        return f"[{concept.name}] {description} (来源:{concept.source_file})"
    if hit.community_summary is not None:
        return f"[社区摘要] {hit.community_summary}"
    return ""


__all__ = ["compile_context", "render_hit"]

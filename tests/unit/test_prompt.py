"""``qa.prompt`` 单测（方案 §3.5.3 step 5，Feature s1-feat-009）。

覆盖：
- 空 hits → 空 context（generator 据此返回 "未找到相关知识点"）。
- 三元组 hit 渲染为 ``[head]--{relation}-->[tail] (来源:file)``。
- concept hit 渲染为 ``[name] description (来源:file)``。
- tiktoken 裁剪到 ``max_context_tokens``（FakeLLM count_tokens 粗估）。
"""

from __future__ import annotations

from nanokb.config import Settings
from nanokb.models import Concept, Confidence, RetrievalHit, Triple
from nanokb.qa.prompt import compile_context, render_hit


class FakeLLMClient:
    """模拟 LLM：count_tokens 用 chars // 4 粗估（与单测约定一致）。"""

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        raise AssertionError("compile_context 不应调用 complete")

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("compile_context 不应调用 embed")

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _triple_hit(head: str, relation: str, tail: str, file: str = "doc.md") -> RetrievalHit:
    return RetrievalHit(
        triple=Triple(
            head=head,
            relation=relation,
            tail=tail,
            confidence=Confidence.EXTRACTED,
            source_file=file,
        ),
        score=1.0,
        source="graph",
    )


def _concept_hit(name: str, desc: str, file: str = "doc.md") -> RetrievalHit:
    return RetrievalHit(
        concept=Concept(
            name=name,
            description=desc,
            source_file=file,
            confidence=Confidence.EXTRACTED,
        ),
        score=1.0,
        source="graph",
    )


# ── 空 hits ──────────────────────────────────────────────────────────


def test_empty_hits_returns_empty_context() -> None:
    settings = Settings()
    ctx = compile_context([], settings, FakeLLMClient())
    assert ctx == ""


# ── 边渲染 ───────────────────────────────────────────────────────────


def test_triple_hit_renders_edge_line() -> None:
    hit = _triple_hit("Transformer", "uses", "Attention", "doc.md")
    line = render_hit(hit)
    assert line == "[Transformer]--uses-->[Attention] (来源:doc.md)"


def test_concept_hit_rendes_description_line() -> None:
    hit = _concept_hit("Transformer", "A neural network architecture.", "paper.md")
    line = render_hit(hit)
    assert line == "[Transformer] A neural network architecture. (来源:paper.md)"


def test_empty_hit_renders_empty_string() -> None:
    hit = RetrievalHit(source="graph")
    assert render_hit(hit) == ""


# ── 多 hit 拼接 ─────────────────────────────────────────────────────


def test_multiple_hits_joined_with_header() -> None:
    hits = [
        _triple_hit("A", "r1", "B"),
        _triple_hit("C", "r2", "D"),
    ]
    settings = Settings()
    ctx = compile_context(hits, settings, FakeLLMClient())
    assert ctx.startswith("已知知识点：")
    assert "[A]--r1-->[B] (来源:doc.md)" in ctx
    assert "[C]--r2-->[D] (来源:doc.md)" in ctx


# ── tiktoken 裁剪 ───────────────────────────────────────────────────


def test_context_truncated_to_max_tokens() -> None:
    # 构造 5 条 hit，限制 max_context_tokens 仅能容纳 header + 1 条 hit
    hits = [_triple_hit("Alpha", "rel", "Beta", f"f{i}.md") for i in range(5)]
    # header "已知知识点：" = 12 chars // 4 = 3 tokens
    # 每条 hit line ~ 30+ chars // 4 ~ 8 tokens
    # 设 max=12 → header(3) + 1 hit(8) = 11 ≤ 12，第 2 hit 8 > (12-11)=1，break
    settings = Settings(max_context_tokens=12)
    ctx = compile_context(hits, settings, FakeLLMClient())
    # 至少保留 1 条 hit（不空）
    assert "[Alpha]--rel-->[Beta] (来源:f0.md)" in ctx
    # 第 2 条不应出现
    assert "f1.md" not in ctx


def test_first_hit_kept_even_if_exceeds_max() -> None:
    """首条 hit 即超 max_tokens 时仍保留（避免完全空 context）。"""
    hits = [_triple_hit("VeryLongEntityName", "very_long_relation", "AnotherLongEntity")]
    settings = Settings(max_context_tokens=1)
    ctx = compile_context(hits, settings, FakeLLMClient())
    assert "VeryLongEntityName" in ctx


def test_max_context_tokens_zero_still_keeps_first_hit() -> None:
    hits = [_triple_hit("A", "r", "B")]
    settings = Settings(max_context_tokens=0)
    ctx = compile_context(hits, settings, FakeLLMClient())
    # max=0 → 首条必入（避免空），后续裁剪
    assert "[A]--r-->[B]" in ctx

"""``qa.generator`` 单测（方案 §3.5.3 step 6，Feature s1-feat-009）。

覆盖 4 条关键行为：
- AC #1：generate 返回带 ``^[source_file]`` 引用的 ``Answer``。
- AC #2：hits 含 INFERRED/AMBIGUOUS → ``used_inferred=True`` + 答案附加推理提示。
- AC #4：空 context → 直接返回 "未找到相关知识点"（不调 LLM、不幻觉）。
- Answer.confidence 取 hits 中最严格者。
"""

from __future__ import annotations

from nanokb.config import Settings
from nanokb.models import Confidence, RetrievalHit, Triple
from nanokb.qa.generator import (
    _INFERRED_WARNING,
    _NO_RESULTS_TEXT,
    generate,
)


class FakeLLMClient:
    """模拟 LLM：complete 返回预设字符串，记录调用。"""

    def __init__(self, response: str = "") -> None:
        self._response = response
        self.complete_calls: int = 0

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        self.complete_calls += 1
        return self._response

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _triple(
    head: str,
    relation: str,
    tail: str,
    confidence: Confidence,
    file: str = "doc.md",
) -> Triple:
    return Triple(
        head=head,
        relation=relation,
        tail=tail,
        confidence=confidence,
        source_file=file,
    )


def _hit(triple: Triple, score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(triple=triple, score=score, source="graph")


# ── AC #1：带 ^[source_file] 引用 ────────────────────────────────────


def test_generate_returns_answer_with_citation() -> None:
    hits = [
        _hit(_triple("Transformer", "uses", "Attention", Confidence.EXTRACTED))
    ]
    response = "Transformer 通过 self-attention 依赖 Attention^[doc.md]。"
    llm = FakeLLMClient(response=response)
    settings = Settings()

    answer = generate("Transformer 如何依赖 Attention？", "已知知识点：...", hits, llm, settings)

    assert answer.text.startswith("Transformer")
    assert "^[doc.md]" in answer.text
    assert answer.citations == ["doc.md"]
    assert answer.used_inferred is False
    assert answer.confidence == Confidence.EXTRACTED


def test_generate_preserves_multiple_citations() -> None:
    hits = [
        _hit(_triple("A", "r1", "B", Confidence.EXTRACTED, "f1.md")),
        _hit(_triple("A", "r2", "C", Confidence.EXTRACTED, "f2.md")),
    ]
    response = "A 的行为：r1 B^[f1.md]，同时 r2 C^[f2.md]。"
    llm = FakeLLMClient(response=response)
    answer = generate("question", "context", hits, llm, Settings())

    assert "^[f1.md]" in answer.text
    assert "^[f2.md]" in answer.text
    assert answer.citations == ["f1.md", "f2.md"]


# ── AC #2：INFERRED/AMBIGUOUS 标记 used_inferred + 附加提示 ────────────


def test_generate_with_inferred_hit_marks_used_inferred_and_appends_warning() -> None:
    hits = [
        _hit(_triple("Transformer", "uses", "Attention", Confidence.EXTRACTED)),
        _hit(_triple("Transformer", "is_a", "Model", Confidence.INFERRED)),
    ]
    response = "Transformer 是一种 Model^[doc.md]。"
    llm = FakeLLMClient(response=response)

    answer = generate("question", "context", hits, llm, Settings())

    assert answer.used_inferred is True
    assert _INFERRED_WARNING in answer.text
    # 原文仍在
    assert "^[doc.md]" in answer.text


def test_generate_with_ambiguous_hit_also_marks_used_inferred() -> None:
    hits = [_hit(_triple("A", "r", "B", Confidence.AMBIGUOUS))]
    response = "A 可能 r B^[doc.md]。"
    llm = FakeLLMClient(response=response)

    answer = generate("q", "ctx", hits, llm, Settings())

    assert answer.used_inferred is True
    assert _INFERRED_WARNING in answer.text


def test_generate_all_extracted_does_not_mark_used_inferred() -> None:
    hits = [_hit(_triple("A", "r", "B", Confidence.EXTRACTED))]
    response = "A r B^[doc.md]。"
    llm = FakeLLMClient(response=response)

    answer = generate("q", "ctx", hits, llm, Settings())

    assert answer.used_inferred is False
    assert _INFERRED_WARNING not in answer.text


# ── AC #4：空 context → 未找到相关知识点（不调 LLM） ──────────────────


def test_empty_context_returns_no_results_without_calling_llm() -> None:
    llm = FakeLLMClient(response="不该被调用")
    answer = generate("question", "", [], llm, Settings())

    assert answer.text == _NO_RESULTS_TEXT
    assert llm.complete_calls == 0
    assert answer.used_inferred is False
    assert answer.citations == []


def test_whitespace_only_context_returns_no_results() -> None:
    llm = FakeLLMClient(response="不该被调用")
    answer = generate("q", "   \n  ", [], llm, Settings())
    assert answer.text == _NO_RESULTS_TEXT
    assert llm.complete_calls == 0


def test_empty_llm_response_falls_back_to_no_results() -> None:
    """LLM 返回空字符串 → 守卫为 no-results（避免空答案）。"""
    llm = FakeLLMClient(response="")
    hits = [_hit(_triple("A", "r", "B", Confidence.EXTRACTED))]
    answer = generate("q", "ctx", hits, llm, Settings())
    assert answer.text == _NO_RESULTS_TEXT


# ── Answer.confidence 选择 ───────────────────────────────────────────


def test_picks_strictest_confidence_from_hits() -> None:
    hits = [
        _hit(_triple("A", "r1", "B", Confidence.INFERRED)),
        _hit(_triple("A", "r2", "C", Confidence.AMBIGUOUS)),
        _hit(_triple("A", "r3", "D", Confidence.EXTRACTED)),
    ]
    response = "answer^[doc.md]"
    llm = FakeLLMClient(response=response)

    answer = generate("q", "ctx", hits, llm, Settings())

    # EXTRACTED 最严格
    assert answer.confidence == Confidence.EXTRACTED


def test_confidence_inferred_when_no_extracted() -> None:
    hits = [
        _hit(_triple("A", "r1", "B", Confidence.INFERRED)),
        _hit(_triple("A", "r2", "C", Confidence.AMBIGUOUS)),
    ]
    llm = FakeLLMClient(response="answer^[doc.md]")
    answer = generate("q", "ctx", hits, llm, Settings())
    # INFERRED 比 AMBIGUOUS 严格
    assert answer.confidence == Confidence.INFERRED

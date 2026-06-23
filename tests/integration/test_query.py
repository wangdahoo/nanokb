"""问答端到端集成测试（方案 §3.5.3，Feature s1-feat-009 AC #1/#2/#3/#4/#7）。

覆盖：
- AC #1：``answer_query`` 返回带 ``^[source_file]`` 引用的答案。
- AC #2：hits 含 INFERRED → ``used_inferred=True`` + 答案附加推理提示。
- AC #3：小写 ``transformer`` 经 normalize 命中 ``Transformer`` 节点（不漏召回）。
- AC #4：不相关实体 → 答案 "未找到相关知识点"（不幻觉）。
- AC #7：召回 < ``min_hit_count`` → ``review_flagged=True``。

全部用 FakeLLMClient 注入，``tmp_path`` 隔离，零真实 LLM 调用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from nanokb import pipeline
from nanokb.config import Settings
from nanokb.models import Answer, Confidence


class FakeLLMClient:
    """模拟 LLM：按调用顺序消费响应；compile/answer 共用同一签名。"""

    def __init__(self, responses: list[str] | None = None, default: str = "") -> None:
        self._responses = list(responses) if responses else []
        self._default = default
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        self.calls.append({"user": user, "response_format": response_format})
        if self._responses:
            return self._responses.pop(0)
        return self._default

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _extract_response() -> str:
    """FakeLLM 抽取响应：Transformer→Attention (EXTRACTED) + Transformer→Model (INFERRED)。"""
    return json.dumps({
        "triples": [
            {
                "head": "Transformer",
                "relation": "uses",
                "tail": "Attention",
                "confidence": "EXTRACTED",
            },
            {
                "head": "Transformer",
                "relation": "is_a",
                "tail": "Model",
                "confidence": "INFERRED",
            },
        ],
        "concepts": [
            {"name": "Transformer", "description": "A sequence model.", "node_type": "model"},
            {"name": "Attention", "description": "A focus mechanism.", "node_type": "mechanism"},
            {"name": "Model", "description": "A model category.", "node_type": "category"},
        ],
    })


def _build_compiled_kb(tmp_path: Path) -> Settings:
    """构造已编译的 KB（Transformer → Attention/Model 图谱）。

    s1-feat-012：``answer_query`` 升级为三路融合后，本夹具默认禁用向量/社区召回，
    使本文件的既有 graph 路行为断言保持稳定（三路融合行为在 test_fusion.py 覆盖）。
    """
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    settings = Settings(
        raw_dir=raw_dir,
        out_dir=out_dir,
        enable_vector_recall=False,
        enable_community_recall=False,
    )
    llm = FakeLLMClient(responses=[_extract_response()])
    pipeline.compile(settings, llm=llm)
    return settings


# ── AC #1：返回带 ^[source_file] 引用的答案 ────────────────────────────


def test_answer_query_returns_cited_answer(tmp_path: Path) -> None:
    settings = _build_compiled_kb(tmp_path)
    # 本场景 2 hits（uses + is_a），调低 min_hit_count 以避免触发 review_flag
    settings = settings.model_copy(update={"min_hit_count": 2})

    # answer_query 需要两次 LLM 调用：NER + generate
    ner = json.dumps({"entities": ["Transformer"]})
    gen = "Transformer 通过 self-attention 机制依赖 Attention^[doc.md]。"
    llm = FakeLLMClient(responses=[ner, gen])

    result = pipeline.answer_query(
        settings, "Transformer 如何依赖 Attention？", llm=llm
    )

    assert isinstance(result.answer, Answer)
    assert "^[doc.md]" in result.answer.text
    assert "doc.md" in result.answer.citations
    # 2 hits（uses + is_a）
    assert len(result.hits) == 2
    assert result.answer.review_flagged is False  # 2 hits >= min_hit_count=2


# ── AC #2：INFERRED → used_inferred + 推理提示 ─────────────────────────


def test_answer_query_marks_used_inferred_when_context_has_inferred(tmp_path: Path) -> None:
    settings = _build_compiled_kb(tmp_path)

    ner = json.dumps({"entities": ["Transformer"]})
    gen = "Transformer 是一种 Model^[doc.md]，使用 Attention^[doc.md]。"
    llm = FakeLLMClient(responses=[ner, gen])

    result = pipeline.answer_query(settings, "q", llm=llm)

    # hits 含 INFERRED (Transformer is_a Model)
    assert any(
        h.triple is not None and h.triple.confidence == Confidence.INFERRED
        for h in result.hits
    )
    assert result.answer.used_inferred is True
    assert "此结论为 AI 推理，建议核实源文件" in result.answer.text


# ── AC #3：小写实体经 normalize 命中 ──────────────────────────────────


def test_answer_query_lowercase_entity_still_hits(tmp_path: Path) -> None:
    settings = _build_compiled_kb(tmp_path)

    ner = json.dumps({"entities": ["transformer"]})  # 小写
    gen = "Transformer^[doc.md] uses Attention^[doc.md]."
    llm = FakeLLMClient(responses=[ner, gen])

    result = pipeline.answer_query(settings, "transformer 是什么？", llm=llm)

    # 即便 NER 返回小写，normalize 后仍命中 Transformer 节点
    assert len(result.hits) >= 1
    assert any(
        h.triple is not None and h.triple.head == "Transformer" for h in result.hits
    )


# ── AC #4：不相关实体 → 未找到相关知识点 ──────────────────────────────


def test_answer_query_unrelated_entity_returns_no_results(tmp_path: Path) -> None:
    settings = _build_compiled_kb(tmp_path)

    ner = json.dumps({"entities": ["QuantumComputing"]})
    llm = FakeLLMClient(responses=[ner, "should not be called"])

    result = pipeline.answer_query(settings, "QuantumComputing 是什么？", llm=llm)

    assert result.hits == []
    assert result.answer.text == "未找到相关知识点"
    assert result.answer.used_inferred is False
    assert result.answer.citations == []
    # generate 不应被调用（context 为空时短路）
    assert len(llm.calls) == 1  # 仅 NER


# ── AC #7：召回 < min_hit_count → review_flagged ─────────────────────


def test_answer_query_low_hit_count_flags_review(tmp_path: Path) -> None:
    settings = Settings(
        raw_dir=Path(),
        out_dir=Path(),
        min_hit_count=10,  # 提高 threshold 触发
    )
    # 复用已编译 KB 但调高 min_hit_count
    settings = _build_compiled_kb(tmp_path)
    settings = settings.model_copy(update={"min_hit_count": 10})

    ner = json.dumps({"entities": ["Transformer"]})
    gen = "answer^[doc.md]"
    llm = FakeLLMClient(responses=[ner, gen])

    result = pipeline.answer_query(settings, "q", llm=llm)

    # 2 hits < min_hit_count=10 → flag
    assert len(result.hits) < 10
    assert result.answer.review_flagged is True


# ── review_flagged=False 当召回充足 ──────────────────────────────────


def test_answer_query_enough_hits_does_not_flag(tmp_path: Path) -> None:
    settings = _build_compiled_kb(tmp_path)
    # min_hit_count=2 与本场景命中数匹配 → 不 flag（max_score 也高）
    settings = settings.model_copy(update={"min_hit_count": 2, "min_confidence_score": 0.3})

    ner = json.dumps({"entities": ["Transformer"]})
    gen = "answer^[doc.md]"
    llm = FakeLLMClient(responses=[ner, gen])

    result = pipeline.answer_query(settings, "q", llm=llm)
    assert result.answer.review_flagged is False


# ── graph 注入复用：跳过 build 也能查询 ───────────────────────────────


def test_answer_query_accepts_injected_graph(tmp_path: Path) -> None:
    """answer_query(graph=...) 允许跳过 graph.json 加载（便于测试与上游复用）。"""
    settings = Settings(raw_dir=tmp_path / "raw", out_dir=tmp_path / "out")
    # 制造 raw 文件以满足冷启动校验
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "doc.md").write_text("content", encoding="utf-8")
    # 制造 graph.json 以通过冷启动校验
    out = tmp_path / "out"
    out.mkdir()
    (out / "graph.json").write_text("{}", encoding="utf-8")

    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_node("Foo", description="Foo entity.", source_file="f.md", confidence="EXTRACTED")
    g.add_node("Bar", description="Bar entity.", source_file="f.md", confidence="EXTRACTED")
    g.add_edge("Foo", "Bar", relation="rel", source_file="f.md", confidence="EXTRACTED")

    ner = json.dumps({"entities": ["Foo"]})
    gen = "Foo 关联 Bar^[f.md]。"
    llm = FakeLLMClient(responses=[ner, gen])

    result = pipeline.answer_query(settings, "q", llm=llm, graph=g)
    assert "^[f.md]" in result.answer.text

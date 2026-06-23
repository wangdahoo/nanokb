"""review 主动学习闭环集成测试（方案 §阶段 5，Feature s1-feat-013 全部 5 条 AC）。

覆盖：
- AC #1：查图谱外问题（命中数 < ``min_hit_count``）→ ``out/review_queue.md`` 追加记录。
- AC #2：融合最高分 < ``min_confidence_score`` → 入队（Medium #2 OR 触发）。
- AC #3：AMBIGUOUS 冲突边查询 → 入队。
- AC #4：``nanokb review`` 列出全部待审条目。
- AC #5：``nanokb review --clear`` 清空文件。

全部用 FakeLLMClient 注入，``tmp_path`` 隔离，零真实 LLM 调用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from nanokb import pipeline
from nanokb.cli import app
from nanokb.config import Settings
from nanokb.stage5_qa.review import REVIEW_QUEUE_FILENAME, ReviewQueue

runner = CliRunner()


class FakeLLMClient:
    """模拟 LLM：按调用顺序消费响应。"""

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


def _extraction(triples: list[dict[str, str]], concepts: list[dict[str, str]]) -> str:
    """构造 SemanticTrack 抽取响应 JSON。"""
    return json.dumps({"triples": triples, "concepts": concepts})


def _build_kb(
    tmp_path: Path,
    triples: list[dict[str, str]],
    concepts: list[dict[str, str]],
) -> Settings:
    """构造已编译 KB（默认禁用向量/社区召回，聚焦 graph 路行为）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# doc\n\ncontent.", encoding="utf-8")

    settings = Settings(
        raw_dir=raw_dir,
        out_dir=out_dir,
        enable_vector_recall=False,
        enable_community_recall=False,
    )
    llm = FakeLLMClient(responses=[_extraction(triples, concepts)])
    pipeline.compile(settings, llm=llm)
    return settings


def _review_path(settings: Settings) -> Path:
    return settings.out_dir / REVIEW_QUEUE_FILENAME


# ── AC #1：命中数 < min_hit_count → 入队 ──────────────────────────────


def test_low_hit_count_appends_review_entry(tmp_path: Path) -> None:
    triples = [
        {"head": "Transformer", "relation": "uses", "tail": "Attention",
         "confidence": "EXTRACTED"},
    ]
    concepts = [
        {"name": "Transformer", "description": "A model.", "node_type": "model"},
        {"name": "Attention", "description": "A mechanism.", "node_type": "mechanism"},
    ]
    settings = _build_kb(tmp_path, triples, concepts)
    # 调高阈值：1 hit < 3 → 触发 low_hit_count
    settings = settings.model_copy(update={"min_hit_count": 3})

    ner = json.dumps({"entities": ["Transformer"]})
    gen = "answer^[doc.md]"
    llm = FakeLLMClient(responses=[ner, gen])

    result = pipeline.answer_query(settings, "Transformer 是什么？", llm=llm)

    assert result.answer.review_flagged is True
    # review_queue.md 已追加一条记录
    entries = ReviewQueue(settings.out_dir).list_pending()
    assert len(entries) == 1
    assert entries[0].question == "Transformer 是什么？"
    assert entries[0].reason == "low_hit_count"


# ── AC #2：max_score < min_confidence_score → 入队（Medium #2 OR） ────


def test_low_confidence_score_appends_review_entry(tmp_path: Path) -> None:
    triples = [
        {"head": "Foo", "relation": "rel", "tail": "Bar", "confidence": "EXTRACTED"},
        {"head": "Foo", "relation": "rel2", "tail": "Baz", "confidence": "EXTRACTED"},
        {"head": "Foo", "relation": "rel3", "tail": "Qux", "confidence": "EXTRACTED"},
    ]
    concepts = [{"name": c, "description": f"{c} desc.", "node_type": "concept"}
                for c in ("Foo", "Bar", "Baz", "Qux")]
    settings = _build_kb(tmp_path, triples, concepts)
    # count 充足（3 hits）；min_confidence_score 设为超出任意 fusion 分数 → 触发
    settings = settings.model_copy(
        update={"min_hit_count": 2, "min_confidence_score": 1.5}
    )

    ner = json.dumps({"entities": ["Foo"]})
    gen = "answer^[doc.md]"
    llm = FakeLLMClient(responses=[ner, gen])

    result = pipeline.answer_query(settings, "Foo 是什么？", llm=llm)

    assert len(result.hits) >= 2  # count 通过
    assert result.answer.review_flagged is True  # max_score < 1.5 触发
    entries = ReviewQueue(settings.out_dir).list_pending()
    assert len(entries) == 1
    assert entries[0].reason == "low_confidence_score"


# ── AC #3：AMBIGUOUS 冲突边 → 入队 ───────────────────────────────────


def test_ambiguous_conflict_edge_appends_review_entry(tmp_path: Path) -> None:
    triples = [
        {"head": "Alpha", "relation": "uses", "tail": "Beta", "confidence": "EXTRACTED"},
        {"head": "Alpha", "relation": "might_use", "tail": "Gamma",
         "confidence": "AMBIGUOUS"},
    ]
    concepts = [
        {"name": "Alpha", "description": "Alpha entity.", "node_type": "concept"},
        {"name": "Beta", "description": "Beta entity.", "node_type": "concept"},
        {"name": "Gamma", "description": "Gamma entity.", "node_type": "concept"},
    ]
    settings = _build_kb(tmp_path, triples, concepts)
    # count 与 score 均放宽，确保唯一触发条件为 AMBIGUOUS
    settings = settings.model_copy(
        update={"min_hit_count": 1, "min_confidence_score": 0.0}
    )

    ner = json.dumps({"entities": ["Alpha"]})
    gen = "answer^[doc.md]"
    llm = FakeLLMClient(responses=[ner, gen])

    result = pipeline.answer_query(settings, "Alpha 用了什么？", llm=llm)

    # hits 含 AMBIGUOUS 三元组
    assert any(
        h.triple is not None and h.triple.confidence == "AMBIGUOUS"
        for h in result.hits
    )
    assert result.answer.review_flagged is True
    entries = ReviewQueue(settings.out_dir).list_pending()
    assert len(entries) == 1
    assert entries[0].reason == "ambiguous_conflict"
    assert "Alpha" in entries[0].entities


# ── AC #4：nanokb review 列出待审条目 ─────────────────────────────────


def test_review_command_lists_pending_entries(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    queue = ReviewQueue(out_dir)
    queue.append("问题一", "low_hit_count", "A, B", "ts1")
    queue.append("问题二", "ambiguous_conflict", "C", "ts2")

    result = runner.invoke(
        app, ["review"], env={"NANOKB_OUT_DIR": str(out_dir)}
    )

    assert result.exit_code == 0
    assert "2" in result.stdout
    assert "问题一" in result.stdout
    assert "问题二" in result.stdout
    assert "low_hit_count" in result.stdout


def test_review_command_empty_queue_message(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = runner.invoke(
        app, ["review"], env={"NANOKB_OUT_DIR": str(out_dir)}
    )
    assert result.exit_code == 0
    assert "空" in result.stdout


# ── AC #5：nanokb review --clear 清空 ─────────────────────────────────


def test_review_clear_empties_queue(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    queue = ReviewQueue(out_dir)
    queue.append("问题一", "low_hit_count", "A", "ts1")
    queue.append("问题二", "low_hit_count", "B", "ts2")
    assert queue.list_pending()

    result = runner.invoke(
        app, ["review", "--clear"], env={"NANOKB_OUT_DIR": str(out_dir)}
    )

    assert result.exit_code == 0
    assert _review_path(Settings(out_dir=out_dir)).read_text(encoding="utf-8") == ""
    assert ReviewQueue(out_dir).list_pending() == []


# ── 闭环：query 触发入队后，nanokb review 可见 ────────────────────────


def test_query_then_review_end_to_end(tmp_path: Path) -> None:
    """一次低召回 query 写入 review_queue.md，随后 ``nanokb review`` 能列出。"""
    triples = [
        {"head": "Foo", "relation": "rel", "tail": "Bar", "confidence": "EXTRACTED"},
    ]
    concepts = [
        {"name": "Foo", "description": "Foo entity.", "node_type": "concept"},
        {"name": "Bar", "description": "Bar entity.", "node_type": "concept"},
    ]
    settings = _build_kb(tmp_path, triples, concepts)
    settings = settings.model_copy(update={"min_hit_count": 5})

    ner = json.dumps({"entities": ["Foo"]})
    gen = "answer^[doc.md]"
    llm = FakeLLMClient(responses=[ner, gen])
    pipeline.answer_query(settings, "Foo 是什么？", llm=llm)

    result = runner.invoke(
        app, ["review"], env={"NANOKB_OUT_DIR": str(settings.out_dir)}
    )
    assert result.exit_code == 0
    assert "Foo 是什么？" in result.stdout

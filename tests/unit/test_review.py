"""``stage5_qa.review`` ``ReviewQueue`` + ``should_flag`` AMBIGUOUS 扩展单测
（方案 §3.5.3 step 7 + AC #3，Feature s1-feat-013）。

覆盖：
- AC #3：hits 含 AMBIGUOUS confidence 三元组 → ``should_flag`` 返回 True（冲突边入队）。
- AMBIGUOUS 不破坏既有 count/score OR 逻辑（无 AMBIGUOUS 且阈值满足 → False）。
- ``ReviewQueue.append`` 写入格式 ``- [ ] <q> | <reason> | <entities> | <ts>``。
- ``ReviewQueue.list_pending`` 解析（含 question 内带 ``|`` 的边界）。
- ``ReviewQueue.clear`` 清空文件。
- ``determine_reason`` / ``collect_entities`` 辅助函数。
"""

from __future__ import annotations

from pathlib import Path

from nanokb.config import Settings
from nanokb.models import Confidence, RetrievalHit, Triple
from nanokb.stage5_qa.review import (
    REVIEW_QUEUE_FILENAME,
    ReviewEntry,
    ReviewQueue,
    collect_entities,
    determine_reason,
    should_flag,
)


def _hit(
    score: float,
    confidence: Confidence = Confidence.EXTRACTED,
    head: str = "A",
    tail: str = "B",
) -> RetrievalHit:
    return RetrievalHit(
        triple=Triple(
            head=head,
            relation="r",
            tail=tail,
            confidence=confidence,
            source_file="f.md",
        ),
        score=score,
        source="graph",
    )


# ── AC #3：AMBIGUOUS 冲突边 → should_flag True ──────────────────────────


def test_ambiguous_hit_triggers_flag_even_with_enough_hits_and_score() -> None:
    """count 与 score 均达标，但含 AMBIGUOUS → 仍触发（AC #3 第三 OR 条件）。"""
    settings = Settings(min_hit_count=2, min_confidence_score=0.3)
    hits = [
        _hit(1.0, Confidence.EXTRACTED),
        _hit(0.9, Confidence.INFERRED),
        _hit(0.4, Confidence.AMBIGUOUS),
    ]
    assert should_flag(hits, settings) is True


def test_no_ambiguous_with_enough_hits_and_score_does_not_trigger() -> None:
    """EXTRACTED + INFERRED（无 AMBIGUOUS）且阈值满足 → False（扩展不破坏既有逻辑）。"""
    settings = Settings(min_hit_count=2, min_confidence_score=0.3)
    hits = [
        _hit(1.0, Confidence.EXTRACTED),
        _hit(0.9, Confidence.INFERRED),
    ]
    assert should_flag(hits, settings) is False


def test_ambiguous_alone_triggers_flag() -> None:
    settings = Settings(min_hit_count=1, min_confidence_score=0.1)
    hits = [_hit(1.0, Confidence.AMBIGUOUS)]
    assert should_flag(hits, settings) is True


# ── determine_reason ───────────────────────────────────────────────────


def test_determine_reason_low_hit_count() -> None:
    settings = Settings(min_hit_count=5, min_confidence_score=0.3)
    assert determine_reason([_hit(1.0)], settings) == "low_hit_count"


def test_determine_reason_low_confidence() -> None:
    settings = Settings(min_hit_count=1, min_confidence_score=0.9)
    assert determine_reason([_hit(0.2)], settings) == "low_confidence_score"


def test_determine_reason_ambiguous_conflict() -> None:
    settings = Settings(min_hit_count=1, min_confidence_score=0.1)
    hits = [_hit(1.0, Confidence.AMBIGUOUS)]
    assert determine_reason(hits, settings) == "ambiguous_conflict"


# ── collect_entities ───────────────────────────────────────────────────


def test_collect_entities_dedups_preserving_order() -> None:
    hits = [
        _hit(1.0, Confidence.EXTRACTED, head="Foo", tail="Bar"),
        _hit(1.0, Confidence.EXTRACTED, head="Bar", tail="Baz"),
    ]
    assert collect_entities(hits) == "Foo, Bar, Baz"


def test_collect_entities_skips_hits_without_triple() -> None:
    no_triple = RetrievalHit(score=0.5, source="community", community_summary="s")
    assert collect_entities([no_triple]) == ""


def test_collect_entities_empty() -> None:
    assert collect_entities([]) == ""


# ── ReviewQueue.append 格式（方案 §阶段 5） ───────────────────────────


def test_append_writes_expected_format(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path)
    queue.append(
        question="Transformer 如何依赖 Attention？",
        reason="low_hit_count",
        entities="Transformer, Attention",
        timestamp="2026-06-23T00:00:00+00:00",
    )
    content = queue.path.read_text(encoding="utf-8")
    assert content == (
        "- [ ] Transformer 如何依赖 Attention？ | low_hit_count | "
        "Transformer, Attention | 2026-06-23T00:00:00+00:00\n"
    )


def test_append_creates_out_dir_if_missing(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    assert not out_dir.exists()
    queue = ReviewQueue(out_dir)
    queue.append("q", "low_hit_count", "A", "ts")
    assert queue.path.exists()
    assert queue.path.parent == out_dir


def test_append_is_additive(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path)
    queue.append("q1", "low_hit_count", "A", "ts1")
    queue.append("q2", "low_confidence_score", "B", "ts2")
    lines = queue.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_append_generates_timestamp_when_none(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path)
    queue.append("q", "low_hit_count", "A")
    entries = queue.list_pending()
    assert len(entries) == 1
    # timestamp 自动生成（非空 ISO 字符串）
    assert entries[0].timestamp
    assert entries[0].reason == "low_hit_count"


# ── ReviewQueue.list_pending 解析 ──────────────────────────────────────


def test_list_pending_parses_entries(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path)
    queue.append("问题一", "low_hit_count", "X, Y", "ts1")
    queue.append("问题二", "ambiguous_conflict", "Z", "ts2")

    entries = queue.list_pending()
    assert entries == [
        ReviewEntry(question="问题一", reason="low_hit_count", entities="X, Y", timestamp="ts1"),
        ReviewEntry(question="问题二", reason="ambiguous_conflict", entities="Z", timestamp="ts2"),
    ]


def test_list_pending_empty_when_file_missing(tmp_path: Path) -> None:
    assert ReviewQueue(tmp_path).list_pending() == []


def test_list_pending_empty_when_file_empty(tmp_path: Path) -> None:
    (tmp_path / REVIEW_QUEUE_FILENAME).write_text("", encoding="utf-8")
    assert ReviewQueue(tmp_path).list_pending() == []


def test_list_pending_skips_non_pending_lines(tmp_path: Path) -> None:
    """已勾选（``- [x]``）与无前缀行被跳过。"""
    path = tmp_path / REVIEW_QUEUE_FILENAME
    path.write_text(
        "- [ ] keep | low_hit_count | A | ts1\n"
        "- [x] done | low_hit_count | B | ts2\n"
        "random junk\n",
        encoding="utf-8",
    )
    entries = ReviewQueue(tmp_path).list_pending()
    assert len(entries) == 1
    assert entries[0].question == "keep"


def test_list_pending_handles_question_with_pipe(tmp_path: Path) -> None:
    """question 含 ``|`` 时，从右固定取后三段，其余归 question。"""
    path = tmp_path / REVIEW_QUEUE_FILENAME
    path.write_text(
        "- [ ] a | b | low_hit_count | X | ts1\n",
        encoding="utf-8",
    )
    entries = ReviewQueue(tmp_path).list_pending()
    assert len(entries) == 1
    assert entries[0].question == "a | b"
    assert entries[0].reason == "low_hit_count"
    assert entries[0].entities == "X"
    assert entries[0].timestamp == "ts1"


def test_list_pending_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / REVIEW_QUEUE_FILENAME
    path.write_text(
        "- [ ] only|two|fields\n"  # 字段不足
        "- [ ] good | low_hit_count | A | ts1\n",
        encoding="utf-8",
    )
    entries = ReviewQueue(tmp_path).list_pending()
    assert len(entries) == 1
    assert entries[0].question == "good"


# ── ReviewQueue.clear ──────────────────────────────────────────────────


def test_clear_empties_file_with_entries(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path)
    queue.append("q1", "low_hit_count", "A", "ts1")
    queue.append("q2", "low_hit_count", "B", "ts2")
    assert queue.list_pending()

    queue.clear()

    assert queue.path.read_text(encoding="utf-8") == ""
    assert queue.list_pending() == []


def test_clear_when_file_missing_creates_empty(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path)
    assert not queue.path.exists()
    queue.clear()
    assert queue.path.exists()
    assert queue.path.read_text(encoding="utf-8") == ""

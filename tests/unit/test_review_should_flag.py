"""``qa.review.should_flag`` 单测（方案 §3.5.3 step 7，Medium #2，Feature s1-feat-009）。

覆盖 OR 触发逻辑：
- ``len(hits) < min_hit_count`` → True（低召回）。
- ``max(hits.score) < min_confidence_score`` → True（低置信度）。
- 两者都满足阈值 → False（答案可信，不入队）。
- 空 hits → True（命中数 0 < min_hit_count）。
"""

from __future__ import annotations

from nanokb.config import Settings
from nanokb.models import Confidence, RetrievalHit, Triple
from nanokb.qa.review import should_flag


def _hit(score: float, confidence: Confidence = Confidence.EXTRACTED) -> RetrievalHit:
    return RetrievalHit(
        triple=Triple(
            head="A",
            relation="r",
            tail="B",
            confidence=confidence,
            source_file="f.md",
        ),
        score=score,
        source="graph",
    )


# ── hit_count < min_hit_count 触发 ─────────────────────────────────


def test_few_hits_triggers_flag() -> None:
    settings = Settings(min_hit_count=3, min_confidence_score=0.3)
    hits = [_hit(1.0), _hit(1.0)]  # 2 < 3
    assert should_flag(hits, settings) is True


def test_exactly_min_hit_count_does_not_trigger_on_count() -> None:
    settings = Settings(min_hit_count=3, min_confidence_score=0.3)
    hits = [_hit(1.0), _hit(1.0), _hit(1.0)]  # 3 == 3，score 也够
    assert should_flag(hits, settings) is False


def test_empty_hits_triggers_flag() -> None:
    settings = Settings(min_hit_count=3, min_confidence_score=0.3)
    assert should_flag([], settings) is True


# ── max_score < min_confidence_score 触发 ───────────────────────────


def test_low_max_score_triggers_flag_even_with_enough_hits() -> None:
    settings = Settings(min_hit_count=2, min_confidence_score=0.5)
    hits = [_hit(0.2), _hit(0.3)]  # 2 hits OK，但 max=0.3 < 0.5
    assert should_flag(hits, settings) is True


def test_max_score_at_threshold_does_not_trigger() -> None:
    settings = Settings(min_hit_count=2, min_confidence_score=0.3)
    hits = [_hit(0.3), _hit(0.3)]  # max == 0.3，not < 0.3
    assert should_flag(hits, settings) is False


# ── 两者均满足 → False ──────────────────────────────────────────────


def test_enough_hits_and_high_score_does_not_trigger() -> None:
    settings = Settings(min_hit_count=3, min_confidence_score=0.3)
    hits = [_hit(1.0), _hit(0.8), _hit(0.6)]
    assert should_flag(hits, settings) is False


# ── OR 逻辑：count 通过但 score 失败 ────────────────────────────────


def test_or_logic_count_ok_score_fail() -> None:
    settings = Settings(min_hit_count=2, min_confidence_score=0.7)
    hits = [_hit(0.5), _hit(0.5), _hit(0.5)]  # 3 OK，max=0.5 < 0.7
    assert should_flag(hits, settings) is True


def test_or_logic_count_fail_score_ok() -> None:
    settings = Settings(min_hit_count=5, min_confidence_score=0.3)
    hits = [_hit(1.0), _hit(1.0)]  # 2 < 5，即便 score=1.0 仍触发
    assert should_flag(hits, settings) is True

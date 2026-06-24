"""Review 判定与待审队列（方案 §3.5.3 step 7 + Medium #2 + AC #3，Feature s1-feat-009 + s1-feat-013）。

``should_flag`` 实现 OR 触发逻辑——满足任一条件即入 ``review_queue.md``：

- ``len(hits) < min_hit_count``：召回命中数过少（知识库覆盖不足或问题超纲）。
- ``max(hits.score) < min_confidence_score``：最高置信度过低（推理/歧义占比过高）。
- hits 含 AMBIGUOUS confidence 三元组（冲突边，AC #3）：歧义关系反哺人工审核。

``ReviewQueue`` 以追加写方式持久化到 ``out/review_queue.md``，格式（方案 §阶段 5）：

``- [ ] <question> | <reason> | <相关实体> | <timestamp>``

不轮转、纯追加；``nanokb review`` 列出待审条目，``nanokb review --clear`` 清空队列。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nanokb.config import Settings
from nanokb.models import Confidence, RetrievalHit

logger = logging.getLogger("nanokb")

#: review_queue.md 文件名（位于 out/ 下）。
REVIEW_QUEUE_FILENAME = "review_queue.md"

#: 待审条目行前缀（markdown 未勾选 checkbox，表示待审核）。
_PENDING_PREFIX = "- [ ]"

#: 条目字段分隔符（前后各一空格，便于人类阅读）。
_FIELD_SEP = " | "


def should_flag(hits: list[RetrievalHit], settings: Settings) -> bool:
    """判定一次问答是否应入 review 队列（OR 触发，Medium #2 + AC #3 AMBIGUOUS 冲突）。

    Args:
        hits: 本次问答的召回命中列表。
        settings: 提供 ``min_hit_count`` / ``min_confidence_score`` 阈值。

    Returns:
        ``True`` 表示命中 review 条件（应入队），``False`` 表示答案可信。
    """
    if len(hits) < settings.min_hit_count:
        return True
    max_score = max((h.score for h in hits), default=0.0)
    if max_score < settings.min_confidence_score:
        return True
    # AC #3：AMBIGUOUS 冲突边入队反哺（confidence 标注主观性风险，方案 §5 风险表）。
    for hit in hits:
        triple = hit.triple
        if triple is not None and triple.confidence == Confidence.AMBIGUOUS:
            return True
    return False


def determine_reason(hits: list[RetrievalHit], settings: Settings) -> str:
    """返回触发 review 的原因标识（与 ``should_flag`` 的 OR 分支顺序一致）。

    用于写入 ``review_queue.md`` 的 ``<reason>`` 字段，便于人工分诊。
    """
    if len(hits) < settings.min_hit_count:
        return "low_hit_count"
    max_score = max((h.score for h in hits), default=0.0)
    if max_score < settings.min_confidence_score:
        return "low_confidence_score"
    return "ambiguous_conflict"


def collect_entities(hits: list[RetrievalHit]) -> str:
    """从 hits 收集相关实体（head/tail 去重保序，逗号分隔）。

    空命中或仅含社区/概念 hit（无 triple）时返回空字符串。
    """
    seen: set[str] = set()
    entities: list[str] = []
    for hit in hits:
        triple = hit.triple
        if triple is None:
            continue
        for node in (triple.head, triple.tail):
            if node and node not in seen:
                seen.add(node)
                entities.append(node)
    return ", ".join(entities)


def _now_iso() -> str:
    """当前 UTC 时间的 ISO 8601 字符串（与 pipeline._now_iso 一致，可字典序排序）。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ReviewEntry:
    """解析后的待审条目。"""

    question: str
    reason: str
    entities: str
    timestamp: str


class ReviewQueue:
    """``out/review_queue.md`` 追加写待审队列（s1-feat-013）。

    格式（方案 §阶段 5）：``- [ ] <question> | <reason> | <相关实体> | <timestamp>``
    纯追加、不轮转；``--clear`` 截断文件为空。
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.path = out_dir / REVIEW_QUEUE_FILENAME

    def append(
        self,
        question: str,
        reason: str,
        entities: str,
        timestamp: str | None = None,
    ) -> None:
        """追加一条待审记录（best-effort，IO 失败仅记日志不抛出）。"""
        ts = timestamp if timestamp is not None else _now_iso()
        line = f"{_PENDING_PREFIX} {question}{_FIELD_SEP}{reason}{_FIELD_SEP}{entities}{_FIELD_SEP}{ts}\n"
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            logger.warning(
                "failed to append review_queue entry",
                exc_info=True,
                extra={"stage": "review"},
            )

    def list_pending(self) -> list[ReviewEntry]:
        """读取并解析全部待审（未勾选 ``- [ ]``）条目。

        已勾选（``- [x]``）或格式不符的行被跳过。
        """
        if not self.path.exists():
            return []
        entries: list[ReviewEntry] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line.startswith(_PENDING_PREFIX):
                continue
            body = line[len(_PENDING_PREFIX) :].lstrip()
            parsed = _split_fields(body)
            if parsed is None:
                continue
            question, reason, entities, timestamp = parsed
            entries.append(
                ReviewEntry(
                    question=question,
                    reason=reason,
                    entities=entities,
                    timestamp=timestamp,
                )
            )
        return entries

    def clear(self) -> None:
        """清空待审队列（截断文件为空内容）。"""
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
        except OSError:
            logger.warning(
                "failed to clear review_queue",
                exc_info=True,
                extra={"stage": "review"},
            )


def _split_fields(body: str) -> tuple[str, str, str, str] | None:
    """将 ``<question> | <reason> | <entities> | <timestamp>`` 拆分为四元组。

    question 可能含 ``|``，故从右固定取 timestamp/entities/reason 三段，其余归 question。
    """
    parts = body.split(_FIELD_SEP)
    if len(parts) < 4:
        return None
    timestamp = parts[-1].strip()
    entities = parts[-2].strip()
    reason = parts[-3].strip()
    question = _FIELD_SEP.join(parts[:-3]).strip()
    return question, reason, entities, timestamp


__all__ = [
    "REVIEW_QUEUE_FILENAME",
    "ReviewEntry",
    "ReviewQueue",
    "collect_entities",
    "determine_reason",
    "should_flag",
]

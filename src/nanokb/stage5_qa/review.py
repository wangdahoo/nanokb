"""Review 判定（方案 §3.5.3 step 7 + Medium #2，Feature s1-feat-009）。

``should_flag`` 实现 OR 触发逻辑——满足任一条件即入 ``review_queue.md``：

- ``len(hits) < min_hit_count``：召回命中数过少（知识库覆盖不足或问题超纲）。
- ``max(hits.score) < min_confidence_score``：最高置信度过低（推理/歧义占比过高）。

``ReviewQueue`` 持久化（``out/review_queue.md`` 追加写）与 ``nanokb review`` 命令
在 s1-feat-013 接入。本 feature 仅实现判定逻辑供 ``answer_query`` 调用。
"""

from __future__ import annotations

from nanokb.config import Settings
from nanokb.models import RetrievalHit


def should_flag(hits: list[RetrievalHit], settings: Settings) -> bool:
    """判定一次问答是否应入 review 队列（OR 触发，Medium #2）。

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
    return False


__all__ = ["should_flag"]

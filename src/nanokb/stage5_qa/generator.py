"""答案生成器（方案 §3.5.3 step 6，Feature s1-feat-009 + s1-feat-013）。

将召回上下文与问题交给 LLM 生成可溯源 ``Answer``：

- **强制引用**：system prompt 指示 LLM 在每条 claim 后附 ``^[source_file]`` 引用。
- **INFERRED/AMBIGUOUS 标记**：上下文 hit 中含推理/歧义 confidence 时，
  ``Answer.used_inferred=True``，答案末尾附加 ``"此结论为 AI 推理，建议核实源文件"``。
- **空上下文守卫**：上下文为空时直接返回 ``"未找到相关知识点"``（不调用 LLM、不幻觉）。
- **Answer.confidence**：取 hits 中最严格者（``EXTRACTED > INFERRED > AMBIGUOUS``）。
- **review_flagged**：调用 ``review.should_flag``（方案 §阶段 5：generator 调用
  review.should_flag）判定是否入 review_queue；实际 append 由 pipeline 编排。

s1-feat-012 的三路融合后由 ``MultiRetriever`` 调用本模块，接口不变。
"""

from __future__ import annotations

import logging
import re

from nanokb.config import Settings
from nanokb.llm.base import LLMClient
from nanokb.models import Answer, Confidence, RetrievalHit
from nanokb.stage5_qa.review import should_flag

logger = logging.getLogger("nanokb")

#: 无知识点时的固定回复（避免 LLM 幻觉，AC #4）。
_NO_RESULTS_TEXT = "未找到相关知识点"

#: 推理提示附加文案（AC #2：上下文含 INFERRED/AMBIGUOUS 时附加）。
_INFERRED_WARNING = "此结论为 AI 推理，建议核实源文件"

#: 引用提取正则——``^[source_file]`` 形式（source_file 内不允许出现 ``]``）。
_CITATION_RE = re.compile(r"\^\[([^\]]+)\]")

#: confidence 严格度排序（值越小越严格；选最严格者作为 Answer.confidence）。
_CONF_RANK: dict[Confidence, int] = {
    Confidence.EXTRACTED: 0,
    Confidence.INFERRED: 1,
    Confidence.AMBIGUOUS: 2,
}

_SYSTEM_PROMPT = (
    "You are a knowledge-base answer generator. Answer the user's question using "
    "ONLY the provided 已知知识点 context. Rules:\n"
    "- Cite every claim as ^[source_file] using the source shown in each context line.\n"
    "- If context is empty or does not address the question, respond exactly: "
    "未找到相关知识点\n"
    "- Do not fabricate facts beyond the context.\n"
    "- Match the question's language (Chinese/English)."
)


def generate(
    question: str,
    context: str,
    hits: list[RetrievalHit],
    llm: LLMClient,
    settings: Settings,
) -> Answer:
    """根据问题与上下文生成 ``Answer``。"""
    if not context.strip():
        return _build_no_results_answer(hits, settings)

    used_inferred = _contains_inferred(hits)
    raw = llm.complete(
        _SYSTEM_PROMPT,
        _build_user_prompt(question, context),
        response_format="text",
        temperature=0.0,
    )
    text = (raw or "").strip()
    if not text:
        logger.warning("generate: LLM returned empty text; returning no-results answer")
        return _build_no_results_answer(hits, settings)

    citations = _CITATION_RE.findall(text)
    confidence = _pick_confidence(hits)
    answer_text = _append_warning(text) if used_inferred else text

    return Answer(
        text=answer_text,
        citations=citations,
        used_inferred=used_inferred,
        confidence=confidence,
        review_flagged=should_flag(hits, settings),
    )


def _build_user_prompt(question: str, context: str) -> str:
    return f"问题：{question}\n\n{context}"


def _build_no_results_answer(hits: list[RetrievalHit], settings: Settings) -> Answer:
    """构造 '未找到相关知识点' 答案（hits 为空时 confidence=AMBIGUOUS）。

    空上下文同样调用 ``should_flag`` 设置 ``review_flagged``（通常命中 low_hit_count）。
    """
    return Answer(
        text=_NO_RESULTS_TEXT,
        citations=[],
        used_inferred=False,
        confidence=_pick_confidence(hits) if hits else Confidence.AMBIGUOUS,
        review_flagged=should_flag(hits, settings),
    )


def _append_warning(text: str) -> str:
    """附加推理提示（独立成段，不破坏原有引用结构）。"""
    return f"{text}\n\n{_INFERRED_WARNING}"


def _contains_inferred(hits: list[RetrievalHit]) -> bool:
    """hits 中是否存在 INFERRED 或 AMBIGUOUS confidence 的三元组（AC #2 触发条件）。"""
    for hit in hits:
        triple = hit.triple
        if triple is None:
            continue
        if triple.confidence in (Confidence.INFERRED, Confidence.AMBIGUOUS):
            return True
    return False


def _pick_confidence(hits: list[RetrievalHit]) -> Confidence:
    """从 hits 中选最严格的 confidence（EXTRACTED > INFERRED > AMBIGUOUS）。

    无 triple hit 时返回 AMBIGUOUS（保守标注，触发 review）。
    """
    if not hits:
        return Confidence.AMBIGUOUS
    best = Confidence.AMBIGUOUS
    best_rank = _CONF_RANK[Confidence.AMBIGUOUS]
    for hit in hits:
        triple = hit.triple
        if triple is None:
            continue
        rank = _CONF_RANK.get(triple.confidence, _CONF_RANK[Confidence.AMBIGUOUS])
        if rank < best_rank:
            best_rank = rank
            best = triple.confidence
    return best


def _extract_citations(text: str) -> list[str]:
    """从答案文本提取 ``^[source_file]`` 引用（去重保序）。"""
    seen: set[str] = set()
    result: list[str] = []
    for match in _CITATION_RE.findall(text):
        if match not in seen:
            seen.add(match)
            result.append(match)
    return result


__all__ = ["generate"]

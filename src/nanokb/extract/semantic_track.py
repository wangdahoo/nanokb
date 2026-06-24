"""语义轨 LLM 抽取（方案 §3.4.3 + §3.6 Medium #4 + Opt #1 v3）。

``SemanticTrack`` 对 ``doc.chunks`` 逐块调用 LLM，prompt 同时要求
``{"triples":[...], "concepts":[{"name","description","node_type"}]}``，
跨块合并结果：

- **Opt #1 v3 同名 concept 冲突**：默认 ``last_write_wins``（按 ``chunk_index``
  升序处理，后到块覆盖前块描述，确定可复现）；``concat_dedup`` 则按句去重拼接。
- **Medium #4 JSON 容错**：``parse_json_loose`` 容错 → 重试 1 次
  （temperature 升至 0.2） → 仍失败则为该块生成 ``AMBIGUOUS`` 哨兵三元组并跳过
  concepts（不崩溃，不中断流水线）。
- **跨块重复三元组保留**：去重交给 ``GraphBuilder.upsert``（按
  ``(source_file, head, relation, tail)`` 主键），本层只负责忠实聚合。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from nanokb.config import Settings
from nanokb.llm.base import LLMClient, parse_json_loose
from nanokb.models import (
    Chunk,
    Concept,
    Confidence,
    Document,
    ExtractionResult,
    Track,
    Triple,
)

logger = logging.getLogger("nanokb")

# 强制 LLM 输出严格 JSON：triples + concepts 两字段。
_SYSTEM_PROMPT = (
    "You are a knowledge-graph extractor. Read the user-supplied text chunk and "
    "extract a knowledge graph as STRICT JSON.\n"
    "\n"
    "Output schema (return ONLY this object, no markdown fences, no prose):\n"
    '{"triples": [{"head": str, "relation": str, "tail": str, '
    '"confidence": "EXTRACTED"|"INFERRED"|"AMBIGUOUS"}], '
    '"concepts": [{"name": str, "description": str, "node_type": str}]}\n'
    "\n"
    "Rules:\n"
    "- head/relation/tail and concept.name must be short normalized entity names "
    "(strip articles, normalize whitespace/case consistently).\n"
    "- concept.description MUST be a non-empty sentence describing the concept; "
    "never null or empty string.\n"
    "- confidence: EXTRACTED = directly stated in text; INFERRED = reasoned from "
    "context; AMBIGUOUS = uncertain or contradictory.\n"
    "- Emit every distinct triple you can find; duplicates across chunks are fine.\n"
    "- If the chunk has no extractable knowledge, return "
    '{"triples": [], "concepts": []}.'
)

# 在句末标点之后的空白处切分，保留标点与前句相连（concat_dedup 策略使用）。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。.!！?？;；\n])\s*")


def _build_user_prompt(chunk: Chunk) -> str:
    """构造单块抽取的 user prompt（携带来源文件与块索引供 LLM 引用）。"""
    return f"source_file: {chunk.source_file}\nchunk_index: {chunk.index}\n\ntext:\n{chunk.text}"


def _split_sentences(text: str) -> list[str]:
    """按中英文句末标点切分（concat_dedup 策略使用）。"""
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _coerce_confidence(raw: Any) -> Confidence:
    """把 LLM 返回的 confidence 字段安全转换为枚举，非法值降级 EXTRACTED。"""
    candidate = str(raw).strip().upper() if raw is not None else ""
    try:
        return Confidence(candidate)
    except ValueError:
        logger.warning("unknown confidence %r, defaulting to EXTRACTED", raw)
        return Confidence.EXTRACTED


class SemanticTrack:
    """语义轨抽取器：逐块 LLM 抽取并合并 (triples, concepts)。

    构造期注入 ``llm`` 与 ``settings``（Opt #1：llm 下沉到 ``__init__``，
    使 ``Extractor`` Protocol 保持纯数据契约）。``settings.concept_description_strategy``
    控制同名 concept 冲突合并策略。

    合并语义（关键不变量）：

    - ``triples``：按 ``chunk_index`` 升序聚合，保留跨块重复（下游 GraphBuilder 幂等去重）。
    - ``concepts``：同名条目按策略合并；``last_write_wins`` 时后到块（``chunk_index``
      较大者）覆盖前块描述。
    - 解析失败块：经重试仍无法解析时，为该块生成单条 AMBIGUOUS 哨兵三元组
      ``(head=<doc_stem>, relation="extraction_failed", tail="chunk_<i>")``，
      该块不贡献 concepts（避免污染合并池）。
    """

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    def extract(self, doc: Document) -> ExtractionResult:
        """对 ``doc.chunks`` 逐块抽取并合并为 ``ExtractionResult``。

        ``doc.chunks`` 为空时返回空结果（由上游流水线负责分块填充）。
        """
        source_file = str(doc.path)
        triples: list[Triple] = []
        merged_concepts: dict[str, Concept] = {}

        # 按 chunk_index 升序处理，保证 last-write-wins 语义确定可复现。
        ordered_chunks = sorted(doc.chunks, key=lambda c: c.index)

        total_chunks = len(ordered_chunks)
        for ci, chunk in enumerate(ordered_chunks, 1):
            logger.info(
                "  chunk %d/%d of %s ...",
                ci,
                total_chunks,
                doc.path.name,
            )
            parsed = self._extract_chunk_with_retry(chunk)
            if parsed is None:
                # Medium #4：解析仍失败 → AMBIGUOUS 哨兵，不崩溃
                logger.warning(
                    "chunk %d of %s: extraction failed after retry, emitting AMBIGUOUS sentinel",
                    chunk.index,
                    source_file,
                )
                triples.append(
                    Triple(
                        head=doc.path.stem or source_file,
                        relation="extraction_failed",
                        tail=f"chunk_{chunk.index}",
                        confidence=Confidence.AMBIGUOUS,
                        source_file=source_file,
                        track=Track.SEMANTIC,
                        chunk_index=chunk.index,
                    )
                )
                continue

            for raw_triple in parsed.get("triples", []):
                triple = self._coerce_triple(raw_triple, source_file, chunk.index)
                if triple is not None:
                    triples.append(triple)

            for raw_concept in parsed.get("concepts", []):
                self._merge_concept(raw_concept, chunk.index, source_file, merged_concepts)

        return ExtractionResult(triples=triples, concepts=list(merged_concepts.values()))

    def _extract_chunk_with_retry(self, chunk: Chunk) -> dict[str, Any] | None:
        """调用 LLM 并解析；首次失败重试 1 次（temperature 升至 0.2）。

        返回符合 ``{"triples": list, "concepts": list}`` 形状的 dict；
        无法解析时返回 ``None``（上游据此降级 AMBIGUOUS 哨兵）。
        """
        user_prompt = _build_user_prompt(chunk)

        raw = self._llm.complete(
            _SYSTEM_PROMPT,
            user_prompt,
            response_format="json",
            temperature=0.0,
        )
        parsed = self._parse_chunk_response(raw)
        if parsed is not None:
            return parsed

        logger.warning(
            "chunk %d: JSON parse failed on first attempt, retrying with temperature=0.2",
            chunk.index,
        )
        raw = self._llm.complete(
            _SYSTEM_PROMPT,
            user_prompt,
            response_format="json",
            temperature=0.2,
        )
        parsed = self._parse_chunk_response(raw)
        if parsed is not None:
            return parsed

        logger.warning("chunk %d: JSON parse failed after retry", chunk.index)
        return None

    @staticmethod
    def _parse_chunk_response(raw: str) -> dict[str, Any] | None:
        """parse_json_loose 容错 + 形状校验（必须含 triples/concepts 两 list 字段）。"""
        parsed = parse_json_loose(raw)
        if not isinstance(parsed, dict):
            return None
        triples = parsed.get("triples")
        concepts = parsed.get("concepts")
        if not isinstance(triples, list) or not isinstance(concepts, list):
            return None
        return {"triples": triples, "concepts": concepts}

    @staticmethod
    def _coerce_triple(raw: Any, source_file: str, chunk_index: int) -> Triple | None:
        """把 LLM 返回的单条三元组 dict 转为 ``Triple``；关键字段缺失返回 None。"""
        if not isinstance(raw, dict):
            return None
        head = str(raw.get("head", "")).strip()
        relation = str(raw.get("relation", "")).strip()
        tail = str(raw.get("tail", "")).strip()
        if not head or not relation or not tail:
            logger.warning(
                "chunk %d: skipping malformed triple (missing head/relation/tail): %r",
                chunk_index,
                raw,
            )
            return None
        return Triple(
            head=head,
            relation=relation,
            tail=tail,
            confidence=_coerce_confidence(raw.get("confidence")),
            source_file=source_file,
            track=Track.SEMANTIC,
            chunk_index=chunk_index,
        )

    def _merge_concept(
        self,
        raw: Any,
        chunk_index: int,
        source_file: str,
        merged: dict[str, Concept],
    ) -> None:
        """合并单个 concept 到聚合池；同名条目按配置策略解决描述冲突。"""
        if not isinstance(raw, dict):
            return
        name = str(raw.get("name", "")).strip()
        if not name:
            return

        description = self._coerce_description(raw.get("description"), name)
        node_type = str(raw.get("node_type") or "concept").strip() or "concept"

        existing = merged.get(name)
        if existing is None:
            merged[name] = Concept(
                name=name,
                description=description,
                source_file=source_file,
                node_type=node_type,
                confidence=Confidence.EXTRACTED,
            )
            return

        # 同名 concept 冲突合并
        if self._settings.concept_description_strategy == "concat_dedup":
            merged[name] = Concept(
                name=name,
                description=self._concat_dedup_description(existing.description, description),
                source_file=source_file,
                node_type=existing.node_type,
                confidence=existing.confidence,
                extra={**existing.extra, "last_chunk_index": chunk_index},
            )
        else:
            # last_write_wins（默认）：chunks 已按 index 升序处理，后到者直接覆盖。
            merged[name] = Concept(
                name=name,
                description=description,
                source_file=source_file,
                node_type=node_type,
                confidence=existing.confidence,
                extra={**existing.extra, "last_chunk_index": chunk_index},
            )

    @staticmethod
    def _coerce_description(raw: Any, name: str) -> str:
        """保证 concept.description 永远非空（AC #1）；缺失时以 name 兜底。"""
        if raw is None:
            return name
        text = str(raw).strip()
        return text if text else name

    @staticmethod
    def _concat_dedup_description(existing: str | None, incoming: str) -> str:
        """concat_dedup 策略：按句去重后拼接，保留出现顺序。"""
        parts: list[str] = []
        if existing:
            parts.extend(_split_sentences(existing))
        parts.extend(_split_sentences(incoming))
        seen: set[str] = set()
        deduped: list[str] = []
        for sentence in parts:
            key = sentence.lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(sentence)
        return " ".join(deduped) if deduped else incoming


__all__ = ["SemanticTrack"]

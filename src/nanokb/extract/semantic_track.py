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

import concurrent.futures
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
        """对 ``doc.chunks`` 抽取并合并为 ``ExtractionResult``。

        支持可配置 chunk 级并发（方案 §3.3，Feature s1-feat-003）：
        ``extract_chunk_concurrency<=1`` 时串行；``>1`` 时用 ``ThreadPoolExecutor``
        并发抽取各 chunk，收集后按 ``chunk_index`` 升序回放合并，**完全保留
        last-write-wins / concat_dedup 确定性**（线程完成顺序不影响回放顺序）。

        两分支统一把单 chunk LLM 异常降级为 AMBIGUOUS 哨兵（方案 A，相对原
        "连累整文档失败"的有意变更）：``_extract_chunk_with_retry`` 只降级 JSON
        解析失败，LLM 调用本身的异常（网络错误 / APIError 等）会被两分支统一
        try/except 捕获 → ``parsed=None`` → AMBIGUOUS 哨兵。

        ``doc.chunks`` 为空时返回空结果（由上游流水线负责分块填充）。
        """
        source_file = str(doc.path)

        # 按 chunk_index 升序处理，保证 last-write-wins 语义确定可复现。
        ordered_chunks = sorted(doc.chunks, key=lambda c: c.index)
        total_chunks = len(ordered_chunks)
        concurrency = max(1, self._settings.extract_chunk_concurrency)

        # ── 阶段 1：抽取（线程安全：_extract_chunk_with_retry 无共享可变状态） ──
        # 收集 (chunk_index, parsed_or_None)；LLM 异常统一降级 None → AMBIGUOUS 哨兵。
        raw_results: list[tuple[int, dict[str, Any] | None]] = []
        if concurrency == 1:
            # 串行回退（零开销，行为与改造前确定性输出一致）
            for ci, chunk in enumerate(ordered_chunks, 1):
                logger.info("  chunk %d/%d of %s ...", ci, total_chunks, doc.path.name)
                try:
                    parsed = self._extract_chunk_with_retry(chunk)
                except Exception:
                    logger.exception(
                        "chunk %d of %s: extraction crashed, degrading to AMBIGUOUS sentinel",
                        chunk.index,
                        source_file,
                    )
                    parsed = None
                raw_results.append((chunk.index, parsed))
        else:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
            future_to_chunk = {
                pool.submit(self._extract_chunk_with_retry, chunk): chunk
                for chunk in ordered_chunks
            }
            try:
                done = 0
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    done += 1
                    logger.info(
                        "  chunk %d/%d of %s done (%d/%d completed)",
                        chunk.index,
                        total_chunks,
                        doc.path.name,
                        done,
                        total_chunks,
                    )
                    # 单 chunk 异常隔离：与串行分支一致，降级 None → AMBIGUOUS 哨兵
                    try:
                        parsed = future.result()
                    except Exception:
                        logger.exception(
                            "chunk %d of %s: concurrent extraction crashed, "
                            "degrading to AMBIGUOUS sentinel",
                            chunk.index,
                            source_file,
                        )
                        parsed = None
                    raw_results.append((chunk.index, parsed))
                pool.shutdown(wait=True)
            except KeyboardInterrupt:
                # Ctrl-C：取消排队中的 future 并立即返回，不阻塞等待在途 LLM HTTP
                # （with 语句默认 shutdown(wait=True) 会卡住主线程直到全部 worker 完成，
                # 这是 build 按 Ctrl-C 不立即退出的根因）。失败安全：抽取阶段不触碰
                # graph/triples/vector 写入，中断只丢弃在途结果（已 cache 的保留）。
                # worker 线程为非守护线程，进程级即时退出由 CLI 层 os._exit(130) 兜底。
                pool.shutdown(wait=False, cancel_futures=True)
                raise

        # ── 阶段 2：按 chunk_index 升序回放合并（确定性，主线程串行） ──
        # as_completed 的乱序到达只影响收集顺序；sort 强制升序回放，
        # 保证 _merge_concept 的 last-write-wins / concat_dedup 按序生效。
        raw_results.sort(key=lambda r: r[0])
        triples: list[Triple] = []
        merged_concepts: dict[str, Concept] = {}
        for chunk_index, parsed in raw_results:
            if parsed is None:
                # Medium #4：解析失败或 LLM 异常 → AMBIGUOUS 哨兵，不崩溃
                logger.warning(
                    "chunk %d of %s: extraction failed, emitting AMBIGUOUS sentinel",
                    chunk_index,
                    source_file,
                )
                triples.append(
                    Triple(
                        head=doc.path.stem or source_file,
                        relation="extraction_failed",
                        tail=f"chunk_{chunk_index}",
                        confidence=Confidence.AMBIGUOUS,
                        source_file=source_file,
                        track=Track.SEMANTIC,
                        chunk_index=chunk_index,
                    )
                )
                continue

            for raw_triple in parsed.get("triples", []):
                triple = self._coerce_triple(raw_triple, source_file, chunk_index)
                if triple is not None:
                    triples.append(triple)

            for raw_concept in parsed.get("concepts", []):
                self._merge_concept(raw_concept, chunk_index, source_file, merged_concepts)

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

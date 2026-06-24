"""SemanticTrack 单测（方案 §3.4.3，Feature s1-feat-006）。

覆盖 4 条验收标准：
- AC #1：含 chunks 的 Document → ExtractionResult(triples, concepts)，
  concepts 每项含非空 description。
- AC #2：chunk1/chunk2 同名 concept 不同 description → 最终为 chunk_index 较大者
  （last-write-wins）。
- AC #3：LLM 返回畸形 JSON（FakeLLMClient）→ 不崩溃，重试 1 次后仍失败则该块
  相关三元组标 AMBIGUOUS。
- AC #4：跨块重复三元组 (head,relation,tail) → 结果列表保留（去重交给 graph_builder）。

全部用 FakeLLMClient 注入预设响应，零真实 LLM 调用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanokb.config import Settings
from nanokb.extract.base import Extractor
from nanokb.extract.semantic_track import SemanticTrack
from nanokb.models import (
    Chunk,
    Confidence,
    Document,
    ExtractionResult,
    Track,
)


class FakeLLMClient:
    """模拟 LLMClient：按调用顺序消费预设响应，耗尽后回落 default。

    记录每次 complete 的入参（temperature / response_format）供断言重试行为。
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        default: str | None = None,
    ) -> None:
        self._responses = list(responses) if responses else []
        self._default = (
            default if default is not None else json.dumps({"triples": [], "concepts": []})
        )
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "response_format": response_format,
                "temperature": temperature,
            }
        )
        if self._responses:
            return self._responses.pop(0)
        return self._default

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


def _make_doc(chunks_texts: list[str], path: str = "doc.md") -> Document:
    """构造含给定 chunks 的 Document（chunk_index 从 0 升序）。"""
    path_obj = Path(path)
    chunks = [
        Chunk(
            index=i,
            text=text,
            token_count=max(1, len(text) // 4),
            source_file=path,
        )
        for i, text in enumerate(chunks_texts)
    ]
    return Document(
        path=path_obj,
        content="\n\n".join(chunks_texts),
        sha256="abc123",
        format="md",
        chunks=chunks,
    )


# ── AC #1：返回 ExtractionResult，concepts 每项含非空 description ─────────


def test_extract_returns_result_with_nonempty_concept_descriptions() -> None:
    llm_response = json.dumps(
        {
            "triples": [
                {
                    "head": "Transformer",
                    "relation": "uses",
                    "tail": "Attention",
                    "confidence": "EXTRACTED",
                }
            ],
            "concepts": [
                {
                    "name": "Transformer",
                    "description": "A neural network architecture based on attention.",
                    "node_type": "model",
                },
                {
                    "name": "Attention",
                    "description": "A mechanism that weights input tokens.",
                    "node_type": "mechanism",
                },
            ],
        }
    )
    fake = FakeLLMClient(default=llm_response)
    track = SemanticTrack(fake, Settings())
    doc = _make_doc(["Some text about Transformer and Attention."])

    result = track.extract(doc)

    assert isinstance(result, ExtractionResult)
    assert len(result.concepts) == 2
    for concept in result.concepts:
        assert concept.description is not None
        assert concept.description.strip()
    names = {c.name for c in result.concepts}
    assert names == {"Transformer", "Attention"}
    assert len(result.triples) == 1
    triple = result.triples[0]
    assert (triple.head, triple.relation, triple.tail) == (
        "Transformer",
        "uses",
        "Attention",
    )
    assert triple.confidence == Confidence.EXTRACTED
    assert triple.track == Track.SEMANTIC
    assert triple.source_file == "doc.md"
    assert triple.chunk_index == 0


def test_concept_missing_description_is_filled_with_name_fallback() -> None:
    # LLM 返回 concept 缺 description → SemanticTrack 必须填非空兜底
    llm_response = json.dumps(
        {
            "triples": [],
            "concepts": [{"name": "Foo", "node_type": "concept"}],
        }
    )
    fake = FakeLLMClient(default=llm_response)
    track = SemanticTrack(fake, Settings())

    result = track.extract(_make_doc(["text"]))

    assert len(result.concepts) == 1
    assert result.concepts[0].name == "Foo"
    assert result.concepts[0].description == "Foo"


# ── AC #2：同名 concept 跨块冲突 last-write-wins ───────────────────────


def test_concept_description_last_write_wins_uses_higher_chunk_index() -> None:
    resp_chunk0 = json.dumps(
        {
            "triples": [],
            "concepts": [
                {
                    "name": "Transformer",
                    "description": "first description from chunk0",
                    "node_type": "model",
                }
            ],
        }
    )
    resp_chunk1 = json.dumps(
        {
            "triples": [],
            "concepts": [
                {
                    "name": "Transformer",
                    "description": "second description from chunk1",
                    "node_type": "model",
                }
            ],
        }
    )
    fake = FakeLLMClient(responses=[resp_chunk0, resp_chunk1])
    track = SemanticTrack(fake, Settings())
    doc = _make_doc(["chunk zero text", "chunk one text"])

    result = track.extract(doc)

    assert len(result.concepts) == 1
    concept = result.concepts[0]
    assert concept.name == "Transformer"
    assert concept.description == "second description from chunk1"


def test_concept_description_strategy_concat_dedup_merges_sentences() -> None:
    settings = Settings(concept_description_strategy="concat_dedup")
    resp_chunk0 = json.dumps(
        {
            "triples": [],
            "concepts": [
                {
                    "name": "Transformer",
                    "description": "Encoder-decoder architecture. Uses self-attention.",
                    "node_type": "model",
                }
            ],
        }
    )
    resp_chunk1 = json.dumps(
        {
            "triples": [],
            "concepts": [
                {
                    "name": "Transformer",
                    "description": "Uses self-attention. Scales to long sequences.",
                    "node_type": "model",
                }
            ],
        }
    )
    fake = FakeLLMClient(responses=[resp_chunk0, resp_chunk1])
    track = SemanticTrack(fake, settings)

    result = track.extract(_make_doc(["c0", "c1"]))

    assert len(result.concepts) == 1
    desc = result.concepts[0].description
    assert desc is not None
    # 按句去重后三条独立句子都应保留，重复的 "Uses self-attention." 只出现一次
    assert "Encoder-decoder architecture." in desc
    assert "Uses self-attention." in desc
    assert "Scales to long sequences." in desc
    assert desc.count("Uses self-attention.") == 1


# ── AC #3：畸形 JSON → 不崩溃 + 重试 1 次 + 降级 AMBIGUOUS ──────────────


def test_malformed_json_retries_once_then_degrades_to_ambiguous() -> None:
    # FakeLLMClient 永远返回畸形 JSON（两次调用都失败）
    broken = "Sorry, I cannot comply. {{ broken json"
    fake = FakeLLMClient(default=broken)
    track = SemanticTrack(fake, Settings())
    doc = _make_doc(["some text that yields no parseable JSON"])

    result = track.extract(doc)

    # 不崩溃：返回正常 ExtractionResult
    assert isinstance(result, ExtractionResult)
    # 重试 1 次：共 2 次 complete 调用
    assert len(fake.calls) == 2
    # 第二次 temperature 升至 0.2（方案 §3.6 Medium #4）
    assert fake.calls[0]["temperature"] == 0.0
    assert fake.calls[1]["temperature"] == 0.2
    # 失败块产生 1 条 AMBIGUOUS 哨兵三元组（"相关三元组标 AMBIGUOUS"）
    assert len(result.triples) == 1
    sentinel = result.triples[0]
    assert sentinel.confidence == Confidence.AMBIGUOUS
    assert sentinel.relation == "extraction_failed"
    assert sentinel.source_file == "doc.md"
    assert sentinel.chunk_index == 0
    # 该块不贡献 concepts
    assert result.concepts == []


def test_malformed_json_then_success_on_retry_does_not_degrade() -> None:
    # 第一次畸形，重试后返回正常 JSON → 不降级，正常抽取
    broken = "not json at all"
    good = json.dumps(
        {
            "triples": [{"head": "A", "relation": "r", "tail": "B", "confidence": "EXTRACTED"}],
            "concepts": [{"name": "A", "description": "entity A", "node_type": "x"}],
        }
    )
    fake = FakeLLMClient(responses=[broken, good])
    track = SemanticTrack(fake, Settings())

    result = track.extract(_make_doc(["text"]))

    assert len(fake.calls) == 2  # 首次 + 重试
    assert len(result.triples) == 1
    assert result.triples[0].confidence == Confidence.EXTRACTED
    assert len(result.concepts) == 1


def test_malformed_json_in_one_chunk_does_not_break_other_chunks() -> None:
    # chunk0 失败、chunk1 成功：失败块降级 AMBIGUOUS，成功块正常贡献
    broken = "broken {{"
    good = json.dumps(
        {
            "triples": [{"head": "X", "relation": "rel", "tail": "Y", "confidence": "EXTRACTED"}],
            "concepts": [{"name": "X", "description": "ok", "node_type": "concept"}],
        }
    )
    # chunk0: broken → retry → broken（仍失败）；chunk1: good（首次成功）
    fake = FakeLLMClient(responses=[broken, broken, good])
    track = SemanticTrack(fake, Settings())

    result = track.extract(_make_doc(["bad chunk", "good chunk"]))

    # 两块各贡献 1 条 triple：chunk0 的 AMBIGUOUS sentinel + chunk1 的真实三元组
    assert len(result.triples) == 2
    by_chunk = {t.chunk_index: t for t in result.triples}
    assert by_chunk[0].confidence == Confidence.AMBIGUOUS
    assert by_chunk[0].relation == "extraction_failed"
    assert by_chunk[1].confidence == Confidence.EXTRACTED
    assert (by_chunk[1].head, by_chunk[1].tail) == ("X", "Y")
    assert len(result.concepts) == 1
    assert result.concepts[0].name == "X"


# ── AC #4：跨块重复三元组保留（去重交给 graph_builder） ────────────────


def test_duplicate_triples_across_chunks_are_preserved() -> None:
    triple = {
        "head": "A",
        "relation": "rel",
        "tail": "B",
        "confidence": "EXTRACTED",
    }
    resp = json.dumps({"triples": [triple], "concepts": []})
    fake = FakeLLMClient(default=resp)
    track = SemanticTrack(fake, Settings())
    doc = _make_doc(["chunk one mentions A rel B", "chunk two also mentions A rel B"])

    result = track.extract(doc)

    # 两块各产出 1 条相同 (A, rel, B) → 结果列表含 2 条（本层不去重）
    assert len(result.triples) == 2
    for t in result.triples:
        assert (t.head, t.relation, t.tail) == ("A", "rel", "B")
    # chunk_index 各自带回，便于下游溯源
    assert {t.chunk_index for t in result.triples} == {0, 1}


# ── 协议与边界 ────────────────────────────────────────────────────────


def test_semantic_track_satisfies_extractor_protocol() -> None:
    fake = FakeLLMClient()
    track = SemanticTrack(fake, Settings())
    # runtime_checkable Protocol 实例检查
    assert isinstance(track, Extractor)


def test_empty_chunks_returns_empty_result() -> None:
    fake = FakeLLMClient(default=json.dumps({"triples": [], "concepts": []}))
    track = SemanticTrack(fake, Settings())
    doc = Document(
        path=Path("empty.md"),
        content="",
        sha256="x",
        format="md",
        chunks=[],
    )

    result = track.extract(doc)

    assert result.triples == []
    assert result.concepts == []
    # 无 chunk 不应触发 LLM 调用
    assert fake.calls == []


def test_malformed_triple_entries_are_skipped_not_raising() -> None:
    # LLM 返回的 triples 数组中混入缺字段/类型错误条目 → 跳过且不崩溃
    llm_response = json.dumps(
        {
            "triples": [
                {"head": "OK", "relation": "rel", "tail": "Fine", "confidence": "EXTRACTED"},
                {"head": "", "relation": "missing_head", "tail": "x"},  # 缺 head
                {"relation": "no_head_no_tail"},  # 缺关键字段
                "not-a-dict",  # 类型错误
                {"head": "Also", "relation": "ok", "tail": "Good"},
            ],
            "concepts": [],
        }
    )
    fake = FakeLLMClient(default=llm_response)
    track = SemanticTrack(fake, Settings())

    result = track.extract(_make_doc(["text"]))

    # 仅 2 条合法三元组保留
    assert len(result.triples) == 2
    heads = [t.head for t in result.triples]
    assert "OK" in heads and "Also" in heads


def test_unknown_confidence_falls_back_to_extracted() -> None:
    llm_response = json.dumps(
        {
            "triples": [{"head": "A", "relation": "r", "tail": "B", "confidence": "WHATEVER"}],
            "concepts": [],
        }
    )
    fake = FakeLLMClient(default=llm_response)
    track = SemanticTrack(fake, Settings())

    result = track.extract(_make_doc(["text"]))

    assert len(result.triples) == 1
    assert result.triples[0].confidence == Confidence.EXTRACTED


def test_chunks_processed_in_index_order_regardless_of_input_order() -> None:
    # 故意乱序传入 chunks：SemanticTrack 必须按 index 升序处理保证 last-write-wins 确定
    path = Path("doc.md")
    chunks = [
        Chunk(index=1, text="chunk one", token_count=2, source_file="doc.md"),
        Chunk(index=0, text="chunk zero", token_count=2, source_file="doc.md"),
    ]
    doc = Document(path=path, content="...", sha256="x", format="md", chunks=chunks)
    resp0 = json.dumps(
        {
            "triples": [],
            "concepts": [{"name": "K", "description": "zero", "node_type": "c"}],
        }
    )
    resp1 = json.dumps(
        {
            "triples": [],
            "concepts": [{"name": "K", "description": "one", "node_type": "c"}],
        }
    )
    # 按 index 升序处理 → 先 resp0（chunk 0）再 resp1（chunk 1）
    fake = FakeLLMClient(responses=[resp0, resp1])
    track = SemanticTrack(fake, Settings())

    result = track.extract(doc)

    assert result.concepts[0].description == "one"  # chunk_index=1 后到覆盖

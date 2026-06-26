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
import re
import threading
import time
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
    """模拟 LLMClient：按 ``chunk_index`` 映射响应（并发安全，方案 §3.3）。

    ``complete`` 从 user prompt 解析 ``chunk_index: N``，按 chunk_index 取响应——
    无论并发下线程完成顺序如何，同一 chunk 始终拿到同一响应序列，保证输出确定。
    这使本 fake 在 ``extract_chunk_concurrency>1`` 下同样确定可复现。

    ``responses`` 支持两种形式：
    - ``dict[int, list[str]]``：``chunk_index → 该 chunk 各次 complete 尝试的响应序列``
      （按调用顺序消费；重试场景用，如 ``{0: [broken, good]}``）。
    - ``list[str]``（便捷形式）：等价于 ``{i: [resp_i]}``（position i → chunk_index i，
      每块单次响应；适用于"每块一个响应、按 index 对齐"的常见场景）。

    ``default``：未命中映射 / 序列耗尽时的兜底响应（每个 chunk 都返回 default，
    天然并发安全）。``calls`` 记录每次 complete 入参供断言重试行为（Lock 保护）。
    """

    _CHUNK_INDEX_RE = re.compile(r"^chunk_index:\s*(\d+)", re.MULTILINE)

    def __init__(
        self,
        responses: dict[int, list[str]] | list[str] | None = None,
        default: str | None = None,
    ) -> None:
        if responses is None:
            self._by_chunk: dict[int, list[str]] = {}
        elif isinstance(responses, dict):
            self._by_chunk = {k: list(v) for k, v in responses.items()}
        else:  # list[str] 便捷形式：position i → chunk_index i（每块单次响应）
            self._by_chunk = {i: [r] for i, r in enumerate(responses)}
        self._default = (
            default if default is not None else json.dumps({"triples": [], "concepts": []})
        )
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _chunk_index_from_user(self, user: str) -> int | None:
        m = self._CHUNK_INDEX_RE.search(user)
        return int(m.group(1)) if m else None

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        with self._lock:
            self.calls.append(
                {
                    "system": system,
                    "user": user,
                    "response_format": response_format,
                    "temperature": temperature,
                }
            )
            chunk_index = self._chunk_index_from_user(user)
            if chunk_index is not None and chunk_index in self._by_chunk:
                seq = self._by_chunk[chunk_index]
                if seq:
                    return seq.pop(0)
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
    fake = FakeLLMClient(responses={0: [resp_chunk0], 1: [resp_chunk1]})
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
    fake = FakeLLMClient(responses={0: [resp_chunk0], 1: [resp_chunk1]})
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
    fake = FakeLLMClient(responses={0: [broken, good]})
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
    fake = FakeLLMClient(responses={0: [broken, broken], 1: [good]})
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
    fake = FakeLLMClient(responses={0: [resp0], 1: [resp1]})
    track = SemanticTrack(fake, Settings())

    result = track.extract(doc)

    assert result.concepts[0].description == "one"  # chunk_index=1 后到覆盖


# ── chunk 级并发（方案 §3.3，Feature s1-feat-003） ──────────────────────


def _result_signature(
    result: ExtractionResult,
) -> tuple[set[tuple[str, str, str, str, int]], dict[str, str]]:
    """把 ExtractionResult 归一为可比较签名（与 chunk 完成顺序无关）。

    triples → {(head, relation, tail, confidence, chunk_index)} 集合（顺序无关）；
    concepts → {name: description}（last-write-wins 结果）。
    """
    triples_sig = {
        (
            t.head,
            t.relation,
            t.tail,
            t.confidence.value,
            t.chunk_index if t.chunk_index is not None else -1,
        )
        for t in result.triples
    }
    concepts_sig = {c.name: (c.description or "") for c in result.concepts}
    return triples_sig, concepts_sig


def test_chunk_concurrency_4_output_identical_to_serial() -> None:
    """AC #2：同一多 chunk 文档，chunk_concurrency=4 与 =1 输出逐字节一致。

    构造 4 chunk、每块不同 triples + 同名 concept 不同 description（触发
    last-write-wins）。用 chunk-aware FakeLLMClient 保证映射与完成顺序无关。
    """
    responses: dict[int, list[str]] = {}
    for i in range(4):
        responses[i] = [
            json.dumps(
                {
                    "triples": [
                        {
                            "head": f"H{i}",
                            "relation": "rel",
                            "tail": f"T{i}",
                            "confidence": "EXTRACTED",
                        }
                    ],
                    "concepts": [
                        {"name": "Shared", "description": f"desc from chunk{i}", "node_type": "c"}
                    ],
                }
            )
        ]
    doc = _make_doc([f"chunk {i}" for i in range(4)])

    track_serial = SemanticTrack(
        FakeLLMClient(responses=responses), Settings(extract_chunk_concurrency=1)
    )
    track_conc = SemanticTrack(
        FakeLLMClient(responses=responses), Settings(extract_chunk_concurrency=4)
    )

    serial_result = track_serial.extract(doc)
    conc_result = track_conc.extract(doc)

    assert _result_signature(serial_result) == _result_signature(conc_result)
    # last-write-wins：chunk3 的 description 应胜出（两种并发度下一致）
    assert serial_result.concepts[0].description == "desc from chunk3"
    assert conc_result.concepts[0].description == "desc from chunk3"


class _RaisingFakeLLMClient:
    """对指定 chunk_index 抛异常、其余返回固定响应（验证 chunk 级异常隔离）。"""

    _CHUNK_RE = re.compile(r"^chunk_index:\s*(\d+)", re.MULTILINE)

    def __init__(self, response: str, raise_on_chunk: int) -> None:
        self._response = response
        self._raise_on = raise_on_chunk

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        m = self._CHUNK_RE.search(user)
        idx = int(m.group(1)) if m else -1
        if idx == self._raise_on:
            raise RuntimeError(f"simulated LLM failure for chunk {idx}")
        return self._response

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


def test_llm_exception_degrades_to_ambiguous_both_concurrency_modes() -> None:
    """AC #3：单 chunk LLM 异常在 chunk=1 与 chunk=4 下都降级 AMBIGUOUS，两分支一致。

    方案 A：相对原"连累整文档失败"的有意变更——单 chunk 异常被隔离，该 chunk
    降级 AMBIGUOUS 哨兵，其余 chunk 正常，文档仍成功产出。
    """
    good = json.dumps(
        {
            "triples": [
                {"head": "OK", "relation": "rel", "tail": "Fine", "confidence": "EXTRACTED"}
            ],
            "concepts": [{"name": "OK", "description": "fine", "node_type": "c"}],
        }
    )
    doc = _make_doc(["good chunk 0", "bad chunk 1", "good chunk 2"])

    track_serial = SemanticTrack(
        _RaisingFakeLLMClient(response=good, raise_on_chunk=1),
        Settings(extract_chunk_concurrency=1),
    )
    track_conc = SemanticTrack(
        _RaisingFakeLLMClient(response=good, raise_on_chunk=1),
        Settings(extract_chunk_concurrency=4),
    )

    serial_result = track_serial.extract(doc)
    conc_result = track_conc.extract(doc)

    # 两分支输出一致
    assert _result_signature(serial_result) == _result_signature(conc_result)

    # chunk 1 降级 AMBIGUOUS 哨兵
    by_chunk_serial = {t.chunk_index: t for t in serial_result.triples}
    by_chunk_conc = {t.chunk_index: t for t in conc_result.triples}
    assert by_chunk_serial[1].confidence == Confidence.AMBIGUOUS
    assert by_chunk_serial[1].relation == "extraction_failed"
    assert by_chunk_conc[1].confidence == Confidence.AMBIGUOUS
    assert by_chunk_conc[1].relation == "extraction_failed"
    # chunk 0/2 正常抽取（EXTRACTED）
    assert by_chunk_serial[0].confidence == Confidence.EXTRACTED
    assert by_chunk_serial[2].confidence == Confidence.EXTRACTED
    assert by_chunk_conc[0].confidence == Confidence.EXTRACTED
    assert by_chunk_conc[2].confidence == Confidence.EXTRACTED
    # 文档仍成功产出（不连累整文档失败），concept 正常
    assert len(serial_result.concepts) == 1
    assert len(conc_result.concepts) == 1


def test_last_write_wins_order_under_concurrency() -> None:
    """AC #5：并发模式下 _merge_concept 仍按 chunk_index 升序 last-write-wins。

    即便高并发度下线程完成顺序乱，回放阶段 sort(chunk_index) 保证后到块（更大
    chunk_index）覆盖前块描述。
    """
    responses: dict[int, list[str]] = {
        i: [
            json.dumps(
                {
                    "triples": [],
                    "concepts": [{"name": "K", "description": f"chunk{i}", "node_type": "c"}],
                }
            )
        ]
        for i in range(6)
    }
    doc = _make_doc([f"c{i}" for i in range(6)])
    track = SemanticTrack(FakeLLMClient(responses=responses), Settings(extract_chunk_concurrency=6))

    result = track.extract(doc)

    # chunk_index 最大者（5）的 description 胜出
    assert len(result.concepts) == 1
    assert result.concepts[0].description == "chunk5"


class _DelayedFakeLLMClient:
    """带人为延迟和线程安全调用计数的 fake（验证并发加速，方案 §5.2）。"""

    def __init__(self, response: str, delay: float = 0.1) -> None:
        self._response = response
        self._delay = delay
        self._call_count = 0
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        return self._call_count

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        with self._lock:
            self._call_count += 1
        time.sleep(self._delay)  # 模拟网络 IO 等待
        return self._response

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


def test_concurrent_extract_faster_than_serial() -> None:
    """AC #4：8 chunk、delay=0.1s、chunk_concurrency=4 → 总耗时远小于串行。

    断言 wall_time < serial_time / concurrency * 1.5（留容差，防 CI 抖动）。
    """
    good = json.dumps({"triples": [], "concepts": []})
    doc = _make_doc([f"chunk {i}" for i in range(8)])
    delay = 0.1
    concurrency = 4

    serial_track = SemanticTrack(
        _DelayedFakeLLMClient(response=good, delay=delay),
        Settings(extract_chunk_concurrency=1),
    )
    start = time.monotonic()
    serial_track.extract(doc)
    serial_elapsed = time.monotonic() - start

    conc_track = SemanticTrack(
        _DelayedFakeLLMClient(response=good, delay=delay),
        Settings(extract_chunk_concurrency=concurrency),
    )
    start = time.monotonic()
    conc_track.extract(doc)
    conc_elapsed = time.monotonic() - start

    # 串行 ~8×0.1=0.8s；并发 4 ~2×0.1=0.2s。断言并发明显快于串行（容差 1.5×）。
    assert conc_elapsed < serial_elapsed / concurrency * 1.5, (
        f"concurrent {conc_elapsed:.3f}s not faster than serial/{concurrency}*1.5 "
        f"= {serial_elapsed / concurrency * 1.5:.3f}s (serial {serial_elapsed:.3f}s)"
    )

"""检索进度报告器单元测试（Feature：query/ask/search 检索过程状态信息）。

覆盖：
- ``NullProgressReporter`` 为空实现（stage 无副作用、进出正常）。
- ``MultiRetriever.recall`` 按路由顺序报告 stage（图谱/向量/社区召回）+ fuse stage，
  且 ``progress=None`` 时退化为空实现。
- ``answer_query`` 完整 stage 序列：加载知识库 → 召回 → 融合重排 → 构建上下文 → 生成答案。

全部用注入的图谱 / FakeLLMClient，不依赖 ChromaDB（零真实索引）。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from nanokb import pipeline
from nanokb.config import Settings
from nanokb.models import Confidence, RetrievalHit, Triple
from nanokb.qa.progress import NullProgressReporter
from nanokb.qa.retriever import MultiRetriever

# ── 测试夹具：记录型 reporter / 假 retriever / 假 LLM ──────────────────


class RecordingReporter:
    """记录所有 stage 消息（按进入顺序），满足 ``ProgressReporter`` 协议。"""

    def __init__(self) -> None:
        self.stages: list[str] = []

    @contextmanager
    def stage(self, message: str) -> Iterator[None]:
        self.stages.append(message)
        yield


class _FakeRetriever:
    """最小 retriever：固定 ``SOURCE`` + 固定 hits，用于 ``MultiRetriever`` 单测。"""

    SOURCE: str

    def __init__(self, source: str, hits: list[RetrievalHit]) -> None:
        self.SOURCE = source
        self._hits = hits

    def recall(self, question: str) -> list[RetrievalHit]:
        return list(self._hits)


class _FakeLLM:
    """``MultiRetriever.fuse`` 仅需 ``count_tokens``；``answer_query`` 需完整签名。"""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else []
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
        return ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _hit(source: str, head: str = "A", tail: str = "B") -> RetrievalHit:
    return RetrievalHit(
        triple=Triple(
            head=head,
            relation="uses",
            tail=tail,
            confidence=Confidence.EXTRACTED,
            source_file="f.md",
        ),
        score=1.0,
        source=source,
    )


# ── NullProgressReporter ──────────────────────────────────────────────


def test_null_reporter_is_noop() -> None:
    """``NullProgressReporter.stage`` 可进出且不产生副作用。"""
    reporter = NullProgressReporter()
    with reporter.stage("anything"):
        pass
    # 可重复进入，无异常即通过
    with reporter.stage("second"):
        pass


# ── MultiRetriever 各路 + fuse stage ──────────────────────────────────


def test_multi_retriever_reports_per_route_and_fuse_stages() -> None:
    """三路 retriever 按 SOURCE 顺序报告召回 stage，fuse 独立成 stage。"""
    retrievers = [
        _FakeRetriever("graph", [_hit("graph")]),
        _FakeRetriever("vector", [_hit("vector")]),
        _FakeRetriever("community", [_hit("community")]),
    ]
    settings = Settings()
    reporter = RecordingReporter()
    multi = MultiRetriever(retrievers, settings, _FakeLLM(), progress=reporter)

    hits = multi.recall("question")

    assert reporter.stages == [
        "图谱召回中...",
        "向量召回中...",
        "社区召回中...",
        "融合重排中...",
    ]
    # 三路命中均保留（不同 source，去重不合并）
    sources = {h.source for h in hits}
    assert sources == {"graph", "vector", "community"}


def test_multi_retriever_default_progress_is_noop() -> None:
    """``progress=None`` 退化为空实现，行为与无进度反馈一致。"""
    retrievers = [_FakeRetriever("graph", [_hit("graph")])]
    settings = Settings()
    multi = MultiRetriever(retrievers, settings, _FakeLLM())

    hits = multi.recall("question")

    assert len(hits) == 1
    assert hits[0].source == "graph"


def test_multi_retriever_unknown_source_falls_back_to_source_name() -> None:
    """未知 SOURCE 退化为 source 名本身（不报错）。"""
    retrievers = [_FakeRetriever("custom", [_hit("custom")])]
    settings = Settings()
    reporter = RecordingReporter()
    multi = MultiRetriever(retrievers, settings, _FakeLLM(), progress=reporter)

    multi.recall("question")

    assert reporter.stages == ["custom中...", "融合重排中..."]


# ── answer_query 完整 stage 序列 ──────────────────────────────────────


def _make_cold_start_safe_settings(tmp_path: Path) -> Settings:
    """构造通过冷启动校验的 settings（raw 有文件 + graph.json 存在）。"""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    (raw / "doc.md").write_text("# Doc\n\nContent.", encoding="utf-8")
    out.mkdir()
    (out / "graph.json").write_text("{}", encoding="utf-8")
    return Settings(
        raw_dir=raw,
        out_dir=out,
        enable_vector_recall=False,  # 禁用向量/社区路 → 仅 graph 路，避免依赖 chroma
        enable_community_recall=False,
    )


def _build_injected_graph() -> nx.MultiDiGraph:
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_edge(
        "Foo", "Bar",
        relation="uses", source_file="f.md", confidence="EXTRACTED",
    )
    return g


def test_answer_query_reports_full_stage_sequence(tmp_path: Path) -> None:
    """``query`` 模式（仅 graph 路）报告完整 stage 序列。"""
    settings = _make_cold_start_safe_settings(tmp_path)
    settings = settings.model_copy(update={"min_hit_count": 1})
    g = _build_injected_graph()

    ner = json.dumps({"entities": ["Foo"]})
    gen = "Foo uses Bar^[f.md]."
    llm = _FakeLLM(responses=[ner, gen])
    reporter = RecordingReporter()

    result = pipeline.answer_query(
        settings, "Foo 是什么？", mode="query", llm=llm, graph=g, progress=reporter
    )

    assert reporter.stages == [
        "加载知识库...",
        "图谱召回中...",
        "融合重排中...",
        "构建上下文...",
        "生成答案中...",
    ]
    assert "^[f.md]" in result.answer.text


def test_answer_query_ask_mode_reports_only_vector_stage(tmp_path: Path) -> None:
    """``ask`` 模式仅向量路：无 chroma → 空召回，stage 仍报告各阶段。"""
    settings = _make_cold_start_safe_settings(tmp_path)
    g = _build_injected_graph()

    llm = _FakeLLM()  # ask 无 NER，向量路无 chroma → 无 generate 调用（空上下文短路）
    reporter = RecordingReporter()

    result = pipeline.answer_query(
        settings, "q", mode="ask", llm=llm, graph=g, progress=reporter
    )

    # ask 模式无 chroma → retrievers 为空 → recall 仅 fuse stage（无路由 stage）
    assert reporter.stages == [
        "加载知识库...",
        "融合重排中...",
        "构建上下文...",
        "生成答案中...",
    ]
    assert result.hits == []
    assert result.answer.text == "未找到相关知识点"


def test_answer_query_default_progress_is_noop(tmp_path: Path) -> None:
    """``progress=None`` 不影响问答结果（与既有行为一致）。"""
    settings = _make_cold_start_safe_settings(tmp_path)
    settings = settings.model_copy(update={"min_hit_count": 1})
    g = _build_injected_graph()

    ner = json.dumps({"entities": ["Foo"]})
    gen = "Foo^[f.md]"
    llm = _FakeLLM(responses=[ner, gen])

    result = pipeline.answer_query(
        settings, "q", mode="query", llm=llm, graph=g
    )

    assert "^[f.md]" in result.answer.text

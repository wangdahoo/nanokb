"""modified 先清后建 + 兜底描述向量集成测试（Feature s1-feat-008 AC #4 + AC #5）。

覆盖：
- AC #4（Medium #2 v3 先清后建）：已 build 文件 A，修改 A 移除某实体 Foo 再 build →
  图中 Foo 若 degree==0 已清；vector_store.delete_by_source(path) 在 index_nodes
  之前调用（FakeVectorStore 记录调用顺序）。
- AC #5（v4 Medium #1 漏抽 Concept 节点）：LLM 漏抽某 Concept 节点 Bar，build 后
  synthesize_fallback 为 Bar 合成兜底描述，且 vector_store.index_nodes 包含
  path::Bar 向量（未被跳过——synthesize 在 index_nodes 之前执行）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from nanokb import pipeline
from nanokb.config import Settings


class FakeLLMClient:
    """模拟 LLMClient：按调用顺序消费预设 JSON 响应。"""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._default = json.dumps({"triples": [], "concepts": []})
        self.complete_calls: int = 0

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        self.complete_calls += 1
        if self._responses:
            return self._responses.pop(0)
        return self._default

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class OrderedFakeVectorStore:
    """记录 delete_by_source / index_nodes 的调用顺序（验证先清后建时序）。"""

    def __init__(self) -> None:
        self.operations: list[tuple[str, str]] = []  # (op_type, source_file)
        self.indexed_node_ids: list[str] = []

    def delete_by_source(self, source_file: str) -> None:
        self.operations.append(("delete", source_file))

    def index_nodes(self, graph: nx.MultiDiGraph, llm: object) -> None:
        for node, data in graph.nodes(data=True):
            sf = str(data.get("source_file", "unknown"))
            self.indexed_node_ids.append(f"{sf}::{node}")
        self.operations.append(("index", ""))


def _settings(raw_dir: Path, out_dir: Path, **kwargs: Any) -> Settings:
    return Settings(raw_dir=raw_dir, out_dir=out_dir, **kwargs)


# ── AC #4：modified 先清后建（Medium #2 v3）────────────────────────


def test_modified_removes_entity_no_residual(tmp_path: Path) -> None:
    """AC #4：修改文件移除实体 Foo → rebuild → 图中 Foo 已清（先清后建）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    doc_path = raw_dir / "doc.md"
    doc_path.write_text("Version 1 with Foo.", encoding="utf-8")

    settings = _settings(raw_dir, out_dir)

    # v1 抽取结果：含 Foo, Bar 两个实体
    v1_response = json.dumps({
        "triples": [
            {"head": "Foo", "relation": "rel", "tail": "Bar", "confidence": "EXTRACTED"},
        ],
        "concepts": [
            {"name": "Foo", "description": "Entity Foo.", "node_type": "concept"},
            {"name": "Bar", "description": "Entity Bar.", "node_type": "concept"},
        ],
    })

    llm = FakeLLMClient([v1_response])
    pipeline.compile(settings, llm=llm)

    # 确认 v1 图含 Foo
    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    assert graph.has_node("Foo")

    # 修改文件（sha256 变更）→ v2 抽取结果不含 Foo
    doc_path.write_text("Version 2 without Foo, only Baz.", encoding="utf-8")
    v2_response = json.dumps({
        "triples": [
            {"head": "Baz", "relation": "rel", "tail": "Bar", "confidence": "EXTRACTED"},
        ],
        "concepts": [
            {"name": "Baz", "description": "Entity Baz.", "node_type": "concept"},
            {"name": "Bar", "description": "Entity Bar v2.", "node_type": "concept"},
        ],
    })

    llm2 = FakeLLMClient([v2_response])
    pipeline.compile(settings, llm=llm2)

    # v2 图中 Foo 已清（degree==0 被 delete_by_source 清理）
    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    assert not graph.has_node("Foo")
    assert graph.has_node("Baz")
    assert graph.has_edge("Baz", "Bar")
    # 不存在旧边 Foo→Bar
    assert not graph.has_edge("Foo", "Bar")


def test_modified_clean_before_rebuild_vector_order(tmp_path: Path) -> None:
    """AC #4：modified 路径 vector_store.delete_by_source 在 index_nodes 之前。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    doc_path = raw_dir / "doc.md"
    doc_path.write_text("v1", encoding="utf-8")

    settings = _settings(raw_dir, out_dir)

    v1 = json.dumps({
        "triples": [{"head": "Foo", "relation": "r", "tail": "Bar", "confidence": "EXTRACTED"}],
        "concepts": [
            {"name": "Foo", "description": "Foo desc.", "node_type": "concept"},
            {"name": "Bar", "description": "Bar desc.", "node_type": "concept"},
        ],
    })

    llm = FakeLLMClient([v1])
    vs = OrderedFakeVectorStore()
    pipeline.compile(settings, llm=llm, vector_store=vs)

    # 修改文件触发 modified
    doc_path.write_text("v2 changed content", encoding="utf-8")
    v2 = json.dumps({
        "triples": [{"head": "Baz", "relation": "r", "tail": "Bar", "confidence": "EXTRACTED"}],
        "concepts": [
            {"name": "Baz", "description": "Baz desc.", "node_type": "concept"},
            {"name": "Bar", "description": "Bar desc.", "node_type": "concept"},
        ],
    })

    vs2 = OrderedFakeVectorStore()
    llm2 = FakeLLMClient([v2])
    pipeline.compile(settings, llm=llm2, vector_store=vs2)

    # 先清后建：delete(doc.md) 出现在 index 之前
    delete_indices = [
        i for i, (op, _) in enumerate(vs2.operations) if op == "delete"
    ]
    index_indices = [
        i for i, (op, _) in enumerate(vs2.operations) if op == "index"
    ]
    assert delete_indices, "expected at least one delete_by_source call"
    assert index_indices, "expected at least one index_nodes call"
    assert max(delete_indices) < min(index_indices), (
        "delete_by_source must be called before index_nodes (先清后建)"
    )


# ── AC #5：LLM 漏抽 Concept → 兜底描述 + 向量索引（v4 Medium #1）────


def test_missed_concept_gets_fallback_and_vector(tmp_path: Path) -> None:
    """AC #5：LLM 漏抽 Concept Bar → synthesize 兜底描述 + 向量含 path::Bar。

    Bar 仅出现在 triple（无 concept），synthesize_fallback 为其合成兜底描述，
    index_nodes（step 7 在 step 6 之后）不为空描述跳过 Bar。
    """
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("Transformer uses Bar mechanism.", encoding="utf-8")

    settings = _settings(raw_dir, out_dir)

    # LLM 返回：Transformer 有 concept，但 Bar 没有（漏抽）
    response = json.dumps({
        "triples": [
            {"head": "Transformer", "relation": "uses", "tail": "Bar", "confidence": "EXTRACTED"},
        ],
        "concepts": [
            {"name": "Transformer", "description": "A model architecture.", "node_type": "concept"},
            # Bar 被漏抽——无 concept 条目
        ],
    })

    llm = FakeLLMClient([response])
    vs = OrderedFakeVectorStore()
    result = pipeline.compile(settings, llm=llm, vector_store=vs)

    # synthesize_fallback 为 Bar 合成了兜底描述
    assert result.synthesized_fallback_count >= 1

    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    assert graph.has_node("Bar")
    bar_desc = graph.nodes["Bar"].get("description", "")
    assert bar_desc.startswith("Bar: ")

    # v4 Medium #1：Bar 的向量被索引（synthesize 在 index_nodes 之前 → Bar 有描述 → 不被跳过）
    assert "doc.md::Bar" in vs.indexed_node_ids, (
        "path::Bar vector must be present — synthesize_fallback ran before index_nodes"
    )


def test_missed_concept_no_vector_store_still_synthesizes(tmp_path: Path) -> None:
    """无 vector_store 时 synthesize_fallback 仍正常执行（向量侧 stub 不影响主线）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("A relates to B.", encoding="utf-8")

    settings = _settings(raw_dir, out_dir)

    # 无 concept，仅 triples
    response = json.dumps({
        "triples": [
            {"head": "A", "relation": "rel", "tail": "B", "confidence": "EXTRACTED"},
        ],
        "concepts": [],
    })

    llm = FakeLLMClient([response])
    result = pipeline.compile(settings, llm=llm)

    # 两个节点均无 concept → synthesize 合成兜底描述
    assert result.synthesized_fallback_count == 2

    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    assert graph.nodes["A"]["description"].startswith("A: ")
    assert graph.nodes["B"]["description"].startswith("B: ")

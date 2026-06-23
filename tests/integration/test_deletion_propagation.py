"""删除传播集成测试（方案 §3.5.1 step 3，Feature s1-feat-008 AC #3）。

覆盖 AC #3（Severe #1）：已 build 文件 A，从 raw/ 删除 A 再 build →
graph.json、ChromaDB（FakeVectorStore）、triples.jsonl（含 delete 标记）均不含
A 来源知识。
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


class FakeVectorStore:
    """模拟 VectorStoreBackend：记录 delete_by_source / index_nodes 调用。"""

    def __init__(self) -> None:
        self.deleted_sources: list[str] = []
        self.indexed_node_ids: list[str] = []

    def delete_by_source(self, source_file: str) -> None:
        self.deleted_sources.append(source_file)

    def index_nodes(self, graph: nx.MultiDiGraph, llm: object) -> None:
        for node, data in graph.nodes(data=True):
            sf = str(data.get("source_file", "unknown"))
            self.indexed_node_ids.append(f"{sf}::{node}")


def _settings(raw_dir: Path, out_dir: Path, **kwargs: Any) -> Settings:
    return Settings(raw_dir=raw_dir, out_dir=out_dir, **kwargs)


def _doc_a_response() -> str:
    return json.dumps({
        "triples": [
            {"head": "Foo", "relation": "rel", "tail": "Bar",
             "confidence": "EXTRACTED"},
        ],
        "concepts": [
            {"name": "Foo", "description": "Entity Foo.", "node_type": "concept"},
            {"name": "Bar", "description": "Entity Bar.", "node_type": "concept"},
        ],
    })


def _doc_b_response() -> str:
    return json.dumps({
        "triples": [
            {"head": "Baz", "relation": "rel", "tail": "Bar",
             "confidence": "EXTRACTED"},
        ],
        "concepts": [
            {"name": "Baz", "description": "Entity Baz.", "node_type": "concept"},
            {"name": "Bar", "description": "Shared entity Bar.", "node_type": "concept"},
        ],
    })


# ── AC #3：删除传播 ─────────────────────────────────────────────────


def test_delete_file_propagates_to_graph_and_triples(tmp_path: Path) -> None:
    """AC #3：删除文件 A 后 build → graph.json / triples.jsonl 不含 A 来源知识。

    Bar 被两文件共享 → 删 A 后 Bar 保留（degree>0）；Foo 仅被 A 引用 → 删除。
    """
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()

    # 初始：两文件共享节点 Bar
    (raw_dir / "a.md").write_text("Foo and Bar.", encoding="utf-8")
    (raw_dir / "b.md").write_text("Baz and Bar.", encoding="utf-8")

    settings = _settings(raw_dir, out_dir)
    llm = FakeLLMClient([_doc_a_response(), _doc_b_response()])
    vs = FakeVectorStore()

    # 首次编译两文件
    pipeline.compile(settings, llm=llm, vector_store=vs)
    assert "a.md::Foo" in vs.indexed_node_ids

    # 删除 a.md
    (raw_dir / "a.md").unlink()

    # 再次编译——新 LLM 实例（不应被调用，因为 b.md 无变更）
    llm2 = FakeLLMClient([])
    vs2 = FakeVectorStore()
    result = pipeline.compile(settings, llm=llm2, vector_store=vs2)

    assert "a.md" in result.changes.deleted
    # a.md 的向量被 delete_by_source
    assert "a.md" in vs2.deleted_sources

    # graph.json：Foo 已删（仅 a.md 引用），Bar 保留（b.md 仍引用）
    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    assert not graph.has_node("Foo")
    assert graph.has_node("Bar")
    assert graph.has_edge("Baz", "Bar")
    # 不存在 a.md 来源的边
    for _, _, data in graph.edges(data=True):
        assert data["source_file"] != "a.md"

    # triples.jsonl 含 delete 标记
    lines = [
        line for line in
        (out_dir / "triples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    delete_records = [
        json.loads(line) for line in lines
        if json.loads(line)["op"] == "delete"
    ]
    assert any(r["source_file"] == "a.md" for r in delete_records)
    assert all(r["schema_version"] for r in delete_records)


def test_delete_isolated_node_removed(tmp_path: Path) -> None:
    """删除文件后仅该文件独有的节点（degree==0）被清理。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()

    (raw_dir / "solo.md").write_text("Unique content.", encoding="utf-8")

    settings = _settings(raw_dir, out_dir)
    llm = FakeLLMClient([_doc_a_response()])
    pipeline.compile(settings, llm=llm)

    # 删除唯一文件
    (raw_dir / "solo.md").unlink()

    llm2 = FakeLLMClient([])
    pipeline.compile(settings, llm=llm2)

    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    # 图为空（所有节点 degree==0 被清理）
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_delete_then_manifest_cleared(tmp_path: Path) -> None:
    """删除文件后 manifest 中该文件的 FileState 被移除。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()

    (raw_dir / "doc.md").write_text("Content.", encoding="utf-8")
    settings = _settings(raw_dir, out_dir)
    llm = FakeLLMClient([_doc_a_response()])
    pipeline.compile(settings, llm=llm)

    manifest_data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "doc.md" in manifest_data["files"]

    (raw_dir / "doc.md").unlink()
    llm2 = FakeLLMClient([])
    pipeline.compile(settings, llm=llm2)

    manifest_data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "doc.md" not in manifest_data["files"]

"""编译流水线端到端集成测试（方案 §3.5.1，Feature s1-feat-008 AC #1 + #2）。

覆盖：
- AC #1：``compile`` 端到端生成 out/graph.json / graph.graphml / triples.jsonl（带
  schema_version）/ manifest.json；节点带 description 与 source_file/confidence。
- AC #2：二次 compile（无变更）→ 不重复处理（manifest 命中，extracted_count=0）。

全部用 FakeLLMClient + FakeVectorStore 注入，tmp_path 隔离，零真实 LLM 调用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from nanokb import pipeline
from nanokb.config import Settings

# ── 测试 doubles ─────────────────────────────────────────────────────


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
        self.index_calls: int = 0

    def delete_by_source(self, source_file: str) -> None:
        self.deleted_sources.append(source_file)

    def index_nodes(self, graph: nx.MultiDiGraph, llm: object) -> None:
        self.index_calls += 1
        for node, data in graph.nodes(data=True):
            sf = str(data.get("source_file", "unknown"))
            self.indexed_node_ids.append(f"{sf}::{node}")


def _settings(raw_dir: Path, out_dir: Path, **kwargs: Any) -> Settings:
    return Settings(raw_dir=raw_dir, out_dir=out_dir, **kwargs)


def _extract_response() -> str:
    """FakeLLM 返回的典型抽取结果：Transformer→Attention + 两个 concept。"""
    return json.dumps({
        "triples": [
            {
                "head": "Transformer",
                "relation": "uses",
                "tail": "Attention",
                "confidence": "EXTRACTED",
            },
            {
                "head": "Transformer",
                "relation": "is_a",
                "tail": "Model",
                "confidence": "INFERRED",
            },
        ],
        "concepts": [
            {
                "name": "Transformer",
                "description": "A neural network architecture for sequence processing.",
                "node_type": "concept",
            },
            {
                "name": "Attention",
                "description": "A mechanism for focusing on relevant parts of input.",
                "node_type": "concept",
            },
        ],
    })


# ── AC #1：端到端编译 ────────────────────────────────────────────────


def test_compile_generates_all_outputs(tmp_path: Path) -> None:
    """AC #1：compile 端到端生成 graph.json/graph.graphml/triples.jsonl/manifest.json。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention mechanism.", encoding="utf-8")

    llm = FakeLLMClient([_extract_response()])
    settings = _settings(raw_dir, out_dir)

    result = pipeline.compile(settings, llm=llm)

    # 抽取了 1 个文件
    assert result.extracted_count == 1
    assert result.skipped == []

    # 四件套全部生成
    assert (out_dir / "graph.json").exists()
    assert (out_dir / "graph.graphml").exists()
    assert (out_dir / "triples.jsonl").exists()
    assert (out_dir / "manifest.json").exists()

    # graph.json 含正确节点/边
    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    assert graph.has_edge("Transformer", "Attention")
    assert graph.has_edge("Transformer", "Model")
    # 节点带 description（来自 concept）
    assert "neural network" in graph.nodes["Transformer"]["description"]
    assert "focusing" in graph.nodes["Attention"]["description"]

    # 边带 source_file 与 confidence
    for _, _, data in graph.edges(data=True):
        assert data["source_file"] == "doc.md"
        assert data["confidence"] in ("EXTRACTED", "INFERRED")

    # triples.jsonl 含 schema_version 的 upsert 记录
    lines = [
        line for line in
        (out_dir / "triples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["schema_version"] == "2"
    assert record["op"] == "upsert"
    assert record["source_file"] == "doc.md"
    assert len(record["triples"]) == 2
    assert len(record["concepts"]) == 2

    # manifest.json 含文件状态与模型身份
    manifest_data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "doc.md" in manifest_data["files"]
    file_state = manifest_data["files"]["doc.md"]
    assert file_state["llm_model"] == settings.llm_model
    assert file_state["extractor_version"] == settings.extractor_version
    assert len(file_state["sha256"]) == 64  # SHA256 hex


# ── AC #2：二次编译无变更不重复处理 ──────────────────────────────────


def test_second_compile_no_changes_skips_processing(tmp_path: Path) -> None:
    """AC #2：二次 compile（无变更）→ manifest 命中，extracted_count=0，无 LLM 调用。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("Some content.", encoding="utf-8")

    llm = FakeLLMClient([_extract_response()])
    settings = _settings(raw_dir, out_dir)

    # 首次编译
    pipeline.compile(settings, llm=llm)
    first_calls = llm.complete_calls
    assert first_calls == 1  # 一个 chunk，一次 LLM 调用

    # 二次编译（无变更）——新 LLM 实例，验证不会被调用
    llm2 = FakeLLMClient([_extract_response()])
    result = pipeline.compile(settings, llm=llm2)

    assert result.extracted_count == 0
    assert llm2.complete_calls == 0  # manifest 命中，不调 LLM


def test_second_compile_graph_unchanged(tmp_path: Path) -> None:
    """二次编译后图谱结构不变（幂等）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("Content.", encoding="utf-8")

    llm = FakeLLMClient([_extract_response(), _extract_response()])
    settings = _settings(raw_dir, out_dir)

    pipeline.compile(settings, llm=llm)
    graph1 = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))

    pipeline.compile(settings, llm=llm)
    graph2 = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))

    assert graph1 == graph2


# ── 向量侧接入验证（FakeVectorStore）────────────────────────────────


def test_compile_with_vector_store_indexes_nodes(tmp_path: Path) -> None:
    """compile(vector_store=fake) 时 step 7 按 path 索引节点（v4 Medium #1 时序）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("Content.", encoding="utf-8")

    llm = FakeLLMClient([_extract_response()])
    vs = FakeVectorStore()
    settings = _settings(raw_dir, out_dir)

    pipeline.compile(settings, llm=llm, vector_store=vs)

    # 向量被索引，id 格式 "{source_file}::{node}"
    assert vs.index_calls == 1
    assert "doc.md::Transformer" in vs.indexed_node_ids
    assert "doc.md::Attention" in vs.indexed_node_ids

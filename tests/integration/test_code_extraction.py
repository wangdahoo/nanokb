"""代码轨端到端集成测试（方案 §3.5.1 + §4 阶段 4，Feature s1-feat-010 AC #5）。

覆盖：
- AC #5：raw/ 下含 .py 文件，``nanokb build``（compile）端到端生成图谱，
  含 calls / defines 关系边（与语义轨结果融合）。

另验证：
- 纯代码 build 不消耗 LLM chat Token（CodeTrack 零 Token）。
- 代码 + 语义双轨融合：同一 build 内 .py 走 CodeTrack、.md 走 SemanticTrack。
- 删除代码文件后图谱 calls/defines 边随之清理（deletion 传播对 code track 同样生效）。
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
    """模拟 LLMClient：仅用于语义轨；记录 complete 调用次数以断言代码轨零 Token。"""

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


def _settings(raw_dir: Path, out_dir: Path, **kwargs: Any) -> Settings:
    return Settings(raw_dir=raw_dir, out_dir=out_dir, **kwargs)


def _semantic_response() -> str:
    """语义轨对 .md 返回的抽取结果（Transformer → uses → Attention）。"""
    return json.dumps({
        "triples": [
            {
                "head": "Transformer",
                "relation": "uses",
                "tail": "Attention",
                "confidence": "EXTRACTED",
            },
        ],
        "concepts": [
            {
                "name": "Transformer",
                "description": "A neural network architecture using attention.",
                "node_type": "concept",
            },
        ],
    })


# ── AC #5：raw/ 含 .py → build 生成含 calls/defines 边的图谱 ─────────


def test_build_with_py_generates_calls_and_defines_edges(tmp_path: Path) -> None:
    """AC #5：raw/ 下含 .py 文件，build 后图谱含 calls/defines 关系边。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "calc.py").write_text(
        "def greet(name):\n"
        "    return format_message(name)\n"
        "\n"
        "def format_message(value):\n"
        "    return 'hello ' + value\n",
        encoding="utf-8",
    )

    llm = FakeLLMClient()
    settings = _settings(raw_dir, out_dir)

    result = pipeline.compile(settings, llm=llm)

    assert result.extracted_count == 1
    assert result.skipped == []

    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)

    relations = {
        data.get("relation") for _, _, data in graph.edges(data=True)
    }
    # 代码轨产出 calls 与 defines 两类关系边
    assert "calls" in relations
    assert "defines" in relations

    # 具体断言：(greet, calls, format_message) 与 (calc, defines, greet)
    edge_set = {
        (u, data.get("relation"), v) for u, v, data in graph.edges(data=True)
    }
    assert ("greet", "calls", "format_message") in edge_set
    assert ("calc", "defines", "greet") in edge_set
    assert ("calc", "defines", "format_message") in edge_set

    # 所有 code 轨边 track=code / confidence=EXTRACTED
    for _, _, data in graph.edges(data=True):
        if data.get("relation") in ("calls", "defines", "contains"):
            assert data.get("track") == "code"
            assert data.get("confidence") == "EXTRACTED"
            assert data.get("source_file") == "calc.py"


def test_pure_code_build_consumes_zero_llm_chat_tokens(tmp_path: Path) -> None:
    """AC #5 / AC #4：仅含 .py 文件的 build 全程零 LLM chat Token。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "mod.py").write_text(
        "def foo(a):\n    return bar(a)\n\ndef bar(b):\n    return b\n",
        encoding="utf-8",
    )

    llm = FakeLLMClient()
    settings = _settings(raw_dir, out_dir)

    pipeline.compile(settings, llm=llm)

    # 纯代码 build → CodeTrack 不调 LLM
    assert llm.complete_calls == 0
    # 图谱仍成功生成
    assert (out_dir / "graph.json").exists()
    assert (out_dir / "triples.jsonl").exists()
    assert (out_dir / "manifest.json").exists()


def test_dual_track_merges_code_and_semantic_results(tmp_path: Path) -> None:
    """AC #5：同一次 build 内 .py 走 CodeTrack、.md 走 SemanticTrack，结果融合到同一图谱。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "mod.py").write_text(
        "def foo():\n    bar()\n\ndef bar():\n    pass\n",
        encoding="utf-8",
    )
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    llm = FakeLLMClient([_semantic_response()])
    settings = _settings(raw_dir, out_dir)

    result = pipeline.compile(settings, llm=llm)

    assert result.extracted_count == 2  # py + md
    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)

    edge_set = {
        (u, data.get("relation"), v, data.get("track"))
        for u, v, data in graph.edges(data=True)
    }
    # 语义轨：Transformer uses Attention（track=semantic）
    assert ("Transformer", "uses", "Attention", "semantic") in edge_set
    # 代码轨：foo calls bar（track=code）
    assert ("foo", "calls", "bar", "code") in edge_set
    # .md 触发一次 LLM 调用，.py 零调用 → 总计 1 次
    assert llm.complete_calls == 1


def test_deletion_propagates_to_code_track_edges(tmp_path: Path) -> None:
    """代码轨边随源文件删除而清理（Severe #1 deletion 传播对 code track 同样生效）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "mod.py").write_text(
        "def foo():\n    bar()\n\ndef bar():\n    pass\n",
        encoding="utf-8",
    )

    llm = FakeLLMClient()
    settings = _settings(raw_dir, out_dir)

    pipeline.compile(settings, llm=llm)
    first_graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    first_graph = nx.node_link_graph(first_graph_data, directed=True, multigraph=True)
    first_edges = {
        (u, data.get("relation"), v)
        for u, v, data in first_graph.edges(data=True)
    }
    assert ("foo", "calls", "bar") in first_edges

    # 删除 mod.py 后重新 build
    (raw_dir / "mod.py").unlink()
    pipeline.compile(settings, llm=llm)

    second_graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    second_graph = nx.node_link_graph(second_graph_data, directed=True, multigraph=True)
    second_edges = {
        (u, data.get("relation"), v)
        for u, v, data in second_graph.edges(data=True)
    }
    assert ("foo", "calls", "bar") not in second_edges


def test_rebuild_idempotent_no_duplicate_code_edges(tmp_path: Path) -> None:
    """代码轨边二次 build 幂等（Medium #9：不累积重复边）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "mod.py").write_text(
        "def foo():\n    bar()\n\ndef bar():\n    pass\n",
        encoding="utf-8",
    )

    settings = _settings(raw_dir, out_dir)
    pipeline.compile(settings, llm=FakeLLMClient())
    pipeline.compile(settings, llm=FakeLLMClient())  # 二次（无变更）

    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    calls_edges = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data.get("relation") == "calls" and u == "foo" and v == "bar"
    ]
    assert len(calls_edges) == 1  # 幂等：仅一条

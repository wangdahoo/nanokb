"""replay 端到端集成测试（方案 §3.5.5，Feature s1-feat-008 AC #6）。

覆盖 AC #6：含失败残留 op 的 triples.jsonl（交错 upsert→delete→upsert），
``replay`` 按去重规则重建出正确图谱（取每文件 ts 最大 op），不消耗 LLM chat Token。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from nanokb import pipeline
from nanokb.config import Settings


def _settings(raw_dir: Path, out_dir: Path) -> Settings:
    return Settings(raw_dir=raw_dir, out_dir=out_dir)


def _write_jsonl(out_dir: Path, records: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    (out_dir / pipeline.TRIPLES_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _triple_dict(head: str, relation: str, tail: str, source_file: str) -> dict[str, Any]:
    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "confidence": "EXTRACTED",
        "source_file": source_file,
        "track": "semantic",
        "chunk_index": 0,
    }


def _concept_dict(name: str, description: str, source_file: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "source_file": source_file,
        "confidence": "EXTRACTED",
        "node_type": "concept",
        "extra": {},
    }


def _upsert_record(
    source_file: str, ts: str, *, triples: list[dict[str, Any]], concepts: list[dict[str, Any]],
    schema_version: str = "2",
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "op": "upsert",
        "source_file": source_file,
        "triples": triples,
        "concepts": concepts,
        "ts": ts,
    }


def _delete_record(source_file: str, ts: str, schema_version: str = "2") -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "op": "delete",
        "source_file": source_file,
        "ts": ts,
    }


# ── AC #6：交错残留 op 的去重回放 ───────────────────────────────────


def test_replay_interleaved_ops_rebuilds_correct_graph(tmp_path: Path) -> None:
    """AC #6：交错 upsert→delete→upsert → replay 取 ts 最大 op 重建正确图谱。"""
    out_dir = tmp_path
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # doc.md 经历三次操作（模拟编译失败后的残留）：
    # ts=1 upsert V1 → ts=2 delete（失败后清理）→ ts=3 upsert V2（最终成功）
    records = [
        _upsert_record("doc.md", "2026-01-01T00:00:01+00:00",
                        triples=[_triple_dict("OLD", "rel", "X", "doc.md")],
                        concepts=[_concept_dict("OLD", "Old version.", "doc.md")]),
        _delete_record("doc.md", "2026-01-01T00:00:02+00:00"),
        _upsert_record("doc.md", "2026-01-01T00:00:03+00:00",
                        triples=[_triple_dict("NEW", "rel", "Y", "doc.md")],
                        concepts=[_concept_dict("NEW", "New version.", "doc.md")]),
    ]
    _write_jsonl(out_dir, records)

    settings = _settings(raw_dir, out_dir)
    result = pipeline.replay(settings)

    # 最终态为 ts=3 的 upsert
    assert result.rebuilt_files == ["doc.md"]
    assert result.deleted_files == []

    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)

    # NEW→Y 存在，OLD→X 不存在
    assert graph.has_edge("NEW", "Y")
    assert not graph.has_node("OLD")
    assert graph.nodes["NEW"]["description"] == "New version."


def test_replay_does_not_consume_llm_tokens(tmp_path: Path) -> None:
    """AC #6：replay 不消耗 LLM chat Token（不从 raw/ 重新抽取）。"""
    out_dir = tmp_path
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    records = [
        _upsert_record("doc.md", "2026-01-01T00:00:01+00:00",
                        triples=[_triple_dict("A", "r", "B", "doc.md")],
                        concepts=[_concept_dict("A", "Entity A.", "doc.md")]),
    ]
    _write_jsonl(out_dir, records)

    settings = _settings(raw_dir, out_dir)
    # replay 不接受 llm 参数——它不调用 LLM
    result = pipeline.replay(settings)

    assert result.rebuilt_files == ["doc.md"]
    # graph.json 正确重建
    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    assert graph.has_edge("A", "B")


def test_replay_multiple_files_converge(tmp_path: Path) -> None:
    """多文件交错操作 → 各自独立收敛到最终态。"""
    out_dir = tmp_path
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    records = [
        # keep.md：upsert → 最终保留
        _upsert_record("keep.md", "2026-01-01T00:00:01+00:00",
                        triples=[_triple_dict("K", "r", "V", "keep.md")],
                        concepts=[_concept_dict("K", "Keep entity.", "keep.md")]),
        # gone.md：upsert → delete → 最终删除
        _upsert_record("gone.md", "2026-01-01T00:00:01+00:00",
                        triples=[_triple_dict("G", "r", "W", "gone.md")],
                        concepts=[_concept_dict("G", "Gone entity.", "gone.md")]),
        _delete_record("gone.md", "2026-01-01T00:00:02+00:00"),
    ]
    _write_jsonl(out_dir, records)

    settings = _settings(raw_dir, out_dir)
    result = pipeline.replay(settings)

    assert result.rebuilt_files == ["keep.md"]
    assert result.deleted_files == ["gone.md"]

    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    assert graph.has_edge("K", "V")
    assert not graph.has_node("G")


def test_replay_synthesizes_fallback_descriptions(tmp_path: Path) -> None:
    """replay 重建后执行 synthesize_fallback_descriptions（step 6）。"""
    out_dir = tmp_path
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # 节点 Bar 仅出现在 triple，无 concept → replay 后 synthesize 兜底
    records = [
        _upsert_record("doc.md", "2026-01-01T00:00:01+00:00",
                        triples=[_triple_dict("A", "rel", "Bar", "doc.md")],
                        concepts=[_concept_dict("A", "Entity A.", "doc.md")]),
    ]
    _write_jsonl(out_dir, records)

    settings = _settings(raw_dir, out_dir)
    result = pipeline.replay(settings)

    assert result.synthesized_fallback_count >= 1

    graph_data = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    assert graph.has_node("Bar")
    assert graph.nodes["Bar"]["description"].startswith("Bar: ")


def test_replay_produces_manifest_and_graph(tmp_path: Path) -> None:
    """replay 输出 graph.json + manifest.json（staging 原子切换）。"""
    out_dir = tmp_path
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    records = [
        _upsert_record("doc.md", "2026-01-01T00:00:01+00:00",
                        triples=[_triple_dict("A", "r", "B", "doc.md")],
                        concepts=[
                            _concept_dict("A", "Entity A.", "doc.md"),
                            _concept_dict("B", "Entity B.", "doc.md"),
                        ]),
    ]
    _write_jsonl(out_dir, records)

    settings = _settings(raw_dir, out_dir)
    pipeline.replay(settings)

    assert (out_dir / "graph.json").exists()
    assert (out_dir / "graph.graphml").exists()
    assert (out_dir / "manifest.json").exists()

    manifest_data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "doc.md" in manifest_data["files"]


def test_replay_after_compile_reproduces_graph(tmp_path: Path) -> None:
    """先 compile 生成 triples.jsonl，再 replay 重建——图结构一致。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("Content about Transformer.", encoding="utf-8")

    settings = _settings(raw_dir, out_dir)

    extract_json = json.dumps({
        "triples": [
            {"head": "Transformer", "relation": "uses", "tail": "Attention",
             "confidence": "EXTRACTED"},
        ],
        "concepts": [
            {"name": "Transformer", "description": "A neural architecture.",
             "node_type": "concept"},
            {"name": "Attention", "description": "A focus mechanism.",
             "node_type": "concept"},
        ],
    })

    class _LocalFakeLLM:
        def __init__(self, resp: str) -> None:
            self._resp = resp
            self.calls = 0

        def complete(self, system: str, user: str,
                     response_format: str = "json", temperature: float = 0.0) -> str:
            self.calls += 1
            return self._resp

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def count_tokens(self, text: str) -> int:
            return max(1, len(text) // 4)

    llm = _LocalFakeLLM(extract_json)
    pipeline.compile(settings, llm=llm)

    graph_after_compile = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))

    result = pipeline.replay(settings)
    assert result.rebuilt_files == ["doc.md"]

    graph_after_replay = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))

    g1 = nx.node_link_graph(graph_after_compile, directed=True, multigraph=True)
    g2 = nx.node_link_graph(graph_after_replay, directed=True, multigraph=True)
    assert set(g1.nodes()) == set(g2.nodes())
    assert set(g1.edges()) == set(g2.edges())

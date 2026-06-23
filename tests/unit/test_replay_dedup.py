"""replay 去重收敛规则单测（方案 §3.5.5，Feature s1-feat-008 AC #6）。

覆盖去重收敛规则的全部分支：
- upsert → delete（同文件）：最终态为 delete（文件不参与重建）。
- delete → upsert（同文件）：最终态为 upsert（upsert 在 delete 之后生效）。
- upsert → upsert（同文件）：最终态为 ts 较大者。
- 同 (source_file, ts) 去重保留 schema_version 最高者。
- 多文件独立收敛。
- 交错 upsert → delete → upsert：最终态为最后一条 upsert。
- schema_version > manifest（降级异常）→ exit 3。
- schema_version < manifest（无迁移函数）→ exit 3。

全部直接测试 pipeline 内部函数，零 LLM 调用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nanokb import pipeline
from nanokb.config import Settings

# ── 辅助构造 ─────────────────────────────────────────────────────────


def _record(
    source_file: str,
    op: str,
    ts: str,
    *,
    schema_version: str = "2",
    triples: list[dict[str, Any]] | None = None,
    concepts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "schema_version": schema_version,
        "op": op,
        "source_file": source_file,
        "ts": ts,
    }
    if op == "upsert":
        rec["triples"] = triples or []
        rec["concepts"] = concepts or []
    return rec


def _write_jsonl(out_dir: Path, records: list[dict[str, Any]]) -> None:
    """将记录列表写入 out/triples.jsonl。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    (out_dir / pipeline.TRIPLES_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── 去重收敛规则 _dedup_converge ─────────────────────────────────────


def test_upsert_then_delete_final_is_delete() -> None:
    """upsert(ts=1) → delete(ts=2)：最终态为 delete。"""
    records = [
        _record("doc.md", "upsert", "2026-01-01T00:00:01+00:00",
                triples=[{"head": "A", "relation": "r", "tail": "B",
                          "confidence": "EXTRACTED", "source_file": "doc.md",
                          "track": "semantic"}]),
        _record("doc.md", "delete", "2026-01-01T00:00:02+00:00"),
    ]
    converged = pipeline._dedup_converge(records)
    assert converged["doc.md"]["op"] == "delete"


def test_delete_then_upsert_final_is_upsert() -> None:
    """delete(ts=1) → upsert(ts=2)：最终态为 upsert。"""
    records = [
        _record("doc.md", "delete", "2026-01-01T00:00:01+00:00"),
        _record("doc.md", "upsert", "2026-01-01T00:00:02+00:00",
                triples=[{"head": "A", "relation": "r", "tail": "B",
                          "confidence": "EXTRACTED", "source_file": "doc.md",
                          "track": "semantic"}]),
    ]
    converged = pipeline._dedup_converge(records)
    assert converged["doc.md"]["op"] == "upsert"


def test_upsert_upsert_keeps_latest_ts() -> None:
    """upsert(ts=1) → upsert(ts=2)：最终态为 ts=2 的 upsert。"""
    records = [
        _record("doc.md", "upsert", "2026-01-01T00:00:01+00:00",
                triples=[{"head": "OLD", "relation": "r", "tail": "B",
                          "confidence": "EXTRACTED", "source_file": "doc.md",
                          "track": "semantic"}]),
        _record("doc.md", "upsert", "2026-01-01T00:00:02+00:00",
                triples=[{"head": "NEW", "relation": "r", "tail": "B",
                          "confidence": "EXTRACTED", "source_file": "doc.md",
                          "track": "semantic"}]),
    ]
    converged = pipeline._dedup_converge(records)
    final = converged["doc.md"]
    assert final["op"] == "upsert"
    assert final["triples"][0]["head"] == "NEW"


def test_same_ts_keeps_highest_schema_version() -> None:
    """同 (source_file, ts) 去重保留 schema_version 最高者。"""
    records = [
        _record("doc.md", "upsert", "2026-01-01T00:00:01+00:00", schema_version="1"),
        _record("doc.md", "upsert", "2026-01-01T00:00:01+00:00", schema_version="2"),
    ]
    converged = pipeline._dedup_converge(records)
    assert converged["doc.md"]["schema_version"] == "2"


def test_multiple_files_converge_independently() -> None:
    """多文件独立收敛：f1 最终 upsert，f2 最终 delete。"""
    records = [
        _record("f1.md", "upsert", "2026-01-01T00:00:01+00:00",
                triples=[{"head": "A", "relation": "r", "tail": "B",
                          "confidence": "EXTRACTED", "source_file": "f1.md",
                          "track": "semantic"}]),
        _record("f2.md", "upsert", "2026-01-01T00:00:01+00:00",
                triples=[{"head": "C", "relation": "r", "tail": "D",
                          "confidence": "EXTRACTED", "source_file": "f2.md",
                          "track": "semantic"}]),
        _record("f2.md", "delete", "2026-01-01T00:00:02+00:00"),
    ]
    converged = pipeline._dedup_converge(records)
    assert converged["f1.md"]["op"] == "upsert"
    assert converged["f2.md"]["op"] == "delete"


def test_interleaved_upsert_delete_upsert_final_is_last_upsert() -> None:
    """交错 upsert → delete → upsert（模拟失败残留）：最终态为最后一条 upsert。"""
    records = [
        _record("doc.md", "upsert", "2026-01-01T00:00:01+00:00",
                triples=[{"head": "V1", "relation": "r", "tail": "B",
                          "confidence": "EXTRACTED", "source_file": "doc.md",
                          "track": "semantic"}]),
        _record("doc.md", "delete", "2026-01-01T00:00:02+00:00"),
        _record("doc.md", "upsert", "2026-01-01T00:00:03+00:00",
                triples=[{"head": "V3", "relation": "r", "tail": "B",
                          "confidence": "EXTRACTED", "source_file": "doc.md",
                          "track": "semantic"}]),
    ]
    converged = pipeline._dedup_converge(records)
    final = converged["doc.md"]
    assert final["op"] == "upsert"
    assert final["triples"][0]["head"] == "V3"


# ── schema_version 校验 ─────────────────────────────────────────────


def test_schema_version_higher_than_manifest_exits_3(tmp_path: Path) -> None:
    """jsonl schema_version > manifest → 拒绝回放 exit 3。"""
    records = [_record("doc.md", "upsert", "2026-01-01T00:00:01+00:00", schema_version="3")]
    _write_jsonl(tmp_path, records)
    # 写入 manifest version=2
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": "2", "files": {}}), encoding="utf-8"
    )
    settings = Settings(raw_dir=tmp_path / "raw", out_dir=tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        pipeline.replay(settings)
    assert exc_info.value.code == 3


def test_schema_version_lower_than_manifest_no_migration_exits_3(tmp_path: Path) -> None:
    """jsonl schema_version < manifest（无迁移函数）→ 拒绝回放 exit 3。"""
    records = [_record("doc.md", "upsert", "2026-01-01T00:00:01+00:00", schema_version="1")]
    _write_jsonl(tmp_path, records)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": "2", "files": {}}), encoding="utf-8"
    )
    settings = Settings(raw_dir=tmp_path / "raw", out_dir=tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        pipeline.replay(settings)
    assert exc_info.value.code == 3


# ── 端到端 replay 重建 ───────────────────────────────────────────────


def test_replay_rebuilds_from_converged_upsert(tmp_path: Path) -> None:
    """replay 从去重收敛后的 upsert 记录重建图谱。"""
    records = [
        _record(
            "doc.md", "upsert", "2026-01-01T00:00:01+00:00",
            triples=[{
                "head": "Transformer", "relation": "uses", "tail": "Attention",
                "confidence": "EXTRACTED", "source_file": "doc.md",
                "track": "semantic", "chunk_index": 0,
            }],
            concepts=[{
                "name": "Transformer", "description": "A neural architecture.",
                "source_file": "doc.md", "confidence": "EXTRACTED",
                "node_type": "concept", "extra": {},
            }],
        ),
    ]
    _write_jsonl(tmp_path, records)
    settings = Settings(raw_dir=tmp_path / "raw", out_dir=tmp_path)

    result = pipeline.replay(settings)

    assert result.rebuilt_files == ["doc.md"]
    assert result.deleted_files == []

    # graph.json 存在且含正确节点/边
    import networkx as nx
    graph_data = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    assert graph.has_edge("Transformer", "Attention")
    assert graph.nodes["Transformer"]["description"] == "A neural architecture."


def test_replay_skips_files_whose_final_op_is_delete(tmp_path: Path) -> None:
    """replay 跳过最终态为 delete 的文件（不参与重建）。"""
    records = [
        _record(
            "keep.md", "upsert", "2026-01-01T00:00:01+00:00",
            triples=[{
                "head": "A", "relation": "r", "tail": "B",
                "confidence": "EXTRACTED", "source_file": "keep.md",
                "track": "semantic",
            }],
        ),
        _record("gone.md", "upsert", "2026-01-01T00:00:01+00:00",
                triples=[{
                    "head": "X", "relation": "r", "tail": "Y",
                    "confidence": "EXTRACTED", "source_file": "gone.md",
                    "track": "semantic",
                }]),
        _record("gone.md", "delete", "2026-01-01T00:00:02+00:00"),
    ]
    _write_jsonl(tmp_path, records)
    settings = Settings(raw_dir=tmp_path / "raw", out_dir=tmp_path)

    result = pipeline.replay(settings)

    assert result.rebuilt_files == ["keep.md"]
    assert result.deleted_files == ["gone.md"]

    import networkx as nx
    graph_data = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)
    assert graph.has_edge("A", "B")
    assert not graph.has_node("X")


def test_replay_empty_jsonl_returns_empty_result(tmp_path: Path) -> None:
    """triples.jsonl 不存在时 replay 返回空结果。"""
    settings = Settings(raw_dir=tmp_path / "raw", out_dir=tmp_path)
    result = pipeline.replay(settings)
    assert result.rebuilt_files == []
    assert result.deleted_files == []

"""编译流水线端到端集成测试（方案 §3.5.1，Feature s1-feat-008 AC #1 + #2 + 缓存/签名扩展）。

覆盖：
- AC #1：``compile`` 端到端生成 out/graph.json / graph.graphml / triples.jsonl（带
  schema_version）/ manifest.json；节点带 description 与 source_file/confidence。
- AC #2：二次 compile（无变更）→ 不重复处理（manifest 命中，extracted_count=0）。
- 缓存/签名扩展（s1-feat-008）：换 embedding_config 命中零 LLM + 向量重建；
  换 extraction 配置（chunk_max_tokens / concept_description_strategy）miss 重抽；
  换 index_config 命中但图/triples.jsonl 重建；改文档内容 sha256 变 miss。

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
    return json.dumps(
        {
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
        }
    )


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
        line
        for line in (out_dir / "triples.jsonl").read_text(encoding="utf-8").splitlines()
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


# ── AC 扩展（Feature s1-feat-008）：三层签名缓存/失效场景 ──────────────
#
# 端到端验证三层配置签名（extraction/index/embedding）切分 + 内容寻址抽取缓存：
# - 换 embedding_config（embedding_model/embedding_provider）→ 缓存命中零 LLM，
#   但向量/图谱下游重建；
# - 换 extraction 配置（chunk_max_tokens/concept_description_strategy/code_languages）
#   → 缓存 miss 重新抽取；
# - 换 index_config（fallback_description_max_edges）→ 缓存命中但图谱/triples.jsonl 重建；
# - 改文档内容（sha256 变）→ 缓存 miss 重新抽取。
#
# 每个用例断言三维度：缓存命中/miss（cached_count）+ complete_calls 数 + 下游重建信号
# （FakeVectorStore.index_calls/deleted_sources 或 triples.jsonl 新增 upsert 记录）。


def test_change_embedding_model_cache_hit_zero_llm(tmp_path: Path) -> None:
    """换 embedding_model recompile：缓存命中（key 不含 embedding），零 LLM，向量重建。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    llm1 = FakeLLMClient([_extract_response()])
    vs = FakeVectorStore()
    settings1 = _settings(raw_dir, out_dir)

    # 首次编译：1 次 LLM 调用，cached_count=0，向量为首次索引
    pipeline.compile(settings1, llm=llm1, vector_store=vs)
    assert llm1.complete_calls == 1
    assert vs.index_calls == 1

    # 换 embedding_model → embedding_config 变 → detector 判 modified → 进入抽取循环
    # 但 cache key = sha256(sha256|extraction_config|llm_model) 不含 embedding → 命中
    llm2 = FakeLLMClient([_extract_response()])
    settings2 = _settings(raw_dir, out_dir, embedding_model="text-embedding-3-large")
    result = pipeline.compile(settings2, llm=llm2, vector_store=vs)

    assert result.cached_count == 1
    assert llm2.complete_calls == 0  # 缓存命中，零 LLM
    # 下游重建信号：modified 触发 delete + re-index
    assert "doc.md" in vs.deleted_sources
    assert vs.index_calls == 2


def test_change_embedding_provider_cache_hit_zero_llm(tmp_path: Path) -> None:
    """换 embedding_provider（区别于 embedding_model）→ 同样缓存命中零 LLM（评审 Opt #1）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    llm1 = FakeLLMClient([_extract_response()])
    vs = FakeVectorStore()
    settings1 = _settings(raw_dir, out_dir)
    pipeline.compile(settings1, llm=llm1, vector_store=vs)
    assert llm1.complete_calls == 1

    # 换 embedding_provider（openai → ollama）：embedding_config 变，extraction_config 不变
    llm2 = FakeLLMClient([_extract_response()])
    settings2 = _settings(raw_dir, out_dir, embedding_provider="ollama")
    result = pipeline.compile(settings2, llm=llm2, vector_store=vs)

    assert result.cached_count == 1
    assert llm2.complete_calls == 0
    assert "doc.md" in vs.deleted_sources
    assert vs.index_calls == 2


def test_change_chunk_max_tokens_cache_miss_reextract(tmp_path: Path) -> None:
    """换 chunk_max_tokens → extraction_config 变 → 缓存 miss 重新抽取。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    llm1 = FakeLLMClient([_extract_response()])
    settings1 = _settings(raw_dir, out_dir)
    pipeline.compile(settings1, llm=llm1)
    assert llm1.complete_calls == 1

    # 换 chunk_max_tokens → extraction_config 变 → cache key 变 → miss
    llm2 = FakeLLMClient([_extract_response()])
    settings2 = _settings(raw_dir, out_dir, chunk_max_tokens=1500)
    result = pipeline.compile(settings2, llm=llm2)

    assert result.cached_count == 0
    assert llm2.complete_calls == 1  # miss → 重新调用 LLM


def test_change_concept_description_strategy_cache_miss(tmp_path: Path) -> None:
    """换 concept_description_strategy → extraction_config 变 → 缓存 miss 重新抽取。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    llm1 = FakeLLMClient([_extract_response()])
    settings1 = _settings(raw_dir, out_dir)
    pipeline.compile(settings1, llm=llm1)
    assert llm1.complete_calls == 1

    # 换 concept_description_strategy → extraction_config 变 → miss
    llm2 = FakeLLMClient([_extract_response()])
    settings2 = _settings(raw_dir, out_dir, concept_description_strategy="concat_dedup")
    result = pipeline.compile(settings2, llm=llm2)

    assert result.cached_count == 0
    assert llm2.complete_calls == 1


def test_change_index_config_cache_hit_graph_rebuilt(tmp_path: Path) -> None:
    """换 fallback_description_max_edges → index_config 变 → 缓存命中但图/triples.jsonl 重建。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    llm1 = FakeLLMClient([_extract_response()])
    settings1 = _settings(raw_dir, out_dir)
    pipeline.compile(settings1, llm=llm1)
    assert llm1.complete_calls == 1

    # 换 fallback_description_max_edges → index_config 变（extraction_config 不变）
    llm2 = FakeLLMClient([_extract_response()])
    settings2 = _settings(raw_dir, out_dir, fallback_description_max_edges=10)
    result = pipeline.compile(settings2, llm=llm2)

    # 缓存命中：extraction_config 未变 → key 不变 → hit
    assert result.cached_count == 1
    assert llm2.complete_calls == 0
    # 图/triples.jsonl 重建信号：modified 触发新 upsert 记录
    records = [
        json.loads(line)
        for line in (out_dir / "triples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    upserts = [r for r in records if r.get("op") == "upsert" and r.get("source_file") == "doc.md"]
    assert len(upserts) == 2  # 首次 added + 二次 modified 各一条


def test_content_change_cache_miss_reextract(tmp_path: Path) -> None:
    """改文档内容 → sha256 变 → 缓存 miss 重新抽取。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    llm1 = FakeLLMClient([_extract_response()])
    settings = _settings(raw_dir, out_dir)
    pipeline.compile(settings, llm=llm1)
    assert llm1.complete_calls == 1

    # 改文档内容 → sha256 变 → cache key 变 → miss
    (raw_dir / "doc.md").write_text("# Transformer v2\n\nDifferent content.", encoding="utf-8")
    llm2 = FakeLLMClient([_extract_response()])
    result = pipeline.compile(settings, llm=llm2)

    assert result.cached_count == 0
    assert llm2.complete_calls == 1  # miss → 重新抽取

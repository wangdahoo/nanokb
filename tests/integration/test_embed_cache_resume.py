"""Embedding Cache 中断重跑集成测试（方案 §4.7，Feature s3-feat-001）。

覆盖验收标准：
- AC1.1：首次 compile embed 调用批次符合预期（miss 切批在 embed_batch 内）。
- AC1.2：删 graph.json 模拟中断且保留 embed_cache → force=True 重跑 embed 调用 == 0。
- AC1.6 / Opt#5：全命中重跑后 vector_store.search(query) 返回预期节点
  （证明 cache 命中的向量仍被 col.upsert 进 ChromaDB 可被召回）。

场景：
  1. 首次 compile → 统计 embed 调用数 N1（>0）。
  2. 删除 graph.json（模拟中断，但保留 out/embed_cache/）。
  3. force=True 重跑 → embed 调用数 N2 == 0（全命中 cache，AC1.2）。
  4. 重跑后 vector_store.search(query) 返回预期节点（AC1.6/Opt#5）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanokb import pipeline
from nanokb.config import Settings
from nanokb.index.vector_store import VectorStore


class CountingEmbedder:
    """记录 embed 调用次数的 embedding 客户端。

    用确定性 hash 把 description 编码进向量维度，使相同 description 产生相同向量
    （便于 search 断言「同 description 召回同节点」）。
    """

    def __init__(self, embedding_dim: int = 8) -> None:
        self._dim = embedding_dim
        self.embed_calls: int = 0
        self.embedded_texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        self.embedded_texts.extend(texts)
        return [self._vec_for(t) for t in texts]

    def _vec_for(self, t: str) -> list[float]:
        """对 description 生成确定性向量（首字节决定方向，便于 search 召回）。"""
        h = abs(hash(t)) % 997
        return [float((h >> i) & 0xFF) / 255.0 for i in range(self._dim)]


class FakeChatLLM:
    """模拟 chat LLM：complete 按序消费预设 JSON；embed 抛错（生文不做 embedding）。"""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._default = json.dumps({"triples": [], "concepts": []})

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        if self._responses:
            return self._responses.pop(0)
        return self._default

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("chat llm should not be called for embedding (decoupled)")

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _settings(raw_dir: Path, out_dir: Path, **kwargs: Any) -> Settings:
    return Settings(raw_dir=raw_dir, out_dir=out_dir, **kwargs)


def _extract_response_with_concepts() -> str:
    """抽取结果：含 Transformer / Attention 两个 concept（带 description）。"""
    return json.dumps(
        {
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
                    "description": "A neural network architecture for sequences.",
                    "node_type": "concept",
                },
                {
                    "name": "Attention",
                    "description": "A mechanism to focus on relevant input tokens.",
                    "node_type": "concept",
                },
            ],
        }
    )


# ══════════════════════════════════════════════════════════════════════
# AC1.1 + AC1.2：首次 embed → 删 graph.json → force 重跑零 embed
# ══════════════════════════════════════════════════════════════════════


def test_resume_after_interrupt_zero_embed_calls(tmp_path: Path) -> None:
    """AC1.2：删 graph.json 模拟中断（保留 embed_cache），force=True 重跑
    embed 调用数 == 0（全命中 cache）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention mechanism.", encoding="utf-8")

    settings = _settings(raw_dir, out_dir, enable_embed_cache=True, embed_concurrency=1)

    # 首次 compile
    chat_llm = FakeChatLLM([_extract_response_with_concepts()])
    embedder = CountingEmbedder(embedding_dim=8)
    pipeline.compile(settings, llm=chat_llm, embedding_client=embedder)

    # 首次 embed 调用数 N1 > 0（至少 1 次维度探测 + 1 次 index_nodes embed）
    n1 = embedder.embed_calls
    assert n1 > 0, "first compile should have called embed at least once"
    # embed_cache 目录已生成文件
    embed_cache_dir = out_dir / "embed_cache"
    assert embed_cache_dir.exists()
    cache_files = list(embed_cache_dir.glob("*.json"))
    assert len(cache_files) >= 1, "embed cache should have entries after first compile"

    # 模拟中断：删除 graph.json（保留 embed_cache）
    (out_dir / "graph.json").unlink()
    assert not (out_dir / "graph.json").exists()
    # embed_cache 仍在
    assert embed_cache_dir.exists()

    # force=True 重跑
    chat_llm2 = FakeChatLLM([_extract_response_with_concepts()])
    embedder2 = CountingEmbedder(embedding_dim=8)
    pipeline.compile(settings, llm=chat_llm2, embedding_client=embedder2, force=True)

    # AC1.2：embed 调用数 == 0（全命中 cache，零 embedding Token）
    # 注意：force 重跑时 _probe_embedding_dim 仍会调用一次 embed（探针），
    # 但 index_nodes 的 embed_batch 全命中 cache，不调用 embedder.embed。
    # 探针计入 embed_calls，故断言 embed_batch 路径的 embed 调用。
    # 为精确断言「cache 命中」，我们检查 embedder2 在 index_nodes 阶段未对
    # description 文本调用 embed：探针只 embed 单条固定探针文本。
    non_probe_calls = [
        t for t in embedder2.embedded_texts if t != "nanokb-embedding-dim-probe"
    ]
    assert non_probe_calls == [], (
        f"force rerun should hit cache for all descriptions; "
        f"non-probe embed texts: {non_probe_calls}"
    )


def test_first_compile_embed_batches_as_expected(tmp_path: Path) -> None:
    """AC1.1：首次 compile embed 批次符合预期（miss 切批在 embed_batch 内）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("Document content about models.", encoding="utf-8")

    settings = _settings(raw_dir, out_dir, enable_embed_cache=True, embed_concurrency=1)

    chat_llm = FakeChatLLM([_extract_response_with_concepts()])
    embedder = CountingEmbedder(embedding_dim=8)
    pipeline.compile(settings, llm=chat_llm, embedding_client=embedder)

    # 首次：description 文本（Transformer/Attention 的 description + 探针）
    # 被分批 embed。2 个节点 description < 64，故 index_nodes 阶段 1 个 batch。
    # 探针 1 次调用。
    description_embeds = [
        t for t in embedder.embedded_texts if t != "nanokb-embedding-dim-probe"
    ]
    # 至少 2 条（Transformer + Attention 的 description）
    assert len(description_embeds) >= 2


# ══════════════════════════════════════════════════════════════════════
# AC1.6 / Opt#5：全命中重跑后 search 返回预期节点
# ══════════════════════════════════════════════════════════════════════


def test_search_returns_expected_node_after_cache_rerun(tmp_path: Path) -> None:
    """AC1.6 / Opt#5：全命中重跑后 vector_store.search(query) 返回预期节点
    （证明 cache 命中的向量仍被 col.upsert 进 ChromaDB 可被召回）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    settings = _settings(raw_dir, out_dir, enable_embed_cache=True, embed_concurrency=1)

    # 首次 compile
    chat_llm = FakeChatLLM([_extract_response_with_concepts()])
    embedder = CountingEmbedder(embedding_dim=8)
    pipeline.compile(settings, llm=chat_llm, embedding_client=embedder)

    # 模拟中断：删除 graph.json（保留 embed_cache + chroma 不删，但重跑会重建 chroma）
    (out_dir / "graph.json").unlink()

    # force=True 重跑（全命中 cache）
    chat_llm2 = FakeChatLLM([_extract_response_with_concepts()])
    embedder2 = CountingEmbedder(embedding_dim=8)
    pipeline.compile(settings, llm=chat_llm2, embedding_client=embedder2, force=True)

    # 重跑后 chroma 已重建，向量来自 cache 命中但仍被 upsert
    vs = VectorStore(out_dir / "chroma", settings.embedding_model, 8)
    assert vs.count() >= 2, "chroma should have vectors after cache-hit rerun (Opt#5)"

    # search 召回：query embedding 用同一 embedder（确定性 hash）
    hits = vs.search("A neural network architecture for sequences.", k=5, embedder=embedder2)
    assert len(hits) > 0, "search should return hits from cache-upserted vectors (AC1.6)"

    # 至少有一个命中节点的 name 在 {Transformer, Attention}
    names = {hit.concept.name for hit in hits}
    assert names & {"Transformer", "Attention"}, (
        f"expected Transformer/Attention in search results; got {names}"
    )


def test_search_returns_hits_first_compile_baseline(tmp_path: Path) -> None:
    """基线对照：首次 compile（无 cache 命中）search 也能返回预期节点，
    确认 AC1.6 测试的有效性（不是偶然通过）。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    settings = _settings(raw_dir, out_dir, enable_embed_cache=True, embed_concurrency=1)
    chat_llm = FakeChatLLM([_extract_response_with_concepts()])
    embedder = CountingEmbedder(embedding_dim=8)
    pipeline.compile(settings, llm=chat_llm, embedding_client=embedder)

    vs = VectorStore(out_dir / "chroma", settings.embedding_model, 8)
    hits = vs.search("A neural network architecture for sequences.", k=5, embedder=embedder)
    assert len(hits) > 0
    names = {hit.concept.name for hit in hits}
    assert names & {"Transformer", "Attention"}

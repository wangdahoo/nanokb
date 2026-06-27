"""index_nodes 端到端并发穿透集成测试（方案 §5.6 AC2.6，Feature s3-feat-003）。

**AC2.6 阻断项**（round 2 Severe #1）：经真实 ``VectorStore.index_nodes`` 调用路径
验证 embedding 阶段并发真实生效——mock embedder 统计「同时 in-flight 的 embed
调用数峰值」，索引 500 节点（单子图，miss ≥ 2 batch），断言峰值 ≥ 2。

这验证了「batch 归属单一化」修复（Severe #1）：``index_nodes`` 一次性把全部
description 传给 ``embed_fn``，``EmbeddingCache.embed_batch`` 独占切批 + 并发。
若两层都切批（旧行为），每次 ``embed_fn`` 只收到 ≤ 64 文本 → 单 batch →
并发恒为 1（峰值 == 1），此测试会失败。

被测链路：``VectorStore.index_nodes`` → ``EmbeddingCache.embed_batch``（并发分支）
→ ``ThreadPoolExecutor`` → mock embedder.embed（峰值统计）。
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import networkx as nx

from nanokb.index.vector_store import VectorStore
from nanokb.llm.embed_cache import EMBED_BATCH_SIZE, EmbeddingCache


class PeakTrackingEmbedder:
    """统计「同时 in-flight embed 调用数峰值」的 embedder。

    进入 embed 时 +1 并更新峰值，退出时 -1。线程安全：峰值与 in-flight 计数
    由 ``threading.Lock`` 保护。可选 ``delay`` 模拟网络 IO（放大并发窗口，
    使峰值统计更稳定）。

    向量生成确定性（sha256），便于后续 search 召回断言。
    """

    def __init__(self, embedding_dim: int = 8, delay: float = 0.0) -> None:
        self._dim = embedding_dim
        self._delay = delay
        self._lock = threading.Lock()
        self._in_flight: int = 0
        self.peak: int = 0
        self.embed_calls: int = 0
        self.embedded_texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.embed_calls += 1
            self._in_flight += 1
            if self._in_flight > self.peak:
                self.peak = self._in_flight
            self.embedded_texts.extend(texts)
        try:
            if self._delay > 0:
                time.sleep(self._delay)
            return [self._vec_for(t) for t in texts]
        finally:
            with self._lock:
                self._in_flight -= 1

    def _vec_for(self, t: str) -> list[float]:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        return [float(h[i % len(h)]) / 255.0 for i in range(self._dim)]


def _make_graph(n_nodes: int, source_file: str = "doc.md") -> nx.MultiDiGraph:
    """构造含 n_nodes 个带 description 节点的单子图（全部同 source_file）。"""
    g = nx.MultiDiGraph()
    for i in range(n_nodes):
        g.add_node(
            f"node-{i:04d}",
            description=f"description for node {i:04d} in knowledge graph",
            source_file=source_file,
            node_type="concept",
        )
    return g


# ══════════════════════════════════════════════════════════════════════
# AC2.6（阻断项）：经 index_nodes 端到端验证峰值 ≥ 2
# ══════════════════════════════════════════════════════════════════════


def test_index_nodes_concurrency_peak_at_least_2(tmp_path: Path) -> None:
    """AC2.6（round 2 Severe #1，阻断项）：经真实 ``VectorStore.index_nodes``
    索引 500 节点（单子图，miss ≥ 2 batch），断言 embedder 同时 in-flight 调用
    数峰值 ≥ 2。

    场景：
    - 构造 500 节点单子图（全部同 source_file，模拟单文件）。
    - embed_concurrency=4，启用 cache（首次 miss）。
    - 500 unique description → ceil(500/64) = 8 batches（≥ 2，满足 AC2.6 前提）。
    - mock embedder 每次 embed 调用 sleep 10ms（放大并发窗口，使峰值稳定 ≥ 2）。

    断言：
    - 峰值 ≥ 2（证明 batch 归属单一化后并发穿透 index_nodes 调用路径）。
    - 8 batches 全部执行（embed 调用数 == 8）。
    - 向量被 upsert（vs.count == 500）。
    - cache 文件已写（中断重跑零成本前提）。
    """
    n_nodes = 500
    expected_batches = -(-n_nodes // EMBED_BATCH_SIZE)  # ceil(500/64) = 8
    assert expected_batches >= 2  # AC2.6 前提

    embedder = PeakTrackingEmbedder(embedding_dim=8, delay=0.01)
    cache = EmbeddingCache(
        tmp_path / "embed_cache",
        "test-model",
        8,
        embedder,
        embed_concurrency=4,
        enable_cache=True,
    )
    vs = VectorStore(tmp_path / "chroma", "test-model", 8)
    graph = _make_graph(n_nodes)

    # 经真实 index_nodes 调用路径（注入 cache.embed_batch）
    vs.index_nodes(graph, embedder, embed_fn=cache.embed_batch)

    # ★ AC2.6 阻断项：峰值 ≥ 2
    assert embedder.peak >= 2, (
        f"AC2.6 BLOCKER: peak in-flight embed calls={embedder.peak}; "
        f"should be ≥ 2 (batch ownership single-point fix, Severe #1). "
        f"If peak==1, index_nodes is double-batching (old bug)."
    )

    # 全部 batches 执行
    assert embedder.embed_calls == expected_batches, (
        f"expected {expected_batches} batches, got {embedder.embed_calls}"
    )
    # 向量被 upsert
    assert vs.count() == n_nodes, (
        f"chroma should have {n_nodes} vectors, got {vs.count()}"
    )
    # cache 文件已写（500 unique → 500 cache 文件）
    cache_files = list((tmp_path / "embed_cache").glob("*.json"))
    assert len(cache_files) == n_nodes, (
        f"cache should have {n_nodes} entries, got {len(cache_files)}"
    )


def test_index_nodes_concurrency_peak_scales_with_workers(tmp_path: Path) -> None:
    """AC2.6 补充：embed_concurrency=4 时峰值应接近 4（充分大的 miss batch 数下）。

    8 batches + 4 workers + delay → 峰值应达 4（第一轮 4 个并发 in-flight）。
    断言峰值 ≥ 3（保守，容许调度抖动），进一步证明并发真实生效而非偶然 ≥ 2。
    """
    n_nodes = 512  # 恰好 8 个满 batch
    embedder = PeakTrackingEmbedder(embedding_dim=8, delay=0.02)
    cache = EmbeddingCache(
        tmp_path / "embed_cache",
        "test-model",
        8,
        embedder,
        embed_concurrency=4,
        enable_cache=True,
    )
    vs = VectorStore(tmp_path / "chroma", "test-model", 8)
    graph = _make_graph(n_nodes)

    vs.index_nodes(graph, embedder, embed_fn=cache.embed_batch)

    # 4 workers + 充分 batches + delay → 峰值应达 4（保守断言 ≥ 3）
    assert embedder.peak >= 3, (
        f"with embed_concurrency=4 and 8 batches, peak should reach ~4; "
        f"got peak={embedder.peak}"
    )
    assert vs.count() == n_nodes


# ══════════════════════════════════════════════════════════════════════
# AC2.6 对照组：embed_concurrency=1 时峰值 == 1（串行）
# ══════════════════════════════════════════════════════════════════════


def test_index_nodes_serial_concurrency_peak_is_1(tmp_path: Path) -> None:
    """对照组：embed_concurrency=1（串行分支）经 index_nodes 端到端，峰值 == 1。
    与 AC2.6 形成对照，证明峰值 ≥ 2 确由并发分支带来，而非测试夹具偶然。"""
    n_nodes = 500
    embedder = PeakTrackingEmbedder(embedding_dim=8, delay=0.01)
    cache = EmbeddingCache(
        tmp_path / "embed_cache",
        "test-model",
        8,
        embedder,
        embed_concurrency=1,  # 串行
        enable_cache=True,
    )
    vs = VectorStore(tmp_path / "chroma", "test-model", 8)
    graph = _make_graph(n_nodes)

    vs.index_nodes(graph, embedder, embed_fn=cache.embed_batch)

    # 串行：峰值恒为 1
    assert embedder.peak == 1, (
        f"serial (embed_concurrency=1) should have peak==1; got {embedder.peak}"
    )
    assert vs.count() == n_nodes


# ══════════════════════════════════════════════════════════════════════
# AC2.6 对照组：embed_fn=None 走原 llm.embed 路径（零回归，Opt#4）
# ══════════════════════════════════════════════════════════════════════


def test_index_nodes_no_embed_fn_uses_llm_directly(tmp_path: Path) -> None:
    """对照组：embed_fn=None 时 index_nodes 走原 ``llm.embed`` 路径（零回归）。
    证明 embed_fn 注入是可选的，不破坏既有调用路径。"""
    n_nodes = 130  # 2 batches（>64 → 切批在 embed_batch 外不存在，index_nodes 不切）
    embedder = PeakTrackingEmbedder(embedding_dim=8, delay=0.01)
    vs = VectorStore(tmp_path / "chroma", "test-model", 8)
    graph = _make_graph(n_nodes)

    # embed_fn=None → index_nodes 直接调 llm.embed(texts) 一次性传入全部
    vs.index_nodes(graph, embedder, embed_fn=None)

    # llm.embed 被调用 1 次（一次性全部 texts），峰值 == 1（无并发）
    assert embedder.embed_calls == 1
    assert embedder.peak == 1
    assert vs.count() == n_nodes


# ══════════════════════════════════════════════════════════════════════
# AC2.4 端到端：并发失败经 index_nodes 上抛 + 已完成 batch 已写 cache
# ══════════════════════════════════════════════════════════════════════


class FailOnNthCallEmbedder:
    """第 N 次 embed 调用抛异常（端到端失败安全验证）。"""

    def __init__(self, fail_on_call: int, embedding_dim: int = 8) -> None:
        self._fail_on_call = fail_on_call
        self._dim = embedding_dim
        self._lock = threading.Lock()
        self.embed_calls: int = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.embed_calls += 1
            n = self.embed_calls
        if n == self._fail_on_call:
            raise RuntimeError(f"injected failure on embed call #{n}")
        return [
            [float(hashlib.sha256(t.encode()).digest()[0]) / 255.0] * self._dim
            for t in texts
        ]


def test_index_nodes_concurrent_failure_propagates(tmp_path: Path) -> None:
    """AC2.4 端到端：并发分支中某 batch embed 抛异常 → index_nodes 抛出
    （不执行 staging_swap）+ 已完成 batch 已写 cache + 排队 batch 被取消。"""
    n_nodes = 512  # 8 batches
    embedder = FailOnNthCallEmbedder(fail_on_call=2, embedding_dim=8)
    cache = EmbeddingCache(
        tmp_path / "embed_cache",
        "test-model",
        8,
        embedder,
        embed_concurrency=4,
        enable_cache=True,
    )
    vs = VectorStore(tmp_path / "chroma", "test-model", 8)
    graph = _make_graph(n_nodes)

    # index_nodes 应抛出 RuntimeError
    import pytest

    with pytest.raises(RuntimeError, match="injected failure"):
        vs.index_nodes(graph, embedder, embed_fn=cache.embed_batch)

    # AC2.7：embed 调用数 ≤ 并发度 + 1（cancel_futures 取消排队 batch）
    assert embedder.embed_calls <= 5, (
        f"embed_calls={embedder.embed_calls} should be ≤ concurrency+1=5; "
        f"queued batches should be cancelled"
    )
    assert embedder.embed_calls < 8, "queued batches were not cancelled"

    # AC2.4：已完成 batch 已写 cache（≥ 1 文件，< 全部）
    cache_files = list((tmp_path / "embed_cache").glob("*.json"))
    assert len(cache_files) >= 1, "completed batches should have written cache"
    assert len(cache_files) < n_nodes, "failure should have aborted before all batches"

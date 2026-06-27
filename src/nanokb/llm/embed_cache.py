"""内容寻址 embedding 缓存（方案 §4 阶段一，Feature s3-feat-001）。

按 ``sha256(description_sha256|embedding_model|embedding_dim)`` 三维 key 缓存
``description → vector``，落盘到 ``out/embed_cache/<key>.json``。key 不含
source_file，故同 description 跨文档自动共享（内容寻址）；value 包裹
``embedding_model`` / ``embedding_dim`` 元数据作为防御层（类似
``VectorStore._ensure_collection`` 的维度校验），不匹配视为 miss。

设计要点（round 2 / round 3）：
- embedder 在 ``__init__`` 构造时注入持有（与 ``RateLimiter`` 注入 ``OpenAIClient``
  范式一致，Medium #5①）。
- ``embed_batch`` 独占 cache 查询 + miss 去重（round 3 Opt#1）+ miss 切批(64) +
  串行 embed + 长度校验（Medium #5②，禁止静默截断）+ 写回 + 原序组装。
  并发分支（embed_concurrency>1）在 feat-003 叠加。
- cache 与并发正交：``enable_cache=False`` 时 get/put 为 no-op，``embed_batch``
  仍走串行/并发 embed（Medium #4）。
- ``embedding_dim == 0``（探测失败）时 ``put`` no-op（Opt#4）——避免探测失败期
  无效缓存堆积。
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path

from nanokb.llm.base import EmbeddingClient
from nanokb.utils.io import atomic_write_json

logger = logging.getLogger("nanokb")

#: miss 切批大小（与 ``VectorStore.EMBED_BATCH_SIZE`` 对齐，取 64 与 embedding
#: provider 的 input 上限一致——智谱 GLM embedding-3 限 64 条）。
EMBED_BATCH_SIZE = 64


class EmbeddingCache:
    """内容寻址 embedding 缓存：``<cache_dir>/<key>.json``。

    key = ``sha256(f"{description_sha256}|{embedding_model}|{embedding_dim}")``，
    不含 source_file（同 description 跨文档共享）。best-effort：可删可重建，
    解析失败 / 维度不匹配视为 miss（复用 ``ExtractionCache`` 的 try/except→None 范式）。

    round 3（Opt#1）：``embed_batch`` 对 miss 文本按 ``self._key(t)`` 先去重再切批
    embed——重复 description（不同 node 同描述，常见于兜底合成节点）只 embed + put
    一次，结果经 ``dedup_map`` 广播回填所有同 key 原始位置。既省 Token 又避免
    「同 key 并发写同一文件」。
    """

    def __init__(
        self,
        cache_dir: Path,
        embedding_model: str,
        embedding_dim: int,
        embedder: EmbeddingClient,
        *,
        embed_concurrency: int = 1,
        enable_cache: bool = True,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._embedder = embedder
        self._embed_concurrency = max(1, embed_concurrency)
        self._enable_cache = enable_cache
        if self._enable_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, description: str) -> str:
        """三维内容寻址 key（description_sha256 | embedding_model | embedding_dim）。"""
        desc_sha = hashlib.sha256(description.encode("utf-8")).hexdigest()
        raw = f"{desc_sha}|{self._embedding_model}|{self._embedding_dim}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, description: str) -> list[float] | None:
        """命中返回向量，miss / 损坏 / 维度不匹配返回 None。

        ``enable_cache=False`` 时恒返回 None（cache 与并发正交，Medium #4）。
        复用 ``ExtractionCache.get`` 的 try/except→None 范式：覆盖 JSONDecodeError
        / KeyError / TypeError / OSError / 维度不匹配等损坏场景，退化为 miss。
        """
        if not self._enable_cache:
            return None
        path = self._cache_dir / f"{self._key(description)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # 防御层：model / dim 不匹配视为过期（miss），不阻断主线
            if data.get("embedding_model") != self._embedding_model:
                return None
            if data.get("embedding_dim") != self._embedding_dim:
                return None
            vector = data.get("vector")
            if not isinstance(vector, list):
                return None
            return vector
        except Exception:
            logger.debug(
                "embed cache miss (corrupt/unreadable): %s",
                path,
                extra={"stage": "compile-vector"},
            )
            return None

    def put(self, description: str, vector: list[float]) -> None:
        """原子写回（``atomic_write_json``）。

        ``enable_cache=False`` 或 ``embedding_dim == 0``（探测失败）时为 no-op
        （Opt#4：避免探测失败期无效缓存堆积）。写盘失败仅 WARNING，不抛。
        """
        if not self._enable_cache or self._embedding_dim == 0:
            return
        path = self._cache_dir / f"{self._key(description)}.json"
        payload = {
            "embedding_model": self._embedding_model,
            "embedding_dim": self._embedding_dim,
            "vector": vector,
        }
        try:
            atomic_write_json(path, payload)
        except Exception:
            logger.warning(
                "embed cache put failed for %s; result kept in memory",
                path,
                extra={"stage": "compile-vector"},
            )

    def embed_batch(
        self,
        texts: list[str],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """批量 embed：查 cache 命中 → miss 去重 → miss 切批(64) → 串行 embed
        → 长度校验 → 写回 → 原序组装。

        本方法封装「查询缓存 + miss 去重 + 批量 embed + 写回」为一个原子动作，
        ``VectorStore.index_nodes`` 调用它替代裸 ``llm.embed(texts)``。

        阶段一（feat-001）只交付串行分支（``embed_concurrency<=1``）；
        并发分支（``embed_concurrency>1``，ThreadPoolExecutor）在 feat-003 叠加。

        Args:
            texts: 待 embed 的 description 文本列表（原序）。
            on_progress: 进度回调 ``(done, total)``，``total`` 为去重后 miss 文本数。

        Returns:
            与 ``texts`` 等长、同序的向量列表。

        Raises:
            RuntimeError: embedder 返回向量数 != 输入 batch 文本数（Medium #5②，
                禁止静默截断残 None）。
        """
        # 1. 查 cache（主线程）；enable_cache=False 时全部视为 miss
        results: list[list[float] | None] = [self.get(t) for t in texts]
        miss_idx = [i for i, r in enumerate(results) if r is None]
        if not miss_idx:
            # 全命中
            return [r for r in results if r is not None]

        # 2. miss 去重（round 3 Opt#1）：按 self._key(t) 去重，重复 description 只
        #    embed + put 一次，结果广播回填所有同 key 原始位置。
        miss_texts_raw = [texts[i] for i in miss_idx]
        seen: dict[str, int] = {}
        unique_miss: list[str] = []
        for t in miss_texts_raw:
            k = self._key(t)
            if k not in seen:
                seen[k] = len(unique_miss)
                unique_miss.append(t)
        dedup_map: dict[str, list[float]] = {}

        # 3. miss 去重后按 batch 切分（EMBED_BATCH_SIZE=64）——embed_batch 独占切批
        batches = [
            (s, unique_miss[s : s + EMBED_BATCH_SIZE])
            for s in range(0, len(unique_miss), EMBED_BATCH_SIZE)
        ]
        concurrency = self._embed_concurrency
        done = 0

        def _do_one(start: int, batch: list[str]) -> int:
            """单个 batch：embed → 长度校验 → 写回 cache + dedup_map。"""
            # ★ Medium #5②：返回前校验长度，禁止静默截断
            vecs = self._embedder.embed(batch)
            if len(vecs) != len(batch):
                logger.error(
                    "embedder returned %d vectors for %d texts; fail-safe abort batch",
                    len(vecs),
                    len(batch),
                    extra={"stage": "compile-vector"},
                )
                raise RuntimeError(
                    f"embedder length mismatch: {len(vecs)} != {len(batch)}"
                )
            for i, v in zip(range(start, start + len(batch)), vecs, strict=True):
                self.put(unique_miss[i], list(map(float, v)))
                dedup_map[self._key(unique_miss[i])] = list(map(float, v))
            return len(batch)

        # 4. 串行分支（concurrency<=1，阶段一基线 / 零回归）
        #    并发分支（concurrency>1，ThreadPoolExecutor + as_completed）在 feat-003。
        if concurrency <= 1:
            for start, batch in batches:
                done += _do_one(start, batch)
                if on_progress:
                    on_progress(done, len(unique_miss))
        else:  # pragma: no cover — 并发分支在 feat-003 实现，本 feature 不交付
            # 占位：feat-003 在此加 ThreadPoolExecutor(max_workers=concurrency)
            # + as_completed 归并 + except BaseException: pool.shutdown(cancel_futures=True)
            for start, batch in batches:
                done += _do_one(start, batch)
                if on_progress:
                    on_progress(done, len(unique_miss))

        # 5. 广播回填：按原始位置查 dedup_map
        for orig_idx in miss_idx:
            results[orig_idx] = dedup_map[self._key(texts[orig_idx])]

        # 至此 results 中所有 None 已被 dedup_map 回填；长度必与 texts 相等。
        # 显式窄化为 list[list[float]] 供 mypy strict 通过。
        return [r for r in results if r is not None]


__all__ = ["EMBED_BATCH_SIZE", "EmbeddingCache"]

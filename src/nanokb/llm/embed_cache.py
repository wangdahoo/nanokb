"""内容寻址 embedding 缓存（方案 §4 阶段一 + §5 阶段二）。

按 ``sha256(description_sha256|embedding_model|embedding_dim)`` 三维 key 缓存
``description → vector``，落盘到 ``out/embed_cache/<key>.json``。key 不含
source_file，故同 description 跨文档自动共享（内容寻址）；value 包裹
``embedding_model`` / ``embedding_dim`` 元数据作为防御层（类似
``VectorStore._ensure_collection`` 的维度校验），不匹配视为 miss。

设计要点（round 2 / round 3）：
- embedder 在 ``__init__`` 构造时注入持有（与 ``RateLimiter`` 注入 ``OpenAIClient``
  范式一致，Medium #5①）。
- ``embed_batch`` 独占 cache 查询 + miss 去重（round 3 Opt#1）+ miss 切批(64) +
  embed + 长度校验（Medium #5②，禁止静默截断）+ 写回 + 原序组装。
  - 串行分支（``embed_concurrency<=1``，feat-001 基线 / 零回归）。
  - 并发分支（``embed_concurrency>1``，feat-003）：``ThreadPoolExecutor`` +
    ``as_completed`` 归并，``with`` + ``except BaseException: cancel_futures=True``
    失败安全（Medium #2）。
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        """批量 embed：查 cache 命中 → miss 去重 → miss 切批(64) → 串行或并发
        embed → 长度校验 → 写回 → 原序组装。

        本方法封装「查询缓存 + miss 去重 + 批量 embed + 写回」为一个原子动作，
        ``VectorStore.index_nodes`` 调用它替代裸 ``llm.embed(texts)``。

        - 串行分支（``embed_concurrency<=1``，feat-001 基线 / 零回归）。
        - 并发分支（``embed_concurrency>1``，feat-003）：``ThreadPoolExecutor`` +
          ``as_completed`` 归并，``with`` + ``except BaseException`` 失败安全
          取消未决 future（Medium #2）。原序组装在并发乱序返回下正确：每个完成
          的 batch 携带自身绝对 start 索引，经 miss_idx 回填原始位置，与完成
          顺序无关（确定性约束）。

        Args:
            texts: 待 embed 的 description 文本列表（原序）。
            on_progress: 进度回调 ``(done, total)``，``total`` 为去重后 miss 文本数。

        Returns:
            与 ``texts`` 等长、同序的向量列表。

        Raises:
            RuntimeError: embedder 返回向量数 != 输入 batch 文本数（Medium #5②，
                禁止静默截断残 None）；或并发中某 batch embed 抛出（fut.result()
                重抛 → except BaseException 取消未决 future → 上抛 index_nodes）。
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
        total_batches = len(batches)
        log_every = max(1, total_batches // 20)

        logger.info(
            "embed_batch: %d texts (%d unique after dedup, %d batches, concurrency=%d)",
            len(texts), len(unique_miss), total_batches, concurrency,
            extra={"stage": "compile-vector"},
        )

        def _log_progress(batches_done: int) -> None:
            """按 ~5% 步进打印 embed 进度（大批量不刷屏，末批必报）。"""
            if batches_done % log_every == 0 or batches_done == total_batches:
                logger.info(
                    "embed_batch: %d/%d texts (%d/%d batches)",
                    done, len(unique_miss), batches_done, total_batches,
                    extra={"stage": "compile-vector"},
                )

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
        #    或并发分支（concurrency>1，阶段二 feat-003：ThreadPoolExecutor +
        #    as_completed + 失败安全取消未决 future，Medium #2）。
        if concurrency <= 1:
            for idx, (start, batch) in enumerate(batches, 1):
                done += _do_one(start, batch)
                _log_progress(idx)
                if on_progress:
                    on_progress(done, len(unique_miss))
        else:
            # ★ Medium #2：with 语句保证退出 shutdown；except BaseException
            #   覆盖 KeyboardInterrupt 与普通异常，显式 cancel_futures=True
            #   取消排队未执行的 future（省 Token），随后 raise 上抛由
            #   index_nodes / pipeline 中断（失败安全，已完成 batch 已写回 cache）。
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                try:
                    futures = {
                        pool.submit(_do_one, s, b): (s, b) for s, b in batches
                    }
                    completed = 0
                    for fut in as_completed(futures):
                        done += fut.result()
                        completed += 1
                        _log_progress(completed)
                        if on_progress:
                            on_progress(done, len(unique_miss))
                except BaseException:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise

        # 5. 广播回填：按原始位置查 dedup_map
        for orig_idx in miss_idx:
            results[orig_idx] = dedup_map[self._key(texts[orig_idx])]

        # 不变式校验：所有位置必须已回填（cache 命中或 dedup_map 广播）。残留 None
        # 说明上游逻辑被破坏——显式 raise 而非静默过滤（旧 [r for r in results if r
        # is not None] 会丢项 + 缩短列表，虽被 index_nodes 长度校验兜住但掩盖根因）。
        out: list[list[float]] = []
        for i, r in enumerate(results):
            if r is None:
                raise RuntimeError(
                    f"embed_batch invariant violated: position {i} unresolved after backfill"
                )
            out.append(r)
        return out


__all__ = ["EMBED_BATCH_SIZE", "EmbeddingCache"]

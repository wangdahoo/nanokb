"""ChromaDB 向量库（方案 §3.4.4 + §3.5.1 step 7，Feature s1-feat-011）。

``VectorStore`` 封装 ChromaDB 持久化客户端，实现 pipeline ``VectorStoreBackend``
协议（delete_by_source / index_nodes），为知识图谱节点提供向量索引与语义召回能力：

- ``_ensure_collection``：读 collection metadata ``embedding_dim``，不匹配则 drop 重建
  （Medium #7 向量侧防御层：切换 embedding_model 导致维度变更时自动重建，避免
  残留旧维度向量污染召回）。
- ``index_nodes``：为每个节点 description 生成 embedding，id=f"{source_file}::{node}"
  upsert（Medium #9 幂等——重跑不累积重复向量）。空描述节点跳过 + WARNING（Opt #2 v3）。
  v4 Medium #1：pipeline 保证 ``synthesize_fallback_descriptions`` 先于此方法执行，
  故节点描述均已就绪（含兜底描述），无节点因空描述被跳过。
- ``delete_by_source``：``where={"source_file":source_file}`` 删除（Severe #1 删除传播 +
  Medium #2 modified 先清后建复用）。
- ``search``：向量召回，返回 ``RetrievalHit`` 列表（s1-feat-012 ``VectorRetriever`` 接入）。

**ChromaDB 语义说明**：collection 的 ``metadata`` 是用户自定义元信息（我们记 embedding_dim/
embedding_model），ChromaDB 的实际维度由首次 upsert 的 embedding 决定。``_ensure_collection``
的维度校验基于 metadata（我们的记录），确保切换模型后整个 collection 被 drop 重建，
使实际存储维度与 metadata 一致。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import InvalidCollectionException

from nanokb.models import RetrievalHit

if TYPE_CHECKING:
    import networkx as nx  # type: ignore[import-untyped]

    from nanokb.llm.base import EmbeddingClient

logger = logging.getLogger("nanokb")

#: ChromaDB collection 名（3-63 字符约束，nanokb 专属）
COLLECTION_NAME = "nanokb_nodes"

#: metadata 中记录 embedding 维度的 key
EMBEDDING_DIM_KEY = "embedding_dim"

#: metadata 中记录 embedding 模型的 key
EMBEDDING_MODEL_KEY = "embedding_model"

#: index_nodes 批量 embed 的最大 batch 大小（避免单次 embed 文本过多）。
#: 取 64 与 embedding provider 的 input 上限对齐（智谱 GLM embedding-3 限 64 条）。
EMBED_BATCH_SIZE = 64

#: search 默认召回数
DEFAULT_SEARCH_K = 10


class VectorStore:
    """ChromaDB 持久化向量库，实现 ``VectorStoreBackend`` 协议。

    构造期通过 ``_ensure_collection`` 建立或校验 collection。所有方法均操作同一个
    PersistentClient（``path`` 目录下的 ChromaDB 实例）。

    Args:
        path: ChromaDB 持久化目录（通常 ``out/chroma``）。
        embedding_model: embedding 模型标识（记入 collection metadata）。
        embedding_dim: embedding 维度（记入 collection metadata，用于 Medium #7 维度校验）。
    """

    def __init__(self, path: Path, embedding_model: str, embedding_dim: int) -> None:
        self._path = Path(path)
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._client = chromadb.PersistentClient(
            path=str(self._path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection_name = COLLECTION_NAME
        self._collection: Any = None  # chromadb Collection（类型存根不完整，用 Any）
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """校验 / 建立 collection（Medium #7 维度防御层）。

        - collection 不存在 → 创建，metadata 记 ``embedding_dim`` / ``embedding_model``。
        - collection 存在但 ``metadata['embedding_dim']`` 与当前不匹配 → drop 重建
          （切换 embedding_model 致维度变更时触发，避免旧维度向量污染）。
        - collection 存在且维度匹配 → 直接复用。
        """
        expected_dim = self._embedding_dim
        desired_metadata: dict[str, Any] = {
            EMBEDDING_DIM_KEY: expected_dim,
            EMBEDDING_MODEL_KEY: self._embedding_model,
        }

        try:
            col = self._client.get_collection(self._collection_name)
        except (InvalidCollectionException, ValueError):
            col = self._client.create_collection(
                self._collection_name, metadata=dict(desired_metadata)
            )
            self._collection = col
            logger.debug(
                "created collection %s with embedding_dim=%d",
                self._collection_name,
                expected_dim,
            )
            return

        meta = col.metadata or {}
        existing_dim = meta.get(EMBEDDING_DIM_KEY)
        if existing_dim != expected_dim:
            logger.warning(
                "embedding_dim mismatch: collection has %r, expected %d; dropping and rebuilding",
                existing_dim,
                expected_dim,
                extra={"stage": "vector-store"},
            )
            self._client.delete_collection(self._collection_name)
            col = self._client.create_collection(
                self._collection_name, metadata=dict(desired_metadata)
            )

        self._collection = col

    def index_nodes(self, graph: nx.MultiDiGraph, llm: EmbeddingClient) -> None:
        """为图中每个节点的 description 生成 embedding 并 upsert（Medium #9 幂等）。

        每个节点以 ``id=f"{source_file}::{node}"`` 作为主键 upsert，保证同文件二次
        index 不累积重复向量。``source_file`` 缺失的节点默认用 ``"unknown"``。

        ``llm`` 仅使用其 ``embed`` 方法（``EmbeddingClient`` 协议）；生文与向量解耦后，
        此处接收的是独立的 embedding 客户端（``LLMClient`` 是其超集，向后兼容）。

        空描述节点跳过 + WARNING（Opt #2 v3）。v4 Medium #1：pipeline 保证
        ``synthesize_fallback_descriptions`` 先于此方法执行，故实际调用时节点描述
        均已就绪（含兜底描述），跳过分支仅在防御性场景生效。
        """
        col = self._collection
        if col is None:
            raise RuntimeError("collection not initialized; call _ensure_collection first")

        items: list[tuple[str, str, str, str]] = []  # (id, source_file, node, description)
        for node, data in graph.nodes(data=True):
            description = data.get("description")
            if not isinstance(description, str) or not description.strip():
                logger.warning(
                    "skip node %r: empty description (synthesize_fallback_descriptions "
                    "should run before index_nodes)",
                    node,
                    extra={"stage": "vector-store", "file": str(node)},
                )
                continue
            source_file = str(data.get("source_file", "unknown"))
            node_id = f"{source_file}::{node}"
            items.append((node_id, source_file, node, description))

        if not items:
            logger.debug("index_nodes: no nodes with descriptions to index")
            return

        # 分批 embed + upsert
        for start in range(0, len(items), EMBED_BATCH_SIZE):
            batch = items[start : start + EMBED_BATCH_SIZE]
            texts = [desc for _, _, _, desc in batch]
            embeddings = llm.embed(texts)
            if len(embeddings) != len(batch):
                logger.error(
                    "embed returned %d vectors for %d texts; skipping batch",
                    len(embeddings),
                    len(batch),
                    extra={"stage": "vector-store"},
                )
                continue

            col.upsert(
                ids=[node_id for node_id, _, _, _ in batch],
                embeddings=[list(map(float, e)) for e in embeddings],
                documents=[desc for _, _, _, desc in batch],
                metadatas=[{"source_file": sf, "node": node} for _, sf, node, _ in batch],
            )

        logger.info(
            "index_nodes: upserted %d node vectors",
            len(items),
            extra={"stage": "vector-store"},
        )

    def delete_by_source(self, source_file: str) -> None:
        """删除 ``source_file`` 的全部向量（Severe #1 + Medium #2 先清后建）。

        ChromaDB ``delete(where={"source_file": source_file})`` 删除所有匹配的向量。
        """
        col = self._collection
        if col is None:
            raise RuntimeError("collection not initialized; call _ensure_collection first")
        col.delete(where={"source_file": source_file})
        logger.debug(
            "delete_by_source(%s)",
            source_file,
            extra={"stage": "vector-store", "file": source_file},
        )

    def search(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        *,
        embedder: EmbeddingClient | None = None,
    ) -> list[RetrievalHit]:
        """向量语义召回：embed query → ChromaDB 近邻查询 → ``RetrievalHit`` 列表。

        Args:
            query: 自然语言查询文本。
            k: 召回数量上限。
            embedder: 用于 embed query 的 embedding 客户端（``EmbeddingClient`` 协议，
                仅用 ``embed``；必须提供）。

        Returns:
            ``RetrievalHit`` 列表，``score = 1.0 - distance``（cosine 距离转相似度），
            ``concept`` 字段携带节点名与 description 供下游展示，``source="vector"``。
        """
        col = self._collection
        if col is None:
            raise RuntimeError("collection not initialized; call _ensure_collection first")
        if embedder is None:
            raise ValueError("search requires an embedder client to embed the query")

        query_embeddings = embedder.embed([query])
        if not query_embeddings:
            return []

        n_results = min(k, col.count()) if col.count() > 0 else k
        if n_results == 0:
            return []

        results = col.query(
            query_embeddings=[list(map(float, query_embeddings[0]))],
            n_results=n_results,
        )

        hits: list[RetrievalHit] = []
        ids_batch = results.get("ids", [[]])
        distances_batch = results.get("distances", [[]])
        metadatas_batch = results.get("metadatas", [[]])
        documents_batch = results.get("documents", [[]])

        if not ids_batch:
            return []

        ids = ids_batch[0]
        distances = distances_batch[0] if distances_batch else [0.0] * len(ids)
        metadatas = metadatas_batch[0] if metadatas_batch else [{}] * len(ids)
        documents = documents_batch[0] if documents_batch else [""] * len(ids)

        from nanokb.models import Concept, Confidence

        for i, node_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 0.0
            document = documents[i] if i < len(documents) else ""
            score = max(0.0, 1.0 - float(distance))
            node_name = str(meta.get("node", node_id))
            source_file = str(meta.get("source_file", ""))
            hits.append(
                RetrievalHit(
                    concept=Concept(
                        name=node_name,
                        description=document or None,
                        source_file=source_file,
                        confidence=Confidence.EXTRACTED,
                    ),
                    score=score,
                    source="vector",
                )
            )

        return hits

    def count(self) -> int:
        """返回 collection 中当前向量总数（测试 / 调试用）。"""
        if self._collection is None:
            return 0
        return int(self._collection.count())

    def list_ids(self) -> list[str]:
        """返回 collection 中全部向量 id（测试 / 调试用）。"""
        if self._collection is None:
            return []
        result = self._collection.get()
        return list(result.get("ids", []))


__all__ = [
    "COLLECTION_NAME",
    "DEFAULT_SEARCH_K",
    "EMBED_BATCH_SIZE",
    "EMBEDDING_DIM_KEY",
    "EMBEDDING_MODEL_KEY",
    "VectorStore",
]

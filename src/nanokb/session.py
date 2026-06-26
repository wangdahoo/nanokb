"""知识库检索会话（s2-feat-006 / 优化方案 M3）。

提供 ``RetrievalSession`` 库级 API：对同一 ``Settings`` 懒加载并缓存 graph /
communities / vector_store，多次 ``answer`` / ``search`` 复用，消除每次调用重复
全量解析 graph.json / communities.json 的开销。

CLI 仍为单次进程直连 ``pipeline``（行为不变）；本模块面向编程式多次查询与未来
常驻/server 模式。所有加载通过 ``pipeline`` 的现有 loader 完成（graph/communities/
vector_store 注入 ``pipeline.answer_query`` / ``pipeline.search_communities`` 的既有
可选参数，不改动它们的签名）。

graph/communities 在 session 生命周期内视为只读（与 retriever 一致）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nanokb import pipeline
from nanokb.config import Settings
from nanokb.index.community import CommunityResult, load_communities
from nanokb.llm.base import EmbeddingClient, LLMClient, make_llm_client
from nanokb.models import RetrievalHit
from nanokb.qa.progress import NullProgressReporter, ProgressReporter

if TYPE_CHECKING:
    from nanokb.index.vector_store import VectorStore
    from nanokb.pipeline import AnswerQueryResult

logger = logging.getLogger("nanokb")


class RetrievalSession:
    """对同一知识库的多次检索会话，懒加载并复用 graph/communities/vector_store。

    用法::

        session = RetrievalSession(settings)
        r1 = session.answer("Transformer 如何依赖 Attention？")  # 首次加载
        r2 = session.answer("RNN 是什么？")                     # 复用缓存
        hits = session.search("深度学习")                       # 复用 graph/communities

    ``llm`` / ``embedding_client`` 可选注入（测试 / 复用现有客户端）；未注入时按
    ``Settings`` 创建。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLMClient | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._settings = settings
        self._llm: LLMClient | None = llm
        self._embedding_client = embedding_client
        # 缓存字段（graph 始终非 None；communities/vector_store 可能为 None——用
        # *_loaded 标志区分"未加载"与"已加载但不可用"，避免 None 时反复重试加载）
        self._graph = None
        self._communities: CommunityResult | None = None
        self._communities_loaded: bool = False
        self._vector_store: VectorStore | None = None
        self._vector_store_loaded: bool = False
        # 加载计数（测试/诊断用，验证缓存命中）
        self._graph_loads: int = 0
        self._communities_loads: int = 0
        self._vector_store_loads: int = 0

    # ── 懒加载（缓存一次） ───────────────────────────────────────────

    def _ensure_graph(self) -> object:
        if self._graph is None:
            self._graph = pipeline._load_graph(self._settings.out_dir)
            self._graph_loads += 1
        return self._graph

    def _ensure_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = make_llm_client(self._settings)
        return self._llm

    def _ensure_communities(self) -> CommunityResult | None:
        if not self._communities_loaded:
            self._communities = load_communities(self._settings.out_dir)
            self._communities_loaded = True
            self._communities_loads += 1
        return self._communities

    def _ensure_vector_store(self) -> VectorStore | None:
        if not self._vector_store_loaded:
            embedder = pipeline._resolve_embedder(
                self._settings, self._embedding_client, self._ensure_llm()
            )
            self._vector_store = pipeline._ensure_vector_store(
                self._settings, embedder, None
            )
            self._vector_store_loaded = True
            self._vector_store_loads += 1
        return self._vector_store

    # ── 公共 API ─────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        *,
        mode: str = "query",
        progress: ProgressReporter | None = None,
    ) -> AnswerQueryResult:
        """复用缓存的 graph/communities/vector_store 执行问答（透传 ``pipeline.answer_query``）。

        按 ``mode`` 懒加载所需资源：``query`` 三路（graph 始终，vector/community 按
        配置）；``ask`` 仅向量；``search`` 仅社区（用 ``search``）。
        """
        graph = self._ensure_graph()
        llm = self._ensure_llm()
        if mode in ("query", "ask"):
            self._ensure_vector_store()
        if mode == "query" and self._settings.enable_community_recall:
            self._ensure_communities()
        return pipeline.answer_query(
            self._settings,
            question,
            mode=mode,  # type: ignore[arg-type]
            llm=llm,
            embedding_client=self._embedding_client,
            graph=graph,
            vector_store=self._vector_store,
            communities=self._communities,
            progress=progress or NullProgressReporter(),
        )

    def search(
        self,
        keyword: str,
        *,
        progress: ProgressReporter | None = None,
    ) -> list[RetrievalHit]:
        """复用缓存的 graph/communities 执行社区宏观检索（透传 ``pipeline.search_communities``）。"""
        graph = self._ensure_graph()
        communities = self._ensure_communities()
        llm = self._ensure_llm()
        return pipeline.search_communities(
            self._settings,
            keyword,
            llm=llm,
            graph=graph,
            communities=communities,
            progress=progress or NullProgressReporter(),
        )


__all__ = ["RetrievalSession"]

"""索引构建阶段包（方案 §3.4.4 + §3.5.1 step 7/8，Feature s1-feat-011）。

stage4 在图谱编译完成后（step 6 ``synthesize_fallback_descriptions`` 之后）执行，
构建三类索引供问答期多路召回消费：

- ``vector_store.VectorStore``：ChromaDB 持久化向量库（实现 ``VectorStoreBackend``
  协议）。``index_nodes`` 为节点 description 生成 embedding，``id=f"{source_file}::{node}"``
  upsert（Medium #9 幂等）。``_ensure_collection`` 校验维度不匹配则 drop 重建（Medium #7）。
  ``delete_by_source`` 支持删除传播 + modified 先清后建。``search`` 向量语义召回。
- ``community.detect_communities``：Leiden 社区发现——折叠平行边 + 对称化 sum
  （Opt #3 v3）+ ``ModularityVertexPartition`` + 社区 LLM 摘要 → ``communities.json``。
- ``keyword_index.build``：关键词倒排索引（中英混合分词）→ ``keywords.json``（v4 Opt #1
  单文件，纳入 staging 原子切换五件套）。

pipeline step 8 调用 ``build_indexes(graph, settings, llm, staging_dir)`` 将 community
+ keyword 两索引写入 staging 目录，由 ``staging_swap`` 原子切换到 out/。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from nanokb.config import Settings
from nanokb.index.community import (
    COMMUNITIES_FILENAME,
    Community,
    CommunityResult,
    detect_communities,
    load_communities,
)
from nanokb.index.keyword_index import (
    KEYWORDS_FILENAME,
    KeywordEntry,
    KeywordIndex,
)
from nanokb.index.keyword_index import (
    build as build_keyword_index,
)
from nanokb.index.keyword_index import (
    load as load_keyword_index,
)
from nanokb.index.vector_store import (
    COLLECTION_NAME,
    DEFAULT_SEARCH_K,
    VectorStore,
)

if TYPE_CHECKING:
    import networkx as nx  # type: ignore[import-untyped]

    from nanokb.llm.base import LLMClient

logger = logging.getLogger("nanokb")


def build_indexes(
    graph: nx.MultiDiGraph,
    settings: Settings,
    llm: LLMClient | None,
    staging_dir: Path,
) -> tuple[CommunityResult, KeywordIndex]:
    """执行 step 8 全部索引构建（community + keyword），写入 ``staging_dir``。

    由 pipeline ``compile`` / ``replay`` 在 step 8 调用。两索引均由 graph 派生，
    纳入 staging 原子切换五件套（v4 Opt #1）。

    Args:
        graph: 已编译的知识图谱（含 synthesize_fallback 兜底描述）。
        settings: 全局配置（读 ``leiden_symmetrize``）。
        llm: LLM 客户端（社区摘要用）；``None`` 时用启发式摘要。
        staging_dir: staging 目录（``communities.json`` + ``keywords.json`` 写入此处）。

    Returns:
        ``(CommunityResult, KeywordIndex)`` —— 社区发现 + 关键词索引结果。
    """
    community_result = detect_communities(graph, settings, llm, staging_dir=staging_dir)
    keyword_result = build_keyword_index(graph, staging_dir=staging_dir)
    logger.info(
        "build_indexes: %d communities, %d keywords",
        len(community_result.communities),
        keyword_result.total_keywords,
        extra={"stage": "build-indexes"},
    )
    return community_result, keyword_result


__all__ = [
    "COLLECTION_NAME",
    "COMMUNITIES_FILENAME",
    "Community",
    "CommunityResult",
    "DEFAULT_SEARCH_K",
    "KEYWORDS_FILENAME",
    "KeywordEntry",
    "KeywordIndex",
    "VectorStore",
    "build_indexes",
    "build_keyword_index",
    "detect_communities",
    "load_communities",
    "load_keyword_index",
]

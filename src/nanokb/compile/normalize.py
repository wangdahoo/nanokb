"""实体归一化（方案 §3.4.4 + §3.5.3 step 3，Medium #10）。

提供 ``normalize_entity``：大小写归一化（lower）+ 连续空白折叠为单空格 + 去前后空格。
``GraphBuilder`` 与 ``GraphRetriever`` 共用此函数，保证抽取端与查询端实体名比对一致
（避免 ``Transformer`` / ``  transformer `` / ``TransFormer`` 漏召回）。

本模块为纯函数工具，不持有状态。``GraphBuilder`` 在 upsert 时保留三元组的原始实体名
（忠实记录抽取结果），归一化由 retriever 在查询比对阶段应用。
"""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_entity(name: str) -> str:
    """归一化实体名：lower + 折叠连续空白 + 去前后空格。

    Medium #10：保证同一实体的不同写法（大小写/空白差异）映射到同一规范形式，
    供 ``GraphRetriever`` 在查图时与图谱节点做规范比对。空字符串原样返回。

    Examples:
        >>> normalize_entity("Transformer")
        'transformer'
        >>> normalize_entity("  Neural   Network ")
        'neural network'
    """
    if not name:
        return ""
    return _WHITESPACE_RE.sub(" ", name).strip().lower()


__all__ = ["normalize_entity"]

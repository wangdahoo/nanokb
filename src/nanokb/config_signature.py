"""三层配置签名 helper（内容寻址）。

来源：技术实施方案 §3.3。三个纯函数将 :class:`Settings` 的相关字段折叠为稳定
的 sha256 签名，供 detector 比对与 manifest 记录用。各层独立，互不干扰：

* ``extraction_config`` —— 决定 ``ExtractionResult``（→ 图谱结构）。
* ``index_config`` —— 决定 fallback 节点描述 + Leiden 社区索引。
* ``embedding_config`` —— 决定 ChromaDB 向量索引。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from nanokb.config import Settings


def _sig(payload: Mapping[str, object]) -> str:
    """稳定内容寻址签名：``sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False))``。

    ``sort_keys=True`` 仅排序 dict 的键；list 元素顺序保留原样，因此调用方对
    集合语义字段（如 ``code_languages``）需在传入前显式 ``sorted``。
    ``ensure_ascii=False`` 保证非 ASCII 配置值在不同平台产生相同签名。
    """
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def extraction_config_signature(s: Settings) -> str:
    """抽取层签名：决定 ``ExtractionResult``（→ 图谱结构）。

    涵盖 ``extractor_version`` / ``chunk_max_tokens`` / ``chunk_overlap_tokens``
    / ``concept_description_strategy`` / ``code_languages``。
    ``code_languages`` 显式 ``sorted`` —— 同集合不同顺序应产生相同签名（集合语义）。
    """
    payload = {
        "extractor_version": s.extractor_version,
        "chunk_max_tokens": s.chunk_max_tokens,
        "chunk_overlap_tokens": s.chunk_overlap_tokens,
        "concept_description_strategy": s.concept_description_strategy,
        "code_languages": sorted(s.code_languages),
    }
    return _sig(payload)


def index_config_signature(s: Settings) -> str:
    """索引层签名：决定 fallback 节点描述 + Leiden 社区。

    涵盖 ``fallback_description_max_edges`` / ``leiden_symmetrize``。
    """
    payload = {
        "fallback_description_max_edges": s.fallback_description_max_edges,
        "leiden_symmetrize": s.leiden_symmetrize,
    }
    return _sig(payload)


def embedding_config_signature(s: Settings) -> str:
    """向量层签名：决定 ChromaDB 向量索引。

    涵盖 ``embedding_model`` / ``embedding_provider``。
    """
    payload = {
        "embedding_model": s.embedding_model,
        "embedding_provider": s.embedding_provider,
    }
    return _sig(payload)


__all__ = [
    "embedding_config_signature",
    "extraction_config_signature",
    "index_config_signature",
]

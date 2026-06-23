"""知识抽取阶段包（方案 §3.4.3 + §3.5.1 step 2b）。

导出：
- ``chunker.chunk_text``：按 max_tokens + overlap 切片，tiktoken 精确计数（s1-feat-004）
- ``base.Extractor`` Protocol：统一 ``extract(doc) -> ExtractionResult`` 契约
  （Severe #2：返回同时携带 triples 与 concepts 的结果，闭合节点描述数据流）
- ``semantic_track.SemanticTrack``：语义轨 LLM 抽取（Opt #1 v3 同名 concept 冲突 last-write-wins）

扩展点：``CodeTrack``（tree-sitter 代码轨）在 s1-feat-010 通过同一 ``Extractor``
接口接入。
"""

from __future__ import annotations

from nanokb.stage2_extract.base import Extractor
from nanokb.stage2_extract.chunker import chunk_text
from nanokb.stage2_extract.semantic_track import SemanticTrack

__all__ = ["Extractor", "SemanticTrack", "chunk_text"]

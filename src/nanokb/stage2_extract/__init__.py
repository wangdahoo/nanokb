"""知识抽取阶段（方案 §3.4.3）。

- ``Extractor`` Protocol：统一 ``extract(doc) -> ExtractionResult`` 契约
  （Severe #2：返回同时携带 triples 与 concepts 的结果，闭合节点描述数据流）。
- ``SemanticTrack``：语义轨，对 ``doc.chunks`` 逐块调用 LLM 抽取 triples + concepts
  并合并跨块结果（Opt #1 v3：同名 concept 描述冲突 last-write-wins）。

扩展点：``CodeTrack``（tree-sitter 代码轨）在 s1-feat-010 通过同一 ``Extractor``
接口接入；``chunker``（分块器）由 s1-feat-004 在本包内补全。
"""

from __future__ import annotations

from nanokb.stage2_extract.base import Extractor
from nanokb.stage2_extract.semantic_track import SemanticTrack

__all__ = ["Extractor", "SemanticTrack"]

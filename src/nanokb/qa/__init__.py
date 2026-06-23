"""阶段 5：问答与主动学习（方案 §3.5.3，Feature s1-feat-009）。

阶段切分：

- **阶段 3 / s1-feat-009（本 feature）**：仅 graph 路（``GraphRetriever``），
  ``ask``/``search`` 为打桩（Opt #5 v3 降级）。
- **阶段 4 / s1-feat-012**：补全 ``VectorRetriever`` / ``CommunityRetriever`` +
  ``MultiRetriever`` 三路融合 + ``build --watch`` + ``search --community``。
- **阶段 5 / s1-feat-013**：完善 ``ReviewQueue`` 持久化与 ``nanokb review`` 命令
  （主动学习闭环）。

本包对外暴露的核心 API：

- ``GraphRetriever``：NER → normalize → 查图 → fuzzy 兜底 → N 跳子图扩展。
- ``compile_context``：把召回 hits 渲染为纯文本上下文（tiktoken 裁剪）。
- ``generate``：生成带 ``^[source_file]`` 引用的 ``Answer``（强制引用 + 推理标记）。
- ``should_flag``：review 判定（OR 触发，Medium #2）。
"""

from __future__ import annotations

from nanokb.qa.generator import generate
from nanokb.qa.prompt import compile_context
from nanokb.qa.retriever import GraphRetriever
from nanokb.qa.review import should_flag

__all__ = [
    "GraphRetriever",
    "compile_context",
    "generate",
    "should_flag",
]

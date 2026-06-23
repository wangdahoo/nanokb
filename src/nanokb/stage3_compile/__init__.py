"""图谱编译阶段包（方案 §3.4.4 + §3.5.1 step 5/6，Feature s1-feat-007）。

导出：
- ``normalize.normalize_entity``：实体归一化（大小写/去空格），
  ``GraphBuilder`` 与 ``GraphRetriever`` 共用（Medium #10）。
- ``graph_builder.GraphBuilder``：在 ``MultiDiGraph`` 上提供 upsert（Medium #9 幂等）、
  ``delete_by_source``（Severe #1 删除传播）、``synthesize_fallback_descriptions``
  （Opt #2 v3 兜底描述，v4 Medium #1 须先于 ``index_nodes``）、``save_graph``
  （JSON 主 ``node_link_data`` + GraphML 副）。
"""

from __future__ import annotations

from nanokb.stage3_compile.graph_builder import GraphBuilder
from nanokb.stage3_compile.normalize import normalize_entity

__all__ = ["GraphBuilder", "normalize_entity"]

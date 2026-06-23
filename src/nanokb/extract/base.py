"""抽取器协议（方案 §3.4.3，Severe #2）。

``Extractor`` 仅声明 ``extract(doc) -> ExtractionResult``：LLM 客户端在具体实现
（如 ``SemanticTrack.__init__``）的构造期注入，使协议本身保持纯粹的数据进出契约
（Opt #1：llm 下沉到 ``__init__``），便于在测试与流水线中替换为 ``FakeLLMClient``
或未来的确定性抽取器（``CodeTrack``）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nanokb.models import Document, ExtractionResult


@runtime_checkable
class Extractor(Protocol):
    """知识抽取器统一契约。

    实现者负责把一篇 ``Document``（含已分块的 ``chunks``）转换为
    ``ExtractionResult(triples, concepts)``：

    - ``triples``：语义三元组列表，携带 ``confidence`` / ``source_file`` /
      ``track`` / ``chunk_index`` 等元数据。
    - ``concepts``：节点概念列表，每项含非空 ``description``（Severe #2 的
      节点描述数据流，下游 ``GraphBuilder`` 与 ``VectorStore`` 据此生成节点向量）。

    实现需保证：解析失败不崩溃（容错 + 降级 AMBIGUOUS）；跨块重复三元组保留
    （去重交给 ``GraphBuilder.upsert``）。
    """

    def extract(self, doc: Document) -> ExtractionResult:
        """从文档中抽取三元组与节点描述。"""
        ...


__all__ = ["Extractor"]

"""知识抽取阶段包（方案 §3.4.3 + §3.5.1 step 2b）。

导出：
- ``chunker.chunk_text``：按 max_tokens + overlap 切片，tiktoken 精确计数（s1-feat-004）
- ``base.Extractor`` Protocol：统一 ``extract(doc) -> ExtractionResult`` 契约
  （Severe #2：返回同时携带 triples 与 concepts 的结果，闭合节点描述数据流）
- ``semantic_track.SemanticTrack``：语义轨 LLM 抽取（Opt #1 v3 同名 concept 冲突 last-write-wins）
- ``code_track.CodeTrack``：代码轨 tree-sitter 确定性抽取（s1-feat-010，零 Token）
- ``build_default_extractor``：按文件扩展名在 SemanticTrack / CodeTrack 间分发的默认抽取器
"""

from __future__ import annotations

import threading

from nanokb.config import Settings
from nanokb.extract.base import Extractor
from nanokb.extract.chunker import chunk_text
from nanokb.extract.code_track import CodeTrack, supported_code_suffixes
from nanokb.extract.semantic_track import SemanticTrack
from nanokb.llm.base import LLMClient
from nanokb.models import Document, ExtractionResult

__all__ = [
    "CodeTrack",
    "DefaultExtractor",
    "Extractor",
    "SemanticTrack",
    "build_default_extractor",
    "chunk_text",
]


class DefaultExtractor:
    """按文件扩展名在语义轨 / 代码轨间分发的默认抽取器。

    - 代码扩展名（``.py`` / ``.js`` / ``.java``）→ ``CodeTrack.extract``（零 Token）。
    - 其余 → ``SemanticTrack.extract``（逐块 LLM 抽取）。

    实现满足 ``Extractor`` Protocol（单一 ``extract`` 方法），供 pipeline 作为单一
    extractor 使用——pipeline 不再按文件挑 extractor，由本类内部按扩展名分发。

    ``CodeTrack`` 构造不依赖 ``llm``；``SemanticTrack`` 按需懒构造（仅当遇到语义轨文件时）。

    线程安全（方案 §3.5，Feature s1-feat-004）：文档级并发下首个语义轨文件并发
    到达时，双重检查锁（DCL）保证 ``SemanticTrack`` 仅构造一次。``SemanticTrack``
    本身是无状态对象（``extract`` 内 triples/merged_concepts 都是方法局部变量），
    单个共享实例跨文档并发调用 ``extract`` 安全。
    """

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings
        self._code_track = CodeTrack(settings)
        self._semantic_track: SemanticTrack | None = None
        self._sem_lock = threading.Lock()  # 保护懒构造（DCL）

    def extract(self, doc: Document) -> ExtractionResult:
        """按 doc 扩展名分发：代码扩展名 → CodeTrack，其余 → SemanticTrack。"""
        if doc.path.suffix.lower() in supported_code_suffixes():
            return self._code_track.extract(doc)
        # 双重检查锁：避免文档级并发首调构造多个 SemanticTrack
        if self._semantic_track is None:
            with self._sem_lock:
                if self._semantic_track is None:
                    self._semantic_track = SemanticTrack(self._llm, self._settings)
        return self._semantic_track.extract(doc)


def build_default_extractor(llm: LLMClient, settings: Settings) -> Extractor:
    """构造默认抽取器：代码轨 + 语义轨按扩展名自动分发。"""
    return DefaultExtractor(llm, settings)

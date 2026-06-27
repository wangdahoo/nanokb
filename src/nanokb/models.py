"""核心数据模型（pydantic v2）。

来源：技术实施方案 §3.2。所有阶段共享的类型在此集中定义，
为后续 feature（抽取/编译/问答）提供契约稳定的数据载体。
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Confidence(str, Enum):
    """三元组置信度三级标注。"""

    EXTRACTED = "EXTRACTED"
    INFERRED = "INFERRED"
    AMBIGUOUS = "AMBIGUOUS"


class Track(str, Enum):
    """知识抽取轨道：语义轨（LLM）或代码轨（tree-sitter）。"""

    SEMANTIC = "semantic"
    CODE = "code"


class Triple(BaseModel):
    """语义三元组：(head, relation, tail) + 元数据。"""

    head: str
    relation: str
    tail: str
    confidence: Confidence
    source_file: str
    track: Track = Track.SEMANTIC
    chunk_index: int | None = None


class Concept(BaseModel):
    """节点概念：name + description（闭合节点描述数据流，Severe #2）。"""

    name: str
    description: str | None = None
    source_file: str
    confidence: Confidence = Confidence.EXTRACTED
    node_type: str = "concept"
    extra: dict[str, object] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Extractor.extract 的统一返回，同时携带三元组与节点描述。"""

    triples: list[Triple] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)


class Chunk(BaseModel):
    """文档分块：索引、文本、token 计数、来源文件。"""

    index: int
    text: str
    token_count: int
    source_file: str


class Document(BaseModel):
    """已加载的文档：原始内容 + sha256 + 分块列表。"""

    path: Path
    content: str
    sha256: str
    format: str
    chunks: list[Chunk] = Field(default_factory=list)


class FileState(BaseModel):
    """manifest 中单个文件的增量状态（含模型身份，Medium #7）。

    三层签名字段（``extraction_config`` / ``index_config`` / ``embedding_config``）
    来自 ``nanokb.config_signature``，供 detector 五维身份比对使用。default ``""``
    保证旧 manifest 反序列化不报错（零迁移）：``""`` ≠ 计算签名 → 首次 compile
    全量 modified 自愈。
    """

    path: str
    sha256: str
    processed_at: str
    # 旧字段（保留，供调试 / VectorStore / 向后兼容）
    extractor_version: str = "1"
    llm_model: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0
    # 新增三层签名（detector 比对用；default "" → 旧 manifest 触发全量 modified）
    extraction_config: str = ""
    index_config: str = ""
    embedding_config: str = ""


class Manifest(BaseModel):
    """增量哈希清单 —— staging 原子切换最后写入的文件。

    round 3（Opt#3 保守变体）：顶层新增 ``total_vectors`` / ``last_compiled_at`` /
    ``last_llm_model`` / ``last_embedding_model`` 四个可选字段（默认值空/0）。
    ``version`` 保持 ``"2"`` 不变——新字段视为「2.x 增量」，Pydantic 可选字段 +
    默认值保证旧 manifest 反序列化不报错（读到默认值时 status 显示「N/A」）。
    这些字段供 ``nanokb status`` 在无运行期进度文件时展示编译统计（向量数 / 模型
    身份），**status 绝不打开 chroma**（保守假设规避跨进程锁，Medium #3 / AC3.4）。
    """

    version: str = "2"
    files: dict[str, FileState] = Field(default_factory=dict)
    # 2.x 增量字段（status 静态展示用；旧 manifest 读默认值显示「N/A」）
    total_vectors: int = 0
    last_compiled_at: str = ""
    last_llm_model: str = ""
    last_embedding_model: str = ""


class RetrievalHit(BaseModel):
    """单路召回命中的中立结构（图/向量/社区三路共用）。"""

    triple: Triple | None = None
    concept: Concept | None = None
    community_summary: str | None = None
    score: float = 0.0
    source: str


class Answer(BaseModel):
    """问答输出：正文 + 引用 + 是否含推理 + review 标记。"""

    text: str
    citations: list[str] = Field(default_factory=list)
    used_inferred: bool = False
    confidence: Confidence
    review_flagged: bool = False


__all__ = [
    "Answer",
    "Chunk",
    "Concept",
    "Confidence",
    "Document",
    "ExtractionResult",
    "FileState",
    "Manifest",
    "RetrievalHit",
    "Track",
    "Triple",
]

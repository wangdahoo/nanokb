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
    """manifest 中单个文件的增量状态（含模型身份，Medium #7）。"""

    path: str
    sha256: str
    processed_at: str
    extractor_version: str = "1"
    llm_model: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0


class Manifest(BaseModel):
    """增量哈希清单 —— staging 原子切换最后写入的文件。"""

    version: str = "2"
    files: dict[str, FileState] = Field(default_factory=dict)


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

"""关键词倒排索引（方案 §3.1 + v4 Opt #1，Feature s1-feat-011）。

``KeywordIndex`` 构建内存倒排索引（keyword → 命中节点列表），持久化为单文件
``keywords.json``（v4 Opt #1：由目录改为单文件，纳入 staging 原子切换五件套）。

关键词来源：节点 ``name`` 整体 + ``description`` 中的词（ASCII 词 2+ 字符、CJK 单字，
小写化，去停用词 / 去标点）。支持中英混合文本。

用途：``search`` 命令的关键词精确召回路（s1-feat-012 ``MultiRetriever`` 可接入）。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from nanokb.utils.io import atomic_write_json

if TYPE_CHECKING:
    import networkx as nx  # type: ignore[import-untyped]

logger = logging.getLogger("nanokb")

#: keywords.json 输出文件名（纳入 staging 原子切换，v4 Opt #1）
KEYWORDS_FILENAME = "keywords.json"

#: ASCII 关键词最小长度（单字符词信息量低，过滤）
MIN_ASCII_TOKEN_LENGTH = 2

#: 倒排索引单关键词最大命中数（避免超高频词膨胀索引）
MAX_HITS_PER_KEYWORD = 200

#: 基础英文停用词（高频无信息量词）
_ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "must", "shall", "can", "need", "dare", "ought", "used",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below", "up", "down",
        "out", "off", "over", "under", "again", "further", "then", "once",
        "and", "or", "but", "if", "while", "an", "no", "nor", "so", "yet",
        "either", "neither", "not", "only", "also", "too", "very",
        "just", "about", "between", "there", "here", "when", "where",
        "why", "how", "all", "each", "every", "few", "more", "most",
        "other", "some", "such", "than", "this", "that", "these", "those",
        "it", "its", "they", "them", "their", "we", "us", "our", "you", "your",
        "he", "him", "his", "she", "her", "my", "me", "what", "which",
        "who", "whom",
    }
)

#: CJK 停用词（高频虚词 / 标点残片）
_CJK_STOPWORDS: frozenset[str] = frozenset(
    {"的", "了", "是", "在", "和", "与", "或", "也", "都", "就", "还", "又",
     "把", "被", "让", "使", "给", "为", "对", "于", "从", "向", "由", "按",
     "这", "那", "其", "之", "而", "以", "及", "等", "但", "如", "若", "则"}
)

#: ASCII 词提取正则（连续字母数字，2 字符起）
_ASCII_WORD_RE = re.compile(
    rf"[A-Za-z][A-Za-z0-9_]{{{MIN_ASCII_TOKEN_LENGTH - 1},}}"
)

#: CJK 字符范围检测（中日韩统一表意文字 + 扩展）
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


class KeywordEntry(BaseModel):
    """倒排索引中单个命中条目（某关键词出现在某节点）。"""

    node: str
    source_file: str = ""
    node_type: str = ""


class KeywordIndex(BaseModel):
    """关键词倒排索引（keyword → 命中条目列表）。"""

    index: dict[str, list[KeywordEntry]] = Field(default_factory=dict)
    total_nodes: int = 0
    total_keywords: int = 0

    def lookup(self, keyword: str) -> list[KeywordEntry]:
        """查询单个关键词（小写化后精确匹配）。

        Returns:
            命中条目列表；关键词不存在返回空列表。
        """
        return self.index.get(keyword.lower().strip(), [])

    def lookup_any(self, keywords: list[str]) -> list[KeywordEntry]:
        """查询多个关键词，返回去重后的全部命中（按命中节点去重，保留首个）。"""
        seen: set[str] = set()
        results: list[KeywordEntry] = []
        for kw in keywords:
            for entry in self.lookup(kw):
                if entry.node not in seen:
                    seen.add(entry.node)
                    results.append(entry)
        return results


def build(
    graph: nx.MultiDiGraph,
    *,
    staging_dir: Path | None = None,
) -> KeywordIndex:
    """从图谱节点构建关键词倒排索引。

    遍历每个节点的 ``name``（整体作为关键词）+ ``description``（分词），构建
    ``keyword → [KeywordEntry]`` 倒排索引。单关键词命中上限 ``MAX_HITS_PER_KEYWORD``
    防止超高频词膨胀。

    Args:
        graph: 知识图谱（``MultiDiGraph``）。
        staging_dir: staging 目录；非 None 时原子写入 ``staging_dir/keywords.json``。

    Returns:
        ``KeywordIndex`` —— 倒排索引 + 统计计数。
    """
    inverted: dict[str, list[KeywordEntry]] = {}

    nodes = list(graph.nodes(data=True))
    for node, data in nodes:
        source_file = str(data.get("source_file", ""))
        node_type = str(data.get("node_type", ""))
        entry = KeywordEntry(node=node, source_file=source_file, node_type=node_type)

        keywords = _extract_keywords_for_node(str(node), data)
        for kw in keywords:
            bucket = inverted.setdefault(kw, [])
            if len(bucket) < MAX_HITS_PER_KEYWORD:
                bucket.append(entry)

    result = KeywordIndex(
        index=inverted,
        total_nodes=len(nodes),
        total_keywords=len(inverted),
    )

    if staging_dir is not None:
        _write_keywords(staging_dir, result)

    logger.info(
        "keyword_index: built %d keywords from %d nodes",
        result.total_keywords,
        result.total_nodes,
        extra={"stage": "keyword-index"},
    )

    return result


def load(out_dir: Path) -> KeywordIndex | None:
    """从 ``out/keywords.json`` 加载关键词索引；文件不存在返回 None。

    供 s1-feat-012 ``MultiRetriever`` / CLI ``search`` 消费。
    """
    path = out_dir / KEYWORDS_FILENAME
    if not path.exists():
        return None
    import json

    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return KeywordIndex.model_validate(data)


# ── 内部辅助 ─────────────────────────────────────────────────────────


def _extract_keywords_for_node(node_name: str, data: dict[str, Any]) -> set[str]:
    """提取节点关键词集合：节点名整体 + 描述分词。

    - 节点名整体（小写化）始终作为关键词（保证精确匹配）。
    - 节点名本身的分词也加入（如 "Transformer" → "transformer"）。
    - description 分词：ASCII 词 2+ 字符 + CJK 单字，去停用词。
    """
    keywords: set[str] = set()

    # 节点名整体（小写化）作为精确匹配关键词
    name_lower = node_name.lower().strip()
    if name_lower and not _is_stopword_ascii(name_lower):
        keywords.add(name_lower)

    # 节点名分词
    keywords |= _tokenize(node_name)

    # description 分词
    description = data.get("description")
    if isinstance(description, str) and description.strip():
        keywords |= _tokenize(description)

    return keywords


def _tokenize(text: str) -> set[str]:
    """混合中英文分词：ASCII 词（2+ 字符）+ CJK 单字，小写化，去停用词。

    - ASCII：``[A-Za-z][A-Za-z0-9_]+``（2 字符起，避免单字符噪声）。
    - CJK：每个汉字单独作为一个 token（简单分词，无分词器依赖）。
    - 数字串（3+ 字符）作为 token（如版本号、技术指标）。
    """
    tokens: set[str] = set()

    # ASCII 词（含数字后缀，如 transformer3）
    for match in _ASCII_WORD_RE.finditer(text):
        word = match.group().lower()
        if not _is_stopword_ascii(word):
            tokens.add(word)

    # 纯数字串（3+ 字符）
    for match in re.finditer(r"\d{3,}", text):
        tokens.add(match.group())

    # CJK 单字
    for char in text:
        if _CJK_CHAR_RE.match(char):
            if char not in _CJK_STOPWORDS:
                tokens.add(char)

    return tokens


def _is_stopword_ascii(word: str) -> bool:
    """英文停用词判定（小写比较）。"""
    return word.lower() in _ENGLISH_STOPWORDS


def _write_keywords(staging_dir: Path, index: KeywordIndex) -> None:
    """原子写入 ``staging_dir/keywords.json``。"""
    staging_dir.mkdir(parents=True, exist_ok=True)
    path = staging_dir / KEYWORDS_FILENAME
    atomic_write_json(path, index.model_dump(mode="json"))


__all__ = [
    "KEYWORDS_FILENAME",
    "KeywordEntry",
    "KeywordIndex",
    "build",
    "load",
]

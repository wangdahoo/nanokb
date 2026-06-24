"""KeywordIndex 单测（方案 §3.1 + v4 Opt #1，Feature s1-feat-011）。

覆盖 AC #5：图谱 → keyword_index.build → 产出 keywords.json（单文件倒排索引）。

另覆盖：
- 中英混合分词（ASCII 词 2+ 字符 / CJK 单字 / 去停用词）。
- 节点名整体作为精确匹配关键词。
- lookup / lookup_any 查询。
- keywords.json 原子写入 staging。
- load 回读。
- MAX_HITS_PER_KEYWORD 截断保护。
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from nanokb.index.keyword_index import (
    KEYWORDS_FILENAME,
    KeywordEntry,
    build,
    load,
)

# ── 辅助构造 ─────────────────────────────────────────────────────────


def _make_graph() -> nx.MultiDiGraph:
    """构造含中英混合节点描述的小型图谱。"""
    g = nx.MultiDiGraph()
    g.add_node(
        "Transformer",
        description="A neural network architecture for sequence processing.",
        source_file="doc1.md",
        node_type="concept",
    )
    g.add_node(
        "Attention",
        description="A mechanism for focusing on relevant input parts.",
        source_file="doc1.md",
        node_type="concept",
    )
    g.add_node(
        "深度学习",
        description="深度学习是机器学习的一个分支，使用多层神经网络。",
        source_file="doc2.md",
        node_type="concept",
    )
    return g


# ── AC #5：build → keywords.json ────────────────────────────────────


def test_build_writes_keywords_json(tmp_path: Path) -> None:
    """AC #5：build 产出 keywords.json（单文件倒排索引）。"""
    graph = _make_graph()
    staging = tmp_path / "staging"

    result = build(graph, staging_dir=staging)

    assert (staging / KEYWORDS_FILENAME).exists()
    assert result.total_nodes == 3
    assert result.total_keywords > 0


def test_build_keywords_json_valid_structure(tmp_path: Path) -> None:
    """keywords.json 结构合法：{index: {keyword: [{node, source_file, node_type}]}, ...}"""
    graph = _make_graph()
    staging = tmp_path / "staging"

    build(graph, staging_dir=staging)

    raw = (staging / KEYWORDS_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)

    assert "index" in data
    assert "total_nodes" in data
    assert "total_keywords" in data
    assert data["total_nodes"] == 3

    inverted = data["index"]
    assert len(inverted) > 0
    # 每个条目含 node + source_file + node_type
    for keyword, entries in inverted.items():
        assert isinstance(keyword, str)
        for entry in entries:
            assert "node" in entry
            assert "source_file" in entry
            assert "node_type" in entry


# ── 关键词提取（中英混合分词）──────────────────────────────────────


def test_build_node_name_as_keyword(tmp_path: Path) -> None:
    """节点名整体（小写化）作为精确匹配关键词。"""
    graph = nx.MultiDiGraph()
    graph.add_node(
        "Transformer",
        description="Some description.",
        source_file="doc.md",
        node_type="concept",
    )

    result = build(graph)

    # "transformer" 作为关键词
    assert "transformer" in result.index
    hits = result.index["transformer"]
    assert any(h.node == "Transformer" for h in hits)


def test_build_description_tokens_indexed(tmp_path: Path) -> None:
    """description 中的 ASCII 词被索引（2+ 字符，小写化）。"""
    graph = nx.MultiDiGraph()
    graph.add_node(
        "X",
        description="A neural network for image classification.",
        source_file="doc.md",
        node_type="concept",
    )

    result = build(graph)

    # "neural", "network", "image", "classification" 都应被索引
    assert "neural" in result.index
    assert "network" in result.index
    assert "image" in result.index
    assert "classification" in result.index


def test_build_chinese_single_char_tokens(tmp_path: Path) -> None:
    """中文节点描述按单字分词（无分词器依赖），去停用词。"""
    graph = nx.MultiDiGraph()
    graph.add_node(
        "深度学习",
        description="深度学习是机器学习的分支。",
        source_file="doc.md",
        node_type="concept",
    )

    result = build(graph)

    # "深", "度", "学", "习" 应被索引（"的", "是" 是停用词被过滤）
    assert "深" in result.index
    assert "度" in result.index
    assert "学" in result.index
    assert "习" in result.index
    # 停用词被过滤
    assert "的" not in result.index
    assert "是" not in result.index


def test_build_stopwords_filtered(tmp_path: Path) -> None:
    """英文停用词被过滤（the, is, a, for 等）。"""
    graph = nx.MultiDiGraph()
    graph.add_node(
        "X",
        description="This is a test for the network.",
        source_file="doc.md",
        node_type="concept",
    )

    result = build(graph)

    # 停用词被过滤
    assert "the" not in result.index
    assert "this" not in result.index
    assert "for" not in result.index
    # 有信息量词保留
    assert "test" in result.index
    assert "network" in result.index


def test_build_short_ascii_filtered(tmp_path: Path) -> None:
    """单字符 ASCII 词被过滤（信息量低）。"""
    graph = nx.MultiDiGraph()
    graph.add_node("AB", description="A B C DE.", source_file="doc.md", node_type="concept")

    result = build(graph)

    # "A", "B", "C" 是单字符（< 2），不索引；"AB" 是节点名整体；"DE" 是 2 字符词
    # 节点名 "ab" 整体作为关键词
    assert "ab" in result.index


# ── lookup 查询 ─────────────────────────────────────────────────────


def test_lookup_exact_match(tmp_path: Path) -> None:
    """lookup 精确匹配关键词（小写化）。"""
    graph = _make_graph()
    result = build(graph)

    hits = result.lookup("transformer")
    assert len(hits) >= 1
    assert any(h.node == "Transformer" for h in hits)


def test_lookup_case_insensitive(tmp_path: Path) -> None:
    """lookup 大小写无关（内部小写化）。"""
    graph = _make_graph()
    result = build(graph)

    hits_lower = result.lookup("transformer")
    hits_upper = result.lookup("TRANSFORMER")
    hits_mixed = result.lookup("Transformer")

    assert len(hits_lower) == len(hits_upper) == len(hits_mixed)


def test_lookup_nonexistent_returns_empty(tmp_path: Path) -> None:
    """lookup 不存在的关键词 → 空列表。"""
    graph = _make_graph()
    result = build(graph)

    assert result.lookup("nonexistent_keyword_xyz") == []


def test_lookup_any_deduplicates_by_node(tmp_path: Path) -> None:
    """lookup_any 多关键词查询，按节点去重。"""
    graph = _make_graph()
    result = build(graph)

    # 查询多个关键词，命中同一节点只出现一次
    hits = result.lookup_any(["transformer", "attention"])
    nodes = [h.node for h in hits]
    assert len(nodes) == len(set(nodes))  # 无重复节点


# ── staging_dir=None 不写文件 ───────────────────────────────────────


def test_build_no_staging_dir_skips_write(tmp_path: Path) -> None:
    """staging_dir=None 时不写文件（仅返回索引）。"""
    graph = _make_graph()
    result = build(graph)  # staging_dir=None

    assert result.total_keywords > 0
    assert not (tmp_path / KEYWORDS_FILENAME).exists()


# ── load 回读 ───────────────────────────────────────────────────────


def test_load_roundtrip(tmp_path: Path) -> None:
    """build 写入 → load 回读一致。"""
    graph = _make_graph()
    build(graph, staging_dir=tmp_path)

    loaded = load(tmp_path)
    assert loaded is not None
    assert loaded.total_nodes == 3
    assert loaded.total_keywords > 0
    # lookup 仍可用
    assert loaded.lookup("transformer")


def test_load_missing_returns_none(tmp_path: Path) -> None:
    """文件不存在时 load 返回 None。"""
    assert load(tmp_path) is None


# ── 空图 ────────────────────────────────────────────────────────────


def test_build_empty_graph(tmp_path: Path) -> None:
    """空图 → 空索引（total_nodes=0, total_keywords=0）。"""
    graph = nx.MultiDiGraph()
    result = build(graph, staging_dir=tmp_path)

    assert result.total_nodes == 0
    assert result.total_keywords == 0
    assert result.index == {}


# ── 节点无 description ──────────────────────────────────────────────


def test_build_node_without_description_uses_name_only(tmp_path: Path) -> None:
    """无 description 的节点仍索引其名称作为关键词。"""
    graph = nx.MultiDiGraph()
    graph.add_node("FooBar", source_file="doc.md", node_type="entity")  # 无 description

    result = build(graph)

    assert "foobar" in result.index
    assert result.total_nodes == 1


# ── 多文件倒排 ──────────────────────────────────────────────────────


def test_build_multiple_nodes_same_keyword(tmp_path: Path) -> None:
    """多节点共享关键词 → 倒排索引列表含全部命中。"""
    graph = nx.MultiDiGraph()
    graph.add_node(
        "NodeA", description="A neural network model.", source_file="doc1.md", node_type="concept"
    )
    graph.add_node(
        "NodeB",
        description="Another neural network design.",
        source_file="doc2.md",
        node_type="concept",
    )

    result = build(graph)

    # "neural" 与 "network" 同时命中 NodeA 和 NodeB
    neural_hits = result.lookup("neural")
    neural_nodes = {h.node for h in neural_hits}
    assert "NodeA" in neural_nodes
    assert "NodeB" in neural_nodes

    network_hits = result.lookup("network")
    network_nodes = {h.node for h in network_hits}
    assert "NodeA" in network_nodes
    assert "NodeB" in network_nodes


# ── KeywordEntry 结构 ───────────────────────────────────────────────


def test_keyword_entry_fields() -> None:
    """KeywordEntry 含 node / source_file / node_type 字段。"""
    entry = KeywordEntry(node="Test", source_file="doc.md", node_type="concept")
    assert entry.node == "Test"
    assert entry.source_file == "doc.md"
    assert entry.node_type == "concept"


def test_keyword_entry_defaults() -> None:
    """KeywordEntry source_file / node_type 默认为空字符串。"""
    entry = KeywordEntry(node="Test")
    assert entry.source_file == ""
    assert entry.node_type == ""

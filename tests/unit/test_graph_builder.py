"""GraphBuilder + normalize_entity 单测（方案 §3.4.4，Feature s1-feat-007）。

覆盖 5 条验收标准：
- AC #1：upsert 插入三元组为边（带 source_file/confidence），A/B 节点带 description
  （来自 concept 或合成）。
- AC #2：同 (source_file,head,relation,tail) 二次 upsert → 该边只有一条（幂等，Medium #9）。
- AC #3：delete_by_source 删该文件全部边 + degree==0 孤立节点，跨文件共享节点保留
  （Severe #1）。
- AC #4：仅作为 triple head 出现无 Concept 的节点 → synthesize_fallback_descriptions
  合成 "{node}: {relation} {tail}; ..." 兜底描述（Opt #2 v3，取前 max_edges 条）。
- AC #5：save_graph 生成 graph.json（node_link_data 保类型）与 graph.graphml。

全部离线，零 LLM 调用。
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from nanokb.compile import GraphBuilder, normalize_entity
from nanokb.config import Settings
from nanokb.models import (
    Concept,
    Confidence,
    ExtractionResult,
    Track,
    Triple,
)

# ── 辅助构造 ─────────────────────────────────────────────────────────


def _triple(
    head: str,
    relation: str,
    tail: str,
    *,
    source_file: str = "doc.md",
    confidence: Confidence = Confidence.EXTRACTED,
    chunk_index: int | None = 0,
) -> Triple:
    return Triple(
        head=head,
        relation=relation,
        tail=tail,
        confidence=confidence,
        source_file=source_file,
        track=Track.SEMANTIC,
        chunk_index=chunk_index,
    )


def _concept(
    name: str,
    description: str,
    *,
    source_file: str = "doc.md",
    node_type: str = "concept",
) -> Concept:
    return Concept(
        name=name,
        description=description,
        source_file=source_file,
        node_type=node_type,
        confidence=Confidence.EXTRACTED,
    )


def _result(
    triples: list[Triple] | None = None,
    concepts: list[Concept] | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        triples=triples or [],
        concepts=concepts or [],
    )


def _edges_with_data(graph: nx.MultiDiGraph, u: str, v: str) -> list[dict[str, object]]:
    """返回 u→v 的全部 multi-edge data（按 key 升序）。"""
    if not graph.has_edge(u, v):
        return []
    return [dict(graph[u][v][key]) for key in sorted(graph[u][v])]


# ── normalize_entity ────────────────────────────────────────────────


def test_normalize_entity_lowercases_and_collapses_whitespace() -> None:
    assert normalize_entity("Transformer") == "transformer"
    assert normalize_entity("  Neural   Network ") == "neural network"
    assert normalize_entity("TransFormer") == "transformer"
    assert normalize_entity("") == ""


# ── AC #1：upsert 插入边 + 节点描述（来自 concept 或合成） ──────────


def test_upsert_creates_edge_with_source_file_and_confidence() -> None:
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    result = _result(
        triples=[_triple("A", "rel", "B", source_file="doc.md")],
        concepts=[
            _concept("A", "entity A description"),
            _concept("B", "entity B description"),
        ],
    )

    gb.upsert(result, "doc.md")

    # 图中存在 A->B 边，带 source_file 与 confidence
    assert graph.has_edge("A", "B")
    edge_data = _edges_with_data(graph, "A", "B")
    assert len(edge_data) == 1
    assert edge_data[0]["source_file"] == "doc.md"
    assert edge_data[0]["confidence"] == "EXTRACTED"
    assert edge_data[0]["relation"] == "rel"

    # A/B 节点带 description（来自 concept）
    assert graph.nodes["A"]["description"] == "entity A description"
    assert graph.nodes["B"]["description"] == "entity B description"
    assert graph.nodes["A"]["node_type"] == "concept"


def test_upsert_node_without_concept_gets_fallback_after_synthesize() -> None:
    # 节点仅出现在 triple（无 Concept）→ upsert 后缺 description，synthesize 后获得兜底描述
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    result = _result(triples=[_triple("A", "uses", "B", source_file="doc.md")])

    gb.upsert(result, "doc.md")

    # upsert 后 A/B 节点存在但 description 缺失（来自 concept 或合成——此处尚无）
    assert graph.has_node("A")
    assert graph.has_node("B")
    assert not graph.nodes["A"].get("description")
    assert not graph.nodes["B"].get("description")

    # synthesize 后 A/B 节点均获得兜底描述
    count = gb.synthesize_fallback_descriptions()
    assert count == 2
    assert graph.nodes["A"]["description"].startswith("A: ")
    assert graph.nodes["B"]["description"].startswith("B: ")


# ── AC #2：同主键二次 upsert 幂等（Medium #9） ───────────────────────


def test_upsert_same_key_twice_is_idempotent() -> None:
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    result = _result(triples=[_triple("A", "rel", "B", source_file="doc.md")])

    gb.upsert(result, "doc.md")
    gb.upsert(result, "doc.md")

    # 该边只有一条（不重复累积）
    edges = _edges_with_data(graph, "A", "B")
    assert len(edges) == 1
    assert graph.number_of_edges() == 1


def test_upsert_same_triple_different_source_keeps_both() -> None:
    # 同 (head, relation, tail) 但不同 source_file → 两条独立边
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(triples=[_triple("A", "rel", "B", source_file="f1.md")]),
        "f1.md",
    )
    gb.upsert(
        _result(triples=[_triple("A", "rel", "B", source_file="f2.md")]),
        "f2.md",
    )

    edges = _edges_with_data(graph, "A", "B")
    assert len(edges) == 2
    sources = {e["source_file"] for e in edges}
    assert sources == {"f1.md", "f2.md"}


def test_upsert_concept_overwrites_description_last_write_wins() -> None:
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(
            triples=[_triple("A", "rel", "B", source_file="doc.md")],
            concepts=[_concept("A", "first description")],
        ),
        "doc.md",
    )
    # 二次 upsert 同节点 A 的 concept → 描述被覆盖
    gb.upsert(
        _result(
            concepts=[_concept("A", "second description")],
        ),
        "doc.md",
    )

    assert graph.nodes["A"]["description"] == "second description"


# ── AC #3：delete_by_source（Severe #1 删除传播） ────────────────────


def test_delete_by_source_removes_edges_and_isolated_nodes() -> None:
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(
            triples=[_triple("A", "rel", "B", source_file="doc.md")],
        ),
        "doc.md",
    )

    gb.delete_by_source("doc.md")

    # 边被删
    assert graph.number_of_edges() == 0
    # 仅被该文件引用的节点（degree==0）被删
    assert not graph.has_node("A")
    assert not graph.has_node("B")


def test_delete_by_source_preserves_cross_file_shared_nodes() -> None:
    # 两文件共享节点 B：f1.md 有 A->B，f2.md 有 C->B；删 f1.md 后 B 保留（f2.md 仍引用）
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(triples=[_triple("A", "rel", "B", source_file="f1.md")]),
        "f1.md",
    )
    gb.upsert(
        _result(triples=[_triple("C", "rel", "B", source_file="f2.md")]),
        "f2.md",
    )

    gb.delete_by_source("f1.md")

    # f1.md 的边被删
    assert not graph.has_edge("A", "B")
    # A 仅被 f1.md 引用 → 删除
    assert not graph.has_node("A")
    # B 跨文件共享（仍被 f2.md 引用，degree>0）→ 保留
    assert graph.has_node("B")
    # f2.md 的边保留
    assert graph.has_edge("C", "B")


def test_delete_by_source_preserves_shared_node_with_bidirectional_edges() -> None:
    # B 既被 f1.md 引用（A→B）又被 f2.md 反向引用（B→D）：删 f1.md 后 B 保留
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(triples=[_triple("A", "rel", "B", source_file="f1.md")]),
        "f1.md",
    )
    gb.upsert(
        _result(triples=[_triple("B", "rel", "D", source_file="f2.md")]),
        "f2.md",
    )

    gb.delete_by_source("f1.md")

    assert not graph.has_node("A")
    assert graph.has_node("B")  # B 仍有出边到 D（f2.md）
    assert graph.has_node("D")
    assert graph.has_edge("B", "D")


def test_delete_by_source_nonexistent_is_noop() -> None:
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(triples=[_triple("A", "rel", "B", source_file="f1.md")]),
        "f1.md",
    )

    gb.delete_by_source("nonexistent.md")

    assert graph.number_of_edges() == 1
    assert graph.has_node("A")
    assert graph.has_node("B")


def test_delete_then_re_upsert_clean_rebuild_medium2() -> None:
    # Medium #2：modified 先清后建——delete_by_source 后再 upsert 新内容
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(triples=[_triple("A", "rel", "B", source_file="doc.md")]),
        "doc.md",
    )

    # 先清
    gb.delete_by_source("doc.md")
    # 再建（modified 后内容变更：实体 A 消失，新增 C->D）
    gb.upsert(
        _result(triples=[_triple("C", "rel", "D", source_file="doc.md")]),
        "doc.md",
    )

    assert not graph.has_node("A")  # 旧实体无残留
    assert not graph.has_node("B")
    assert graph.has_edge("C", "D")


# ── AC #4：synthesize_fallback_descriptions（Opt #2 v3） ─────────────


def test_synthesize_fallback_for_head_only_node() -> None:
    # Foo 仅作为 triple head 出现，无 Concept → 兜底描述 "{node}: {relation} {tail}; ..."
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(
            triples=[
                _triple("Foo", "uses", "Bar", source_file="doc.md"),
                _triple("Foo", "calls", "Baz", source_file="doc.md"),
            ],
        ),
        "doc.md",
    )

    count = gb.synthesize_fallback_descriptions()

    # Foo/Bar/Baz 三个节点均无 Concept → 均合成
    assert count == 3
    foo_desc = graph.nodes["Foo"]["description"]
    assert foo_desc.startswith("Foo: ")
    # 出边组装："uses Bar" 与 "calls Baz" 片段（取前 fallback_description_max_edges=5 条）
    assert "uses Bar" in foo_desc
    assert "calls Baz" in foo_desc


def test_synthesize_fallback_respects_max_edges() -> None:
    # fallback_description_max_edges=2：Foo 有 3 条出边，兜底描述仅含前 2 条
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings(fallback_description_max_edges=2))
    gb.upsert(
        _result(
            triples=[
                _triple("Foo", "r1", "T1", source_file="doc.md"),
                _triple("Foo", "r2", "T2", source_file="doc.md"),
                _triple("Foo", "r3", "T3", source_file="doc.md"),
            ],
        ),
        "doc.md",
    )

    gb.synthesize_fallback_descriptions()

    desc = graph.nodes["Foo"]["description"]
    # 恰好 2 个片段（"; " 分隔 → 1 个分号）
    assert desc.count("; ") == 1
    assert "r1 T1" in desc
    assert "r2 T2" in desc
    assert "r3 T3" not in desc


def test_synthesize_fallback_uses_incoming_edges_when_no_outgoing() -> None:
    # Sink 仅作为 tail（入边），无出边 → 兜底描述用入边的 "{relation} {head}"
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(
            triples=[_triple("Src", "feeds", "Sink", source_file="doc.md")],
        ),
        "doc.md",
    )

    gb.synthesize_fallback_descriptions()

    desc = graph.nodes["Sink"]["description"]
    assert desc.startswith("Sink: ")
    assert "feeds Src" in desc  # 入边片段：relation + neighbor(head)


def test_synthesize_fallback_skips_nodes_with_existing_description() -> None:
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(
            triples=[_triple("A", "rel", "B", source_file="doc.md")],
            concepts=[_concept("A", "explicit description")],  # A 有 Concept 描述
        ),
        "doc.md",
    )

    count = gb.synthesize_fallback_descriptions()

    # 仅 B 合成（A 有显式描述跳过）
    assert count == 1
    assert graph.nodes["A"]["description"] == "explicit description"
    assert graph.nodes["B"]["description"].startswith("B: ")


def test_synthesize_fallback_isolates_no_op() -> None:
    # 无 incident edges 的节点（理论不会出现，防御性测试）→ 不合成
    graph = nx.MultiDiGraph()
    graph.add_node("Lonely")
    gb = GraphBuilder(graph, Settings())

    count = gb.synthesize_fallback_descriptions()

    assert count == 0
    assert "description" not in graph.nodes["Lonely"]


def test_synthesize_fallback_returns_zero_when_all_have_descriptions() -> None:
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(
            triples=[_triple("A", "rel", "B", source_file="doc.md")],
            concepts=[
                _concept("A", "desc A"),
                _concept("B", "desc B"),
            ],
        ),
        "doc.md",
    )

    assert gb.synthesize_fallback_descriptions() == 0


# ── AC #5：save_graph（graph.json node_link_data + graph.graphml） ────


def test_save_graph_writes_json_and_graphml(tmp_path: Path) -> None:
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(
            triples=[_triple("A", "rel", "B", source_file="doc.md")],
            concepts=[_concept("A", "desc A"), _concept("B", "desc B")],
        ),
        "doc.md",
    )

    staging = tmp_path / "staging"
    gb.save_graph(staging)

    graph_json = staging / "graph.json"
    graph_graphml = staging / "graph.graphml"
    assert graph_json.exists()
    assert graph_graphml.exists()

    # graph.json 为 node_link_data，可解析回等价图（保类型）
    data = json.loads(graph_json.read_text(encoding="utf-8"))
    assert data["directed"] is True
    assert data["multigraph"] is True
    node_ids = {n["id"] for n in data["nodes"]}
    assert node_ids == {"A", "B"}
    # JSON 保留 description 等属性类型
    node_a = next(n for n in data["nodes"] if n["id"] == "A")
    assert node_a["description"] == "desc A"

    # graph.graphml 可被 networkx 读回
    g_back = nx.read_graphml(graph_graphml)
    assert set(g_back.nodes()) == {"A", "B"}


def test_save_graph_preserves_dict_attrs_in_json_sidecar(tmp_path: Path) -> None:
    # concept.extra 为 dict → graph.json 保类型保留，graph.graphml 转 JSON 字符串
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    result = _result(
        triples=[_triple("A", "rel", "B", source_file="doc.md")],
        concepts=[
            Concept(
                name="A",
                description="entity A",
                source_file="doc.md",
                node_type="concept",
                confidence=Confidence.EXTRACTED,
                extra={"meta": 42, "tags": ["x", "y"]},
            )
        ],
    )
    gb.upsert(result, "doc.md")

    staging = tmp_path / "staging"
    gb.save_graph(staging)

    # JSON sidecar 保留 dict 原始类型
    data = json.loads((staging / "graph.json").read_text(encoding="utf-8"))
    node_a = next(n for n in data["nodes"] if n["id"] == "A")
    assert node_a["extra"] == {"meta": 42, "tags": ["x", "y"]}

    # GraphML 副本可成功写入（dict 已清洗为 JSON 字符串，不触发 TypeError）
    g_back = nx.read_graphml(staging / "graph.graphml")
    assert "A" in g_back.nodes


def test_save_graph_roundtrip_via_node_link_data(tmp_path: Path) -> None:
    # 端到端往返：save → 读取 graph.json → 重建图 → 比较结构
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    gb.upsert(
        _result(
            triples=[
                _triple("Transformer", "uses", "Attention", source_file="ml.md"),
                _triple("Transformer", "is_a", "Model", source_file="ml.md"),
            ],
            concepts=[_concept("Transformer", "A neural architecture.")],
        ),
        "ml.md",
    )

    staging = tmp_path / "staging"
    gb.save_graph(staging)

    data = json.loads((staging / "graph.json").read_text(encoding="utf-8"))
    rebuilt = nx.node_link_graph(data, directed=True, multigraph=True)

    assert set(rebuilt.nodes()) == {"Transformer", "Attention", "Model"}
    assert rebuilt.has_edge("Transformer", "Attention")
    assert rebuilt.has_edge("Transformer", "Model")
    assert rebuilt.nodes["Transformer"]["description"] == "A neural architecture."


def test_save_graph_creates_staging_dir_if_missing(tmp_path: Path) -> None:
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())
    staging = tmp_path / "nested" / "staging"
    assert not staging.exists()

    gb.save_graph(staging)

    assert (staging / "graph.json").exists()
    assert (staging / "graph.graphml").exists()


# ── 集成：upsert → delete → synthesize → save 全流程 ─────────────────


def test_full_upsert_delete_synthesize_save_flow(tmp_path: Path) -> None:
    # 模拟 pipeline 的 step 5→6→10：两文件 upsert → 删一文件 → 合成兜底 → 序列化
    graph = nx.MultiDiGraph()
    gb = GraphBuilder(graph, Settings())

    gb.upsert(
        _result(
            triples=[
                _triple("A", "rel", "B", source_file="f1.md"),
                _triple("B", "rel", "C", source_file="f1.md"),
            ],
            concepts=[_concept("B", "shared node")],
        ),
        "f1.md",
    )
    gb.upsert(
        _result(
            triples=[_triple("C", "rel", "D", source_file="f2.md")],
        ),
        "f2.md",
    )

    # 删除 f1.md（C 跨文件共享保留，A/B 视情况）
    gb.delete_by_source("f1.md")
    assert not graph.has_node("A")  # 仅 f1.md 引用 → 删除
    assert graph.has_node("C")  # 跨文件共享 → 保留

    # 合成兜底（D 仅出现在 f2.md 的 tail，无 Concept）
    gb.synthesize_fallback_descriptions()

    # 序列化（不崩溃）
    staging = tmp_path / "staging"
    gb.save_graph(staging)
    assert (staging / "graph.json").exists()
    assert (staging / "graph.graphml").exists()

"""``RetrievalSession`` 单测（s2-feat-006 / 优化方案 M3）。

验证库级 RetrievalSession 在同一会话内对 graph / communities / vector_store 各仅
加载一次（懒加载 + 缓存命中），多次 answer / search 复用缓存。用 monkeypatch spy
计数 pipeline 的 loader 调用，避免依赖真实知识库落盘。
"""

from __future__ import annotations

import networkx as nx

import nanokb.session as session_mod
from nanokb import pipeline
from nanokb.config import Settings
from nanokb.index.community import CommunityResult
from nanokb.session import RetrievalSession


class FakeLLMClient:
    """最小 LLM stub：满足 RetrievalSession / loader 对 embed 的需要。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, *args: object, **kwargs: object) -> str:
        self.calls.append("complete")
        return '{"entities": []}'

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _make_session(tmp_path) -> RetrievalSession:
    settings = Settings(out_dir=tmp_path)
    return RetrievalSession(settings, llm=FakeLLMClient())


def test_session_caches_graph_communities_vector_store(monkeypatch, tmp_path) -> None:
    """s2-feat-006：两次 answer 后 graph/communities/vector_store 各仅加载 1 次。"""
    session = _make_session(tmp_path)

    calls = {"graph": 0, "comm": 0, "vs": 0, "answer": 0}

    def fake_load_graph(out_dir):
        calls["graph"] += 1
        return nx.MultiDiGraph()

    def fake_load_communities(out_dir):
        calls["comm"] += 1
        return CommunityResult(communities=[], total_nodes=0)

    def fake_ensure_vector_store(settings, embedder, explicit):
        calls["vs"] += 1
        return None  # 不可用（测试不依赖真实 chroma）

    def fake_answer_query(settings, question, **kwargs):
        calls["answer"] += 1
        # 断言注入的缓存对象就是 session 持有的
        assert kwargs.get("graph") is session._graph
        assert kwargs.get("vector_store") is session._vector_store
        return "ANSWER_RESULT"

    monkeypatch.setattr(pipeline, "_load_graph", fake_load_graph)
    monkeypatch.setattr(session_mod, "load_communities", fake_load_communities)
    monkeypatch.setattr(pipeline, "_ensure_vector_store", fake_ensure_vector_store)
    monkeypatch.setattr(pipeline, "_resolve_embedder", lambda *a, **k: session._llm)
    monkeypatch.setattr(pipeline, "answer_query", fake_answer_query)

    # 第一次 answer：触发各 loader 一次
    assert session.answer("q1", mode="query") == "ANSWER_RESULT"
    assert calls == {"graph": 1, "comm": 1, "vs": 1, "answer": 1}
    assert session._graph_loads == 1
    assert session._communities_loads == 1
    assert session._vector_store_loads == 1

    # 第二次 answer：复用缓存，loader 不再调用
    session.answer("q2", mode="query")
    assert calls == {"graph": 1, "comm": 1, "vs": 1, "answer": 2}
    assert session._graph_loads == 1
    assert session._communities_loads == 1
    assert session._vector_store_loads == 1


def test_session_search_reuses_cached_graph_and_communities(monkeypatch, tmp_path) -> None:
    """s2-feat-006：answer 后 search 复用已缓存的 graph/communities，不重新加载。"""
    session = _make_session(tmp_path)

    calls = {"graph": 0, "comm": 0, "search": 0}

    monkeypatch.setattr(pipeline, "_load_graph", lambda out_dir: (calls.__setitem__("graph", calls["graph"] + 1), nx.MultiDiGraph())[1])
    monkeypatch.setattr(
        session_mod,
        "load_communities",
        lambda out_dir: (calls.__setitem__("comm", calls["comm"] + 1), CommunityResult(communities=[], total_nodes=0))[1],
    )
    monkeypatch.setattr(pipeline, "_ensure_vector_store", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_resolve_embedder", lambda *a, **k: session._llm)
    monkeypatch.setattr(pipeline, "answer_query", lambda *a, **k: "A")
    monkeypatch.setattr(
        pipeline,
        "search_communities",
        lambda *a, **k: (calls.__setitem__("search", calls["search"] + 1), [])[1],
    )

    session.answer("q", mode="query")
    assert calls["graph"] == 1 and calls["comm"] == 1

    session.search("深度学习")
    # search 复用缓存：graph/communities 不重新加载
    assert calls["graph"] == 1 and calls["comm"] == 1
    assert calls["search"] == 1


def test_session_ask_mode_skips_communities(monkeypatch, tmp_path) -> None:
    """s2-feat-006：ask 模式（仅向量）不触发 communities 加载。"""
    session = _make_session(tmp_path)
    calls = {"comm": 0}

    monkeypatch.setattr(pipeline, "_load_graph", lambda out_dir: nx.MultiDiGraph())
    monkeypatch.setattr(
        session_mod,
        "load_communities",
        lambda out_dir: (calls.__setitem__("comm", calls["comm"] + 1), None)[1],
    )
    monkeypatch.setattr(pipeline, "_ensure_vector_store", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_resolve_embedder", lambda *a, **k: session._llm)
    monkeypatch.setattr(pipeline, "answer_query", lambda *a, **k: "A")

    session.answer("q", mode="ask")
    assert calls["comm"] == 0  # ask 模式不加载 communities

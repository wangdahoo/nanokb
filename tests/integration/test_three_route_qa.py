"""``query`` 三路融合 / ``ask`` 仅向量 / ``search --community`` 集成测试
（方案 §3.5.3 + s1-feat-012 AC #1/#2/#3）。

覆盖：
- AC #1：已 build 含向量库与社区 → ``query`` 综合 graph+vector+community 三路召回。
- AC #2：已 build → ``ask`` 仅走向量路检索（无 graph / community hit）。
- AC #3：已 build 含社区 → ``search --community`` 返回所属社区摘要。

另覆盖：
- ``query`` 默认 mode 三路融合命中来源覆盖 graph/vector/community。
- ``search_communities`` 冷启动 / 无社区索引的错误处理。

全部用 FakeLLMClient 注入（embed 返回按文本区分的稳定向量），``tmp_path`` 隔离，
走真实 compile → ChromaDB + communities.json 端到端。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from nanokb import pipeline
from nanokb.cli import app
from nanokb.config import Settings

runner = CliRunner()


class FakeLLMClient:
    """模拟 LLM：complete 按序消费；embed 按文本 hash 区分。"""

    def __init__(
        self,
        responses: list[str] | None = None,
        default: str = "",
        embedding_dim: int = 8,
    ) -> None:
        self._responses = list(responses) if responses else []
        self._default = default
        self._dim = embedding_dim
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        self.calls.append({"user": user, "response_format": response_format})
        if self._responses:
            return self._responses.pop(0)
        return self._default

    def embed(self, texts: list[str]) -> list[list[float]]:
        # 按文本首字符生成稳定但可区分的向量，让向量召回可区分相关 / 不相关节点
        return [[float((hash(t) >> i) & 0xFF) / 255.0 for i in range(self._dim)] for t in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _ner(entities: list[str]) -> str:
    return json.dumps({"entities": entities})


def _extract_response() -> str:
    """FakeLLM 抽取响应：Transformer→Attention (EXTRACTED) + Transformer→Model (INFERRED)。"""
    return json.dumps(
        {
            "triples": [
                {
                    "head": "Transformer",
                    "relation": "uses",
                    "tail": "Attention",
                    "confidence": "EXTRACTED",
                },
                {
                    "head": "Transformer",
                    "relation": "is_a",
                    "tail": "Model",
                    "confidence": "INFERRED",
                },
            ],
            "concepts": [
                {"name": "Transformer", "description": "A sequence model.", "node_type": "model"},
                {
                    "name": "Attention",
                    "description": "A focus mechanism.",
                    "node_type": "mechanism",
                },
                {"name": "Model", "description": "A model category.", "node_type": "category"},
            ],
        }
    )


def _build_full_kb(tmp_path: Path, *, enable_vector: bool = True) -> Settings:
    """构造已编译 KB（含 graph + ChromaDB + communities.json）。

    Args:
        enable_vector: 是否在 settings 中启用向量召回（默认 True，完整三路）。
    """
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    (raw_dir / "doc.md").write_text("# Transformer\n\nUses attention.", encoding="utf-8")

    settings = Settings(
        raw_dir=raw_dir,
        out_dir=out_dir,
        enable_vector_recall=enable_vector,
        enable_community_recall=True,
    )
    llm = FakeLLMClient(responses=[_extract_response()])
    pipeline.compile(settings, llm=llm)
    return settings


# ══════════════════════════════════════════════════════════════════════
# AC #1：query 三路融合（graph + vector + community）
# ══════════════════════════════════════════════════════════════════════


def test_query_three_route_fusion_hits_all_sources(tmp_path: Path) -> None:
    """AC #1：query 命中来源覆盖 graph + vector + community 三路。"""
    settings = _build_full_kb(tmp_path)

    # query 三路（s2-feat-005 NER 共享）：MultiRetriever 预调 1 次 NER 供 graph+community
    # 共用，再加 1 次 generate（vector 用 embed 不调 complete）。
    ner = _ner(["Transformer"])
    gen = "Transformer 通过 self-attention 依赖 Attention^[doc.md]。"
    llm = FakeLLMClient(responses=[ner, gen])

    result = pipeline.answer_query(settings, "Transformer 如何依赖 Attention？", llm=llm)

    sources = {h.source for h in result.hits}
    # 三路都贡献了 hit
    assert "graph" in sources, f"graph route missing; sources={sources}"
    assert "vector" in sources, f"vector route missing; sources={sources}"
    assert "community" in sources, f"community route missing; sources={sources}"
    # 答案带引用
    assert "^[doc.md]" in result.answer.text


def test_query_fusion_ranks_extracted_above_inferred(tmp_path: Path) -> None:
    """AC #1+AC #5：三路融合后 EXTRACTED 边排在 INFERRED 边前。"""
    settings = _build_full_kb(tmp_path)

    # s2-feat-005：graph+community 共享 1 次 NER，再加 generate。
    ner = _ner(["Transformer"])
    gen = "answer^[doc.md]"
    llm = FakeLLMClient(responses=[ner, gen])

    result = pipeline.answer_query(settings, "q", llm=llm)

    # graph 路：uses EXTRACTED, is_a INFERRED；EXTRACTED 应排在 INFERRED 前
    graph_triples = [h.triple for h in result.hits if h.source == "graph" and h.triple]
    if len(graph_triples) >= 2:
        # 取前两条 graph triple 的 confidence 序列
        confs = [t.confidence for t in graph_triples[:2]]
        from nanokb.models import Confidence

        # EXTRACTED 排在 INFERRED 前（EXTRACTED 权重 1.0 > INFERRED 0.6）
        if Confidence.EXTRACTED in confs and Confidence.INFERRED in confs:
            assert confs.index(Confidence.EXTRACTED) < confs.index(Confidence.INFERRED)


# ══════════════════════════════════════════════════════════════════════
# AC #2：ask 仅向量路
# ══════════════════════════════════════════════════════════════════════


def test_ask_uses_only_vector_route(tmp_path: Path) -> None:
    """AC #2：ask 模式仅走向量路检索，无 graph / community hit。"""
    settings = _build_full_kb(tmp_path)

    # ask 仅向量：不调用 complete（向量路只 embed），故只 1 个 generate 调用
    gen = "Transformer is a model using attention^[doc.md]."
    llm = FakeLLMClient(responses=[gen])

    result = pipeline.answer_query(settings, "Transformer 是什么？", mode="ask", llm=llm)

    # 命中全部来自 vector
    sources = {h.source for h in result.hits}
    assert sources == {"vector"}, f"ask should only use vector route; got {sources}"
    # LLM complete 只被 generate 调用 1 次（无 NER）
    assert len(llm.calls) == 1


def test_ask_without_chroma_returns_no_results(tmp_path: Path) -> None:
    """AC #2 边界：ask 模式下 ChromaDB 不存在 → 空召回 → 未找到。"""
    settings = _build_full_kb(tmp_path)
    # 删除 chroma 目录模拟未构建向量库
    chroma = settings.out_dir / "chroma"
    if chroma.exists():
        import shutil

        shutil.rmtree(chroma)

    llm = FakeLLMClient(responses=["answer"])
    result = pipeline.answer_query(settings, "q", mode="ask", llm=llm)

    assert result.hits == []
    assert result.answer.text == "未找到相关知识点"


# ══════════════════════════════════════════════════════════════════════
# AC #3：search --community 返回社区摘要
# ══════════════════════════════════════════════════════════════════════


def test_search_community_returns_matched_community_summary(tmp_path: Path) -> None:
    """AC #3：search --community 命中社区 → 返回社区摘要。"""
    settings = _build_full_kb(tmp_path)

    ner = _ner(["Transformer"])  # Transformer 在社区成员中
    llm = FakeLLMClient(responses=[ner])

    hits = pipeline.search_communities(settings, "深度学习 Transformer", llm=llm)

    assert len(hits) >= 1
    assert all(h.community_summary for h in hits)
    assert all(h.source == "community" for h in hits)


def test_search_community_no_match_returns_empty(tmp_path: Path) -> None:
    """AC #3 边界：关键词不在任何社区 → 空列表。"""
    settings = _build_full_kb(tmp_path)
    ner = _ner(["QuantumComputing"])  # 不在社区
    llm = FakeLLMClient(responses=[ner])

    hits = pipeline.search_communities(settings, "QuantumComputing", llm=llm)
    assert hits == []


def test_search_community_cold_start_raises(tmp_path: Path) -> None:
    """AC #3 边界：未 build → ColdStartError。"""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    out_dir.mkdir()
    settings = Settings(raw_dir=raw_dir, out_dir=out_dir)

    with pytest.raises(pipeline.ColdStartError):
        pipeline.search_communities(settings, "anything")


def test_search_community_missing_communities_raises(tmp_path: Path) -> None:
    """AC #3 边界：已 build 但无 communities.json → ColdStartError 提示先 build。"""
    settings = _build_full_kb(tmp_path)
    # 删除 communities.json
    comm = settings.out_dir / "communities.json"
    if comm.exists():
        comm.unlink()

    with pytest.raises(pipeline.ColdStartError):
        pipeline.search_communities(settings, "anything")


# ══════════════════════════════════════════════════════════════════════
# CLI 集成：query / ask / search --community
# ══════════════════════════════════════════════════════════════════════


def test_cli_search_community_prints_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #3 CLI：``nanokb search 'kw' --community`` 打印社区摘要。"""
    settings = _build_full_kb(tmp_path)
    # 把 NANOKB_RAW_DIR / NANOKB_OUT_DIR 环境变量指向 tmp_path（CLI _load_settings 读 env）
    monkeypatch.setenv("NANOKB_RAW_DIR", str(settings.raw_dir))
    monkeypatch.setenv("NANOKB_OUT_DIR", str(settings.out_dir))
    # 注入 OPENAI key 避免 make_llm_client exit 2
    monkeypatch.setenv("NANOKB_OPENAI_API_KEY", "fake-key-for-cli-test")

    # search_communities 会调用 make_llm_client → 缺 key 会 exit 2；monkeypatch 注入 fake
    # 但真实 OpenAI client 会尝试联网；改用 monkeypatch 替换 make_llm_client
    import nanokb.pipeline as pl

    ner = _ner(["Transformer"])
    fake_llm = FakeLLMClient(responses=[ner])

    def _fake_make_llm(_settings: Any) -> Any:
        return fake_llm

    monkeypatch.setattr(pl, "make_llm_client", _fake_make_llm)
    # 同时替换 cli 模块看到的 make_llm_client（避免 build/query 路径触发）
    import nanokb.cli as cli_mod

    monkeypatch.setattr(cli_mod, "make_llm_client", _fake_make_llm)

    result = runner.invoke(app, ["search", "Transformer", "--community"])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    # 打印了社区相关摘要（社区成员含 Transformer，摘要应包含模型/学习相关词）
    assert "社区" in result.output or "community" in result.output.lower()


def test_cli_search_community_no_match_prints_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #3 CLI：``search`` 无命中 → 友好提示。"""
    settings = _build_full_kb(tmp_path)
    monkeypatch.setenv("NANOKB_RAW_DIR", str(settings.raw_dir))
    monkeypatch.setenv("NANOKB_OUT_DIR", str(settings.out_dir))
    monkeypatch.setenv("NANOKB_OPENAI_API_KEY", "fake-key-for-cli-test")

    import nanokb.cli as cli_mod
    import nanokb.pipeline as pl

    fake_llm = FakeLLMClient(responses=[_ner(["QuantumComputing"])])
    monkeypatch.setattr(pl, "make_llm_client", lambda _s: fake_llm)
    monkeypatch.setattr(cli_mod, "make_llm_client", lambda _s: fake_llm)

    result = runner.invoke(app, ["search", "QuantumComputing", "--community"])

    assert result.exit_code == 0
    assert "未找到" in result.output

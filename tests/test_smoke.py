"""Smoke tests —— 验证包可导入、配置与数据模型默认值、CLI 六命令可路由。

覆盖 Phase 0 验收标准中 "至少 1 个 smoke test 通过" 的要求。
"""

from __future__ import annotations

from typer.testing import CliRunner

import nanokb
from nanokb.cli import app
from nanokb.config import Settings
from nanokb.logging_setup import get_logger, setup_logging
from nanokb.models import (
    Answer,
    Chunk,
    Concept,
    Confidence,
    Document,
    ExtractionResult,
    FileState,
    Manifest,
    RetrievalHit,
    Track,
    Triple,
)

runner = CliRunner()


def test_package_version_is_defined() -> None:
    assert nanokb.__version__ == "0.1.0"


def test_all_models_are_importable() -> None:
    # 触发各模型基本构造，确认 pydantic 注册无误
    triple = Triple(
        head="Transformer",
        relation="depends_on",
        tail="Attention",
        confidence=Confidence.EXTRACTED,
        source_file="paper.md",
    )
    assert triple.track == Track.SEMANTIC
    assert triple.chunk_index is None

    concept = Concept(name="Attention", source_file="paper.md")
    assert concept.node_type == "concept"
    assert concept.description is None

    result = ExtractionResult(triples=[triple], concepts=[concept])
    assert len(result.triples) == 1
    assert len(result.concepts) == 1

    chunk = Chunk(index=0, text="hi", token_count=1, source_file="paper.md")
    document = Document(
        path="paper.md",  # type: ignore[arg-type]
        content="hi",
        sha256="0" * 64,
        format="md",
        chunks=[chunk],
    )
    assert document.chunks[0].token_count == 1

    file_state = FileState(path="paper.md", sha256="0" * 64, processed_at="2026-01-01T00:00:00Z")
    assert file_state.extractor_version == "1"

    manifest = Manifest()
    assert manifest.version == "2"
    assert manifest.files == {}

    hit = RetrievalHit(source="graph", score=0.9)
    assert hit.triple is None

    answer = Answer(text="answer", confidence=Confidence.INFERRED)
    assert answer.used_inferred is False
    assert answer.review_flagged is False


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.chunk_max_tokens == 3000
    assert settings.chunk_overlap_tokens == 200
    assert settings.code_languages == ["python", "javascript", "java"]
    assert settings.concept_description_strategy == "last_write_wins"
    assert settings.leiden_symmetrize == "sum"


def test_logging_setup_does_not_raise(tmp_path) -> None:  # type: ignore[no-untyped-def]
    logger = setup_logging(tmp_path / "out", verbose=True)
    assert logger is get_logger()
    logger.info("smoke log line", extra={"stage": "test", "file": "smoke.py"})


def test_cli_help_lists_six_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.stdout
    for cmd in ("build", "query", "ask", "search", "status", "review"):
        assert cmd in result.stdout, f"missing {cmd} in help output"


def test_cli_status_empty_state(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # raw/ 与 out/ 均不存在 → 空状态输出
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "raw/" in result.stdout
    assert "0" in result.stdout
    assert "未编译" in result.stdout

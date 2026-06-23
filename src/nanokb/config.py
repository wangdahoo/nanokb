"""配置管理（pydantic-settings）。

来源：技术实施方案 §3.3。通过环境变量（前缀 NANOKB_）与 .env 文件覆盖默认值。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_env_file() -> Path:
    """定位项目根目录的 ``.env``，锚定到 ``pyproject.toml`` 所在位置。

    pydantic-settings 默认按 CWD 解析相对 ``env_file``，导致从非项目根目录
    运行时读不到 ``.env``。这里向上查找 ``pyproject.toml`` 作为项目根锚点，
    使 ``.env`` 加载与运行目录无关。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent / ".env"
    return here.parent.parent / ".env"


class Settings(BaseSettings):
    """Nano KB 全局配置。

    所有字段均可通过 ``NANOKB_<FIELD_NAME>`` 环境变量覆盖，
    或在项目根 ``.env`` 文件中配置（``env_nested_delimiter="__"``）。
    ``.env`` 的定位与运行目录无关，始终锚定到项目根。
    """

    model_config = SettingsConfigDict(
        env_prefix="NANOKB_",
        env_file=_project_env_file(),
        env_nested_delimiter="__",
        extra="ignore",
    )

    # ── 目录 ────────────────────────────────────────────────────────
    raw_dir: Path = Path("raw")
    out_dir: Path = Path("out")

    # ── LLM ─────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    llm_model: str = "glm-5.1"
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None  # OpenAI 兼容端点（如智谱 GLM https://open.bigmodel.cn/api/paas/v4）
    anthropic_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    # ── Embedding ───────────────────────────────────────────────────
    embedding_provider: Literal["openai", "ollama"] = "openai"
    embedding_model: str = "text-embedding-3-small"

    # ── 图谱 ────────────────────────────────────────────────────────
    graph_serialization: Literal["json", "graphml"] = "json"
    extractor_version: str = "1"

    # ── 分块 ────────────────────────────────────────────────────────
    chunk_max_tokens: int = 3000
    chunk_overlap_tokens: int = 200

    # ── 代码轨 ──────────────────────────────────────────────────────
    code_languages: list[str] = ["python", "javascript", "java"]

    # ── 抽取策略 ────────────────────────────────────────────────────
    concept_description_strategy: Literal["last_write_wins", "concat_dedup"] = "last_write_wins"
    fallback_description_max_edges: int = 5
    leiden_symmetrize: Literal["sum", "max"] = "sum"

    # ── 检索/问答 ───────────────────────────────────────────────────
    retrieval_hops: int = 2
    max_context_tokens: int = 4000
    enable_vector_recall: bool = True
    enable_community_recall: bool = True
    fuzzy_match_cutoff: float = 0.8
    min_hit_count: int = 3
    min_confidence_score: float = 0.3


__all__ = ["Settings"]

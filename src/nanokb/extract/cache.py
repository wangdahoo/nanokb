"""内容寻址抽取缓存（方案 §3.5 / plan §3.3）。

按 ``sha256|extraction_config|llm_model`` 三维 key 缓存 ``ExtractionResult``，
落盘到 ``out/extract_cache/<key>.json``。key 不含 source_file，故同内容不同路径
自动共享同一缓存条目；correctness 由 pipeline 的 ``_normalize_result_source``
在加载时覆盖 source_file 兜底。best-effort：可删可重建，解析失败视为 miss。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from nanokb.models import ExtractionResult
from nanokb.utils.io import atomic_write_json

logger = logging.getLogger("nanokb")


class ExtractionCache:
    """内容寻址抽取缓存：``<cache_dir>/<key>.json``。

    key = ``sha256(f"{sha256}|{extraction_config}|{llm_model}")``，不含 source_file
    （同内容不同路径自动共享；correctness 由 pipeline._normalize_result_source
    在加载时覆盖 rel_key 保证）。best-effort：可删可重建，解析失败视为 miss。
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = Path(cache_dir)

    def _key(self, sha256: str, extraction_config: str, llm_model: str) -> str:
        raw = f"{sha256}|{extraction_config}|{llm_model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, sha256: str, extraction_config: str, llm_model: str) -> ExtractionResult | None:
        path = self._dir / f"{self._key(sha256, extraction_config, llm_model)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ExtractionResult.model_validate(data)
        except Exception:
            # 覆盖 JSONDecodeError / pydantic ValidationError / OSError 等损坏场景，
            # 退化为 miss（best-effort，不阻断主线）。
            logger.debug("extract cache miss (corrupt/unreadable): %s", path)
            return None

    def put(
        self,
        sha256: str,
        extraction_config: str,
        llm_model: str,
        result: ExtractionResult,
    ) -> None:
        path = self._dir / f"{self._key(sha256, extraction_config, llm_model)}.json"
        atomic_write_json(path, result.model_dump(mode="json"))


__all__ = ["ExtractionCache"]

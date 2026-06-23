"""UnstructuredLoader：多格式文档文本抽取（方案 §3.4.2 + AC #1/#2）。

- ``.md`` / ``.txt``：直接 ``Path.read_text(encoding='utf-8')``，不依赖 unstructured。
- ``.pdf`` / ``.docx``：走 ``unstructured.partition.auto.partition``，按元素 ``text``
  以空行连接为完整文本（懒导入，避免拖慢包导入与冷启动）。

``supports`` 判定扩展名是否落在 text / binary 集合内；直接对不支持的扩展名调用
``load`` 抛 ``UnsupportedFormatError``，与 ``LoaderRegistry`` 错误口径一致。
"""

from __future__ import annotations

import logging
from pathlib import Path

from nanokb.loaders.base import UnsupportedFormatError

logger = logging.getLogger("nanokb")

# 纯文本扩展名（直接 read_text，不经 unstructured）
_TEXT_EXTS: frozenset[str] = frozenset({".md", ".txt"})

# 二进制/富文本扩展名（交由 unstructured.partition 抽取）
_UNSTRUCTURED_EXTS: frozenset[str] = frozenset({".pdf", ".docx"})


class UnstructuredLoader:
    """处理 .md/.txt（纯文本直读）与 .pdf/.docx（unstructured.partition）。

    扩展名集合通过构造参数可覆写，便于测试与未来扩展（如新增 .rst/.org）。
    """

    def __init__(
        self,
        *,
        text_extensions: frozenset[str] = _TEXT_EXTS,
        binary_extensions: frozenset[str] = _UNSTRUCTURED_EXTS,
    ) -> None:
        self._text_exts = text_extensions
        self._binary_exts = binary_extensions

    def supports(self, path: Path) -> bool:
        """扩展名命中 text 或 binary 集合即声明支持（大小写不敏感）。"""
        suffix = path.suffix.lower()
        return suffix in self._text_exts or suffix in self._binary_exts

    def load(self, path: Path) -> str:
        """加载文件并返回纯文本。

        ``.md``/``.txt`` 直接 ``read_text``；``.pdf``/``.docx`` 走 unstructured.partition。
        其余扩展名抛 ``UnsupportedFormatError``（registry 路径下不会到达——supports 已过滤）。
        """
        suffix = path.suffix.lower()
        if suffix in self._text_exts:
            return path.read_text(encoding="utf-8")
        if suffix in self._binary_exts:
            return self._partition_with_unstructured(path)
        raise UnsupportedFormatError(
            f"UnstructuredLoader does not support extension {suffix!r}: {path}"
        )

    @staticmethod
    def _partition_with_unstructured(path: Path) -> str:
        """懒加载并调用 ``unstructured.partition.auto.partition``。

        懒导入使包导入零成本、冷启动不触发 unstructured 重依赖；调用时才解析模块属性，
        因此测试用例可通过 ``monkeypatch.setattr(unstructured.partition.auto, "partition", fake)``
        在调用点打桩。元素 ``text`` 以空行拼接，空文本元素被跳过。
        """
        from unstructured.partition.auto import partition

        elements = partition(filename=str(path))
        parts = [str(el.text) for el in elements if getattr(el, "text", None)]
        return "\n\n".join(parts)


__all__ = ["UnstructuredLoader"]

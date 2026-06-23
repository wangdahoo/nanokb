"""CodeLoader：源代码原文加载（方案 §3.4.2 + s1-feat-010 AC #1）。

处理 ``.py`` / ``.js`` / ``.java`` 三种扩展名，直接 ``read_text`` 返回源码原文，
供 ``CodeTrack`` 用 tree-sitter 做确定性抽取（零 Token）。与 ``UnstructuredLoader``
并列注册到 ``LoaderRegistry``，二者支持的扩展名集合互斥。

``supports`` 判定扩展名是否落在受支持集合内（大小写不敏感）；扩展名集合可通过
构造参数覆写，便于测试与未来扩展（如新增 ``.ts`` / ``.go``）。
"""

from __future__ import annotations

from pathlib import Path

from nanokb.loaders.base import UnsupportedFormatError

#: 受支持的代码扩展名（与 detector.SUPPORTED_SUFFIXES / config.code_languages 对齐）
_CODE_EXTS: frozenset[str] = frozenset({".py", ".js", ".java"})


class CodeLoader:
    """源代码原文加载器：``.py`` / ``.js`` / ``.java`` 直读为纯文本。

    不做任何解析或转换——解析由 ``CodeTrack`` 负责（tree-sitter 确定性抽取）。
    本 loader 仅保证源码字节以 UTF-8 文本形式进入 ``Document.content``。
    """

    def __init__(self, *, extensions: frozenset[str] = _CODE_EXTS) -> None:
        self._exts = extensions

    def supports(self, path: Path) -> bool:
        """扩展名命中受支持集合即声明支持（大小写不敏感）。"""
        return path.suffix.lower() in self._exts

    def load(self, path: Path) -> str:
        """加载源码文件并返回 UTF-8 纯文本。

        Raises:
            UnsupportedFormatError: 扩展名不在受支持集合（registry 路径下不会到达——
                ``supports`` 已过滤）。
        """
        suffix = path.suffix.lower()
        if suffix not in self._exts:
            raise UnsupportedFormatError(
                f"CodeLoader does not support extension {suffix!r}: {path}"
            )
        return path.read_text(encoding="utf-8")


__all__ = ["CodeLoader"]

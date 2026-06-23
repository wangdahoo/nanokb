"""文档加载器抽象层（方案 §3.4.2 + §3.6）。

- ``DocumentLoader`` Protocol + ``LoaderRegistry``（按注册顺序选首个 supports 的 loader）
- ``UnstructuredLoader``：.md/.txt 直读 + .pdf/.docx 走 unstructured.partition
- ``CodeLoader``：.py/.js/.java 直读源码原文（供 CodeTrack tree-sitter 确定性抽取）
- ``UnsupportedFormatError``：无 loader 支持时由 ``LoaderRegistry.load`` 抛出
"""

from __future__ import annotations

from nanokb.loaders.base import (
    DocumentLoader,
    LoaderRegistry,
    UnsupportedFormatError,
)
from nanokb.loaders.code_loader import CodeLoader
from nanokb.loaders.unstructured_loader import UnstructuredLoader

__all__ = [
    "CodeLoader",
    "DocumentLoader",
    "LoaderRegistry",
    "UnstructuredLoader",
    "UnsupportedFormatError",
]

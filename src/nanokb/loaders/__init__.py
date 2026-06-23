"""文档加载器抽象层（方案 §3.4.2 + §3.6）。

- ``DocumentLoader`` Protocol + ``LoaderRegistry``（按注册顺序选首个 supports 的 loader）
- ``UnstructuredLoader``：.md/.txt 直读 + .pdf/.docx 走 unstructured.partition
- ``UnsupportedFormatError``：无 loader 支持时由 ``LoaderRegistry.load`` 抛出

扩展点：``LoaderRegistry.register`` 接受任意 ``DocumentLoader`` 实现；
``CodeLoader``（.py/.js/.java）在 s1-feat-010 通过同一接口接入。
"""

from __future__ import annotations

from nanokb.loaders.base import (
    DocumentLoader,
    LoaderRegistry,
    UnsupportedFormatError,
)
from nanokb.loaders.unstructured_loader import UnstructuredLoader

__all__ = [
    "DocumentLoader",
    "LoaderRegistry",
    "UnstructuredLoader",
    "UnsupportedFormatError",
]

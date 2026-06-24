"""文档加载器抽象层（方案 §3.4.2 + §3.6）。

定义：
- ``DocumentLoader`` Protocol（``supports`` / ``load``，``@runtime_checkable``）
- ``UnsupportedFormatError``：无 loader 支持时由 ``LoaderRegistry.load`` 抛出
- ``LoaderRegistry``：按注册顺序遍历 loader，``load`` 选首个 ``supports`` 的 loader 结果；
  全部不支持则抛 ``UnsupportedFormatError``

扩展点：``LoaderRegistry.register`` 接受任意实现 ``DocumentLoader`` 的对象。
``CodeLoader``（.py/.js/.java，方案 s1-feat-010）通过同一接口接入，本 feature 不实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


class UnsupportedFormatError(Exception):
    """请求的文件扩展名无注册 loader 支持（方案 §3.6）。"""


@runtime_checkable
class DocumentLoader(Protocol):
    """文档加载器统一契约。

    ``supports`` 判定能否处理给定路径（通常按扩展名）；``load`` 抽取并返回纯文本。
    注册到 ``LoaderRegistry`` 后，由 registry 按 ``supports`` 结果分发。
    """

    def supports(self, path: Path) -> bool:
        """是否支持加载该路径（通常按扩展名判定）。"""
        ...

    def load(self, path: Path) -> str:
        """加载并返回文件纯文本内容。"""
        ...


class LoaderRegistry:
    """按注册顺序选首个 ``supports`` 的 loader 完成分发。

    ``register`` 追加 loader 到内部有序列表；``load`` 自前向后遍历，首个
    ``supports(path)`` 为真的 loader 的返回值即为结果。若所有 loader 均不声明支持，
    抛 ``UnsupportedFormatError``（方案 §3.6 错误处理：跳过并记日志由上层负责）。
    """

    def __init__(self) -> None:
        self._loaders: list[DocumentLoader] = []

    def register(self, loader: DocumentLoader) -> None:
        """追加一个 loader 到注册表末尾（后注册者优先级低）。"""
        self._loaders.append(loader)

    @property
    def loaders(self) -> tuple[DocumentLoader, ...]:
        """已注册 loader 的只读视图（用于测试与诊断）。"""
        return tuple(self._loaders)

    def load(self, path: Path) -> str:
        """返回首个 ``supports`` 的 loader 的加载结果；无 loader 支持时抛错。"""
        for loader in self._loaders:
            if loader.supports(path):
                return loader.load(path)
        raise UnsupportedFormatError(f"no registered loader supports this file: {path}")


__all__ = ["DocumentLoader", "LoaderRegistry", "UnsupportedFormatError"]

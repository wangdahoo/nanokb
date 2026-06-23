"""检索进度报告器（提升 query/ask/search 命令的交互体验）。

定义 ``ProgressReporter`` 协议：pipeline / MultiRetriever 在各检索阶段调用
``stage(message)`` 进入一个上下文管理器（退出即表示该阶段完成）。CLI 注入基于
``rich`` 的实现（``RichProgressReporter``，见 ``cli.py``），库调用 / 测试不注入时
使用 ``NullProgressReporter``（空实现，行为与无进度反馈时完全一致）。

**解耦约定**：本模块为纯领域层，不导入 ``rich``。Rich 表现层实现放在 ``cli.py``，
保证 retriever / pipeline 不耦合任何终端 UI 库。

**stage 粒度**（由调用方编排，非本模块决定）：
- ``answer_query``：加载知识库 → 各路召回 → 融合重排 → 构建上下文 → 生成答案。
- ``MultiRetriever.recall``：每个 retriever 一路 stage（按 ``SOURCE`` 取标签），fuse 独立 stage。
- ``search_communities``：加载社区索引 → 社区召回。
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

#: 各路 retriever 的中文标签（``SOURCE`` → 友好名称），供 stage 消息 / CLI 摘要复用。
_SOURCE_LABELS: dict[str, str] = {
    "graph": "图谱召回",
    "vector": "向量召回",
    "community": "社区召回",
}


class ProgressReporter(Protocol):
    """检索阶段进度报告协议。

    实现方提供 ``stage(message)``，返回一个上下文管理器：进入表示阶段开始，
    退出表示阶段完成。``NullProgressReporter`` 为空实现；``RichProgressReporter``
    （CLI 层）基于 ``rich.console.Console.status`` 实现带 spinner + 耗时的持久日志。
    """

    def stage(self, message: str) -> AbstractContextManager[None]:
        """进入一个检索阶段（如 ``"图谱召回中..."``），退出即完成该阶段。"""
        ...


class NullProgressReporter:
    """空进度报告器：``stage`` 返回不产生任何副作用的上下文管理器。

    作为 ``progress=None`` 时的默认值，保证库调用 / 测试与"无进度反馈"行为一致。
    """

    def stage(self, message: str) -> AbstractContextManager[None]:
        return nullcontext()


__all__ = ["NullProgressReporter", "ProgressReporter"]

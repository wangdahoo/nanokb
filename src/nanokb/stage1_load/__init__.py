"""stage1_load —— 增量检测 + 文档加载编排阶段包。

导出（s1-feat-004）：
- ``detector.detect_changes``：比对 manifest 四维身份返回 ChangeSet（added/modified/deleted）
- ``detector.ChangeSet``：三互斥集合
- ``detector.WatchQueue``：watchdog debounce + queue 事件处理器
- ``detector.start_watch``：启动 Observer + 单 worker 串行消费（Medium #3 并发安全核心）
- ``detector.WatchContext``：监听上下文（stop() 清理资源）
- ``ingest.ingest``：编排 detect → load → chunk 填充 Document.chunks
- ``ingest.ingest_file``：加载单个文件为 Document
- ``ingest.IngestResult``：ingest 输出（changes + documents）
"""

from __future__ import annotations

from nanokb.stage1_load.detector import (
    DEBOUNCE_SECONDS,
    SUPPORTED_SUFFIXES,
    ChangeSet,
    WatchContext,
    WatchQueue,
    detect_changes,
    start_watch,
)
from nanokb.stage1_load.ingest import IngestResult, ingest, ingest_file

__all__ = [
    "DEBOUNCE_SECONDS",
    "SUPPORTED_SUFFIXES",
    "ChangeSet",
    "IngestResult",
    "WatchContext",
    "WatchQueue",
    "detect_changes",
    "ingest",
    "ingest_file",
    "start_watch",
]

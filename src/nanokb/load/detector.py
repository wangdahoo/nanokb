"""增量检测与 watchdog 监听（方案 §3.5.1 step 1 + Medium #7 + §3.5.2 watch）。

两块职责：

1. **detect_changes**：比对 raw_dir 与 manifest，返回 ``ChangeSet{added, modified, deleted}``
   三互斥集合。判定 modified 的五维身份（任一变更即触发重抽取）：
   - ``sha256``（文件内容变更）
   - ``extraction_config``（抽取配置签名：extractor_version / chunk_* /
     concept_description_strategy / code_languages 折叠）
   - ``llm_model``（LLM 身份变更 → 抽取结果可能不同）
   - ``index_config``（索引层签名：fallback_description_max_edges / leiden_symmetrize）
   - ``embedding_config``（向量层签名：embedding_model / embedding_provider）

2. **WatchQueue / start_watch**：watchdog Observer 回调仅累计事件 + debounce 500ms，
   到期将本窗口内所有变更路径并入 ``queue.Queue``；单工作线程串行消费（方案 §3.5.2
   Medium #3 并发安全核心：回调只入队，compile worker 单线程消费，内存 graph 线程独占，
   无需加锁）。
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from nanokb.config import Settings
from nanokb.config_signature import (
    embedding_config_signature,
    extraction_config_signature,
    index_config_signature,
)
from nanokb.models import Manifest
from nanokb.utils.hashing import sha256_file

logger = logging.getLogger("nanokb")

# 受监听 / 增量扫描的文档扩展名（与 cli.py _SUPPORTED_SUFFIXES 对齐）
SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {".md", ".txt", ".pdf", ".docx", ".py", ".js", ".java"}
)

# debounce 窗口（秒）—— 方案 §3.5.2 拍定 500ms
DEBOUNCE_SECONDS: float = 0.5


class ChangeSet(BaseModel):
    """增量检测结果：added / modified / deleted 三互斥集合。

    三个列表均为相对 ``raw_dir`` 的路径字符串（与 ``Manifest.files`` key 口径一致），
    按字典序排序保证可复现。集合构造上互斥（一个文件不可能同时出现在两个集合中）。
    """

    added: list[str] = Field(default_factory=list)
    modified: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """是否存在任何变更（无变更时 compile 可直接跳过）。"""
        return bool(self.added or self.modified or self.deleted)


def _is_supported(path: str | Path) -> bool:
    """路径扩展名是否受支持。"""
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def _iter_supported_files(root: Path) -> list[Path]:
    """递归列出 raw_dir 下受支持扩展名的文件（按路径排序，保证可复现）。"""
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _rel_key(path: Path, root: Path) -> str:
    """返回相对 raw_dir 的路径字符串（manifest key 约定）。"""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def detect_changes(raw_dir: Path, manifest: Manifest, settings: Settings) -> ChangeSet:
    """比对 raw_dir 与 manifest，返回三互斥集合。

    判定规则：
    - 磁盘有、manifest 无 → ``added``
    - 磁盘无、manifest 有 → ``deleted``
    - 都有但五维身份（sha256 / extraction_config / llm_model / index_config /
      embedding_config）任一变更 → ``modified``
    - 都有且五维全部一致 → 不在任何集合（unchanged）

    三集合构造上互斥（added ∩ modified = ∅，因为 added 要求 manifest 中无记录）。
    """
    current_paths = _iter_supported_files(raw_dir)
    current_keys: dict[str, Path] = {_rel_key(p, raw_dir): p for p in current_paths}

    extraction_sig = extraction_config_signature(settings)
    index_sig = index_config_signature(settings)
    embedding_sig = embedding_config_signature(settings)

    added: list[str] = []
    modified: list[str] = []

    for rel_key, path in current_keys.items():
        state = manifest.files.get(rel_key)
        if state is None:
            added.append(rel_key)
            continue
        # 五维身份比对（任一变更 → modified）
        digest = sha256_file(path)
        if (
            state.sha256 != digest
            or state.extraction_config != extraction_sig
            or state.llm_model != settings.llm_model
            or state.index_config != index_sig
            or state.embedding_config != embedding_sig
        ):
            modified.append(rel_key)

    # deleted：manifest 有记录但磁盘不存在
    deleted: list[str] = [
        rel_key for rel_key in manifest.files if rel_key not in current_keys
    ]

    return ChangeSet(
        added=sorted(added),
        modified=sorted(modified),
        deleted=sorted(deleted),
    )


# --------------------------------------------------------------------------- #
# watchdog 监听（Medium #3 queue 模型）
# --------------------------------------------------------------------------- #


class WatchQueue(FileSystemEventHandler):
    """watchdog 事件 → debounce → queue.Queue 串行消费。

    Observer 回调（独立线程）仅累计事件到 ``_pending`` 集合并启动/重置 debounce 计时器；
    计时器到期后将本窗口内所有变更路径去重排序后逐个入队。compile worker 单线程从
    ``queue`` 串行消费（FIFO），内存 graph 由该线程独占访问，无需加锁。

    用法：直接作为 ``observer.schedule(handler, ...)`` 的事件处理器，或单测中直接调用
    ``on_any_event`` 注入伪事件以验证 debounce + 入队行为。
    """

    #: 关闭信号 —— 入队后 worker 收到此对象即退出
    SENTINEL: object = object()

    def __init__(self, *, debounce_seconds: float = DEBOUNCE_SECONDS) -> None:
        super().__init__()
        self._debounce_seconds = debounce_seconds
        self.queue: queue.Queue[object] = queue.Queue()
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    # -- watchdog 回调 -------------------------------------------------------- #

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Observer 线程回调：累计受支持路径并重置 debounce 计时器。

        对 moved 事件同时追踪 dest_path（新位置作为 added 处理）。
        非受支持扩展名 / 目录事件忽略。
        """
        if event.is_directory:
            return

        candidates: list[str] = [str(event.src_path)]
        if event.event_type == "moved":
            dest = getattr(event, "dest_path", None)
            if dest:
                candidates.append(str(dest))

        with self._lock:
            added_any = False
            for path in candidates:
                if _is_supported(path):
                    self._pending.add(path)
                    added_any = True
            if added_any:
                self._reset_timer_locked()

    # -- 内部 ----------------------------------------------------------------- #

    def _reset_timer_locked(self) -> None:
        """（调用方持有 _lock）取消旧计时器并启动新的 debounce 计时器。"""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce_seconds, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        """debounce 到期：将累计的 pending 路径去重排序后逐个入队。"""
        with self._lock:
            if not self._pending:
                return
            paths = sorted(self._pending)
            self._pending.clear()
            self._timer = None
        for p in paths:
            self.queue.put(p)

    # -- 生命周期 ------------------------------------------------------------- #

    def close(self) -> None:
        """发送关闭信号到 queue（worker 收 SENTINEL 后退出）。"""
        # 取消未触发的 debounce 计时器，避免泄漏
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self.queue.put(self.SENTINEL)

    def flush_now(self) -> None:
        """立即刷新 pending 路径入队（测试辅助：跳过等待 debounce 窗口）。"""
        self._flush()


class WatchContext:
    """``start_watch`` 返回的监听上下文，封装 Observer + worker 的生命周期。

    调用 ``stop()`` 后 Observer 停止、worker 收 SENTINEL 退出，资源清理完毕。
    """

    def __init__(
        self,
        observer: BaseObserver,
        worker: threading.Thread,
        watch_queue: WatchQueue,
    ) -> None:
        self.observer = observer
        self.worker = worker
        self.watch_queue = watch_queue

    def stop(self) -> None:
        """停止 Observer 并通知 worker 退出，等待线程结束。"""
        self.observer.stop()
        self.observer.join(timeout=5.0)
        self.watch_queue.close()
        self.worker.join(timeout=5.0)


def start_watch(
    raw_dir: Path,
    on_change: Callable[[str], None],
    *,
    debounce_seconds: float = DEBOUNCE_SECONDS,
    recursive: bool = True,
) -> WatchContext:
    """启动 watchdog Observer 监听 ``raw_dir``，变更经 debounce 入队后单线程消费。

    Args:
        raw_dir: 监听目录。
        on_change: compile worker 对每个变更路径调用的回调（在 worker 线程中执行）。
            回调异常被捕获并记日志（不中断 worker）。
        debounce_seconds: debounce 窗口（默认 500ms）。
        recursive: 是否递归监听子目录。

    Returns:
        ``WatchContext`` —— 调用 ``.stop()`` 清理资源。

    并发模型（方案 §3.5.2 Medium #3）：
    - Observer 线程：on_any_event → 累计 pending → debounce 计时器
    - debounce 计时器线程：到期 → flush pending 到 queue
    - worker 线程：从 queue 串行取路径 → 调用 ``on_change``
    内存 graph 仅由 worker 线程访问，无锁。
    """
    watch_queue = WatchQueue(debounce_seconds=debounce_seconds)

    def _worker_loop() -> None:
        while True:
            item = watch_queue.queue.get()
            if item is WatchQueue.SENTINEL:
                break
            try:
                on_change(str(item))
            except Exception:
                logger.exception(
                    "compile worker error on %s",
                    item,
                    extra={"stage": "watch", "file": str(item)},
                )

    worker = threading.Thread(target=_worker_loop, daemon=True, name="nanokb-compile-worker")
    worker.start()

    observer = Observer()
    observer.schedule(watch_queue, str(raw_dir), recursive=recursive)
    observer.start()

    logger.info(
        "watch started on %s (debounce=%.3fs, recursive=%s)",
        raw_dir,
        debounce_seconds,
        recursive,
        extra={"stage": "watch", "file": str(raw_dir)},
    )

    return WatchContext(observer=observer, worker=worker, watch_queue=watch_queue)


__all__ = [
    "DEBOUNCE_SECONDS",
    "SUPPORTED_SUFFIXES",
    "ChangeSet",
    "WatchContext",
    "WatchQueue",
    "detect_changes",
    "start_watch",
]

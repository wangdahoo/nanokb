"""运行时进度文件（方案 §6 阶段三，Feature s3-feat-004）。

跨进程可见的编译进度：build 周期性原子写 ``out/.build_progress.json``，status 只读。
**status 绝不打开 ChromaDB**（保守假设规避跨进程锁，Medium #3）——向量数从 progress
或 manifest 读，不从 chroma 读。该设计在任何锁语义下都稳健（最小依赖原则）。

设计要点（round 2 / round 3）：
- **heartbeat 与业务 flush 解耦（Medium #1）**：daemon Thread 每 ``PROGRESS_HEARTBEAT_INTERVAL_SEC``
  只刷 ``heartbeat_ts``，业务计数 ``update_extract`` / ``update_vector`` 按阈值(每 5 文件)/
  阶段切换 flush。即便单大文档耗时 30-60s 也不会因 heartbeat 过期被误报僵尸。
- **check_liveliness 次级判据兜底（Medium #1）**：heartbeat 过期时 sleep ``recheck_sec``
  后重读，``extract.completed`` / ``vector.indexed_nodes`` 有增长仍判 alive。即便 heartbeat
  timer 异常停了，只要计数在增长就不误报僵尸。
- **Opt#6 显式 timer 启动落点**：``__init__`` 末尾 ``if self._enabled: self._start_heartbeat_timer()``；
  ``enabled=False`` 时不起 timer 且所有方法 no-op（零回归）。
- **Opt#3 recheck 可配置**：``check_liveliness(out_dir, *, recheck_sec=None)``，默认
  ``PROGRESS_LIVENESS_RECHECK_SEC``（1s），可由调用方传入 ``settings.progress_liveliness_recheck_sec``。
- **os._exit(130) 兼容**：每次 flush 直接落盘，不依赖 finally / atexit；
  ``interrupted()`` 在异常上抛前执行，``KeyboardInterrupt`` / ``Exception`` 均覆盖。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from nanokb.utils.io import atomic_write_json

logger = logging.getLogger("nanokb")

# ── 常量 ────────────────────────────────────────────────────────────────

#: 进度文件名（落盘于 out 目录）。
PROGRESS_FILENAME = ".build_progress.json"

#: heartbeat 判活超时阈值（秒）——超过即视为疑似僵尸，进入次级判据。
PROGRESS_HEARTBEAT_TIMEOUT_SEC = 60

#: heartbeat timer 刷新间隔（秒）——后台 daemon 线程每 ``N`` 秒只刷 heartbeat_ts。
#: round 2 Medium #1：与业务 flush 解耦，慢文件不误报。
PROGRESS_HEARTBEAT_INTERVAL_SEC = 10

#: check_liveliness 二次采样等待秒数（默认）——round 3 Opt#3：可由调用方覆盖。
PROGRESS_LIVENESS_RECHECK_SEC = 1


# ── 数据模型 ────────────────────────────────────────────────────────────


class BuildStage(str, Enum):
    """编译阶段枚举（status 据此渲染阶段化进度）。"""

    DETECT = "detect"
    EXTRACT = "extract"
    GRAPH = "graph"
    VECTOR = "vector"
    INDEX = "index"
    FINALIZE = "finalize"
    DONE = "done"
    INTERRUPTED = "interrupted"


class ExtractProgress(BaseModel):
    """阶段 A 抽取进度。"""

    total: int = 0
    completed: int = 0
    cached: int = 0
    skipped: int = 0


class VectorProgress(BaseModel):
    """step 7 向量索引进度。"""

    total_nodes: int = 0
    indexed_nodes: int = 0


class BuildProgress(BaseModel):
    """``out/.build_progress.json`` 的顶层 schema（Pydantic v2）。

    新字段以默认值保证向后兼容（旧文件缺字段时降级为默认值，不报错）。
    """

    schema_version: str = "1"
    pid: int = 0
    stage: BuildStage = BuildStage.DETECT
    started_at: str = ""
    heartbeat_ts: str = ""
    force: bool = False
    extract: ExtractProgress = Field(default_factory=ExtractProgress)
    vector: VectorProgress = Field(default_factory=VectorProgress)
    message: str = ""


# ── Writer ─────────────────────────────────────────────────────────────


def _now_iso() -> str:
    """当前 UTC 时间的 ISO 8601 字符串（可字典序排序，跨进程时戳一致）。"""
    return datetime.now(timezone.utc).isoformat()


class BuildProgressWriter:
    """编译进度写入器：业务计数 update + 后台 heartbeat timer 双轨落盘。

    线程模型：
    - 一把 ``threading.Lock`` 保护「序列化内存对象 + atomic_write」临界区。
    - daemon heartbeat Thread 每 ``PROGRESS_HEARTBEAT_INTERVAL_SEC`` 只刷 ``heartbeat_ts``
      + 原子写（不动业务计数，Medium #1 解耦）。
    - 业务 ``set_stage`` 必写；``update_extract`` / ``update_vector`` 按 ``FLUSH_EVERY``
      阈值或 ``force_flush=True`` 写。

    进程退出处理（§6.4）：
    - ``done()``：停 timer → 删除进度文件（完成后删除避免 status 误判还在跑）。
    - ``interrupted()``：停 timer → 写 ``INTERRUPTED`` 保留文件供诊断 → join timer。
    - 未捕获异常 / kill：文件残留 + heartbeat 过期 → status 识别僵尸（或计数增长判 alive）。

    ``enabled=False`` 时所有方法 no-op、不起 timer（零回归，Opt#6）。
    """

    #: 业务计数 update 触发落盘的阈值（每 N 次 update 强制 flush）。
    FLUSH_EVERY: int = 5

    def __init__(self, out_dir: Path, force: bool, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._path = Path(out_dir) / PROGRESS_FILENAME
        self._lock = threading.Lock()
        self._progress = BuildProgress(
            schema_version="1",
            pid=os.getpid(),
            stage=BuildStage.DETECT,
            started_at=_now_iso(),
            heartbeat_ts=_now_iso(),
            force=force,
        )
        self._pending: int = 0
        self._stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None

        # round 3（Opt#6）显式落点：构造即启动 heartbeat timer（仅 enabled 时）。
        if self._enabled:
            self._start_heartbeat_timer()

    # ── heartbeat timer ──────────────────────────────────────────────

    def _start_heartbeat_timer(self) -> None:
        """启动 daemon heartbeat 线程：循环 sleep + 刷 heartbeat_ts + 原子写。

        ``Event.wait`` 既 sleep 又可被 ``set()`` 唤醒（精确停止）。daemon=True 保证
        主进程退出时自动结束，不阻塞 ``os._exit``。
        """
        self._stop = threading.Event()

        def _loop() -> None:
            assert self._stop is not None
            while not self._stop.wait(PROGRESS_HEARTBEAT_INTERVAL_SEC):
                with self._lock:
                    self._progress.heartbeat_ts = _now_iso()
                    self._write_locked()

        t = threading.Thread(target=_loop, daemon=True, name="nanokb-build-heartbeat")
        t.start()
        self._heartbeat_thread = t

    def _stop_timer(self) -> None:
        """停止 heartbeat timer：set Event 唤醒 → join 线程确保退出后再继续。

        join 后才能安全删除/改写文件，避免 heartbeat 在 done/interrupted 后又写回旧内容。
        """
        if self._stop is not None:
            self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5.0)
            self._heartbeat_thread = None
        self._stop = None

    # ── 内部落盘（须持锁） ───────────────────────────────────────────

    def _write_locked(self) -> None:
        """原子写当前 progress 到磁盘（调用方须持有 self._lock）。

        每次 flush 都刷新 ``heartbeat_ts``（即便业务 update 也顺带刷新心跳，
        使活跃 build 始终被判 alive）。使用 ``mode="json"`` 保证 Enum 序列化为值字符串，
        跨进程读取时可经 ``BuildProgress.model_validate`` 还原。
        """
        self._progress.heartbeat_ts = _now_iso()
        atomic_write_json(self._path, self._progress.model_dump(mode="json"))

    def _flush_if_pending(self, force: bool) -> None:
        """按阈值或强制标志落盘（调用方须持有 self._lock）。"""
        self._pending += 1
        if force or self._pending >= self.FLUSH_EVERY:
            self._write_locked()
            self._pending = 0

    # ── 业务 API ─────────────────────────────────────────────────────

    def set_stage(self, stage: BuildStage, message: str = "") -> None:
        """切换阶段（必写）：更新 stage/message 后立即 flush。"""
        if not self._enabled:
            return
        with self._lock:
            self._progress.stage = stage
            if message:
                self._progress.message = message
            self._write_locked()
            self._pending = 0

    def update_extract(
        self,
        *,
        total: int | None = None,
        completed_delta: int = 0,
        cached_delta: int = 0,
        skipped_delta: int = 0,
        force_flush: bool = False,
    ) -> None:
        """更新抽取计数：按 ``FLUSH_EVERY`` 阈值或 ``force_flush`` 决定是否落盘。"""
        if not self._enabled:
            return
        with self._lock:
            if total is not None:
                self._progress.extract.total = total
            self._progress.extract.completed += completed_delta
            self._progress.extract.cached += cached_delta
            self._progress.extract.skipped += skipped_delta
            self._flush_if_pending(force_flush)

    def update_vector(
        self,
        *,
        total: int | None = None,
        indexed_delta: int = 0,
        force_flush: bool = False,
    ) -> None:
        """更新向量索引计数：按 ``FLUSH_EVERY`` 阈值或 ``force_flush`` 决定是否落盘。"""
        if not self._enabled:
            return
        with self._lock:
            if total is not None:
                self._progress.vector.total_nodes = total
            self._progress.vector.indexed_nodes += indexed_delta
            self._flush_if_pending(force_flush)

    # ── 退出处理 ─────────────────────────────────────────────────────

    def done(self) -> None:
        """正常完成：停 timer → 删除进度文件（避免 status 误判还在跑）。

        AC3.2：完成后文件删除（或标记 DONE 后由下次 build 清理）。
        """
        if not self._enabled:
            return
        self._stop_timer()
        with self._lock:
            self._progress.stage = BuildStage.DONE
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass

    def interrupted(self) -> None:
        """被中断：停 timer → 写 ``INTERRUPTED`` 保留文件供诊断（AC3.3）。

        保留文件使 status 能展示「上次编译中断」+ 中断阶段 + 重跑零成本提示。
        下次 build 启动时 ``__init__`` 的初始 flush 会覆盖旧 INTERRUPTED 文件。
        """
        if not self._enabled:
            return
        self._stop_timer()
        with self._lock:
            self._progress.stage = BuildStage.INTERRUPTED
            self._progress.heartbeat_ts = _now_iso()
            self._write_locked()


# ── 只读 API（供 status 命令使用，绝不打开 chroma） ────────────────────


def read_progress(out_dir: Path) -> BuildProgress | None:
    """读取进度文件；不存在 / 损坏返回 None（降级到静态产物展示，AC3.5）。

    只读纯 JSON，零 ChromaDB 句柄（保守假设规避跨进程锁，Medium #3 / AC3.4）。
    """
    path = Path(out_dir) / PROGRESS_FILENAME
    if not path.exists():
        return None
    try:
        return BuildProgress.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        logger.debug("failed to parse %s; degrading to None", path, exc_info=True)
        return None


def _heartbeat_fresh(progress: BuildProgress) -> bool:
    """heartbeat 是否在 ``PROGRESS_HEARTBEAT_TIMEOUT_SEC`` 内。"""
    if not progress.heartbeat_ts:
        return False
    try:
        ts = datetime.fromisoformat(progress.heartbeat_ts)
    except (ValueError, TypeError):
        return False
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (now - ts).total_seconds()
    return age < PROGRESS_HEARTBEAT_TIMEOUT_SEC


def is_alive(progress: BuildProgress) -> bool:
    """build 进程是否仍活跃：DONE/INTERRUPTED 视为不活跃；否则看 heartbeat 是否 fresh。"""
    if progress.stage in (BuildStage.DONE, BuildStage.INTERRUPTED):
        return False
    return _heartbeat_fresh(progress)


def check_liveliness(out_dir: Path, *, recheck_sec: float | None = None) -> bool:
    """heartbeat 过期时的次级判据（Medium #1）：sleep ``recheck_sec`` 后重读，计数增长即 alive。

    流程：
    1. 读取 ``p0``；None / DONE / INTERRUPTED → False。
    2. heartbeat fresh → True（无需等待）。
    3. heartbeat 过期 → sleep ``recheck_sec``（默认 ``PROGRESS_LIVENESS_RECHECK_SEC``，
       round 3 Opt#3 可配置）→ 重读 ``p1``；``extract.completed`` 或
       ``vector.indexed_nodes`` 有增长 → True（不误报僵尸，AC3.7）；否则 False（AC3.6）。
    """
    if recheck_sec is None:
        recheck_sec = PROGRESS_LIVENESS_RECHECK_SEC

    p0 = read_progress(out_dir)
    if p0 is None or p0.stage in (BuildStage.DONE, BuildStage.INTERRUPTED):
        return False
    if _heartbeat_fresh(p0):
        return True

    time.sleep(recheck_sec)
    p1 = read_progress(out_dir)
    if p1 is None:
        return False
    return (
        p1.extract.completed > p0.extract.completed
        or p1.vector.indexed_nodes > p0.vector.indexed_nodes
    )


__all__ = [
    "PROGRESS_FILENAME",
    "PROGRESS_HEARTBEAT_INTERVAL_SEC",
    "PROGRESS_HEARTBEAT_TIMEOUT_SEC",
    "PROGRESS_LIVENESS_RECHECK_SEC",
    "BuildProgress",
    "BuildProgressWriter",
    "BuildStage",
    "ExtractProgress",
    "VectorProgress",
    "check_liveliness",
    "is_alive",
    "read_progress",
]

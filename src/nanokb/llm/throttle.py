"""线程安全的全局速率限制器（方案 §3.2.2，Feature s1-feat-001）。

将原 ``OpenAIClient._throttle`` 的"实例时间戳"语义提升为进程级、线程安全、
可跨 provider 共享的节流原语。``interval<=0`` 时无锁快速返回（零开销，
适配 ollama 本地服务 / 无限流场景）。

设计要点：
- 锁内 ``time.sleep``：sleep 期间持有锁会阻塞其他 acquire 调用，但这正是
  "最小间隔"RPM 节流的期望行为——串行化请求发出时机，控制全局请求速率。
- 与 SDK 内置重试（``max_retries`` 指数退避）和应用层 ``RateLimitError`` 退避
  （``_compute_backoff``）正交：RateLimiter 控制**主动请求间隔**，重试机制
  处理**被动 429 响应**，两者叠加安全（退避期间不 acquire）。
- ``threading.Lock`` 非公平（不保证 FIFO 唤醒顺序）。RPM 节流场景下 acquire
  等待数等于并发度（个位数到几十），starvation 风险可忽略，记录为已知特性。
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """确保两次 ``acquire()`` 之间至少间隔 ``interval`` 秒（线程安全）。

    用 ``threading.Lock`` 保护 ``_last_ts`` 的 read-modify-write。锁内 sleep
    串行化所有线程的请求——这正是 RPM 节流的目的（控制全局请求速率）。
    ``interval<=0`` 时直接返回，无锁竞争开销。

    该实例可跨线程、跨 provider client 共享，作为进程级全局节流原语。
    """

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._last_ts: float | None = None

    def acquire(self) -> None:
        """阻塞直至距上次 acquire 至少 ``interval`` 秒。

        ``interval<=0`` 时无锁快速返回（零开销）。否则在锁内计算需等待的
        时间并 sleep，随后更新 ``_last_ts``。锁内 sleep 串行化所有并发
        acquire，确保全局请求速率不超 ``1 / interval`` QPS。
        """
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if self._last_ts is not None:
                wait = self._interval - (now - self._last_ts)
                if wait > 0:
                    time.sleep(wait)
                    now = time.monotonic()
            self._last_ts = now


__all__ = ["RateLimiter"]

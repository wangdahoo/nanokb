"""RateLimiter 单测（方案 §3.2.2，Feature s1-feat-001）。

覆盖 4 条验收标准：
- AC #1：``interval<=0`` 时 acquire 立即返回（无 sleep、无锁竞争开销）。
- AC #2：``interval=0.05s``，10 线程并发 acquire，任意两次实际 acquire 时间
  间隔 ≥ 0.05s（线程安全节流生效，无竞态）。
- AC #3：单线程连续 acquire（``interval=0.1s``）3 次，每次间隔 ≥ 0.1s，
  行为与原 ``OpenAIClient._throttle`` 等价。
- AC #4：``pytest tests/unit/test_throttle.py`` 全部通过。
"""

from __future__ import annotations

import threading
import time

from nanokb.llm.throttle import RateLimiter

#: 间隔断言下界容差，吸收 OS 调度抖动 / ``time.sleep`` 定时器精度，避免 CI flaky。
#: 注意 RateLimiter 请求 sleep 至少 interval 秒，但 Windows 等平台 ``time.sleep``
#: 定时器粒度较粗（默认 ~15.6ms），可能提前数毫秒返回——该提前量与原
#: ``OpenAIClient._throttle`` 行为完全一致（同一 ``time.sleep`` 调用），故容差
#: 仅吸收平台定时器抖动，不削弱对"节流生效"的断言（无节流时间隔为 ~µs 级）。
_TOLERANCE = 0.02


def test_acquire_returns_immediately_when_interval_zero() -> None:
    """AC #1：interval=0 时无锁快速返回，多线程并发 acquire 总耗时极短。"""
    limiter = RateLimiter(interval=0.0)
    n = 50
    barrier = threading.Barrier(n)

    def worker() -> None:
        barrier.wait()
        limiter.acquire()

    start = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start

    # 50 次无 sleep 的 acquire 应在百毫秒级内完成；放宽到 1.0s 防止 CI 抖动。
    assert elapsed < 1.0, f"interval=0 path took {elapsed:.3f}s, expected no-op"


def test_acquire_returns_immediately_when_interval_negative() -> None:
    """AC #1 边界：interval<0 也走"无限流"快速返回分支。"""
    limiter = RateLimiter(interval=-1.0)
    start = time.monotonic()
    for _ in range(100):
        limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.5, f"negative-interval path took {elapsed:.3f}s"


def test_concurrent_acquire_enforces_minimum_interval() -> None:
    """AC #2：interval=0.05s，10 线程并发 acquire，相邻间隔 ≥ interval。"""
    interval = 0.05
    limiter = RateLimiter(interval=interval)
    n = 10
    timestamps: list[float] = []
    ts_lock = threading.Lock()
    barrier = threading.Barrier(n)

    def worker() -> None:
        barrier.wait()
        limiter.acquire()
        with ts_lock:
            timestamps.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(timestamps) == n, "some threads did not record a timestamp"

    timestamps.sort()
    diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    min_diff = min(diffs)
    assert min_diff >= interval - _TOLERANCE, (
        f"min adjacent interval {min_diff:.4f}s < {interval}s (tol {_TOLERANCE}s); "
        f"diffs={[round(d, 4) for d in diffs]}"
    )


def test_sequential_acquire_equivalent_to_original_throttle() -> None:
    """AC #3：单线程连续 acquire（interval=0.1s）3 次，每次间隔 ≥ interval。

    与原 ``OpenAIClient._throttle`` 语义等价（首次立即返回，后续每次至少间隔
    interval）。
    """
    interval = 0.1
    limiter = RateLimiter(interval=interval)

    stamps: list[float] = []
    for _ in range(3):
        limiter.acquire()
        stamps.append(time.monotonic())

    # 首次 acquire 无前序时间戳，立即返回；后续两次每次间隔 ≥ interval。
    gap1 = stamps[1] - stamps[0]
    gap2 = stamps[2] - stamps[1]
    assert gap1 >= interval - _TOLERANCE, f"gap1 {gap1:.4f}s < {interval}s"
    assert gap2 >= interval - _TOLERANCE, f"gap2 {gap2:.4f}s < {interval}s"


def test_zero_interval_limiter_shared_across_threads() -> None:
    """AC #1 补充：interval=0 的共享实例跨线程安全（无异常、无阻塞）。"""
    limiter = RateLimiter(interval=0.0)
    errors: list[BaseException] = []
    err_lock = threading.Lock()

    def worker() -> None:
        try:
            for _ in range(20):
                limiter.acquire()
        except BaseException as exc:  # noqa: BLE001 - 记录任意异常供断言
            with err_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"shared zero-interval limiter raised: {errors}"

"""BuildProgressWriter / read_progress 单测（方案 §6.7，Feature s3-feat-004）。

覆盖验收标准：
- 原子写 + read_progress 往返。
- flush 阈值（FLUSH_EVERY=5：4 次 update 不写、第 5 次写、set_stage 必写、force_flush）。
- AC3.8：heartbeat 与业务 flush 解耦（业务空转仍每 interval 刷 heartbeat_ts）。
- AC3.9 / Opt#6：enabled=True 有 daemon heartbeat 线程；enabled=False 无线程且全方法 no-op。
- AC3.2：done() 删除文件。
- AC3.3：interrupted() 写 INTERRUPTED 保留文件。
- AC3.5：read_progress 损坏 / 不存在降级到 None。
- AC3.6：heartbeat 过期且无增长 → 僵尸（check_liveliness False）。
- AC3.7 / Medium #1：heartbeat 过期但计数增长 → 编译进行中（check_liveliness True）。
- AC3.10 / Opt#3：check_liveliness 的 recheck_sec 可配置。
- is_alive：fresh→True / 过期→False / DONE,INTERRUPTED→False。

全部离线，tmp_path 隔离。被测代码：``src/nanokb/utils/progress.py``（Feature s3-feat-004）。
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import nanokb.utils.progress as progress_mod
from nanokb.utils.io import atomic_write_json
from nanokb.utils.progress import (
    PROGRESS_FILENAME,
    BuildProgress,
    BuildProgressWriter,
    BuildStage,
    ExtractProgress,
    VectorProgress,
    check_liveliness,
    is_alive,
    read_progress,
)


def _progress_path(out_dir: Path) -> Path:
    return out_dir / PROGRESS_FILENAME


def _write_progress_file(
    out_dir: Path,
    *,
    stage: BuildStage = BuildStage.EXTRACT,
    heartbeat_age_sec: float = 0.0,
    completed: int = 0,
    indexed_nodes: int = 0,
) -> None:
    """直接写一个可控的 BuildProgress JSON（供 check_liveliness / is_alive 夹具）。"""
    hb = datetime.now(timezone.utc) - timedelta(seconds=heartbeat_age_sec)
    prog = BuildProgress(
        schema_version="1",
        pid=99999,
        stage=stage,
        started_at=hb.isoformat(),
        heartbeat_ts=hb.isoformat(),
        extract=ExtractProgress(total=10, completed=completed, cached=0, skipped=0),
        vector=VectorProgress(total_nodes=10, indexed_nodes=indexed_nodes),
    )
    atomic_write_json(_progress_path(out_dir), prog.model_dump(mode="json"))


# ══════════════════════════════════════════════════════════════════════
# 原子写 + read_progress 往返
# ══════════════════════════════════════════════════════════════════════


def test_atomic_write_and_read_roundtrip(out_dir: Path) -> None:
    """set_stage 原子写后 read_progress 能还原（跨进程 JSON 可读）。"""
    writer = BuildProgressWriter(out_dir, force=False)
    try:
        writer.set_stage(BuildStage.EXTRACT, message="extracting docs")
        assert _progress_path(out_dir).exists()
        prog = read_progress(out_dir)
        assert prog is not None
        assert prog.stage == BuildStage.EXTRACT
        assert prog.message == "extracting docs"
        assert prog.pid > 0
    finally:
        writer.done()


# ══════════════════════════════════════════════════════════════════════
# flush 阈值（FLUSH_EVERY=5）
# ══════════════════════════════════════════════════════════════════════


def test_flush_threshold_four_updates_no_flush(out_dir: Path) -> None:
    """4 次 update_extract（pending<5）不落盘，文件 completed 仍为 set_stage 时的 0。

    heartbeat interval 默认 10s，本测试 <1s 完成，heartbeat 不会干扰磁盘内容。
    """
    assert BuildProgressWriter.FLUSH_EVERY == 5
    writer = BuildProgressWriter(out_dir, force=False)
    try:
        writer.set_stage(BuildStage.EXTRACT)
        # 4 次更新，均不达阈值 → 不落盘
        for _ in range(4):
            writer.update_extract(completed_delta=1)
        prog = read_progress(out_dir)
        assert prog is not None
        assert prog.extract.completed == 0, "4 updates (< FLUSH_EVERY) should not flush"
    finally:
        writer.done()


def test_flush_threshold_fifth_update_flushes(out_dir: Path) -> None:
    """第 5 次 update（pending==FLUSH_EVERY）触发落盘，文件 completed==5。"""
    writer = BuildProgressWriter(out_dir, force=False)
    try:
        writer.set_stage(BuildStage.EXTRACT)
        for _ in range(5):
            writer.update_extract(completed_delta=1)
        prog = read_progress(out_dir)
        assert prog is not None
        assert prog.extract.completed == 5, "5th update should flush completed=5"
    finally:
        writer.done()


def test_set_stage_always_flushes(out_dir: Path) -> None:
    """set_stage 必写：即便 pending 未达阈值也立即落盘。"""
    writer = BuildProgressWriter(out_dir, force=False)
    try:
        writer.set_stage(BuildStage.EXTRACT)
        writer.update_extract(completed_delta=1)  # pending=1，不达阈值
        writer.set_stage(BuildStage.GRAPH)  # 必写
        prog = read_progress(out_dir)
        assert prog is not None
        assert prog.stage == BuildStage.GRAPH
        # set_stage(GRAPH) 落盘时顺带把内存中 completed=1 一起写入
        assert prog.extract.completed == 1
    finally:
        writer.done()


def test_force_flush_overrides_threshold(out_dir: Path) -> None:
    """force_flush=True 单次 update 即落盘（不受 FLUSH_EVERY 约束）。"""
    writer = BuildProgressWriter(out_dir, force=False)
    try:
        writer.set_stage(BuildStage.VECTOR)
        writer.update_vector(total=100, indexed_delta=3, force_flush=True)
        prog = read_progress(out_dir)
        assert prog is not None
        assert prog.vector.total_nodes == 100
        assert prog.vector.indexed_nodes == 3
    finally:
        writer.done()


# ══════════════════════════════════════════════════════════════════════
# AC3.8：heartbeat 与业务 flush 解耦（Medium #1）
# ══════════════════════════════════════════════════════════════════════


def test_heartbeat_decoupled_from_business_flush(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC3.8：业务空转期间 heartbeat_ts 仍每 interval 刷新（timer 独立于业务 flush）。

    把 heartbeat interval 调到 0.2s，set_stage 后 sleep 0.7s（不调用任何业务 update），
    断言 heartbeat_ts 已被 timer 刷新（且 stage 仍为 EXTRACT，业务计数未被动）。
    """
    monkeypatch.setattr(progress_mod, "PROGRESS_HEARTBEAT_INTERVAL_SEC", 0.2)
    writer = BuildProgressWriter(out_dir, force=False)
    try:
        writer.set_stage(BuildStage.EXTRACT)
        before = read_progress(out_dir)
        assert before is not None
        ts0 = before.heartbeat_ts

        # 业务空转 0.7s（> 3× interval），timer 应回写 heartbeat 多次
        time.sleep(0.7)

        after = read_progress(out_dir)
        assert after is not None
        assert after.heartbeat_ts != ts0, "heartbeat_ts should be refreshed by timer"
        assert after.stage == BuildStage.EXTRACT, "timer must not touch business stage"
        assert after.extract.completed == 0, "timer must not touch business counters"
    finally:
        writer.done()


# ══════════════════════════════════════════════════════════════════════
# AC3.9 / Opt#6：timer 启动落点 + enabled=False no-op
# ══════════════════════════════════════════════════════════════════════


def _heartbeat_thread() -> threading.Thread | None:
    return next(
        (t for t in threading.enumerate() if t.name == "nanokb-build-heartbeat"),
        None,
    )


def test_enabled_starts_daemon_heartbeat_thread(out_dir: Path) -> None:
    """AC3.9 / Opt#6：enabled=True 构造即有活跃 daemon heartbeat 线程。"""
    writer = BuildProgressWriter(out_dir, force=False, enabled=True)
    try:
        t = _heartbeat_thread()
        assert t is not None, "daemon heartbeat thread should be running when enabled"
        assert t.daemon is True, "heartbeat thread must be daemon"
    finally:
        writer.done()
        # done() 停 timer 后线程退出（join 完成）
        assert _heartbeat_thread() is None or not _heartbeat_thread().is_alive()  # type: ignore[union-attr]


def test_disabled_no_thread_and_all_noop(out_dir: Path) -> None:
    """AC3.9 / Opt#6：enabled=False 无 heartbeat 线程，所有方法 no-op（零回归）。"""
    writer = BuildProgressWriter(out_dir, force=False, enabled=False)
    assert writer._heartbeat_thread is None
    assert writer._stop is None

    # 所有方法 no-op，不抛异常、不写文件
    writer.set_stage(BuildStage.EXTRACT)
    writer.update_extract(completed_delta=1)
    writer.update_vector(indexed_delta=1)
    assert not _progress_path(out_dir).exists(), "disabled writer must not write file"

    # done / interrupted 也不抛、不产生文件
    writer.done()
    writer.interrupted()
    assert not _progress_path(out_dir).exists()


# ══════════════════════════════════════════════════════════════════════
# AC3.2：done() 删除文件
# ══════════════════════════════════════════════════════════════════════


def test_done_deletes_progress_file(out_dir: Path) -> None:
    """AC3.2：正常完成 → 进度文件被删除（避免 status 误判还在跑）。"""
    writer = BuildProgressWriter(out_dir, force=False)
    writer.set_stage(BuildStage.EXTRACT)
    assert _progress_path(out_dir).exists()
    writer.done()
    assert not _progress_path(out_dir).exists(), "done() must delete progress file"


# ══════════════════════════════════════════════════════════════════════
# AC3.3：interrupted() 写 INTERRUPTED 保留文件
# ══════════════════════════════════════════════════════════════════════


def test_interrupted_keeps_file_marked_interrupted(out_dir: Path) -> None:
    """AC3.3：中断 → 写 stage=INTERRUPTED 保留文件供 status 诊断。"""
    writer = BuildProgressWriter(out_dir, force=False)
    writer.set_stage(BuildStage.VECTOR)
    writer.interrupted()
    assert _progress_path(out_dir).exists(), "interrupted() must keep the file"
    prog = read_progress(out_dir)
    assert prog is not None
    assert prog.stage == BuildStage.INTERRUPTED


# ══════════════════════════════════════════════════════════════════════
# AC3.5：read_progress 降级（不存在 / 损坏）
# ══════════════════════════════════════════════════════════════════════


def test_read_progress_nonexistent_returns_none(out_dir: Path) -> None:
    """AC3.5：文件不存在 → None（降级到静态产物展示）。"""
    assert read_progress(out_dir) is None


def test_read_progress_corrupted_returns_none(out_dir: Path) -> None:
    """AC3.5：损坏 JSON → None（best-effort 降级，不抛）。"""
    _progress_path(out_dir).write_text("{ not valid json @@", encoding="utf-8")
    assert read_progress(out_dir) is None


# ══════════════════════════════════════════════════════════════════════
# is_alive
# ══════════════════════════════════════════════════════════════════════


def test_is_alive_fresh_heartbeat_true(out_dir: Path) -> None:
    _write_progress_file(out_dir, stage=BuildStage.EXTRACT, heartbeat_age_sec=5.0)
    prog = read_progress(out_dir)
    assert prog is not None
    assert is_alive(prog) is True


def test_is_alive_stale_heartbeat_false(out_dir: Path) -> None:
    _write_progress_file(out_dir, stage=BuildStage.EXTRACT, heartbeat_age_sec=120.0)
    prog = read_progress(out_dir)
    assert prog is not None
    assert is_alive(prog) is False


def test_is_alive_done_false(out_dir: Path) -> None:
    _write_progress_file(out_dir, stage=BuildStage.DONE, heartbeat_age_sec=0.0)
    prog = read_progress(out_dir)
    assert prog is not None
    assert is_alive(prog) is False


def test_is_alive_interrupted_false(out_dir: Path) -> None:
    _write_progress_file(out_dir, stage=BuildStage.INTERRUPTED, heartbeat_age_sec=0.0)
    prog = read_progress(out_dir)
    assert prog is not None
    assert is_alive(prog) is False


# ══════════════════════════════════════════════════════════════════════
# AC3.6 / AC3.7 / AC3.10：check_liveliness
# ══════════════════════════════════════════════════════════════════════


def test_check_liveliness_stale_no_growth_returns_false(out_dir: Path) -> None:
    """AC3.6：heartbeat 过期且两次读取间计数无增长 → 僵尸（False）。"""
    _write_progress_file(out_dir, stage=BuildStage.EXTRACT, heartbeat_age_sec=120.0, completed=3)
    # recheck 期间无写入 → 无增长
    assert check_liveliness(out_dir, recheck_sec=0.05) is False


def test_check_liveliness_stale_but_growing_returns_true(out_dir: Path) -> None:
    """AC3.7 / Medium #1：heartbeat 过期但计数增长 → 编译进行中（True）。"""
    _write_progress_file(out_dir, stage=BuildStage.EXTRACT, heartbeat_age_sec=120.0, completed=3)

    # 模拟 build 仍在推进：recheck_sec 后把 completed 写到 4
    def _grow() -> None:
        time.sleep(0.1)
        _write_progress_file(
            out_dir, stage=BuildStage.EXTRACT, heartbeat_age_sec=120.0, completed=4
        )

    threading.Thread(target=_grow, daemon=True).start()
    assert check_liveliness(out_dir, recheck_sec=0.2) is True


def test_check_liveliness_fresh_heartbeat_short_circuits_true(out_dir: Path) -> None:
    """heartbeat fresh → 直接 True（不进入 recheck 等待）。"""
    _write_progress_file(out_dir, stage=BuildStage.VECTOR, heartbeat_age_sec=2.0)
    start = time.monotonic()
    assert check_liveliness(out_dir, recheck_sec=5.0) is True
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, "fresh heartbeat must short-circuit without sleeping recheck_sec"


def test_check_liveliness_recheck_configurable(out_dir: Path) -> None:
    """AC3.10 / Opt#3：recheck_sec 取自传入参数（可配置），实际等待≈该值。"""
    _write_progress_file(out_dir, stage=BuildStage.EXTRACT, heartbeat_age_sec=120.0, completed=0)
    start = time.monotonic()
    assert check_liveliness(out_dir, recheck_sec=0.3) is False
    elapsed = time.monotonic() - start
    assert elapsed >= 0.25, "should sleep ~recheck_sec before second sample"


def test_check_liveliness_done_returns_false(out_dir: Path) -> None:
    """DONE 阶段 → False（不进入 recheck）。"""
    _write_progress_file(out_dir, stage=BuildStage.DONE, heartbeat_age_sec=0.0)
    assert check_liveliness(out_dir, recheck_sec=5.0) is False


def test_check_liveliness_default_recheck_constant(out_dir) -> None:  # type: ignore[no-untyped-def]
    """未传 recheck_sec 时取 PROGRESS_LIVENESS_RECHECK_SEC 默认值（1.0）。"""
    assert progress_mod.PROGRESS_LIVENESS_RECHECK_SEC == 1


# ══════════════════════════════════════════════════════════════════════
# 并发安全：业务 update 与 heartbeat 临界区受锁保护
# ══════════════════════════════════════════════════════════════════════


def test_concurrent_update_and_heartbeat_no_corruption(
    out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """业务 update 与 heartbeat timer 并发写不产生损坏 JSON（锁保护临界区）。

    Windows 上并发 os.replace 偶发共享冲突时 read_progress 会 best-effort 降级为
    None（正确行为）；本测试断言「不抛异常 + 非 None 结果结构合法」，而非「绝不 None」。
    """
    monkeypatch.setattr(progress_mod, "PROGRESS_HEARTBEAT_INTERVAL_SEC", 0.01)
    writer = BuildProgressWriter(out_dir, force=False)
    try:
        writer.set_stage(BuildStage.EXTRACT)

        def _hammer() -> None:
            for _ in range(50):
                writer.update_extract(completed_delta=1, force_flush=True)

        t = threading.Thread(target=_hammer)
        t.start()
        # 并发读取：每次要么 None（瞬时降级），要么合法 BuildProgress，绝不抛 / 绝不半写
        while t.is_alive():
            prog = read_progress(out_dir)
            assert prog is None or prog.stage == BuildStage.EXTRACT
        t.join()
        # 收尾后最终态确定：completed==50
        final = None
        for _ in range(20):
            final = read_progress(out_dir)
            if final is not None:
                break
            time.sleep(0.01)
        assert final is not None
        assert final.extract.completed == 50
    finally:
        writer.done()

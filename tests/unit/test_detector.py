"""detector 增量检测与 watchdog queue 模型单测（方案 §3.5.1 step 1 + §3.5.2 + AC #2-#5）。

覆盖：
- AC #2：manifest 已记录且 sha256 未变 + 四维身份全一致 → 不在任一集合
- AC #3：llm_model 变更 → modified（模型身份触发重抽取）
- AC #4：文件从磁盘删除 → deleted
- AC #5：watchdog 并发写入 → debounce 窗口后路径并入 queue → 单 worker 串行消费
- 附加：added / extractor_version 变更 / embedding_model 变更 / 三集合互斥 / ingest 编排
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from nanokb.config import Settings
from nanokb.config_signature import (
    embedding_config_signature,
    extraction_config_signature,
    index_config_signature,
)
from nanokb.load import (
    ChangeSet,
    IngestResult,
    WatchQueue,
    detect_changes,
    ingest,
    ingest_file,
    start_watch,
)
from nanokb.loaders import LoaderRegistry, UnstructuredLoader
from nanokb.models import FileState, Manifest
from nanokb.utils.hashing import sha256_file

# --------------------------------------------------------------------------- #
# 辅助：构造 manifest FileState / Settings
# --------------------------------------------------------------------------- #


def _file_state(
    path: Path,
    raw_dir: Path,
    *,
    llm_model: str = "glm-5.1",
    embedding_model: str = "text-embedding-3-small",
    extractor_version: str = "1",
    **sig_overrides: Any,
) -> FileState:
    """根据磁盘文件当前内容构造 FileState（sha256 取真实值）。

    用传入的 llm_model/embedding_model/extractor_version 及额外签名覆盖构造一个
    对应 Settings，计算三层签名填入 FileState，使 unchanged 用例在五维比对下通过。
    """
    rel = str(path.relative_to(raw_dir))
    sig_settings = Settings(
        llm_model=llm_model,
        embedding_model=embedding_model,
        extractor_version=extractor_version,
        **sig_overrides,
    )
    return FileState(
        path=rel,
        sha256=sha256_file(path),
        processed_at="2026-01-01T00:00:00Z",
        extractor_version=extractor_version,
        llm_model=llm_model,
        embedding_model=embedding_model,
        extraction_config=extraction_config_signature(sig_settings),
        index_config=index_config_signature(sig_settings),
        embedding_config=embedding_config_signature(sig_settings),
    )


def _settings(**overrides: Any) -> Settings:
    """构造 Settings，默认使用 glm-5.1（与项目默认一致，使用内置 cl100k_base 离线可用）。"""
    defaults: dict[str, Any] = {
        "llm_model": "glm-5.1",
        "embedding_model": "text-embedding-3-small",
        "extractor_version": "1",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _write(raw_dir: Path, name: str, content: str = "hello world\n") -> Path:
    p = raw_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# AC #2：sha256 未变 + 四维一致 → 不在任一集合
# --------------------------------------------------------------------------- #


def test_unchanged_file_not_in_any_set(tmp_path: Path) -> None:
    """AC #2：manifest 已记录，sha256 未变且四维身份全一致 → added/modified/deleted 均不含。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    p = _write(raw_dir, "a.md", "unchanged content")

    manifest = Manifest()
    rel = "a.md"
    manifest.files[rel] = _file_state(p, raw_dir)

    changes = detect_changes(raw_dir, manifest, _settings())

    assert changes.added == []
    assert changes.modified == []
    assert changes.deleted == []
    assert not changes.has_changes


# --------------------------------------------------------------------------- #
# AC #3：llm_model 变更 → modified
# --------------------------------------------------------------------------- #


def test_llm_model_change_triggers_modified(tmp_path: Path) -> None:
    """AC #3：manifest 记录 llm_model='gpt-4o-mini'，settings 改为 'gpt-4o' → modified。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    p = _write(raw_dir, "paper.md", "Transformer content")

    manifest = Manifest()
    manifest.files["paper.md"] = _file_state(p, raw_dir, llm_model="gpt-4o-mini")

    changes = detect_changes(raw_dir, manifest, _settings(llm_model="gpt-4o"))

    assert "paper.md" in changes.modified
    assert "paper.md" not in changes.added
    assert "paper.md" not in changes.deleted


def test_extractor_version_change_triggers_modified(tmp_path: Path) -> None:
    """extractor_version 变更（如抽取器升级 1 → 2）→ modified。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    p = _write(raw_dir, "a.md")

    manifest = Manifest()
    manifest.files["a.md"] = _file_state(p, raw_dir, extractor_version="1")

    changes = detect_changes(raw_dir, manifest, _settings(extractor_version="2"))

    assert "a.md" in changes.modified


def test_embedding_model_change_triggers_modified(tmp_path: Path) -> None:
    """embedding_model 变更 → modified（向量侧需重建）。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    p = _write(raw_dir, "a.md")

    manifest = Manifest()
    manifest.files["a.md"] = _file_state(
        p, raw_dir, embedding_model="text-embedding-3-small"
    )

    changes = detect_changes(
        raw_dir, manifest, _settings(embedding_model="text-embedding-3-large")
    )

    assert "a.md" in changes.modified


def test_sha256_change_triggers_modified(tmp_path: Path) -> None:
    """文件内容变更（sha256 不同）→ modified。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(raw_dir, "a.md", "new content")

    manifest = Manifest()
    # manifest 记录的是旧 sha256（与磁盘不一致）
    manifest.files["a.md"] = FileState(
        path="a.md",
        sha256="0" * 64,
        processed_at="2026-01-01T00:00:00Z",
        llm_model="glm-5.1",
        embedding_model="text-embedding-3-small",
    )

    changes = detect_changes(raw_dir, manifest, _settings())

    assert "a.md" in changes.modified


# --------------------------------------------------------------------------- #
# AC #4：文件删除 → deleted
# --------------------------------------------------------------------------- #


def test_deleted_file_appears_in_deleted(tmp_path: Path) -> None:
    """AC #4：磁盘上不存在但 manifest 有记录 → deleted。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # raw_dir 下不创建 a.md，但 manifest 有记录
    manifest = Manifest()
    manifest.files["a.md"] = FileState(
        path="a.md",
        sha256="0" * 64,
        processed_at="2026-01-01T00:00:00Z",
    )

    changes = detect_changes(raw_dir, manifest, _settings())

    assert "a.md" in changes.deleted
    assert changes.added == []
    assert changes.modified == []


def test_remove_file_from_disk_then_detect_shows_deleted(tmp_path: Path) -> None:
    """AC #4 完整流程：先记录 → 从磁盘删除 → detect_changes 显示 deleted。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    p = _write(raw_dir, "doc.md", "initial")

    manifest = Manifest()
    manifest.files["doc.md"] = _file_state(p, raw_dir)

    # 首次检测：无变更
    changes1 = detect_changes(raw_dir, manifest, _settings())
    assert changes1.deleted == []

    # 删除文件
    p.unlink()

    # 再次检测：deleted
    changes2 = detect_changes(raw_dir, manifest, _settings())
    assert "doc.md" in changes2.deleted


# --------------------------------------------------------------------------- #
# added 集合
# --------------------------------------------------------------------------- #


def test_new_file_appears_in_added(tmp_path: Path) -> None:
    """磁盘有文件但 manifest 无记录 → added。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(raw_dir, "new.md", "new file")
    manifest = Manifest()

    changes = detect_changes(raw_dir, manifest, _settings())

    assert "new.md" in changes.added
    assert changes.modified == []
    assert changes.deleted == []


# --------------------------------------------------------------------------- #
# 三集合互斥性
# --------------------------------------------------------------------------- #


def test_three_sets_are_mutually_exclusive(tmp_path: Path) -> None:
    """added ∩ modified ∩ deleted = ∅（一个文件不可能同时出现在两个集合）。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # added 文件
    _write(raw_dir, "added.md", "new")
    # unchanged 文件
    p_keep = _write(raw_dir, "keep.md", "stable")
    # modified 文件（sha256 不同）
    _write(raw_dir, "mod.md", "changed content")

    manifest = Manifest()
    manifest.files["keep.md"] = _file_state(p_keep, raw_dir)
    manifest.files["mod.md"] = FileState(
        path="mod.md",
        sha256="0" * 64,  # 与磁盘不一致 → modified
        processed_at="2026-01-01T00:00:00Z",
    )
    manifest.files["gone.md"] = FileState(
        path="gone.md",
        sha256="0" * 64,
        processed_at="2026-01-01T00:00:00Z",
    )

    changes = detect_changes(raw_dir, manifest, _settings())

    added = set(changes.added)
    modified = set(changes.modified)
    deleted = set(changes.deleted)

    assert "added.md" in added
    assert "keep.md" not in (added | modified | deleted)
    assert "mod.md" in modified
    assert "gone.md" in deleted

    # 互斥断言
    assert added & modified == set()
    assert added & deleted == set()
    assert modified & deleted == set()


def test_changeset_results_are_sorted(tmp_path: Path) -> None:
    """added/modified/deleted 列表按字典序排序（可复现）。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in ("c.md", "a.md", "b.md"):
        _write(raw_dir, name)

    manifest = Manifest()
    changes = detect_changes(raw_dir, manifest, _settings())

    assert changes.added == ["a.md", "b.md", "c.md"]


# --------------------------------------------------------------------------- #
# 扩展名过滤 / 子目录
# --------------------------------------------------------------------------- #


def test_unsupported_extensions_ignored(tmp_path: Path) -> None:
    """非受支持扩展名（.log/.json）不进入 added。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(raw_dir, "doc.md", "keep")
    (raw_dir / "debug.log").write_text("noise", encoding="utf-8")
    (raw_dir / "config.json").write_text("{}", encoding="utf-8")

    manifest = Manifest()
    changes = detect_changes(raw_dir, manifest, _settings())

    assert changes.added == ["doc.md"]


def test_nested_subdirectory_files_detected(tmp_path: Path) -> None:
    """递归扫描子目录下的文档。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(raw_dir, "sub/note.md", "nested")
    _write(raw_dir, "top.md", "top level")

    manifest = Manifest()
    changes = detect_changes(raw_dir, manifest, _settings())

    assert "sub/note.md" in changes.added
    assert "top.md" in changes.added


def test_nonexistent_raw_dir_all_manifest_files_deleted(tmp_path: Path) -> None:
    """raw_dir 不存在时，manifest 所有记录视为 deleted。"""
    raw_dir = tmp_path / "raw"  # 不创建
    manifest = Manifest()
    manifest.files["a.md"] = FileState(
        path="a.md", sha256="x" * 64, processed_at="2026-01-01T00:00:00Z"
    )

    changes = detect_changes(raw_dir, manifest, _settings())

    assert changes.deleted == ["a.md"]
    assert changes.added == []
    assert changes.modified == []


# --------------------------------------------------------------------------- #
# WatchQueue：debounce + 入队（AC #5 前半段）
# --------------------------------------------------------------------------- #


class _FakeEvent:
    """模拟 watchdog FileSystemEvent。"""

    def __init__(
        self,
        src_path: str,
        *,
        event_type: str = "modified",
        is_directory: bool = False,
        dest_path: str | None = None,
    ) -> None:
        self.src_path = src_path
        self.event_type = event_type
        self.is_directory = is_directory
        if dest_path is not None:
            self.dest_path = dest_path


def test_watch_queue_debounces_before_enqueuing() -> None:
    """AC #5：并发事件在 debounce 窗口内不立即入队，窗口结束后统一入队。"""
    wq = WatchQueue(debounce_seconds=0.1)

    # 模拟并发写入多文件
    for name in ("a.md", "b.md", "c.md"):
        wq.on_any_event(_FakeEvent(f"/raw/{name}"))

    # debounce 窗口内：queue 为空
    assert wq.queue.empty()

    # 等待 debounce 结束（0.1s + 余量）
    time.sleep(0.25)

    # 窗口结束后：三个路径均已入队
    enqueued: list[str] = []
    while not wq.queue.empty():
        item = wq.queue.get_nowait()
        assert item is not wq.SENTINEL
        enqueued.append(str(item))

    assert sorted(enqueued) == ["/raw/a.md", "/raw/b.md", "/raw/c.md"]


def test_watch_queue_deduplicates_repeated_events_for_same_path() -> None:
    """同一路径多次事件（如连续保存）debounce 后只入队一次。"""
    wq = WatchQueue(debounce_seconds=0.05)

    for _ in range(5):
        wq.on_any_event(_FakeEvent("/raw/same.md"))

    time.sleep(0.15)

    enqueued: list[str] = []
    while not wq.queue.empty():
        item = wq.queue.get_nowait()
        if item is not wq.SENTINEL:
            enqueued.append(str(item))

    assert enqueued == ["/raw/same.md"]


def test_watch_queue_resets_debounce_on_new_event() -> None:
    """窗口内有新事件到达时 debounce 计时器重置（窗口延长）。"""
    wq = WatchQueue(debounce_seconds=0.1)

    wq.on_any_event(_FakeEvent("/raw/a.md"))
    time.sleep(0.06)  # 接近但未到窗口
    wq.on_any_event(_FakeEvent("/raw/b.md"))  # 重置计时器
    time.sleep(0.06)  # 从第二个事件算仅过 0.06s < 0.1s

    # 窗口未结束：无入队
    assert wq.queue.empty()

    time.sleep(0.1)  # 现在到窗口了

    enqueued: list[str] = []
    while not wq.queue.empty():
        item = wq.queue.get_nowait()
        if item is not wq.SENTINEL:
            enqueued.append(str(item))

    assert sorted(enqueued) == ["/raw/a.md", "/raw/b.md"]


def test_watch_queue_ignores_unsupported_extensions() -> None:
    """非受支持扩展名事件不入队。"""
    wq = WatchQueue(debounce_seconds=0.01)
    wq.on_any_event(_FakeEvent("/raw/debug.log"))
    wq.on_any_event(_FakeEvent("/raw/config.json"))
    wq.on_any_event(_FakeEvent("/raw/notes.md"))

    time.sleep(0.05)

    enqueued: list[str] = []
    while not wq.queue.empty():
        item = wq.queue.get_nowait()
        if item is not wq.SENTINEL:
            enqueued.append(str(item))

    assert enqueued == ["/raw/notes.md"]


def test_watch_queue_ignores_directory_events() -> None:
    """目录事件忽略。"""
    wq = WatchQueue(debounce_seconds=0.01)
    wq.on_any_event(_FakeEvent("/raw/subdir", is_directory=True))
    wq.flush_now()

    assert wq.queue.empty()


def test_watch_queue_close_sends_sentinel() -> None:
    """close() 入队 SENTINEL，worker 据此退出。"""
    wq = WatchQueue(debounce_seconds=1.0)
    wq.close()

    item = wq.queue.get_nowait()
    assert item is wq.SENTINEL


def test_watch_queue_moved_event_tracks_dest_path() -> None:
    """moved 事件追踪 dest_path（新位置作为 added 处理）。"""
    wq = WatchQueue(debounce_seconds=0.01)
    wq.on_any_event(
        _FakeEvent("/raw/old.md", event_type="moved", dest_path="/raw/new.md")
    )

    time.sleep(0.05)

    enqueued: list[str] = []
    while not wq.queue.empty():
        item = wq.queue.get_nowait()
        if item is not wq.SENTINEL:
            enqueued.append(str(item))

    # old.md 和 new.md 都应入队
    assert "/raw/old.md" in enqueued
    assert "/raw/new.md" in enqueued


# --------------------------------------------------------------------------- #
# AC #5 完整：单 worker 串行消费（无并发执行 compile）
# --------------------------------------------------------------------------- #


def test_start_watch_single_worker_consumes_serially(tmp_path: Path) -> None:
    """AC #5：并发写入多文件 → debounce 后并入 queue → 单 worker 串行消费。

    用Semaphore(max=1) 非阻塞获取检测：若 worker 并发执行 on_change，
    acquire(blocking=False) 会失败并记录 concurrency_violation。
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    concurrency_violations: list[str] = []
    active_lock = threading.Lock()
    is_active = False
    processed: list[str] = []
    processed_lock = threading.Lock()

    def on_change(path: str) -> None:
        nonlocal is_active
        with active_lock:
            if is_active:
                concurrency_violations.append(path)
                return
            is_active = True
        try:
            # 模拟 compile 工作（持锁期间睡眠）
            time.sleep(0.02)
            with processed_lock:
                processed.append(path)
        finally:
            with active_lock:
                is_active = False

    ctx = start_watch(raw_dir, on_change, debounce_seconds=0.05)

    try:
        # 并发写入多文件（触发 watchdog 事件）
        files = []
        for name in ("a.md", "b.md", "c.md", "d.md"):
            p = raw_dir / name
            p.write_text(f"content {name}", encoding="utf-8")
            files.append(name)

        # 等待 debounce + worker 处理（watchdog 事件传播 + debounce + 串行处理 4 文件）
        # 4 文件 × 20ms = 80ms 最少 + debounce 50ms + watchdog 延迟余量
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with processed_lock:
                if len(processed) >= len(files):
                    break
            time.sleep(0.05)
    finally:
        ctx.stop()

    # 所有文件均被处理
    with processed_lock:
        assert sorted(processed) == sorted(f"/{raw_dir.name}/{n}" for n in files) or (
            # watchdog 可能以绝对或相对路径回调；只验证文件名都在
            all(n in "/".join(processed) for n in files)
            and len(processed) == len(files)
        )

    # 核心断言：无并发执行
    assert concurrency_violations == [], (
        f"worker executed on_change concurrently: {concurrency_violations}"
    )


def test_start_watch_stop_cleans_up_observer_and_worker(tmp_path: Path) -> None:
    """stop() 后 worker 线程退出，资源清理完毕。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    def on_change(path: str) -> None:
        pass

    ctx = start_watch(raw_dir, on_change, debounce_seconds=0.01)

    # 立即停止
    ctx.stop()

    # worker 线程已退出（SENTINEL 收到后终止）
    assert not ctx.worker.is_alive()


# --------------------------------------------------------------------------- #
# ingest 编排（附加覆盖）
# --------------------------------------------------------------------------- #


def _make_registry() -> LoaderRegistry:
    reg = LoaderRegistry()
    reg.register(UnstructuredLoader())
    return reg


def test_ingest_returns_changes_and_documents(tmp_path: Path) -> None:
    """ingest 编排 detect → load → chunk，返回 changes + documents。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(raw_dir, "a.md", "short content")

    manifest = Manifest()
    result = ingest(raw_dir, manifest, _make_registry(), _settings())

    assert isinstance(result, IngestResult)
    assert "a.md" in result.changes.added
    assert "a.md" in result.documents

    doc = result.documents["a.md"]
    assert doc.format == "md"
    assert doc.sha256 == sha256_file(raw_dir / "a.md")
    assert len(doc.chunks) >= 1
    assert doc.chunks[0].source_file == "a.md"


def test_ingest_long_document_chunks_filled(tmp_path: Path) -> None:
    """ingest 对长文档填充多块 chunks。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    long_text = "The quick brown fox jumps. " * 200
    _write(raw_dir, "long.md", long_text)

    manifest = Manifest()
    result = ingest(
        raw_dir, manifest, _make_registry(), _settings(chunk_max_tokens=50, chunk_overlap_tokens=10)
    )

    doc = result.documents["long.md"]
    assert len(doc.chunks) >= 2
    for chunk in doc.chunks:
        assert chunk.token_count <= 50


def test_ingest_skips_unsupported_files(tmp_path: Path) -> None:
    """不支持的扩展名被 LoaderRegistry 抛 UnsupportedFormatError，ingest 跳过并记日志。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(raw_dir, "ok.md", "good")
    (raw_dir / "bad.xyz").write_text("unsupported", encoding="utf-8")

    manifest = Manifest()
    result = ingest(raw_dir, manifest, _make_registry(), _settings())

    # .md 正常加载
    assert "ok.md" in result.documents
    # .xyz 跳过（不在 documents），但仍出现在 changes.added（detect_changes 按扩展名过滤）
    assert "bad.xyz" not in result.documents
    assert "ok.md" in result.changes.added


def test_ingest_file_loads_single_document(tmp_path: Path) -> None:
    """ingest_file 加载单文件为 Document。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    p = _write(raw_dir, "note.md", "a note")

    doc = ingest_file(p, raw_dir, _make_registry(), _settings())

    assert doc.path == p
    assert doc.content == "a note"
    assert doc.format == "md"
    assert len(doc.chunks) == 1


# --------------------------------------------------------------------------- #
# ChangeSet.has_changes
# --------------------------------------------------------------------------- #


def test_changeset_has_changes_true_when_any_nonempty() -> None:
    cs = ChangeSet(added=["a.md"])
    assert cs.has_changes
    cs2 = ChangeSet(modified=["b.md"])
    assert cs2.has_changes
    cs3 = ChangeSet(deleted=["c.md"])
    assert cs3.has_changes


def test_changeset_has_changes_false_when_all_empty() -> None:
    cs = ChangeSet()
    assert not cs.has_changes

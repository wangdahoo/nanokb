"""ExtractionCache 内容寻址缓存单测（plan §3.3 + Phase 4/7 + Feature s1-feat-007）。

覆盖验收标准：
- put→get 往返一致（``model_dump`` 相等）。
- 未 put 的 key → get 返回 None。
- 三维（sha256 / extraction_config / llm_model）任一变化 → key 不同 → get 返回 None。
- 跨文件去重：同 key 二次 put 覆盖第一次（key 与 source_file 无关）。
- 损坏 JSON 文件被忽略返回 None（best-effort 容错：JSONDecodeError / ValidationError）。

用 ``tmp_path`` 隔离，零真实 LLM / 网络。
被测代码：``src/nanokb/extract/cache.py``（Feature s1-feat-004）。
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from nanokb.extract.cache import ExtractionCache
from nanokb.models import Concept, Confidence, ExtractionResult, Track, Triple

# --------------------------------------------------------------------------- #
# 辅助构造
# --------------------------------------------------------------------------- #

_SHA = "a" * 64
_EXTRACTION_SIG = "extraction_sig_value"
_LLM_MODEL = "glm-5.1"


def _result(
    *,
    head: str = "A",
    relation: str = "rel",
    tail: str = "B",
    concept: str = "A",
    description: str = "desc",
    source_file: str = "doc.md",
) -> ExtractionResult:
    """构造带一条 triple + 一条 concept 的 ExtractionResult。"""
    return ExtractionResult(
        triples=[
            Triple(
                head=head,
                relation=relation,
                tail=tail,
                confidence=Confidence.EXTRACTED,
                source_file=source_file,
                track=Track.SEMANTIC,
                chunk_index=0,
            )
        ],
        concepts=[
            Concept(
                name=concept,
                description=description,
                source_file=source_file,
                confidence=Confidence.EXTRACTED,
                node_type="concept",
            )
        ],
    )


def _expected_key_path(cache_dir: Path, sha: str, sig: str, llm_model: str) -> Path:
    """复现 ExtractionCache._key 的落盘路径，供手动写入损坏文件。"""
    raw = f"{sha}|{sig}|{llm_model}"
    key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


# --------------------------------------------------------------------------- #
# AC #1：put→get 往返一致
# --------------------------------------------------------------------------- #


def test_put_get_roundtrip(tmp_path: Path) -> None:
    """put 后以同三维 key get → 返回结果的 model_dump 与原 result 相等。"""
    cache = ExtractionCache(tmp_path / "cache")
    original = _result(head="X", relation="uses", tail="Y", concept="X", description="d")
    cache.put(_SHA, _EXTRACTION_SIG, _LLM_MODEL, original)

    got = cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL)
    assert got is not None
    assert got.model_dump(mode="json") == original.model_dump(mode="json")


def test_put_get_roundtrip_empty_result(tmp_path: Path) -> None:
    """空 triples/concepts 的 ExtractionResult 往返也一致。"""
    cache = ExtractionCache(tmp_path / "cache")
    original = ExtractionResult()
    cache.put(_SHA, _EXTRACTION_SIG, _LLM_MODEL, original)

    got = cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL)
    assert got is not None
    assert got.model_dump(mode="json") == original.model_dump(mode="json")
    assert got.triples == []
    assert got.concepts == []


# --------------------------------------------------------------------------- #
# AC #2：未 put 的 key → None
# --------------------------------------------------------------------------- #


def test_get_missing_key_returns_none(tmp_path: Path) -> None:
    """未 put 的三维 key → get 返回 None（路径不存在分支）。"""
    cache = ExtractionCache(tmp_path / "cache")
    assert cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL) is None


def test_get_on_nonexistent_cache_dir_returns_none(tmp_path: Path) -> None:
    """cache_dir 本身不存在时 get 返回 None（path.exists() 为 False，不抛异常）。"""
    cache = ExtractionCache(tmp_path / "does_not_exist")
    assert cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL) is None


# --------------------------------------------------------------------------- #
# AC #3：三维变化 key 不同
# --------------------------------------------------------------------------- #


def test_different_sha256_yields_different_key(tmp_path: Path) -> None:
    """改 sha256 → key 不同 → get 返回 None；原 key 仍命中。"""
    cache = ExtractionCache(tmp_path / "cache")
    cache.put(_SHA, _EXTRACTION_SIG, _LLM_MODEL, _result())

    assert cache.get("b" * 64, _EXTRACTION_SIG, _LLM_MODEL) is None
    # 原 key 仍命中
    assert cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL) is not None


def test_different_extraction_config_yields_different_key(tmp_path: Path) -> None:
    """改 extraction_config → key 不同 → get 返回 None。"""
    cache = ExtractionCache(tmp_path / "cache")
    cache.put(_SHA, _EXTRACTION_SIG, _LLM_MODEL, _result())

    assert cache.get(_SHA, "different_sig", _LLM_MODEL) is None


def test_different_llm_model_yields_different_key(tmp_path: Path) -> None:
    """改 llm_model → key 不同 → get 返回 None。"""
    cache = ExtractionCache(tmp_path / "cache")
    cache.put(_SHA, _EXTRACTION_SIG, _LLM_MODEL, _result())

    assert cache.get(_SHA, _EXTRACTION_SIG, "gpt-4o") is None


# --------------------------------------------------------------------------- #
# 跨文件去重：同 key 二次 put 覆盖（key 与 source_file 无关）
# --------------------------------------------------------------------------- #


def test_same_key_overwrites_previous_value(tmp_path: Path) -> None:
    """同三维 key 二次 put（来自不同 source_file 的结果）覆盖第一次。

    key 不含 source_file → 同内容不同路径自动共享同一缓存条目；
    pipeline._normalize_result_source 在加载时盖 rel_key 保证 correctness。
    """
    cache = ExtractionCache(tmp_path / "cache")
    first = _result(head="A1", tail="B1", source_file="one.md")
    second = _result(head="A2", tail="B2", source_file="two.md")

    cache.put(_SHA, _EXTRACTION_SIG, _LLM_MODEL, first)
    cache.put(_SHA, _EXTRACTION_SIG, _LLM_MODEL, second)

    got = cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL)
    assert got is not None
    # 命中第二次（覆盖语义）
    assert got.model_dump(mode="json") == second.model_dump(mode="json")
    assert got.model_dump(mode="json") != first.model_dump(mode="json")


def test_key_is_independent_of_source_file(tmp_path: Path) -> None:
    """key 不含 source_file → 同三维 key 不同路径命中同一条目（跨文件共享）。

    pipeline._normalize_result_source 在加载后盖 rel_key 保证 correctness；
    此处验证缓存层只看 (sha256, extraction_config, llm_model)。
    """
    cache = ExtractionCache(tmp_path / "cache")
    # 来自 path-a.md 的结果
    cache.put(_SHA, _EXTRACTION_SIG, _LLM_MODEL, _result(source_file="path-a.md"))

    # 任何"调用方"（即便是 path-b.md 的文件）以同三维 key 查询都命中
    got = cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL)
    assert got is not None
    assert got.triples[0].source_file == "path-a.md"


# --------------------------------------------------------------------------- #
# AC #4：损坏 JSON 文件被忽略（best-effort 容错）
# --------------------------------------------------------------------------- #


def test_corrupt_json_file_returns_none(tmp_path: Path) -> None:
    """手动写入非法 JSON 到 <key>.json → get 返回 None，不抛异常（JSONDecodeError）。"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = ExtractionCache(cache_dir)

    corrupt_path = _expected_key_path(cache_dir, _SHA, _EXTRACTION_SIG, _LLM_MODEL)
    corrupt_path.write_text("{not valid json", encoding="utf-8")

    # get 不抛异常，返回 None
    assert cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL) is None


def test_invalid_extraction_result_schema_returns_none(tmp_path: Path) -> None:
    """合法 JSON 但 schema 不匹配（Triple 缺必填字段）→ 视为损坏返回 None。

    覆盖 pydantic ValidationError 路径：get 内的 except Exception 捕获后视为 miss。
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = ExtractionCache(cache_dir)

    bad_path = _expected_key_path(cache_dir, _SHA, _EXTRACTION_SIG, _LLM_MODEL)
    # 合法 JSON，但 Triple 缺 head/relation/tail/confidence/source_file 等必填字段
    bad_path.write_text(
        json.dumps({"triples": [{"head": "A"}], "concepts": []}),
        encoding="utf-8",
    )

    assert cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL) is None


def test_corrupt_file_does_not_block_subsequent_put(tmp_path: Path) -> None:
    """损坏文件被忽略后，同 key 仍可被 put 覆盖为合法值并命中。"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = ExtractionCache(cache_dir)

    corrupt_path = _expected_key_path(cache_dir, _SHA, _EXTRACTION_SIG, _LLM_MODEL)
    corrupt_path.write_text("{not valid json", encoding="utf-8")

    # 损坏文件被忽略
    assert cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL) is None

    # put 覆盖后正常命中（atomic_write_json 原子替换）
    original = _result(head="Z", tail="W")
    cache.put(_SHA, _EXTRACTION_SIG, _LLM_MODEL, original)
    got = cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL)
    assert got is not None
    assert got.model_dump(mode="json") == original.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# 并发安全（方案 §5.3，Feature s1-feat-004）
# --------------------------------------------------------------------------- #


def test_concurrent_put_different_keys_no_corruption(tmp_path: Path) -> None:
    """多线程并发 put 不同 key → 各自独立落盘，无文件损坏，全部可命中。

    ExtractionCache 经 atomic_write_text（tempfile.mkstemp 唯一名 + os.replace）
    写不同 key 的不同文件，并发安全。文档级并发（doc_concurrency>1）下 cache.put
    会被多 worker 并发调用。
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = ExtractionCache(cache_dir)
    n = 12

    shas = [f"{i:064d}" for i in range(n)]
    results = [_result(head=f"H{i}", tail=f"T{i}", concept=f"C{i}") for i in range(n)]
    errors: list[BaseException] = []
    err_lock = threading.Lock()

    def worker(idx: int) -> None:
        try:
            cache.put(shas[idx], _EXTRACTION_SIG, _LLM_MODEL, results[idx])
        except BaseException as exc:  # noqa: BLE001
            with err_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent put raised: {errors}"
    # 每个 key 都能命中且内容正确（无损坏）
    for i in range(n):
        got = cache.get(shas[i], _EXTRACTION_SIG, _LLM_MODEL)
        assert got is not None, f"key {i} missing after concurrent put"
        assert got.model_dump(mode="json") == results[i].model_dump(mode="json")


def test_concurrent_put_same_key_eventually_consistent(tmp_path: Path) -> None:
    """多线程并发 put 同 key（内容寻址：同 key 即同内容）→ 最终一致、无损坏。

    内容寻址缓存中同 key 意味着同 (sha|sig|model)，对应同一份内容；atomic_write_text
    的 ``os.replace`` 原子替换保证目标文件永不损坏（要么旧值要么完整新值）。

    平台注记：Windows 上对**同一目标**的并发 ``os.replace`` 可能瞬时抛
    ``PermissionError``（文件锁，另一线程持有句柄）——这是 OS 已知行为，非缓存
    缺陷；pipeline 层已 try/except 捕获（"result kept in memory"）。故本测试只断言
    「无非 PermissionError 致命错误 + 最终可读到一致内容」，不要求每次 put 都成功。
    先做一次 warm-up put 保证目标存在且内容为 shared，随后并发覆写同 key。
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = ExtractionCache(cache_dir)
    shared = _result(head="Shared", tail="Value", concept="K")

    # warm-up：保证目标文件存在且内容为 shared（即使并发覆写全失败，最终值仍一致）
    cache.put(_SHA, _EXTRACTION_SIG, _LLM_MODEL, shared)

    n = 8
    errors: list[BaseException] = []
    err_lock = threading.Lock()

    def worker() -> None:
        try:
            cache.put(_SHA, _EXTRACTION_SIG, _LLM_MODEL, shared)
        except BaseException as exc:  # noqa: BLE001
            with err_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 仅容忍 Windows 同目标 os.replace 的瞬时 PermissionError；其它异常视为缺陷
    fatal = [e for e in errors if not isinstance(e, PermissionError)]
    assert not fatal, f"non-permission error during concurrent same-key put: {fatal}"

    # 最终一致：cache 未损坏，可读到 shared 内容（warm-up 或某次覆写成功）
    got = cache.get(_SHA, _EXTRACTION_SIG, _LLM_MODEL)
    assert got is not None
    assert got.model_dump(mode="json") == shared.model_dump(mode="json")

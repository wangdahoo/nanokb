"""三层配置签名 helper 单测（plan §3.3 + Phase 7 + Feature s1-feat-007）。

覆盖验收标准：
- 三签名确定性（同 Settings 两次调用相等，且为 64 位 hex）。
- 各归属字段变化触发对应签名变化（extraction/index/embedding 各自的覆盖字段）。
- code_languages 反序产生相同签名（集合语义，显式 ``sorted`` 正确性）。
- 三签名互不干扰（改 extraction 字段不影响 index/embedding 签名，反之亦然）。
- 非 ASCII 配置值签名稳定（``ensure_ascii=False``，评审 Optimization #2）。

被测代码：``src/nanokb/config_signature.py``（Feature s1-feat-001）。
"""

from __future__ import annotations

from typing import Any

from nanokb.config import Settings
from nanokb.config_signature import (
    embedding_config_signature,
    extraction_config_signature,
    index_config_signature,
)

# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #


def _settings(**overrides: Any) -> Settings:
    """构造 Settings，提供与项目默认一致的基础值，避免依赖 .env / 环境变量。"""
    defaults: dict[str, Any] = {
        "extractor_version": "1",
        "chunk_max_tokens": 3000,
        "chunk_overlap_tokens": 200,
        "concept_description_strategy": "last_write_wins",
        "code_languages": ["python", "javascript", "java"],
        "fallback_description_max_edges": 5,
        "leiden_symmetrize": "sum",
        "embedding_model": "text-embedding-3-small",
        "embedding_provider": "openai",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _assert_hex_64(sig: str) -> None:
    """断言签名是 64 位小写 hex（sha256 标准输出）。"""
    assert len(sig) == 64
    int(sig, 16)


# --------------------------------------------------------------------------- #
# 三签名确定性
# --------------------------------------------------------------------------- #


def test_three_signatures_deterministic_and_hex() -> None:
    """同 Settings 调用两次，三签名均一致，且均为 64 位 hex 字符串。"""
    s = _settings()
    for fn in (
        extraction_config_signature,
        index_config_signature,
        embedding_config_signature,
    ):
        sig_a = fn(s)
        sig_b = fn(s)
        assert sig_a == sig_b
        _assert_hex_64(sig_a)


def test_three_signatures_mutually_distinct() -> None:
    """三签名涵盖字段不同 → 默认 Settings 下互不相等。"""
    s = _settings()
    sigs = {
        extraction_config_signature(s),
        index_config_signature(s),
        embedding_config_signature(s),
    }
    assert len(sigs) == 3


# --------------------------------------------------------------------------- #
# extraction_config_signature 字段覆盖
# --------------------------------------------------------------------------- #


def test_extraction_signature_changes_on_each_field() -> None:
    """extraction 归属字段任一变化触发签名变化：extractor_version /
    chunk_max_tokens / chunk_overlap_tokens / concept_description_strategy /
    code_languages。"""
    base_sig = extraction_config_signature(_settings())

    cases: list[dict[str, Any]] = [
        {"extractor_version": "2"},
        {"chunk_max_tokens": 4096},
        {"chunk_overlap_tokens": 100},
        {"concept_description_strategy": "concat_dedup"},
        {"code_languages": ["python"]},
    ]
    for override in cases:
        changed = extraction_config_signature(_settings(**override))
        assert changed != base_sig, f"extraction sig 未变化: {override}"
        _assert_hex_64(changed)


# --------------------------------------------------------------------------- #
# code_languages 顺序无关（集合语义，评审 Optimization #4/#6）
# --------------------------------------------------------------------------- #


def test_extraction_signature_code_languages_order_invariant() -> None:
    """code_languages 反序产生相同签名（extraction_config_signature 内显式 sorted）。

    直接验证 ``sorted`` 正确性：json.dumps(sort_keys=True) 只排 dict 键不排 list
    元素 → 同集合不同顺序若漏掉 sorted 会产生不同签名。
    """
    forward = _settings(code_languages=["python", "javascript"])
    reverse = _settings(code_languages=["javascript", "python"])
    assert extraction_config_signature(forward) == extraction_config_signature(reverse)

    # 不同集合（多一个语言）仍可区分
    extended = _settings(code_languages=["python", "javascript", "java"])
    assert extraction_config_signature(extended) != extraction_config_signature(forward)


# --------------------------------------------------------------------------- #
# index_config_signature 字段覆盖
# --------------------------------------------------------------------------- #


def test_index_signature_changes_on_each_field() -> None:
    """index 归属字段任一变化触发签名变化：fallback_description_max_edges /
    leiden_symmetrize。"""
    base_sig = index_config_signature(_settings())

    cases: list[dict[str, Any]] = [
        {"fallback_description_max_edges": 10},
        {"leiden_symmetrize": "max"},
    ]
    for override in cases:
        changed = index_config_signature(_settings(**override))
        assert changed != base_sig, f"index sig 未变化: {override}"
        _assert_hex_64(changed)


# --------------------------------------------------------------------------- #
# embedding_config_signature 字段覆盖
# --------------------------------------------------------------------------- #


def test_embedding_signature_changes_on_each_field() -> None:
    """embedding 归属字段任一变化触发签名变化：embedding_model / embedding_provider。"""
    base_sig = embedding_config_signature(_settings())

    cases: list[dict[str, Any]] = [
        {"embedding_model": "text-embedding-3-large"},
        {"embedding_provider": "ollama"},
    ]
    for override in cases:
        changed = embedding_config_signature(_settings(**override))
        assert changed != base_sig, f"embedding sig 未变化: {override}"
        _assert_hex_64(changed)


# --------------------------------------------------------------------------- #
# 三签名互不干扰
# --------------------------------------------------------------------------- #


def test_extraction_field_change_does_not_affect_index_or_embedding() -> None:
    """改 extraction 归属字段 → extraction 签名变，index/embedding 签名不变。"""
    base = _settings()
    changed = _settings(chunk_max_tokens=8192)

    assert extraction_config_signature(changed) != extraction_config_signature(base)
    assert index_config_signature(changed) == index_config_signature(base)
    assert embedding_config_signature(changed) == embedding_config_signature(base)


def test_index_field_change_does_not_affect_extraction_or_embedding() -> None:
    """改 index 归属字段 → index 签名变，extraction/embedding 签名不变。"""
    base = _settings()
    changed = _settings(fallback_description_max_edges=8)

    assert index_config_signature(changed) != index_config_signature(base)
    assert extraction_config_signature(changed) == extraction_config_signature(base)
    assert embedding_config_signature(changed) == embedding_config_signature(base)


def test_embedding_field_change_does_not_affect_extraction_or_index() -> None:
    """改 embedding 归属字段 → embedding 签名变，extraction/index 签名不变。"""
    base = _settings()
    changed = _settings(embedding_model="text-embedding-3-large")

    assert embedding_config_signature(changed) != embedding_config_signature(base)
    assert extraction_config_signature(changed) == extraction_config_signature(base)
    assert index_config_signature(changed) == index_config_signature(base)


# --------------------------------------------------------------------------- #
# 非 ASCII 配置值签名稳定（评审 Optimization #2）
# --------------------------------------------------------------------------- #


def test_signatures_stable_with_non_ascii_values() -> None:
    """非 ASCII 配置值（中文 extractor_version / 中文 code_languages 条目）不引发
    编码异常，签名仍为 64 位 hex 且确定。

    ``_sig`` 用 ``ensure_ascii=False`` 保证 UTF-8 字节直接哈希，跨平台稳定；
    若误用默认 ``ensure_ascii=True``，json.dumps 会产出 ``\\uXXXX`` 转义，
    不同平台/版本可能产生不同字节序列。
    """
    s1 = _settings(extractor_version="版本-1", code_languages=["python", "蟒蛇"])
    s2 = _settings(extractor_version="版本-1", code_languages=["python", "蟒蛇"])

    sig1 = extraction_config_signature(s1)
    sig2 = extraction_config_signature(s2)
    assert sig1 == sig2
    _assert_hex_64(sig1)

    # 不同非 ASCII 值仍可区分
    s3 = _settings(extractor_version="版本-2", code_languages=["python", "蟒蛇"])
    assert extraction_config_signature(s3) != sig1


def test_non_ascii_code_languages_order_invariant() -> None:
    """含非 ASCII 条目的 code_languages 反序仍产生相同签名。"""
    forward = _settings(code_languages=["python", "蟒蛇", "日本語"])
    reverse = _settings(code_languages=["日本語", "蟒蛇", "python"])
    assert extraction_config_signature(forward) == extraction_config_signature(reverse)

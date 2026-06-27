"""EmbeddingCache 单测（方案 §4.7，Feature s3-feat-001）。

覆盖验收标准：
- AC1.3：损坏 cache 文件视为 miss 重 embed。
- AC1.4：enable_embed_cache=False, embed_concurrency=1 零回归（正交性，Medium #4）。
- AC1.5：embedding_dim=0（探测失败）时 put 不写 cache 文件（Opt#4）。
- get 四态：命中 / miss / 损坏 / 维度不匹配。
- embed_batch：cache 命中 / miss / 二次全命中 / 原序组装 / 长度校验 raise /
  去重（round 3 Opt#1）。
- enable_embed_cache=False 正交性：get/put no-op 但 embed_batch 仍返回正确向量。

全部离线，用 tmp_path 隔离，FakeEmbedder 返回可控维度 embedding。
被测代码：``src/nanokb/llm/embed_cache.py``（Feature s3-feat-001）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nanokb.llm.embed_cache import EMBED_BATCH_SIZE, EmbeddingCache

# ── 测试 doubles ─────────────────────────────────────────────────────


class FakeEmbedder:
    """模拟 EmbeddingClient，返回固定维度 embedding 并记录调用。"""

    def __init__(self, embedding_dim: int = 8) -> None:
        self._dim = embedding_dim
        self.embed_calls: int = 0
        self.embedded_texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        self.embedded_texts.extend(texts)
        # 用文本首字符的 ord 编码进向量，便于断言「同文本同向量」
        return [
            [float(ord(t[0]) if t else 0) / 100.0] + [0.0] * (self._dim - 1)
            if self._dim > 0
            else []
            for t in texts
        ]


class ShortFakeEmbedder:
    """返回向量数 < 输入文本数的 embedder（验证长度校验 raise，Medium #5②）。"""

    def __init__(self) -> None:
        self.embed_calls: int = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        # 故意少返回一条
        return [[0.1, 0.2] for _ in texts[:-1]]


# ── 辅助：复现 _key 落盘路径，供手动写入损坏文件 ──────────────────────


def _expected_key_path(
    cache_dir: Path,
    description: str,
    embedding_model: str,
    embedding_dim: int,
) -> Path:
    desc_sha = hashlib.sha256(description.encode("utf-8")).hexdigest()
    raw = f"{desc_sha}|{embedding_model}|{embedding_dim}"
    key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


# ══════════════════════════════════════════════════════════════════════
# get 四态：命中 / miss / 损坏 / 维度不匹配
# ══════════════════════════════════════════════════════════════════════


def test_get_hit_returns_vector(tmp_path: Path) -> None:
    """put 后 get 同 description 返回向量。"""
    cache = EmbeddingCache(
        tmp_path / "cache", "test-model", 8, FakeEmbedder(8), enable_cache=True
    )
    vec = [0.1] * 8
    cache.put("hello", vec)
    got = cache.get("hello")
    assert got is not None
    assert got == vec


def test_get_miss_returns_none(tmp_path: Path) -> None:
    """未 put 的 description → get 返回 None。"""
    cache = EmbeddingCache(
        tmp_path / "cache", "test-model", 8, FakeEmbedder(8), enable_cache=True
    )
    assert cache.get("never-put") is None


def test_get_corrupt_json_returns_none(tmp_path: Path) -> None:
    """损坏 JSON 文件视为 miss（best-effort，不抛异常）。"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = EmbeddingCache(cache_dir, "test-model", 8, FakeEmbedder(8), enable_cache=True)
    corrupt = _expected_key_path(cache_dir, "corrupted", "test-model", 8)
    corrupt.write_text("{not valid json", encoding="utf-8")
    assert cache.get("corrupted") is None


def test_get_wrong_vector_type_returns_none(tmp_path: Path) -> None:
    """合法 JSON 但 vector 字段非 list → 视为 miss。"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = EmbeddingCache(cache_dir, "test-model", 8, FakeEmbedder(8), enable_cache=True)
    bad = _expected_key_path(cache_dir, "badvec", "test-model", 8)
    bad.write_text(
        json.dumps({"embedding_model": "test-model", "embedding_dim": 8, "vector": "not-a-list"}),
        encoding="utf-8",
    )
    assert cache.get("badvec") is None


def test_get_model_mismatch_returns_none(tmp_path: Path) -> None:
    """embedding_model 不匹配 → 视为过期（miss）。"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = EmbeddingCache(cache_dir, "new-model", 8, FakeEmbedder(8), enable_cache=True)
    # 用旧 model 写入
    bad = _expected_key_path(cache_dir, "hello", "new-model", 8)
    bad.write_text(
        json.dumps(
            {"embedding_model": "old-model", "embedding_dim": 8, "vector": [0.1] * 8}
        ),
        encoding="utf-8",
    )
    assert cache.get("hello") is None


def test_get_dim_mismatch_returns_none(tmp_path: Path) -> None:
    """embedding_dim 不匹配 → 视为过期（miss）。"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = EmbeddingCache(cache_dir, "test-model", 8, FakeEmbedder(8), enable_cache=True)
    # 用 dim=4 写入
    bad = _expected_key_path(cache_dir, "hello", "test-model", 8)
    bad.write_text(
        json.dumps(
            {"embedding_model": "test-model", "embedding_dim": 4, "vector": [0.1] * 4}
        ),
        encoding="utf-8",
    )
    assert cache.get("hello") is None


# ══════════════════════════════════════════════════════════════════════
# put 原子写 + dim=0 no-op（AC1.5，Opt#4）
# ══════════════════════════════════════════════════════════════════════


def test_put_writes_cache_file(tmp_path: Path) -> None:
    """put 后 cache 目录下存在对应 key 文件，内容含 model/dim/vector。"""
    cache_dir = tmp_path / "cache"
    cache = EmbeddingCache(cache_dir, "test-model", 8, FakeEmbedder(8), enable_cache=True)
    cache.put("hello", [0.1] * 8)
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["embedding_model"] == "test-model"
    assert data["embedding_dim"] == 8
    assert data["vector"] == [0.1] * 8


def test_put_dim_zero_is_noop(tmp_path: Path) -> None:
    """AC1.5：embedding_dim=0（探测失败）时 put 不写文件（Opt#4）。"""
    cache_dir = tmp_path / "cache"
    cache = EmbeddingCache(cache_dir, "test-model", 0, FakeEmbedder(0), enable_cache=True)
    cache.put("hello", [0.1] * 8)
    assert not cache_dir.exists() or not list(cache_dir.glob("*.json"))


def test_put_overwrite_same_key(tmp_path: Path) -> None:
    """同 description 二次 put 覆盖（atomic_write_json 原子替换）。"""
    cache = EmbeddingCache(
        tmp_path / "cache", "test-model", 8, FakeEmbedder(8), enable_cache=True
    )
    cache.put("hello", [0.1] * 8)
    cache.put("hello", [0.9] * 8)
    got = cache.get("hello")
    assert got == [0.9] * 8


def test_put_disable_cache_is_noop(tmp_path: Path) -> None:
    """enable_cache=False 时 put 不写文件。"""
    cache_dir = tmp_path / "cache"
    cache = EmbeddingCache(
        cache_dir, "test-model", 8, FakeEmbedder(8), enable_cache=False
    )
    cache.put("hello", [0.1] * 8)
    # disable_cache 时不 mkdir，目录可能不存在
    assert not cache_dir.exists() or not list(cache_dir.glob("*.json"))


# ═══════════════════════════════embed_batch：命中 / miss / 原序组装 ═══
# ══════════════════════════════════════════════════════════════════════


def test_embed_batch_all_miss_embeds_and_caches(tmp_path: Path) -> None:
    """全 miss：embed_batch 调用 embedder.embed，写回 cache，返回等长向量。"""
    embedder = FakeEmbedder(8)
    cache = EmbeddingCache(tmp_path / "cache", "test-model", 8, embedder, enable_cache=True)
    texts = ["alpha", "beta", "gamma"]
    vecs = cache.embed_batch(texts)
    assert len(vecs) == 3
    # embed 被调用
    assert embedder.embed_calls >= 1
    # 已写回 cache（二次 get 命中）
    for t in texts:
        assert cache.get(t) is not None


def test_embed_batch_partial_hit_only_misses_embedded(tmp_path: Path) -> None:
    """部分命中：只对 miss 文本调用 embed（命中项不重复 embed）。"""
    embedder = FakeEmbedder(8)
    cache = EmbeddingCache(tmp_path / "cache", "test-model", 8, embedder, enable_cache=True)
    # 预置 cache
    cache.put("alpha", [0.5] * 8)
    embedder.embed_calls = 0  # 重置

    vecs = cache.embed_batch(["alpha", "beta"])
    assert len(vecs) == 2
    # alpha 命中，只 embed beta（1 次 embed 调用）
    assert embedder.embed_calls == 1
    # alpha 的向量是 cache 里的
    assert vecs[0] == [0.5] * 8


def test_embed_batch_second_run_all_hit_zero_embed(tmp_path: Path) -> None:
    """二次全命中：embed 调用次数 == 0（AC1.2 核心，中断重跑零成本）。"""
    embedder = FakeEmbedder(8)
    cache = EmbeddingCache(tmp_path / "cache", "test-model", 8, embedder, enable_cache=True)
    texts = ["alpha", "beta", "gamma"]
    cache.embed_batch(texts)
    first_calls = embedder.embed_calls

    # 二次：全命中
    embedder.embed_calls = 0
    vecs = cache.embed_batch(texts)
    assert embedder.embed_calls == 0
    assert len(vecs) == 3
    assert first_calls >= 1


def test_embed_batch_preserves_order(tmp_path: Path) -> None:
    """原序组装：返回向量顺序与输入 texts 顺序一致。"""
    embedder = FakeEmbedder(8)
    cache = EmbeddingCache(tmp_path / "cache", "test-model", 8, embedder, enable_cache=True)
    texts = ["a", "b", "c", "d", "e"]
    vecs = cache.embed_batch(texts)
    # FakeEmbedder 用首字符 ord 编码，可断言顺序对应
    for i, t in enumerate(texts):
        expected_first = float(ord(t[0])) / 100.0
        assert vecs[i][0] == expected_first


def test_embed_batch_corrupt_file_re_embeds(tmp_path: Path) -> None:
    """AC1.3：损坏 cache 文件视为 miss 重新 embed，其余命中。"""
    embedder = FakeEmbedder(8)
    cache_dir = tmp_path / "cache"
    cache = EmbeddingCache(cache_dir, "test-model", 8, embedder, enable_cache=True)
    # 预置全部命中
    cache.embed_batch(["good", "bad", "ugly"])
    embedder.embed_calls = 0

    # 损坏 "bad" 对应的 cache 文件
    corrupt = _expected_key_path(cache_dir, "bad", "test-model", 8)
    corrupt.write_text("{corrupt", encoding="utf-8")

    vecs = cache.embed_batch(["good", "bad", "ugly"])
    assert len(vecs) == 3
    # 只有 "bad" 重新 embed（1 次调用）
    assert embedder.embed_calls == 1
    # bad 现在已写回 cache
    assert cache.get("bad") is not None


# ══════════════════════════════════════════════════════════════════════
# embed_batch：长度校验 raise（Medium #5②）
# ══════════════════════════════════════════════════════════════════════


def test_embed_batch_length_mismatch_raises(tmp_path: Path) -> None:
    """Medium #5②：embedder 返回向量数 != 输入 batch 数 → raise RuntimeError。"""
    embedder = ShortFakeEmbedder()
    cache = EmbeddingCache(tmp_path / "cache", "test-model", 2, embedder, enable_cache=True)
    with pytest.raises(RuntimeError, match="embedder length mismatch"):
        cache.embed_batch(["a", "b", "c"])


# ══════════════════════════════════════════════════════════════════════
# embed_batch：去重（round 3 Opt#1）
# ══════════════════════════════════════════════════════════════════════


def test_embed_batch_dedup_repeated_descriptions(tmp_path: Path) -> None:
    """Opt#1：重复 description 去重——embed 调用数 == unique 文本数，广播回填。"""
    embedder = FakeEmbedder(8)
    cache = EmbeddingCache(tmp_path / "cache", "test-model", 8, embedder, enable_cache=True)
    # 5 个文本，其中 "dup" 重复 3 次，"unique" 1 次 → 2 个 unique
    texts = ["dup", "unique", "dup", "dup", "unique"]
    vecs = cache.embed_batch(texts)

    # 返回长度 == 输入长度
    assert len(vecs) == 5
    # embed 调用次数 == 1（一个 batch 含 2 unique 文本）
    assert embedder.embed_calls == 1
    # 同 key 的原始位置返回向量逐元素相等（广播回填）
    # "dup" 在位置 0, 2, 3
    assert vecs[0] == vecs[2] == vecs[3]
    # "unique" 在位置 1, 4
    assert vecs[1] == vecs[4]
    # dup 与 unique 不同
    assert vecs[0] != vecs[1]


def test_embed_batch_dedup_only_unique_embedded(tmp_path: Path) -> None:
    """Opt#1：embedded_texts 只含 unique 文本（无重复 embed）。"""
    embedder = FakeEmbedder(8)
    cache = EmbeddingCache(tmp_path / "cache", "test-model", 8, embedder, enable_cache=True)
    texts = ["x", "x", "x"]
    cache.embed_batch(texts)
    # 只 embed 一次 "x"
    assert embedder.embedded_texts == ["x"]


# ══════════════════════════════════════════════════════════════════════
# embed_batch：enable_embed_cache=False 正交性（AC1.4 / Medium #4）
# ══════════════════════════════════════════════════════════════════════


def test_embed_batch_disable_cache_still_returns_vectors(tmp_path: Path) -> None:
    """AC1.4 / Medium #4：enable_cache=False 时 get/put no-op，但 embed_batch
    仍返回正确向量（cache 与并发正交）。"""
    embedder = FakeEmbedder(8)
    cache_dir = tmp_path / "cache"
    cache = EmbeddingCache(
        cache_dir, "test-model", 8, embedder, enable_cache=False
    )
    vecs = cache.embed_batch(["alpha", "beta"])
    assert len(vecs) == 2
    # embed 被调用（未走 cache）
    assert embedder.embed_calls == 1
    # get 恒 None（no-op）
    assert cache.get("alpha") is None
    # cache 目录无文件（put no-op）
    assert not cache_dir.exists() or not list(cache_dir.glob("*.json"))


def test_embed_batch_disable_cache_zero_regression(tmp_path: Path) -> None:
    """AC1.4：enable_embed_cache=False, embed_concurrency=1 行为与改造前一致
    （embed 调用数 == 无 cache 时）。"""
    embedder = FakeEmbedder(8)
    cache = EmbeddingCache(
        tmp_path / "cache",
        "test-model",
        8,
        embedder,
        embed_concurrency=1,
        enable_cache=False,
    )
    texts = ["a", "b", "c"]
    vecs = cache.embed_batch(texts)
    # 无 cache 时 3 个文本进 1 个 batch（< 64），1 次 embed 调用
    assert embedder.embed_calls == 1
    assert len(vecs) == 3


# ══════════════════════════════════════════════════════════════════════
# embed_batch：切批（EMBED_BATCH_SIZE=64）
# ══════════════════════════════════════════════════════════════════════


def test_embed_batch_large_input_splits_into_batches(tmp_path: Path) -> None:
    """超过 EMBED_BATCH_SIZE 的输入被切批，embed 调用数 == ceil(N/64)。"""
    embedder = FakeEmbedder(8)
    cache = EmbeddingCache(tmp_path / "cache", "test-model", 8, embedder, enable_cache=True)
    n = EMBED_BATCH_SIZE + 10  # 74 → 2 batches
    texts = [f"text-{i}" for i in range(n)]
    vecs = cache.embed_batch(texts)
    assert len(vecs) == n
    import math
    expected_batches = math.ceil(n / EMBED_BATCH_SIZE)
    assert embedder.embed_calls == expected_batches


def test_embed_batch_on_progress_callback(tmp_path: Path) -> None:
    """on_progress 回调被调用，参数为 (done, total_unique_miss)。"""
    embedder = FakeEmbedder(8)
    cache = EmbeddingCache(tmp_path / "cache", "test-model", 8, embedder, enable_cache=True)
    progress_calls: list[tuple[int, int]] = []

    def on_progress(done: int, total: int) -> None:
        progress_calls.append((done, total))

    texts = [f"t-{i}" for i in range(70)]  # 2 batches
    cache.embed_batch(texts, on_progress=on_progress)
    assert len(progress_calls) == 2
    # total == unique miss == 70
    assert progress_calls[-1] == (70, 70)


# ══════════════════════════════════════════════════════════════════════
# 空输入
# ══════════════════════════════════════════════════════════════════════


def test_embed_batch_empty_input_returns_empty(tmp_path: Path) -> None:
    """空 texts → 空向量列表，不调用 embed。"""
    embedder = FakeEmbedder(8)
    cache = EmbeddingCache(tmp_path / "cache", "test-model", 8, embedder, enable_cache=True)
    assert cache.embed_batch([]) == []
    assert embedder.embed_calls == 0

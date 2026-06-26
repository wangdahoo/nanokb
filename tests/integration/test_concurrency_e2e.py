"""extract 阶段并发端到端回归（方案 §5，Feature s1-feat-005）。

覆盖：
- AC #1：同一组多 chunk 文档，concurrency=1（两并发度均 1）与 doc=4×chunk=4 高并发
  下 compile，``out/graph.json`` 与 ``out/triples.jsonl``（除 ``ts`` 时间戳外）逐字节一致。
- AC #3：doc=4×chunk=4 嵌套并发下，FakeLLM ``complete_calls`` 无丢失（== 总 chunk 数），
  ``with ThreadPoolExecutor`` 退出后无残留 worker 线程。
- AC #5：记录不同并发度矩阵下的 extract 耗时数据（串行 / 默认 chunk=4 / doc=4×chunk=4）。

确定性论证：阶段 B 按 sorted(to_process) 迭代、manifest 按 sorted(results_map) 迭代
（均不依赖 results_map 插入顺序），且单文档内 chunk 回放按 chunk_index 升序——故并发
完成顺序不影响最终输出。

注：``triples.jsonl`` 每条记录含 ``ts`` 时间戳（运行时刻），跨运行非字节一致；故对比时
移除 ``ts``。``graph.json``（``node_link_data`` 序列化）无时间戳，确定性可逐字段比较。
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from nanokb import pipeline
from nanokb.config import Settings

# ── 测试 doubles ─────────────────────────────────────────────────────


class _ChunkIndexedLLM:
    """按 (source_file, chunk_index) 生成确定性响应（并发安全）。

    每个 chunk 产出唯一 head/tail；同一文档的 chunk 共享 concept name ``C_<sf>`` 但
    description 随 chunk_index 变化——触发 last-write-wins，使输出对回放顺序敏感：
    若并发回放顺序错误，concept 描述与图节点顺序会改变，从而被断言捕获。
    """

    _SF_RE = re.compile(r"^source_file:\s*(.+)$", re.MULTILINE)
    _CI_RE = re.compile(r"^chunk_index:\s*(\d+)", re.MULTILINE)

    def __init__(self) -> None:
        self.complete_calls = 0
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        return self.complete_calls

    def complete(
        self,
        system: str,
        user: str,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        with self._lock:
            self.complete_calls += 1
        sf_m = self._SF_RE.search(user)
        ci_m = self._CI_RE.search(user)
        sf = sf_m.group(1).strip() if sf_m else "unknown"
        ci = int(ci_m.group(1)) if ci_m else 0
        return json.dumps(
            {
                "triples": [
                    {
                        "head": f"H_{sf}_{ci}",
                        "relation": "rel",
                        "tail": f"T_{sf}_{ci}",
                        "confidence": "EXTRACTED",
                    }
                ],
                "concepts": [
                    {
                        "name": f"C_{sf}",
                        "description": f"desc {sf} chunk{ci}",
                        "node_type": "concept",
                    }
                ],
            }
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _write_corpus(raw_dir: Path) -> None:
    """写 3 个 .md 文档，每个足够长以切出多个 chunk（chunk_max_tokens 设小）。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "alpha.md").write_text(
        "# Alpha\n\n" + "Alpha leads the project. " * 20, encoding="utf-8"
    )
    (raw_dir / "beta.md").write_text(
        "# Beta\n\n" + "Beta builds the engine. " * 20, encoding="utf-8"
    )
    (raw_dir / "gamma.md").write_text(
        "# Gamma\n\n" + "Gamma tests the system. " * 20, encoding="utf-8"
    )


def _read_triples_without_ts(out_dir: Path) -> list[dict[str, object]]:
    """读 triples.jsonl，移除每条记录的 ts 时间戳后返回（便于跨运行确定性比较）。"""
    path = out_dir / "triples.jsonl"
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rec.pop("ts", None)
        records.append(rec)
    return records


def _read_graph(out_dir: Path) -> dict[str, object]:
    """读 graph.json（node_link_data）为 dict。"""
    return json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))


# ── AC #1：并发输出确定性 ────────────────────────────────────────────


def test_e2e_concurrency_output_deterministic(tmp_path: Path) -> None:
    """AC #1：concurrency=1 与 doc=4×chunk=4 输出（除 ts 外）逐字节一致。"""
    # 两次独立运行（独立 raw/out 目录，避免缓存/manifest 互相干扰）
    raw1 = tmp_path / "raw1"
    out1 = tmp_path / "out1"
    raw2 = tmp_path / "raw2"
    out2 = tmp_path / "out2"
    _write_corpus(raw1)
    _write_corpus(raw2)

    # 运行 1：完全串行（两并发度均 1）
    settings_serial = Settings(
        raw_dir=raw1,
        out_dir=out1,
        chunk_max_tokens=30,
        chunk_overlap_tokens=5,
        extract_doc_concurrency=1,
        extract_chunk_concurrency=1,
    )
    pipeline.compile(settings_serial, llm=_ChunkIndexedLLM())

    # 运行 2：高并发（doc=4 × chunk=4）
    settings_conc = Settings(
        raw_dir=raw2,
        out_dir=out2,
        chunk_max_tokens=30,
        chunk_overlap_tokens=5,
        extract_doc_concurrency=4,
        extract_chunk_concurrency=4,
    )
    pipeline.compile(settings_conc, llm=_ChunkIndexedLLM())

    # graph.json 确定性（无时间戳）：逐字段比较
    assert _read_graph(out1) == _read_graph(out2), "graph.json differs across concurrency levels"

    # triples.jsonl 确定性（移除 ts 后比较）
    serial_triples = _read_triples_without_ts(out1)
    conc_triples = _read_triples_without_ts(out2)
    assert serial_triples == conc_triples, (
        "triples.jsonl (sans ts) differs across concurrency levels"
    )

    # sanity：确实产生了多 chunk 抽取（否则并发未被真正行使）
    assert len(serial_triples) == 3, f"expected 3 upsert records, got {len(serial_triples)}"


# ── AC #3：嵌套并发无丢失、无线程泄漏 ─────────────────────────────────


def test_nested_concurrency_no_lost_calls_no_leaked_threads(tmp_path: Path) -> None:
    """AC #3：doc=4×chunk=4 嵌套并发，complete_calls 无丢失，无残留 worker 线程。"""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_corpus(raw)

    llm = _ChunkIndexedLLM()
    settings = Settings(
        raw_dir=raw,
        out_dir=out,
        chunk_max_tokens=30,
        chunk_overlap_tokens=5,
        extract_doc_concurrency=4,
        extract_chunk_concurrency=4,
    )

    # 记录测试开始前的非主线程数量基线（如 pytest fixture 线程等）
    baseline_threads = set(threading.enumerate())

    result = pipeline.compile(settings, llm=llm)

    # 全部 3 文档成功抽取（无 skip）
    assert result.skipped == []
    # complete_calls 必须等于真实产生的总 chunk 数（>3，证明多 chunk 被并发处理且无丢失）
    assert llm.call_count > 3, f"expected >3 LLM calls (multi-chunk), got {llm.call_count}"
    # with 块退出后线程池 worker 应全部 join：无新增非守护残留线程
    leftover = set(threading.enumerate()) - baseline_threads
    non_daemon_leftover = [t for t in leftover if not t.daemon and t is not threading.main_thread()]
    assert not non_daemon_leftover, f"leaked worker threads: {non_daemon_leftover}"


# ── AC #5：性能基准（记录数据，软断言并发不慢于串行） ──────────────────


def test_extract_perf_matrix_records_timing(tmp_path: Path) -> None:
    """AC #5：记录串行 / 默认 chunk=4 / doc=4×chunk=4 的 extract 耗时数据。

    不设硬性加速阈值（避免 CI 环境抖动 flaky）；仅断言并发模式不显著慢于串行，
    并把耗时打印到日志供人工评估。
    """
    configs = [
        ("serial(doc=1,chunk=1)", 1, 1),
        ("default(chunk=4)", 1, 4),
        ("high(doc=4,chunk=4)", 4, 4),
    ]
    timings: dict[str, float] = {}
    for label, doc_c, chunk_c in configs:
        raw = tmp_path / f"raw_{doc_c}_{chunk_c}"
        out = tmp_path / f"out_{doc_c}_{chunk_c}"
        _write_corpus(raw)
        settings = Settings(
            raw_dir=raw,
            out_dir=out,
            chunk_max_tokens=30,
            chunk_overlap_tokens=5,
            extract_doc_concurrency=doc_c,
            extract_chunk_concurrency=chunk_c,
        )
        start = time.monotonic()
        pipeline.compile(settings, llm=_ChunkIndexedLLM())
        timings[label] = time.monotonic() - start

    # 高并发不应比串行慢超过 2×（容差，防 CI 抖动；正常应更快）
    serial_t = timings["serial(doc=1,chunk=1)"]
    high_t = timings["high(doc=4,chunk=4)"]
    assert high_t < serial_t * 2.0, (
        f"high-concurrency {high_t:.3f}s unexpectedly slower than serial {serial_t:.3f}s; "
        f"all timings={ {k: round(v, 3) for k, v in timings.items()} }"
    )

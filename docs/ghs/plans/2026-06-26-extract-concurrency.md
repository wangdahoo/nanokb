# 技术方案：extract 阶段并发处理（v2 修订版）

> **修订说明**：本版在 v1 基础上逐条修正 plan-reviewer 裁决的 4 处 Medium（节流重构 2 处、chunk 并发两分支语义不自洽、默认值与"零回归"表述矛盾），并酌情采纳 Optimization 建议。修订处以「【修订：回应 Medium #X】」或「【采纳 Opt #X】」标注，保留原方案结构与已核对通过的核心论证。代码标识符、字段名、枚举值、文件路径、日志/错误信息用英文。

---

## 1. 目标与范围

### 1.1 解决的问题
nanokb 编译流水线的 extract 阶段存在两个串行瓶颈，导致大文档库冷启动 / 增量重编译耗时长（LLM 网络 IO 等待是绝对瓶颈）：

- **chunk 级串行**：`SemanticTrack.extract`（`semantic_track.py:103`，原循环 `116-150`）逐 chunk 同步阻塞调用 `self._llm.complete`，单文档内多 chunk 串行等待。
- **文档级串行**：`pipeline.compile` 阶段 A（`pipeline.py:281`）的 `for path` 循环，多文档串行 ingest + extract + cache。

本方案让两层都支持可配置并发，在不改变现有同步 `complete()` 契约、不引入 asyncio 全链路改造的前提下，显著缩短 extract 耗时。

### 1.2 不解决的问题（显式排除）
- **不改造阶段 B**（破坏性变更：graph upsert / vector index / manifest / staging 切换）——阶段 B 在设计上已等待阶段 A 全部完成才开始（`pipeline.py:357`），边界天然清晰，无需并发化（CPU/IO 混合且涉及原子写，并发收益小风险大）。**【采纳 Opt #6 补充】**：阶段 B 的 embedding 调用（`_resolve_embedder` 复用 chat_llm 的情形，`pipeline.py:816-842`）天然共享 chat 的 RateLimiter，但阶段 A 已 join、无并发竞争，安全。
- **不引入 asyncio / AsyncOpenAI**——全项目无 async 基础设施，改造代价远超收益。
- **不并发化 CodeTrack**——它是 tree-sitter CPU 密集确定性抽取（零 Token），线程并发受 GIL 限制收益小，且已线程安全无需特殊处理（若未来需榨 CPU 并行可单独评估进程池）。**【采纳 Opt #6】**：用户文档（M3）须提示"**纯代码库（全 `.py/.js/.java`）不建议开 `extract_doc_concurrency`**——只会创建多个线程抢 GIL 跑 tree-sitter，无加速反而有线程开销"。
- **不改变缓存内容寻址语义**——`ExtractionCache` 的 key = `sha256(sha256|extraction_config|llm_model)`，不含 source_file；并发 put 已通过 `atomic_write_text`（`tempfile.mkstemp` 唯一名 + `os.replace`）保证安全，无需改造。
- **不改变确定性契约**——`_merge_concept`（`semantic_track.py:226-273`）的 last-write-wins（按 `chunk_index` 升序）/ concat_dedup 语义必须保留。

### 1.3 成功标准
- 单文档 N chunk 时，extract 耗时从 ~N×latency 降至 ~N/concurrency×latency（受 RPM 节流上限约束）。
- 多文档 M 个时，阶段 A 耗时从 ~Σdoc_latency 降至 ~M/doc_concurrency（受 RPM 节流约束）。
- 输出结果与串行模式**逐字节一致**（确定性回归测试固化）。
- **【修订：回应 Medium #3】** **两个并发度均为 1 时**（`extract_doc_concurrency=1` 且 `extract_chunk_concurrency=1`）行为与当前完全一致（零回归）。删除原"默认文档级"的歧义修饰——默认配置（doc=1, chunk=4）**已启用 chunk 级并发**，并非全串行（见 §3.1 默认决策记录）。
- 单文件 / 单 chunk 失败仍被隔离标记 skip，不崩溃流水线（保留现有失败安全语义）。

---

## 2. 方案选型：ThreadPoolExecutor

### 2.1 取舍分析

| 维度 | ThreadPoolExecutor（推荐） | asyncio 全链路 |
|------|---------------------------|----------------|
| **适配现有同步 `complete()`** | ✅ 直接复用，零侵入 | ❌ 需 `AsyncOpenAI`/`AsyncAnthropic` + 全链路 `async def` |
| **GIL 影响** | ✅ 网络 IO 等待释放 GIL，LLM 调用是纯 IO 等待，线程并发有效 | ✅ 原生并发 |
| **现有基础设施** | ✅ Python 标准库 `concurrent.futures`，无新依赖 | ❌ 全项目无 async/await，三 provider client 均同步 |
| **确定性保留** | ✅ "并发收集 + 主线程按序回放" 易实现 | 同等可行但代价更高 |
| **Python 3.10 约束** | ✅ 无版本问题 | ⚠️ 无原生 `TaskGroup`（3.11+），需 `gather` 兼容写法 |
| **改造范围** | 小（3 个模块 + 配置） | 大（全链路 async 化 + 三 provider 重写） |

**结论**：采用 `concurrent.futures.ThreadPoolExecutor`。LLM 调用是网络 IO 等待（不受 GIL 影响），线程池是最小侵入且充分有效的方案。

### 2.2 两层并发模型

```
                    LLM API（受全局 RateLimiter 节流）
                         ▲
              ┌──────────┴──────────┐
              │ 共享 LLMClient 实例 │  ← 单实例，线程安全（节流重构后）
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐   文档级并发（doc_concurrency）
          │              │              │     ThreadPoolExecutor #外层
     ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
     │  Doc A  │    │  Doc B  │    │  Doc C  │
     │extract()│    │extract()│    │extract()│
     └────┬────┘    └─────────┘    └─────────┘
          │
    ┌─────┼─────┐   chunk 级并发（chunk_concurrency）
    │     │     │     ThreadPoolExecutor #内层（SemanticTrack 内部）
   c0    c1    c2
```

- **外层（文档级）**：`pipeline.compile` 阶段 A 用 `ThreadPoolExecutor` 并发执行 (ingest → cache.get → extract → cache.put)，主线程归并 `results_map`/`sha_map`/`skipped`。
- **内层（chunk 级）**：`SemanticTrack.extract` 内部用独立 `ThreadPoolExecutor` 并发抽取各 chunk，收集后按 `chunk_index` 升序回放合并。
- **对 LLM API 的实际并发请求数** = doc_concurrency × chunk_concurrency，由全局 `RateLimiter` 统一节流（即使乘积超过 RPM 上限，RateLimiter 自动串行化，安全）。

**【采纳 Opt #1：嵌套线程池线程预算分析】**

嵌套线程池的线程数 = doc_concurrency ×（1 + chunk_concurrency）（每个 doc worker 自身 1 线程 + 其内部 chunk 线程池最多 chunk_concurrency 线程）。两种典型场景：
- **默认场景（doc=1, chunk=4）**：最多 1×(1+4)=5 线程，开销可忽略。
- **高并发场景（doc=4, chunk=4）**：最多 4×(1+4)=20 线程，Python 默认线程栈约 8MB，20 线程虚拟地址空间 ~160MB（实际常驻内存远低），现代 OS 完全可接受。

线程池创建/销毁开销：每个 doc worker 在 `with ThreadPoolExecutor(...)` 块内创建并销毁一个 chunk 线程池。对"大量小文档（如 1000 个 1-chunk markdown）"场景，doc=1 时 1000 次线程池创建×4 线程，单次创建约 0.1ms 量级，总开销 ~0.4s，相对 LLM 网络耗时（每 chunk 数百 ms）可忽略。但 doc>1 时嵌套创建倍增（doc=4 × 250 文档/worker × 4 chunk 线程池/文档），仍属可接受范围。

**替代方案讨论（单层扁平线程池）**：可考虑"总并发度封顶"的单层 `ThreadPoolExecutor`，把 (doc, chunk) 作为 task 单元扁平提交。优点是线程数固定（无嵌套创建）；缺点是**破坏两层并发的确定性回放边界**——chunk 回放必须在单文档内完成才能保证 last-write-wins，扁平池跨文档的 chunk 回放需额外同步原语，复杂度上升。综合权衡，**保留两层嵌套模型**（默认场景开销极小，高并发场景可接受，且确定性边界清晰）。

---

## 3. 详细设计

### 3.1 并发度配置（`config.py`）

新增两个 `Settings` 字段：

```python
# ── 抽取并发 ─────────────────────────────────────────────────────
# 文档级并发度：阶段 A 同时处理的文件数。0/1 = 串行（默认，向后兼容）。
extract_doc_concurrency: int = 1
# chunk 级并发度：单文档内同时抽取的 chunk 数。0/1 = 串行。默认 4（chunk 级是主要瓶颈）。
extract_chunk_concurrency: int = 4
```

**语义规则**（在 `pipeline.py` / `semantic_track.py` 使用处统一规整）：
- 值 ≤ 0 或 == 1 → 串行回退（`max(1, value)`，避免 `ThreadPoolExecutor(max_workers=0)` 报错）。
- 与 `llm_request_interval` 的协调：实际并发请求数 = doc×chunk，但全局 `RateLimiter`（见 §3.2）保证对 API 的请求速率不超 RPM 限额——即使并发度乘积远大于限额，RateLimiter 会自动串行化节流。用户无需手动计算"并发度 ≤ 60/RPM"，但文档中建议保持合理乘积以避免线程空等。

**环境变量覆盖**：`NANOKB_EXTRACT_DOC_CONCURRENCY` / `NANOKB_EXTRACT_CHUNK_CONCURRENCY`（pydantic-settings 自动支持）。

**【修订：回应 Medium #3】默认值决策记录（必须在方案中显式记录）**：

| 字段 | 默认值 | 决策理由 |
|------|--------|----------|
| `extract_doc_concurrency` | **1** | 严格向后兼容；文档级并发涉及 ingest/CodeTrack（GIL 受限）、`ExtractionCache.put`（虽原子写但放大竞争面），风险面更大，由用户显式开启。 |
| `extract_chunk_concurrency` | **4** | chunk 级是**绝对瓶颈**（纯 LLM IO 等待，GIL 释放充分）；确定性回放（§3.3.3 `raw_results.sort`）+ RateLimiter 节流（§3.2.2）保证安全；开箱即用即有 4× 加速收益。 |

**明确承认**：默认配置（doc=1, chunk=4）**已启用 chunk 级并发**。这意味着升级后，所有语义轨文档的 extract 默认走并发分支（线程池创建、RateLimiter 介入、并发回放路径全部激活）。风险面通过以下手段兜底：
1. §3.3.3 确定性回放保证输出与串行逐字节一致；
2. §5.1 回归测试基线改为"**默认配置（chunk=4）必须与 chunk=1 输出逐字节一致**"（见修订后的 §5.1）；
3. §3.3.2 两分支异常语义已统一（见 Medium #2 修订）。

> 若评审或实施方倾向"严格零回归"，可临时把 `extract_chunk_concurrency` 默认改为 1，待 M1 线上验证后再调为 4。本方案**推荐保持默认 4**（收益/风险比优）。

### 3.2 LLM 节流重构（前置依赖，M0）

#### 3.2.1 问题确认（修正 snapshot）

经源码核实，**当前仅 `OpenAIClient` 实现了 `_throttle()`**：
- `OpenAIClient._throttle`（`openai_client.py:62-74`）：用实例字段 `self._last_call_ts` 做 read-modify-write，**非线程安全**；`make_llm_client` 注入了 `request_interval` / `rate_limit_retries` / `max_retries`（`base.py:137-145`）。
- `AnthropicClient`（`anthropic_client.py:32-131`）：**完全无节流逻辑**，`make_llm_client` 也未传入 `request_interval`（`base.py:156-164`）。
- `OllamaClient`（`ollama_client.py:24-82`）：**完全无节流**，无应用层重试，`make_llm_client` 未传任何节流参数（`base.py:169-173`）。

并发后风险：
1. OpenAIClient 的 `_throttle` 竞态失效（多线程同时读到旧 `_last_call_ts`，同时 sleep，节流失效 → 突破 RPM → 429）。
2. AnthropicClient / OllamaClient 本就无限流，并发下更易触发 429（ollama 本地服务通常无需限流，但 anthropic 云服务需要）。

#### 3.2.2 新增 `llm/throttle.py`：线程安全 RateLimiter

```python
"""线程安全的全局速率限制器。

将原 OpenAIClient._throttle 的"实例时间戳"语义提升为进程级、线程安全、
可跨 provider 共享的节流原语。interval<=0 时无锁快速返回（零开销）。
"""
from __future__ import annotations
import threading
import time


class RateLimiter:
    """确保两次 acquire() 之间至少间隔 interval 秒（线程安全）。

    用 threading.Lock 保护 _last_ts 的 read-modify-write。锁内 sleep 串行化
    所有线程的请求——这正是 RPM 节流的目的（控制全局请求速率）。interval<=0
    时直接返回，无锁竞争开销。
    """

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._last_ts: float | None = None

    def acquire(self) -> None:
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
```

**设计要点**：
- 锁内 `time.sleep`：虽然 sleep 期间持有锁会阻塞其他 acquire 调用，但这正是"最小间隔"节流的期望行为——串行化请求发出时机。对于 `interval=0`（无限流，ollama 场景）路径，`_interval <= 0` 提前返回，无锁开销。
- 进程级单例：由 `make_llm_client` 创建一个 `RateLimiter` 实例注入各 provider client，使三 provider 共享同一节流（全局 RPM 语义）。
- 与 SDK 内置重试（`max_retries` 指数退避）和应用层 `RateLimitError` 退避（`_compute_backoff`）正交：RateLimiter 控制**主动请求间隔**，重试机制处理**被动 429 响应**，两者叠加安全（退避期间不 acquire）。

**【采纳 Opt #5：锁公平性已知特性】**：`threading.Lock` 非公平（不保证 FIFO 唤醒顺序）。RPM 节流场景下，acquire 等待数 = 并发度（个位数到几十），远小于系统线程数，starvation 风险可忽略。记录为已知特性：若未来高并发场景出现 starvation，可改用 `queue.Queue` + 单消费线程或 `threading.Condition` 实现公平排队（本次不在范围）。

#### 3.2.3 改造三个 provider client

- **OpenAIClient**：移除 `self._last_call_ts`（`openai_client.py:50`）和 `_throttle` 方法体（`openai_client.py:62-74`）；`__init__` 签名移除 `request_interval`，新增 `rate_limiter: RateLimiter` 参数持有注入实例；`complete()` / `embed()` 调用 `self._rate_limiter.acquire()` 替代原 `self._throttle()`。保留 `_compute_backoff` 和 `RateLimitError` 应用层重试不变（`rate_limit_retries` 仍保留供 `_compute_backoff` 使用）。
- **AnthropicClient**：新增可选 `rate_limiter: RateLimiter | None = None` 构造参数；`complete()` 开头 `if self._rate_limiter: self._rate_limiter.acquire()`。
- **OllamaClient**：同 AnthropicClient（本地服务通常 interval=0，RateLimiter 无开销）。

**【修订：回应 Medium #4】`complete` 改造后伪代码（明确 `acquire()` 注入位置）**：

经核对原 `OpenAIClient.complete`（`openai_client.py:84-132`），`self._throttle()` 在 `for attempt in range(total_attempts)` 循环**内部**（`line 99`），即**每次请求（含 `RateLimitError` 重试的那次）前都先节流**。改造后必须**保留这一位置**（放到循环外会导致重试时不节流，重试请求可能因无 interval 间隔再次触发 429）：

```python
def complete(self, system, user, response_format="json", temperature=0.0) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    use_json = response_format == "json"
    total_attempts = self._rate_limit_retries + 1
    for attempt in range(total_attempts):
        self._rate_limiter.acquire()   # 【保留在 for attempt 循环内部】与原 _throttle() 位置一致
        try:
            if use_json:
                resp = self._client.chat.completions.create(
                    model=self._model, messages=messages, temperature=temperature,
                    response_format={"type": "json_object"},
                )
            else:
                resp = self._client.chat.completions.create(
                    model=self._model, messages=messages, temperature=temperature,
                )
        except RateLimitError:
            if attempt >= self._rate_limit_retries:
                logger.error("rate limit exhausted after %d app-level retries", self._rate_limit_retries)
                raise
            backoff = self._compute_backoff(attempt)
            logger.warning("rate limited (attempt %d/%d), backing off %.1fs", attempt + 1, total_attempts, backoff)
            time.sleep(backoff)
            continue
        return resp.choices[0].message.content or ""
    raise RuntimeError("unreachable")  # pragma: no cover
```

`embed` 改造：原 `embed`（`openai_client.py:134-142`）的 `self._throttle()` 在方法开头（`line 137`，无重试循环，位置无歧义），改造为 `self._rate_limiter.acquire()` 放原位置即可：

```python
def embed(self, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    self._rate_limiter.acquire()   # 【原 _throttle() 位置，无歧义】
    resp = self._client.embeddings.create(model=self._embedding_model, input=texts)
    return [item.embedding for item in resp.data]
```

**关键约束**：`acquire()` 必须在 `for attempt` 循环**内部**、`try: resp = ...` **之前**。这样保证：首次请求节流 + 每次 `RateLimitError` 退避后的重试请求也节流，与原 `_throttle` 语义逐字节一致。

#### 3.2.4 改造 `make_llm_client` / `make_embedding_client`（`base.py`）

**`make_llm_client` 改造**（chat 端，`base.py:123-177`）——创建进程级共享 `RateLimiter` 注入三个 provider：

```python
def make_llm_client(settings: Settings) -> LLMClient:
    # 创建 chat 端进程级共享 RateLimiter（基于 llm_request_interval）
    chat_rate_limiter = RateLimiter(settings.llm_request_interval)
    if settings.llm_provider == "openai":
        return OpenAIClient(..., rate_limiter=chat_rate_limiter, ...)
    if settings.llm_provider == "anthropic":
        return AnthropicClient(..., rate_limiter=chat_rate_limiter, ...)
    if settings.llm_provider == "ollama":
        return OllamaClient(..., rate_limiter=chat_rate_limiter, ...)
```

**【修订：回应 Medium #1】`make_embedding_client` 改造**（embedding 端，`base.py:180-228`）：

经核对源码，`make_embedding_client` 在 `embedding_provider="openai"` 分支会**独立构造一个 `OpenAIClient`**（`base.py:205-224`），当前传入 `request_interval=settings.llm_request_interval`（`base.py:222`）和 `rate_limit_retries`（`base.py:223`）。`OpenAIClient.__init__` 签名变更（移除 `request_interval`、新增 `rate_limiter`）后，此处**必须同步改造**，否则类型错误或 embedding 端静默丢失节流。

**设计决策：embedding 端创建独立 `RateLimiter`**（与 chat 解耦），理由：embedding 与 chat 多为不同端点/厂商（如 chat 走 DeepSeek、embedding 走智谱 GLM `embedding-3` 或本地 Ollama），限额不同，应独立节流；且 `make_embedding_client` 是 `make_llm_client` 的局部调用、拿不到 chat 的 `RateLimiter` 实例（chat 的 `RateLimiter` 是 `make_llm_client` 内局部变量）。

```python
def make_embedding_client(settings: Settings) -> EmbeddingClient:
    provider = settings.embedding_provider
    if provider == "ollama":
        # Ollama 本地服务，interval 通常为 0，RateLimiter 无开销；但为签名一致仍注入独立实例
        embed_rate_limiter = RateLimiter(settings.llm_request_interval)
        return OllamaClient(..., rate_limiter=embed_rate_limiter, ...)

    if provider == "openai":
        key = settings.embedding_api_key or settings.openai_api_key
        ...
        # 【修订：回应 Medium #1】embedding 端创建【独立】RateLimiter，与 chat 解耦
        embed_rate_limiter = RateLimiter(settings.llm_request_interval)
        return OpenAIClient(
            api_key=key.get_secret_value(),
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
            base_url=base_url,
            max_retries=settings.llm_max_retries,
            rate_limiter=embed_rate_limiter,   # 替代原 request_interval=...
            rate_limit_retries=settings.llm_rate_limit_retries,
        )
    ...
```

**`_resolve_embedder` 情形 2 的 RateLimiter 共享安全性**（`pipeline.py:816-842`）：

`_resolve_embedder` 在"未配置独立 embedding 端点"（`embedding_provider=openai` 且无独立 key/url）时返回 `chat_llm`（`pipeline.py:834-841`），此时 embedding 复用 chat client，**天然共享 chat 的 `RateLimiter`**。这是安全的，因为：
- embedding 发生在**阶段 B**（`vector_store` index 阶段，`pipeline.py` 阶段 B 代码）；
- 阶段 A（含所有 chunk/doc 级 LLM 调用）**已 join 完成**（阶段 B 在阶段 A 全部结束后才开始）；
- 故 embedding 调用与 chat extract 调用**不存在并发竞争**，共享同一 `RateLimiter` 不会出现多线程同时 acquire。

> 综上：chat 端、独立 embedding 端各有自己的 `RateLimiter`；复用 chat_llm 做 embedding 时天然共享 chat 的 `RateLimiter`（阶段 B，无并发）。三类场景节流语义都正确。

### 3.3 chunk 级并发改造（`SemanticTrack.extract`）

#### 3.3.1 核心思路：先并发收集，再按序回放合并

`_coerce_triple`（`semantic_track.py:201`）/ `_parse_chunk_response`（`semantic_track.py:189`）是 `@staticmethod` 无共享状态，线程安全；但 `_merge_concept`（`semantic_track.py:226`）**依赖 chunk_index 升序**（last-write-wins 后到者覆盖，`line 264-273`；concat_dedup 按序拼接，`line 255-263`），不能边并发边合并。

**改造后流程**：
1. **并发抽取阶段**：用 `ThreadPoolExecutor(max_workers=chunk_concurrency)` 对每个 chunk 调用 `_extract_chunk_with_retry(chunk)`，收集 `list[tuple[int, dict | None]]`（chunk_index, parsed_or_None）。`_extract_chunk_with_retry`（`semantic_track.py:154-187`）本身无共享可变状态（只读 `self._llm` / `self._settings`），线程安全。
2. **按序回放合并阶段**（主线程，串行）：将收集结果按 `chunk_index` 升序排序，依次执行原来的 triples append / `_merge_concept` / AMBIGUOUS 哨兵逻辑，**完全保留 last-write-wins 确定性**。

#### 3.3.2 改造后伪代码（`semantic_track.py`）

```python
import concurrent.futures

def extract(self, doc: Document) -> ExtractionResult:
    source_file = str(doc.path)
    ordered_chunks = sorted(doc.chunks, key=lambda c: c.index)
    total_chunks = len(ordered_chunks)
    concurrency = max(1, self._settings.extract_chunk_concurrency)

    # 【采纳 Opt #3】chunk 索引映射：chunk.index → Chunk，回放取值更稳健（理论上 index 可能非连续）
    chunk_by_index: dict[int, Chunk] = {c.index: c for c in ordered_chunks}

    # ── 阶段 1：抽取（线程安全：_extract_chunk_with_retry 无共享可变状态） ──
    raw_results: list[tuple[int, dict[str, Any] | None]] = []
    if concurrency == 1:
        # 串行回退
        for ci, chunk in enumerate(ordered_chunks, 1):
            logger.info("  chunk %d/%d of %s ...", ci, total_chunks, doc.path.name)
            # 【修订：回应 Medium #2 方案 A】串行分支也 try/except，与并发分支异常语义一致
            try:
                parsed = self._extract_chunk_with_retry(chunk)
            except Exception:
                logger.exception(
                    "chunk %d of %s: extraction crashed, degrading to AMBIGUOUS sentinel",
                    chunk.index, source_file,
                )
                parsed = None
            raw_results.append((chunk.index, parsed))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_chunk = {
                pool.submit(self._extract_chunk_with_retry, chunk): chunk
                for chunk in ordered_chunks
            }
            # 【采纳 Opt #4】并发分支补 per-chunk 进度日志（完成计数）
            done = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                done += 1
                logger.info(
                    "  chunk %d/%d of %s done (%d/%d completed)",
                    chunk.index, total_chunks, doc.path.name, done, total_chunks,
                )
                # 单 chunk 异常隔离：与串行分支一致，降级 None → AMBIGUOUS 哨兵
                try:
                    parsed = future.result()
                except Exception:
                    logger.exception(
                        "chunk %d of %s: concurrent extraction crashed, degrading to AMBIGUOUS sentinel",
                        chunk.index, source_file,
                    )
                    parsed = None
                raw_results.append((chunk.index, parsed))

    # ── 阶段 2：按 chunk_index 升序回放合并（确定性，主线程串行） ──
    raw_results.sort(key=lambda r: r[0])
    triples: list[Triple] = []
    merged_concepts: dict[str, Concept] = {}
    for chunk_index, parsed in raw_results:
        if parsed is None:
            # 【采纳 Opt #2】AMBIGUOUS 哨兵构造须与原 semantic_track.py:131-141 逐字段一致
            triples.append(
                Triple(
                    head=doc.path.stem or source_file,
                    relation="extraction_failed",
                    tail=f"chunk_{chunk_index}",
                    confidence=Confidence.AMBIGUOUS,
                    source_file=source_file,
                    track=Track.SEMANTIC,
                    chunk_index=chunk_index,
                )
            )
            continue
        for raw_triple in parsed.get("triples", []):
            triple = self._coerce_triple(raw_triple, source_file, chunk_index)
            if triple is not None:
                triples.append(triple)
        for raw_concept in parsed.get("concepts", []):
            self._merge_concept(raw_concept, chunk_index, source_file, merged_concepts)

    return ExtractionResult(triples=triples, concepts=list(merged_concepts.values()))
```

**【修订：回应 Medium #2】两分支异常语义统一说明（方案 A）**：

经核对源码，`_extract_chunk_with_retry`（`semantic_track.py:154-187`）只把 **JSON 解析失败**（`_parse_chunk_response` 返回 None）降级为 None；**LLM 调用本身的异常**（网络错误、`APIError`、透传的 `RateLimitError` 等）会直接抛出，不被该方法捕获。原 `extract` 循环（`semantic_track.py:116-150`）**没有 try/except 包裹** `_extract_chunk_with_retry`（`line 123`），故原语义是：单 chunk LLM 异常 → 整个文档 `extract` 崩溃 → 传播到 `pipeline` 层 `except Exception` → 该文档进 `skipped`。

**本方案选择方案 A（语义更友好）**：串行分支也用 try/except 把单 chunk LLM 异常降级为 None → AMBIGUOUS 哨兵（与并发分支对齐）。这样两分支行为一致，满足 §1.3 / §5.1 的"逐字节一致"硬门槛。

**明确声明：这是相对原行为的有意变更**：
- **原行为**：单 chunk LLM 异常连累整文档失败（文档进 skipped，无输出）。
- **新行为**：单 chunk LLM 异常被隔离，该 chunk 降级为 AMBIGUOUS 哨兵，文档其余 chunk 正常抽取，文档仍成功产出（带部分 AMBIGUOUS）。

**理由**：chunk 级异常隔离与文档级异常隔离（`_process_one_file` 已有的 try/except）语义一致——单点失败不应连累整体，且 AMBIGUOUS 哨兵在图谱中可追溯。这是比"连累整文档"更合理的失败语义。**须补单测**：见 §5.3 chunk 级异常隔离测试（新增"LLM 抛异常 → 该 chunk 降级 AMBIGUOUS，其余 chunk 正常"用例）。

> 若评审倾向严格向后兼容（方案 B），可改为并发分支不降级、`future.result()` 异常 re-raise 出 `extract`。本方案**推荐方案 A**（更好的失败隔离）。两分支必须一致，这一点是硬性要求。

#### 3.3.3 确定性保证
- `raw_results.sort(key=lambda r: r[0])` 强制按 `chunk_index` 升序，无论线程完成顺序如何，回放顺序确定。
- `as_completed` 的乱序到达只影响收集顺序，不影响最终回放顺序。
- 当 `concurrency == 1` 时走串行分支，无 ThreadPoolExecutor 开销，输出与并发分支逐字节一致（两分支异常语义已统一，见 §3.3.2 Medium #2 修订）。

### 3.4 文档级并发改造（`pipeline.compile` 阶段 A）

#### 3.4.1 核心思路：并发执行单文件处理函数，主线程归并

将阶段 A 循环体（ingest → cache.get → extract → cache.put）提取为内部函数 `_process_one_file(path)`，返回 `(path, result | None, sha256 | None, skipped_flag)`，用 `ThreadPoolExecutor` 并发执行，主线程收集 `as_completed` 结果归并到 `results_map` / `sha_map` / `skipped`。

#### 3.4.2 改造后伪代码（`pipeline.py` 阶段 A）

```python
import concurrent.futures

def _process_one_file(path: str) -> tuple[str, ExtractionResult | None, str | None, bool]:
    """单文件处理：ingest → cache.get → extract → cache.put。

    返回 (path, result_or_None, sha256_or_None, is_skipped)。
    线程安全：ingest_file/cache 无共享状态；extractor 共享单例（见 §3.5 懒构造修复）。
    """
    abs_path = raw_dir / path
    try:
        doc = ingest_file(abs_path, raw_dir, registry, settings)
    except UnsupportedFormatError as exc:
        logger.warning("skip unsupported file: %s (%s)", path, exc, extra={...})
        return (path, None, None, True)
    except Exception:
        logger.exception("ingest failed for %s", path, extra={...})
        return (path, None, None, True)

    cached = cache.get(doc.sha256, extraction_sig, settings.llm_model)
    if cached is not None:
        logger.info("cache hit %s → ...", path, extra={...})
        return (path, _normalize_result_source(cached, path), doc.sha256, False)

    try:
        result = extractor.extract(doc)
    except Exception:
        logger.exception("extraction failed for %s", path, extra={...})
        return (path, None, None, True)

    try:
        cache.put(doc.sha256, extraction_sig, settings.llm_model, result)
    except Exception:
        logger.warning("cache put failed for %s; result kept in memory", path, extra={...})

    logger.info("extracted %s → ...", path, extra={...})
    return (path, _normalize_result_source(result, path), doc.sha256, False)


# ── 阶段 A 并发执行 ──
concurrency = max(1, settings.extract_doc_concurrency)
if concurrency == 1:
    # 串行回退（零回归）
    for idx, path in enumerate(to_process, 1):
        logger.info("[%d/%d] ingesting %s ...", idx, total, path, extra={...})
        p, result, sha, skipped_flag = _process_one_file(path)
        if skipped_flag:
            skipped.append(p)
        elif result is not None:
            results_map[p] = result
            sha_map[p] = sha or ""
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_one_file, path): path for path in to_process}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                p, result, sha, skipped_flag = future.result()
            except Exception:
                # 兜底：_process_one_file 内部异常未捕获时的最终隔离
                logger.exception("unexpected failure processing %s", path, extra={...})
                skipped.append(path)
                continue
            if skipped_flag:
                skipped.append(p)
            elif result is not None:
                results_map[p] = result
                sha_map[p] = sha or ""
```

#### 3.4.3 归并安全性
- `results_map` / `sha_map` / `skipped` 只在主线程写（`as_completed` 循环在主线程），worker 线程只返回不可变元组 → 无锁、无竞态。
- `_normalize_result_source` 返回新 `ExtractionResult` 实例，每文件独立。
- 阶段 B 读取 `results_map` / `sha_map` 时阶段 A 已 join（`with ThreadPoolExecutor` 退出即全部完成），边界与改造前一致。

### 3.5 DefaultExtractor 懒构造竞态修复（`extract/__init__.py`）

`DefaultExtractor.extract`（`__init__.py:50`）的 `self._semantic_track` 懒构造在文档级并发首个语义轨文件并发到达时存在竞态（可能构造多个实例，虽各实例等价不致命，但不优雅）。

**修复**：双重检查锁（DCL）或构造期预热。推荐 DCL（最小改动）：

```python
import threading

class DefaultExtractor:
    def __init__(self, llm, settings):
        self._llm = llm
        self._settings = settings
        self._code_track = CodeTrack(settings)
        self._semantic_track: SemanticTrack | None = None
        self._sem_lock = threading.Lock()  # 保护懒构造

    def extract(self, doc: Document) -> ExtractionResult:
        if doc.path.suffix.lower() in supported_code_suffixes():
            return self._code_track.extract(doc)
        # 双重检查锁：避免并发首调构造多个 SemanticTrack
        if self._semantic_track is None:
            with self._sem_lock:
                if self._semantic_track is None:
                    self._semantic_track = SemanticTrack(self._llm, self._settings)
        return self._semantic_track.extract(doc)
```

> `SemanticTrack` 本身是无状态对象（仅持有 `self._llm` / `self._settings` 引用，`semantic_track.py:99-101`），构造后 `extract` 内部无实例级可变状态（triples/merged_concepts 都是方法局部变量），因此单个共享实例跨文档并发调用 `extract` 是安全的（每文档独立的局部变量）。

### 3.6 进度反馈并发安全

经核实，`compile` 流程当前**不使用** `RichProgressReporter`（它仅用于 qa 路径：query/ask/search，见 `cli.py:210`/`242`/`276`——三处均为 `RichProgressReporter(console)`，已逐行核对）。compile 阶段 A 用 `logger.info("[%d/%d] ...")` 反馈进度。

- **Python 标准 `logging` 是线程安全的**（`Handler.acquire`/`release` 内置锁），并发 worker 线程的 `logger.info` 调用天然原子，无需改造。
- `[idx/total]` 计数在并发模式下语义弱化（`idx` 不再代表严格顺序），但日志仅为可读性提示，不影响正确性。可选优化：改用完成计数（`done += 1`）替代 `idx`，但属锦上添花，非必需。
- **未来扩展**：若后续要给 compile 加 Rich 进度条（如 `console.status`），需用原子计数器 + 主线程刷新，本次不在范围内。

---

## 4. 错误处理与回滚

### 4.1 失败隔离策略

| 失败层级 | 隔离方式 | 语义保留 |
|----------|----------|----------|
| **单 chunk 解析失败** | `_extract_chunk_with_retry` 返回 None → AMBIGUOUS 哨兵（原有逻辑，`semantic_track.py:168-187`） | 单 chunk 不崩溃，不影响同文档其他 chunk |
| **【修订：回应 Medium #2】单 chunk LLM 异常（解析外）** | **串行 + 并发两分支统一** try/except → 降级为 None → AMBIGUOUS 哨兵（§3.3.2 方案 A） | chunk 级异常隔离，两分支行为一致；**相对原行为的有意变更**（原为连累整文档失败） |
| **单文档 extract 异常** | `_process_one_file` 内 try/except → 返回 skipped_flag=True（原有逻辑保留） | 单文档失败标 skip，不影响其他文档 |
| **单文档 ingest 异常** | 同上（`UnsupportedFormatError` / 通用 Exception 分支保留） | — |
| **cache.put 异常** | try/except 仅警告，result 保留在内存（原有逻辑） | 缓存失败不丢结果 |
| **worker 线程未捕获异常** | `as_completed` 外层 try/except 兜底（§3.4.2）→ 标 skip | 最终防线，不崩溃池 |

### 4.2 可回滚开关

- **配置级回滚**：`extract_doc_concurrency=1` 且 `extract_chunk_concurrency=1` → 完全串行（两处都有 `concurrency == 1` 串行分支），行为与改造前**在确定性层面**一致。**【修订：回应 Medium #3 注记】**：chunk 异常语义已统一为"单 chunk 异常降级 AMBIGUOUS"（方案 A），这与原"连累整文档"行为不同，故 `chunk=1` 串行分支相对**改造前原始行为**有一处有意变更（见 §3.3.2）。若需完全复刻原"连累"行为，须额外实现方案 B（不降级、re-raise）——本方案不推荐。
- **环境变量回滚**：`NANOKB_EXTRACT_DOC_CONCURRENCY=1 NANOKB_EXTRACT_CHUNK_CONCURRENCY=1 nanokb build`。
- **代码级回滚**：RateLimiter 在 `interval<=0` 时无锁返回，OpenAIClient 节流行为与改造前（单线程）等价。

---

## 5. 测试策略

### 5.1 确定性回归测试（核心，必须通过）

- **复用** `tests/unit/test_semantic_track.py` 现有用例（顺序/合并/retry/AMBIGUOUS 哨兵），验证 `extract_chunk_concurrency=1` 和 `=4` 两种配置下输出**完全一致**（`ExtractionResult` 深度相等断言）。
- **【修订：回应 Medium #3】回归测试基线更新**：默认配置（`extract_chunk_concurrency=4`）必须与 `extract_chunk_concurrency=1` 输出**逐字节一致**——即"默认配置即并发，默认配置必须确定性回归通过"。不能只测 chunk=1。
- **新增**：构造多 chunk 文档，FakeLLMClient 对不同 chunk_index 返回含同名 concept 的不同 description，验证 last-write-wins（默认策略）和 concat_dedup（`semantic_track.py:255-263`）在并发模式下仍按 chunk_index 升序生效（非按线程完成顺序）。
- **新增**：端到端对比测试——同一组输入文件，分别在 `concurrency=1` / `concurrency=4` 下运行 `compile`，断言 `triples.jsonl` 与 `out/graph.json` 逐字节一致。
- **【修订：回应 Medium #2 新增】chunk 级 LLM 异常隔离单测**：FakeLLMClient 对某 chunk_index 的 `complete` 抛异常 → 验证 chunk=1 与 chunk=4 两种配置下，**该 chunk 都降级为 AMBIGUOUS 哨兵**（`relation="extraction_failed"`, `confidence=Confidence.AMBIGUOUS`），其余 chunk 正常抽取，两分支输出一致。这是方案 A 有意变更的回归保护。

### 5.2 并发正确性测试（注入带延迟的 FakeLLMClient）

扩展现有 `FakeLLMClient` 模式（`test_semantic_track.py` 等多处已有），新增带延迟 + 计数的变体：

```python
class DelayedFakeLLMClient:
    """带人为延迟和调用计数的 fake client，用于验证并发加速与线程安全。"""
    def __init__(self, responses: dict, delay: float = 0.1):
        self._responses = responses
        self._delay = delay
        self._call_count = 0
        self._lock = threading.Lock()
    def complete(self, system, user, response_format="json", temperature=0.0):
        with self._lock:
            self._call_count += 1
            idx = self._call_count
        time.sleep(self._delay)  # 模拟网络 IO 等待
        return self._responses.get(idx, '{"triples": [], "concepts": []}')
```

测试用例：
- **加速验证**：N=8 chunk、delay=0.2s、chunk_concurrency=4 → 总耗时应 ~0.4s（2 轮×0.2s），远小于串行的 1.6s。断言 `wall_time < serial_time / concurrency * 1.5`（留容差）。
- **线程安全验证**：并发调用 `complete` 时 `_call_count` 正确递增（无丢失/重复），所有响应都被收集。
- **RateLimiter 验证**：注入 `RateLimiter(interval=0.05)`，N=10 并发请求 → 测量实际请求间隔均 ≥ 0.05s（线程安全节流生效）。

### 5.3 错误隔离测试
- **chunk 级解析失败**：FakeLLMClient 返回非 JSON → 验证该 chunk 降级 AMBIGUOUS 哨兵（原有行为），其他 chunk 正常抽取。
- **【修订：回应 Medium #2 新增】chunk 级 LLM 异常**：FakeLLMClient 对某 chunk_index 的 `complete` 直接 `raise APIError(...)`（解析外异常）→ 验证 chunk=1 与 chunk=4 配置下都降级 AMBIGUOUS 哨兵，文档其余 chunk 正常，两分支一致。
- **文档级**：FakeLLMClient 对某文件抛异常 → 验证该 path 进 `skipped`，其他文件正常进 `results_map`，阶段 B 正常执行。
- **cache.put 并发**（补测 snapshot 指出的覆盖缺口）：多线程并发 `cache.put` 不同 key → 验证无文件损坏（原子写）；同 key 并发 → 最终一致。

### 5.4 RateLimiter 单元测试（新增 `tests/unit/test_throttle.py`）
- `interval<=0` → acquire 立即返回，无 sleep。
- 多线程并发 acquire → 实际间隔 ≥ interval（线程安全）。
- 单线程连续 acquire → 行为与原 `_throttle` 等价。

**【采纳 Opt #7】三者叠加集成测试**：补一个"`acquire` 节流 + SDK `max_retries` 指数退避 + 应用层 `RateLimitError` backoff"三者叠加的集成测——FakeLLMClient 模拟间歇 429，注入 `RateLimiter(interval=0.05)`，验证三者叠加不会双重等待导致请求超时或顺序异常（总耗时符合 `节流间隔 + backoff` 的预期叠加，无死锁/无请求丢失）。

### 5.5 集成测试
- **复用** `tests/integration/test_cold_start.py`、`test_compile_md.py`、`test_modified_clean_rebuild.py`、`test_deletion_propagation.py`：在**默认配置（`extract_chunk_concurrency=4`）**下应全部通过（确定性回归）。
- **新增**：上述集成测试的并发变体（设置 `extract_doc_concurrency=4`），验证端到端输出一致。

**【采纳 Opt #8】嵌套线程池线程安全专项**：补一个直接断言"doc×chunk 嵌套并发下 `_call_count` 无丢失、无线程泄漏"的用例——设置 doc=4、chunk=4、共 16 个并发 chunk，FakeLLMClient 带 `threading.Lock` 保护的计数器，断言最终 `_call_count == 16`（无丢失）；测试结束后用 `threading.enumerate()` 断言无残留 worker 线程（`ThreadPoolExecutor` 的 `with` 块退出应已 join 全部）。

---

## 6. 分阶段实施计划

### M0：LLM 节流重构（前置依赖，可独立交付）
**目标**：修复 `_throttle` 线程安全问题 + 统一三 provider 节流语义。

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1 | `src/nanokb/llm/throttle.py`（新增） | `RateLimiter` 类 |
| 2 | `src/nanokb/llm/openai_client.py` | 移除 `_last_call_ts`/`_throttle`（`line 50,62-74`），注入 `rate_limiter`；`complete` 的 `acquire()` **保留在 `for attempt` 循环内部**（§3.2.3）；`embed` 放原 `_throttle` 位置 |
| 3 | `src/nanokb/llm/anthropic_client.py` | 新增 `rate_limiter` 参数，`complete` 调 `acquire` |
| 4 | `src/nanokb/llm/ollama_client.py` | 同上 |
| 5 | `src/nanokb/llm/base.py` | `make_llm_client` 创建 chat 端共享 `RateLimiter`；**【修订：回应 Medium #1】`make_embedding_client` 创建 embedding 端独立 `RateLimiter`**（openai 分支 `base.py:216`、ollama 分支 `base.py:199` 同步改造） |
| 6 | `tests/unit/test_throttle.py`（新增） | RateLimiter 线程安全单元测试 |

- **依赖**：无（独立）
- **回滚点**：RateLimiter `interval<=0` 无锁返回，单线程行为不变；`test_throttle.py` + 现有 `test_llm_factory.py` 全绿即可合并。
- **验证**：现有所有测试零回归（此时还未引入并发调用）。

### M1：chunk 级并发（依赖 M0）
**目标**：`SemanticTrack.extract` 支持 chunk 级并发抽取。

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1 | `src/nanokb/config.py` | 新增 `extract_chunk_concurrency: int = 4` |
| 2 | `src/nanokb/extract/semantic_track.py` | `extract` 改为"并发收集 + 按序回放合并"（§3.3）；**【修订：回应 Medium #2】两分支统一 try/except 异常隔离（方案 A）** |
| 3 | `tests/unit/test_semantic_track.py` | 确定性回归（chunk=1 vs 默认 chunk=4）+ 并发加速 + chunk 异常隔离（含 LLM 抛异常场景）用例 |
| 4 | `tests/unit/test_semantic_track.py` | `DelayedFakeLLMClient` 工具类 |

- **依赖**：M0（节流线程安全是 chunk 并发的前提）
- **回滚点**：`extract_chunk_concurrency=1` 串行分支。
- **验证**：chunk=1 / =4 输出逐字节一致（含 LLM 异常场景两分支一致）；带延迟 fake client 验证加速。

### M2：文档级并发 + DefaultExtractor 懒构造修复（依赖 M0，可与 M1 并行）
**目标**：`pipeline.compile` 阶段 A 支持文档级并发。

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1 | `src/nanokb/config.py` | 新增 `extract_doc_concurrency: int = 1` |
| 2 | `src/nanokb/extract/__init__.py` | `DefaultExtractor` DCL 懒构造修复（§3.5） |
| 3 | `src/nanokb/pipeline.py` | 阶段 A 提取 `_process_one_file` + `ThreadPoolExecutor` 并发（§3.4） |
| 4 | `tests/unit/test_pipeline.py` 或集成测试 | 文档级并发 + 异常隔离 + 归并正确性 |
| 5 | `tests/unit/test_extraction_cache.py` | 并发 put 补测 |

- **依赖**：M0（节流）；与 M1 无代码冲突（不同文件，仅 `config.py` 共享但加不同字段）
- **回滚点**：`extract_doc_concurrency=1` 串行分支。
- **验证**：默认配置（doc=1）集成测试零回归；并发变体输出一致。

### M3：端到端验证与文档（依赖 M1 + M2）
**目标**：全链路并发验证 + 用户文档。

| 步骤 | 内容 |
|------|------|
| 1 | 端到端对比测试：concurrency=1 vs 高并发，`triples.jsonl`/`graph.json` 逐字节一致 |
| 2 | 性能基准：记录不同并发度下的 extract 耗时（文档数 × chunk 数矩阵） |
| 3 | 更新 README / 配置文档：说明两个并发度字段、与 `llm_request_interval` 的关系、推荐配置；**【采纳 Opt #6】提示"纯代码库（全 CodeTrack）不建议开 doc 并发"** |

---

## 7. 风险与权衡

### 7.1 GIL 限制
- **风险**：CodeTrack 是 CPU 密集（tree-sitter 解析），线程并发受 GIL 限制无加速。
- **缓解**：CodeTrack 不在并发改造范围（零 Token，通常非瓶颈）；语义轨 LLM 调用是纯 IO 等待，GIL 在 IO 等待时释放，线程并发有效。已显式排除 CodeTrack 进程池改造（YAGNI）。**【采纳 Opt #6】**：用户文档提示纯代码库不建议开 doc 并发。

### 7.2 RPM 限流与并发的交互
- **风险**：doc×chunk 并发度乘积过大时，大量线程在 RateLimiter 上排队空等，线程数膨胀。**【采纳 Opt #1】**线程预算分析见 §2.2（默认 doc=1×chunk=4=5 线程，高并发 doc=4×chunk=4=20 线程，均可接受）。
- **缓解**：RateLimiter 自动串行化保证不突破 RPM；默认 `chunk_concurrency=4`、`doc_concurrency=1` 乘积=4，对大多数 API 限额安全。文档建议用户根据 provider 限额（如 60 RPM → interval≈1s）合理设置，避免线程数过大（线程池 `max_workers` 已封顶）。

### 7.3 节流锁内 sleep 的吞吐影响
- **权衡**：RateLimiter 锁内 `time.sleep` 会串行化所有 acquire（即使多个线程本可并行发起不同请求）。这是"最小间隔"节流的固有语义——控制全局速率。
- **缓解**：`interval=0`（ollama / 无限流场景）无锁快速返回，零开销。对需要高并发的场景，用户可设 `llm_request_interval=0` 依赖 SDK 内置 429 重试。**【采纳 Opt #5】锁非公平（不保证 FIFO）为已知特性，starvation 风险可忽略。**

### 7.4 内存占用
- **风险**：文档级并发时，多个文档的 `ExtractionResult` 同时驻留内存（串行模式下逐个处理）。
- **缓解**：`ExtractionResult` 是纯数据载体，单文档结果通常 KB 级；并发度默认 1，高并发由用户显式开启，可接受。

### 7.5 日志顺序
- **风险**：并发模式下 `logger.info("[%d/%d] ...")` 的 `idx` 不再代表严格处理顺序，日志交错可读性下降。**【采纳 Opt #4】**：chunk 级并发分支已补 per-chunk 完成计数日志（`done += 1`）。
- **缓解**：标准 logging 线程安全（每条原子）；`idx` 仅提示性，不影响正确性。

### 7.6 确定性回归（最高优先级保障）
- **风险**：并发回放顺序错误会破坏 last-write-wins，导致 concept description 不确定。
- **缓解**：`raw_results.sort(key=chunk_index)` 强制升序回放；确定性回归测试（concurrency=1 vs 高并发输出逐字节对比）作为合并硬门槛。**【修订：回应 Medium #3】**基线已更新为"默认配置（chunk=4）必须与 chunk=1 逐字节一致"。

### 7.7 两分支异常语义一致性（新增）
- **风险**：若串行/并发两分支异常处理不一致，§5.1 逐字节一致性无法成立（原 v1 方案此风险）。
- **缓解**：**【修订：回应 Medium #2】**两分支已统一为方案 A（单 chunk LLM 异常降级 AMBIGUOUS），并补单测覆盖。

---

## 附录：改造文件清单

| 文件 | 操作 | Milestone |
|------|------|-----------|
| `src/nanokb/llm/throttle.py` | 新增 | M0 |
| `src/nanokb/llm/openai_client.py` | 修改（移除 `_throttle`/`_last_call_ts`，注入 RateLimiter；`complete` 的 `acquire()` 在 `for attempt` 内部） | M0 |
| `src/nanokb/llm/anthropic_client.py` | 修改（新增 RateLimiter 接入） | M0 |
| `src/nanokb/llm/ollama_client.py` | 修改（同上） | M0 |
| `src/nanokb/llm/base.py` | 修改（`make_llm_client` chat 端共享 RateLimiter；`make_embedding_client` embedding 端独立 RateLimiter） | M0 |
| `src/nanokb/config.py` | 修改（+2 字段） | M1, M2 |
| `src/nanokb/extract/semantic_track.py` | 修改（并发收集 + 按序回放；两分支统一异常隔离） | M1 |
| `src/nanokb/extract/__init__.py` | 修改（DefaultExtractor DCL） | M2 |
| `src/nanokb/pipeline.py` | 修改（阶段 A 并发） | M2 |
| `tests/unit/test_throttle.py` | 新增 | M0 |
| `tests/unit/test_semantic_track.py` | 修改（并发回归 + LLM 异常隔离两分支一致 + DelayedFakeLLMClient） | M1 |
| `tests/unit/test_extraction_cache.py` | 修改（并发 put 补测） | M2 |
| `tests/integration/*` | 新增并发变体（默认配置 chunk=4 回归 + doc=4 并发变体 + 三者叠加 + 嵌套线程安全专项） | M3 |

---

## 附录：v2 修订摘要（供 reviewer 复核）

| 修订项 | 位置 | 内容 |
|--------|------|------|
| Medium #1 | §3.2.4 | 补 `make_embedding_client` 改造伪代码：embedding 端创建**独立** `RateLimiter(settings.llm_request_interval)`；说明 `_resolve_embedder` 情形 2 复用 chat_llm 时天然共享 chat 的 RateLimiter（阶段 B，无并发竞争） |
| Medium #2 | §3.3.2 / §4.1 / §5.1 / §5.3 / §7.7 | 选方案 A：串行分支也 try/except 降级 None→AMBIGUOUS，明确声明相对原行为的有意变更（原为连累整文档），补两分支一致的 LLM 异常隔离单测 |
| Medium #3 | §1.3 / §3.1 / §5.1 / §5.5 / §7.6 | 修正 §1.3 为"两个并发度均为 1 时零回归"；记录默认决策（chunk=4 开箱收益 > 风险面）；回归基线改为"默认 chunk=4 必须与 chunk=1 逐字节一致" |
| Medium #4 | §3.2.3 | 补 `complete` 改造后伪代码：`acquire()` **保留在 `for attempt` 循环内部、`try: resp=...` 之前**（与原 `_throttle` 位置一致）；`embed` 同理（原位置无歧义） |
| Opt #1 | §2.2 / §7.2 | 嵌套线程池线程预算分析（默认 5 线程、高并发 20 线程）+ 单层扁平池替代方案讨论（保留嵌套） |
| Opt #2 | §3.3.2 | AMBIGUOUS 哨兵构造照原 `semantic_track.py:131-141` 逐字段补全（confidence/source_file/track/chunk_index/head/tail） |
| Opt #3 | §3.3.2 | 回放阶段 chunk 取值用 `dict[int, Chunk]` 映射 |
| Opt #4 | §3.3.2 / §7.5 | 并发分支补 per-chunk 进度日志（`done += 1`） |
| Opt #5 | §3.2.2 / §7.3 | RateLimiter 锁非公平记录为已知特性 |
| Opt #6 | §1.2 / §3.6 / M3 / §7.1 | 用户文档提示"纯代码库不建议开 doc 并发" |
| Opt #7 | §5.4 | 三者叠加（节流 + SDK 重试 + 应用层 backoff）集成测试 |
| Opt #8 | §5.5 | 嵌套线程池线程安全专项（`_call_count` 无丢失、无线程泄漏） |
| Opt #9 | §3.6 | `cli.py:210/242/276` 行号已逐行核对，全部正确（均为 `RichProgressReporter(console)`，属 query/ask/search） |
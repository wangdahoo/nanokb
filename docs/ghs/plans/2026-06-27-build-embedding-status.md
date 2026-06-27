# build 编译提速、embedding 缓存/并发、跨进程进度可见性与 status 命令增强

## 修订记录

### round 2

本轮基于首轮评审报告（FAIL：1 Severe + 5 Medium + 5 Optimization）逐条修正。下表列出每条处置结论。

| 编号 | 问题摘要 | 处置 | 落点章节 |
|------|----------|------|----------|
| Severe #1 | 双重 batch 切分（index_nodes 外层 64 + embed_batch 内层 64）致并发恒为 1 | **采纳方案 A**（评审推荐）：batch 归属单一化。`index_nodes` 一次性把**全部** description 文本传给 `embed_fn(all_texts)`，自身不再按 64 切批传给 embed（仅保留按 64 分批 `col.upsert`）；`EmbeddingCache.embed_batch` 独占 cache 查询→miss 切批(64)→ThreadPoolExecutor 并发→写回→原序组装。新增 AC2.6「经 index_nodes 端到端验证并发」。 | §4.6 / §5.1 / §5.2 / AC2.x |
| Medium #1 | heartbeat 仅随业务 flush 更新，慢文件（30-60s）超 60s 阈值误报僵尸 | **采纳**：①起后台 daemon timer 线程每 10s 只刷 heartbeat（不影响业务计数）；②`is_alive` 增加次级判据——`extract.completed`/`vector.indexed_nodes` 两次读取间有增长即视为 alive，即便 heartbeat 过期。 | §6.2 / §6.3 / §6.5 / AC3.x |
| Medium #2 | 并发分支未用 `with`，embed 失败时线程池不 shutdown、排队 future 仍耗 Token | **采纳**：改 `with ThreadPoolExecutor(...) as pool:`；`except BaseException` 调 `pool.shutdown(wait=False, cancel_futures=True)` 后 `raise`。新增 AC2.7。 | §5.2 / AC2.x |
| Medium #3 | §6.1「问题确认」+ DuckDB 锁断言，与脚注「无需验证」矛盾 | **采纳**：标题与正文改为「保守假设」，明确「锁机制未实证，但 status 不依赖 chroma 的设计在任何锁语义下都稳健」，实证降为实施期可选 follow-up。 | §6.1 / §9 |
| Medium #4 | `enable_embed_cache=False` 与 `embed_concurrency` 语义冲突（关 cache 是否关并发） | **采纳**：cache 与并发**正交可独立开关**。即便 `enable_embed_cache=False` 仍构造 `EmbeddingCache`（get/put 为 no-op 磁盘跳过），`embed_batch` 仍提供并发。在 §4.5/§9 显式声明。 | §4.4 / §4.5 / §4.6 / §9 |
| Medium #5 | ①embedder 注入方式悬空；②并发分支 zip 静默截断残 None | **采纳**：①`EmbeddingCache.__init__` 持有 embedder（与 RateLimiter 注入范式一致），`embed_batch(texts, on_progress=None)`；②`embed_batch` 拿到 vecs 校验 `len(vecs)==len(batch)`，不符记 ERROR 并对该 batch raise（失败安全，对齐 index_nodes 既有 skip-batch 语义），禁止静默截断。 | §4.3 / §5.2 |
| Opt #1 | 矩阵缺 `triples.jsonl` append + replay 去重成本 | **采纳**：矩阵补一行（replay 路径，IO Bound 中），§8 注明超大规模可索引化。 | §2.1 / §8 |
| Opt #2 | `synthesize_fallback_descriptions` 最坏 O(N×E) 评级偏低 | **采纳**：矩阵 step6 标注「大图需关注」，§8 补可选优化（按 source_file 分批）。 | §2.1 / §8 |
| Opt #3 | Manifest 新增字段未 bump version | **采纳（保守变体）**：保持 `version="2"`，在 §7.3 注明「version 不变，新字段为 2.x 增量」，避免误判既有 `== "2"` 逻辑。 | §7.3 |
| Opt #4 | `_probe_embedding_dim` 失败（dim=0）时 cache 行为 | **采纳**：§4.4 增加一行——dim=0 时不写 cache，避免探测失败期无效缓存堆积。 | §4.4 |
| Opt #5 | 并发 put 安全性论断缺既有先例佐证 | **采纳**：§5.2/§5.4 显式引用 `ExtractionCache.put` 在 `extract_doc_concurrency>1` 下的现网并发先例（与 embed cache 同构）。 | §5.2 / §5.4 |
### round 3

二轮评审已 PASS（0 Severe / 0 Medium / 6 Optimization）。用户要求把 6 个非阻断 Optimization 一并修掉再 finalize。下表逐条列出处置结论，**全部采纳**，落点章节见右列。本轮不新增任何 Severe/Medium 级设计变更。

| 编号 | 问题摘要 | 处置 | 落点章节 |
|------|----------|------|----------|
| Opt#1 | `embed_batch` 未对 miss 文本去重，重复 description 重复 embed + 同 key 并发写 | **采纳**：`embed_batch` 内按 `self._key(t)` 对 miss_texts **先去重**再切批 embed（重复文本只 embed + put 一次），结果用 `dedup_map: key -> vector` **广播回填**所有同 key 原始位置；§5.2 措辞「零写冲突」改为「不同 key 写不同文件（安全）；同 key 经去重后不并发写」。给出去重 + 广播回填伪代码。 | §5.2 |
| Opt#2 | 跨子图串行 `index_nodes` 限制「多小文件」聚合并发收益 | **采纳**：§8 T2 注明边界——T2 并发收益仅在「单次 index_nodes 内」生效；多小文件（如 10×30 节点）聚合并发度仍为 1，不应被误读为对所有工作负载等比例加速。把「跨 path 累积 texts 一次 embed」/「跨 path 并发 index_nodes（注意 upsert 锁）」列为**可选 follow-up**（不在本方案范围）。避免 AC2.6 被误读。 | §8 T2 / AC2.6 |
| Opt#3 | `check_liveliness` 固定 2s 阻塞拖慢 status 命令 | **采纳**：`PROGRESS_LIVENESS_RECHECK_SEC` 改为 Settings 可配置（`progress_liveliness_recheck_sec`），默认调小到 **1s**；§7.2 场景 4 输出前加 spinner 提示「正在复核进程存活…」改善体感；注明调小 recheck 与 heartbeat interval(10s) 的协调（缩短 recheck 不影响 heartbeat 主判据，仅缩短「heartbeat 过期后等待二次采样」的窗口）。 | §6.2 / §6.5 / §6.6 / §7.2 |
| Opt#4 | `VectorStoreBackend` Protocol 签名未同步更新 | **采纳**：§4.6 显式注明「同步更新 `VectorStoreBackend` Protocol 签名」为 `def index_nodes(self, graph, llm, *, embed_fn=None, on_progress=None) -> None`；embed_fn/on_progress 为可选关键字参数，默认 None 时走原 `llm.embed` 路径，保证测试 mock VectorStore 仍兼容。 | §4.6 |
| Opt#5 | AC 未显式断言「cache 命中后向量仍被 upsert 进 ChromaDB（可被 search）」 | **采纳**：阶段一补一条 AC1.6「cache 全命中重跑后，`vector_store.search(query)` 返回预期节点」（防未来误改 index_nodes 跳过 cache 命中项的 upsert）。 | AC1.x |
| Opt#6 | heartbeat timer 启动时机在伪代码中未显式落点 | **采纳**：§6.3 `BuildProgressWriter.__init__` 伪代码补一行 `if self._enabled: self._start_heartbeat_timer()`；注明 `enable_build_progress=False` 时 writer 为 no-op、不起 timer（与 §6.6 一致）；给出 `_start_heartbeat_timer` 的最小伪代码（daemon Thread 循环 sleep + 刷 heartbeat）。 | §6.3 |

**保留不动的部分**：
- round 2 修复的 Severe #1 + 5 Medium + 5 Opt（首轮）全部保留，不回退。
- 5 阶段结构、依赖图、数据模型、status mock 输出、风险与回滚章节保持。
- 本轮只动 Optimization，不新增 Severe/Medium 级设计变更。
- 首轮 Reviewer Notes 肯定的内容（key 三维设计、独立缓存目录、best-effort 降级矩阵、5 阶段依赖图与文件冲突分析、复用 `atomic_write_json`/`RateLimiter`/`as_completed`、`except KeyboardInterrupt: writer.interrupted(); raise` 在 `pipeline.compile` 内部的位置、RateLimiter 并发正确性论断、status 绝不打开 chroma 的决策）。

---

## 1. 背景与目标

### 1.1 背景

nanokb 的 `pipeline.compile` 是一条「检测 → 抽取 → 图构建 → 向量索引 → 社区/关键词索引 → 原子落盘」的两阶段流水线。在长期使用中暴露出五个相互关联的痛点：

1. **embedding 无缓存**：`ExtractionCache` 只缓存 `ExtractionResult`（triples/concepts），不缓存 `description→vector`。step7（`VectorStore.index_nodes`）每次重跑都重新调用 `embedder.embed()`，吃 embedding Token。尤其在编译被中断（Ctrl-C）后重跑时，所有已计算过的 embedding 重新付费。
2. **embedding 阶段串行**：`index_nodes` 内部是 `for start in range(0, len(items), EMBED_BATCH_SIZE)` 的串行 batch 循环，对比阶段 A 的 `SemanticTrack.extract`（`ThreadPoolExecutor` chunk 并发）完全缺失并发能力。
3. **全流程无系统化瓶颈清单**：step3-11 各阶段的 bound 类型（LLM/IO/CPU）、是否缓存、是否并发没有成文矩阵，提速方向靠猜。
4. **跨进程进度不可见**：build 进程跑的时候，另一个终端的 `status` 进程只能读 `out/graph.json`/`manifest.json`——而这两者是阶段 B 最后 `staging_swap` 原子切换后才生效，运行期间读到的是上一轮的旧产物，无法反映当前进度。更严重的是：`status` 命令若尝试打开 `out/chroma/`（ChromaDB PersistentClient），会与正在写的 build 进程发生锁冲突。
5. **status 命令信息量不足**：当前 `status` 只输出「raw/ 下 N 个文档 | out/ 已编译/未编译」，不展示当前阶段、extract 进度、索引重建进度。

### 1.2 目标

- **G1**：内容寻址的 embedding 缓存，step7 中断后重跑零 embedding Token 成本。
- **G2**：embedding 阶段支持可配置并发，复用 SemanticTrack 的失败安全 + 乱序回放范式，**且在 `index_nodes` 真实调用路径下并发真实生效**（round 2 新增强约束，对应 Severe #1）。
- **G3**：产出系统化的全流程瓶颈矩阵 + 按 ROI 排序的提速建议清单。
- **G4**：运行时进度文件，build 周期性原子写、status 只读；并**以保守假设规避** ChromaDB 跨进程锁问题（round 2 措辞修正，对应 Medium #3）。
- **G5**：增强 `status` 命令，展示阶段化进度。

### 1.3 范围

- **纳入**：上述 5 目标的实现，覆盖 `src/nanokb/`（vector_store、新建 embed cache、pipeline、新建 progress 模块、cli）与测试。
- **不纳入**：阶段 B（破坏性变更）的断点续传/checkpoint 机制（仅作风险记录，独立 feature 处理）；embedding provider 侧的批处理 API 升级；社区 LLM 摘要的并发化（默认走启发式零 Token，ROI 低）。
- **非目标**：不改 `ExtractionCache` 三维 key 设计（已稳定）；不改 `staging_swap` 原子切换语义（已稳定）；不要求 `embed_concurrency>1` 必须开启 cache（round 2 明确：二者正交，见 Medium #4）。

## 2. 现状分析（瓶颈矩阵）

### 2.1 全流程 Step × Bound × Cache × 并发 矩阵

**阶段 A（抽取，失败安全）**

| 操作 | LLM Bound | IO Bound | CPU Bound | Cache | 并发 | 备注 |
|------|-----------|----------|-----------|-------|------|------|
| `detect_changes`（五维身份比对） | - | 中（遍历磁盘） | 低 | - | 串行 | 无需并发 |
| `ingest_file`（Unstructured 解析） | - | **高** | 中 | - | **文档级并发**（`extract_doc_concurrency`，默认 1） | GIL 下纯解析收益有限 |
| `ExtractionCache.get` | - | 中（磁盘读 JSON） | 低 | **有** | - | sha256\|config\|model 三维 key |
| `SemanticTrack.extract`（LLM 抽取） | **高**（贵） | - | 低 | **有**（结果落盘） | **chunk 级并发**（`extract_chunk_concurrency`，默认 4） | ThreadPoolExecutor |
| `ExtractionCache.put` | - | 高（磁盘写） | 低 | - | - | `atomic_write_json` |

**阶段 B（破坏性变更，统一执行）**

| Step | 操作 | LLM Bound | IO Bound | CPU Bound | Cache | 并发 | 瓶颈级别 |
|------|------|-----------|----------|-----------|-------|------|----------|
| step3 | deletion 级联（`delete_by_source`） | - | 低 | 低 | - | 串行 | 极低 |
| step4 | modified 清旧 | - | 低 | 低 | - | 串行 | 极低 |
| step5 | graph 构建（`upsert`） | - | 中 | 低 | - | 串行 | 低（幂等要求，不宜并发） |
| step6 | `synthesize_fallback_descriptions` | - | - | 中（**大图需关注**：最坏 O(N×E)，稠密边图十万节点级可成非 LLM 瓶颈） | - | 串行 | 低（中小图）/ 中（大图） |
| **step7** | **向量索引（`index_nodes`）** | **高**（embed） | **高**（ChromaDB upsert） | 低 | **无** | **串行** | **关键瓶颈** |
| step8 | community/keyword 索引 | 低（可选 LLM 摘要） | 中 | **高**（Leiden） | - | 串行 | 中（CPU 密集，算法内在串行） |
| step9 | manifest 更新 | - | 低 | 低 | - | 串行 | 极低 |
| step10-11 | 序列化 + `staging_swap` | - | 高（磁盘） | 低 | - | 串行 | 低（原子切换要求） |
| (replay) | `out/triples.jsonl` append + `pipeline.replay` 读全量去重重建 | - | **中**（append 写小，但 replay 读全量去重随行数线性增长） | 低 | - | 串行 | 低（中小库）/ 中（超大库线性成本，可索引化） |

### 2.2 约束与限制

- **幂等约束**：step5 `upsert` 必须保证「先删同主键边再插」，并发会破坏幂等；step7 `col.upsert(ids=...)` 已是幂等（同 id 覆盖），但 ChromaDB `collection.upsert` 非完全线程安全（内部 SQLite 写锁），**并发 upsert 有锁冲突风险**。
- **原子性约束**：step10-11 `staging_swap` 必须串行（manifest 最后写作为「提交点」）。
- **失败安全约束**：阶段 A 不触碰 graph/chroma/triples.jsonl 写；阶段 B 任一步骤失败需保持上一轮产物可用（`staging_swap` 保证）。
- **跨进程约束**：ChromaDB `PersistentClient` 持有 DuckDB + SQLite 句柄，build 写入期间第二个进程打开同目录存在锁冲突风险（详见 §6.1，**保守假设**）。
- **确定性约束**：`SemanticTrack.extract` 按 `chunk_index` 升序回放合并保证输出确定；embedding 输出本身与顺序无关（向量是逐 text 独立的），并发不影响确定性。

## 3. 总体设计

### 3.1 阶段拆分与依赖图

```
┌─────────────────────────────────────────────────────────────┐
│  批次 A（可并行，无文件冲突）                                  │
│                                                              │
│  阶段一: Embedding Cache          阶段三: 运行时进度文件       │
│  (新 llm/embed_cache.py          (新 utils/progress.py        │
│   + 改 vector_store.py)           + 改 pipeline.py)           │
│         │                              │                      │
│         ▼                              ▼                      │
│  阶段二: Embedding 并发            阶段四: status 命令增强     │
│  (改 vector_store.py              (改 cli.py                  │
│   + 配置项)                        + 复用 RichProgressReporter)│
│                                                              │
│  阶段五: 瓶颈排查报告（纯文档，与上述全程并行）                  │
└─────────────────────────────────────────────────────────────┘

依赖关系：
  阶段二 ──依赖──▶ 阶段一（共享 EmbeddingCache 读写，cache 命中后的 miss 才并发 embed）
  阶段四 ──依赖──▶ 阶段三（status 读 .build_progress.json）
  阶段一/三/五 ──互相独立──▶ 可同批次并行
```

> **round 2 说明（Severe #1）**：阶段二「并发 embed」的逻辑现已下沉到 `EmbeddingCache.embed_batch` 内部（见 §5.2），`index_nodes` 不再自带 batch 切分传给 embed。因此「阶段二」实现量集中在 `embed_cache.py` 内的并发分支与配置项；阶段一交付的 `embed_batch` 框架（cache 查询 + miss 组装）即阶段二并发的承载点，二者在同一文件内串行演进。

### 3.2 并行批次建议（供 sprint 拆分）

| 批次 | 特征 | 阶段 |
|------|------|------|
| **批次 A** | 无文件冲突，3 个阶段可并行实现 | 阶段一、阶段三、阶段五 |
| **批次 B** | 依赖批次 A 的产物，2 个阶段串行（各自依赖前置） | 阶段二（依赖一）、阶段四（依赖三） |

**文件冲突分析**：
- 阶段一/二：都改 `vector_store.py` + 新建/扩展 `llm/embed_cache.py` → **必须串行**（二在一之后）。
- 阶段三/四：阶段三改 `pipeline.py` + 新建 `utils/progress.py`；阶段四改 `cli.py` + 读 `utils/progress.py` → 文件冲突仅限 `utils/progress.py`（三写四读），可批次内并行定义接口、批次 B 实现消费。
- 阶段五：纯文档（`.ghs/plans/` 或 `docs/`），零冲突。

### 3.3 设计原则贯穿

- **最小变更**：复用 `atomic_write_json`、`RateLimiter`、`ThreadPoolExecutor(as_completed)`、`ExtractionCache` 的内容寻址范式、`RichProgressReporter` 表现层。
- **best-effort + 可禁用**：所有新机制都有配置开关（默认开），损坏/异常退化为旧行为（参照 `ExtractionCache` 解析失败即 miss）。
- **可回滚**：每阶段独立交付，删除新增文件 + 回退配置项即恢复旧行为，不破坏既有 `out/` 产物。
- **可测试**：每阶段给出单元/集成测试策略。

## 4. 阶段一：Embedding Cache（目标 G1）

### 4.1 数据模型与 key 设计

**key 公式**：
```
embed_key = sha256(f"{description_sha256}|{embedding_model}|{embedding_dim}")
```
- `description_sha256` = `sha256(description.encode("utf-8")).hexdigest()`（先对文本求哈希，避免超长 key 拼接）。
- `embedding_model`：复用 `settings.embedding_model`（与 `ExtractionCache` 的 `llm_model` 维度对齐）。
- `embedding_dim`：embedding 实际输出维度（由 `_probe_embedding_dim` 探测，与 `VectorStore` metadata 的 `embedding_dim` 一致）。

**为什么含 dim 而非只含 model**：同名 model 在不同 provider 端点可能输出不同维度（如 GLM embedding-3 支持可配维度），dim 作为额外维度保证同 model 不同 dim 不串。

**为什么不含 source_file**：与 `ExtractionCache` 一致——同 description 不同 source_file 自动共享缓存（内容寻址，跨文档去重）。

### 4.2 存储布局

- **独立目录** `out/embed_cache/`（**不复用** `out/extract_cache/`）。理由：生命周期不同——换 `extraction_config`/`llm_model` 时 extract cache 失效但 embed cache 仍有效；换 `embedding_model`/`embedding_dim` 时反之。独立目录便于按维度独立清理，且不污染既有 extract cache 逻辑。
- **单条一文件**：`out/embed_cache/<embed_key>.json`，与 `ExtractionCache` 的 `<key>.json` 范式一致。
- **value 格式**（JSON）：
  ```json
  {
    "embedding_model": "text-embedding-3-small",
    "embedding_dim": 1536,
    "vector": [0.0123, -0.0456, "..."]
  }
  ```
  - 包裹一层元数据而非裸存 `list[float]`：读取时可校验 `embedding_model`/`embedding_dim` 与当前配置匹配（防御层，类似 `VectorStore._ensure_collection` 的维度校验），不匹配视为 miss。
  - vector 用 JSON 数组（复用 `atomic_write_json`），1536 维 float 约 20KB/条，可接受；不引入 numpy `.npy`（破坏原子写范式 + 增依赖）。

### 4.3 读写流程（`EmbeddingCache`）

新增 `src/nanokb/llm/embed_cache.py`。**round 2（Medium #5①）**：embedder 在构造时注入并持有，与 `RateLimiter` 注入 `OpenAIClient` 的范式一致；`embed_batch` 不再外传 embedder。

```python
class EmbeddingCache:
    """内容寻址 embedding 缓存：out/embed_cache/<key>.json。

    key = sha256(f"{description_sha256}|{embedding_model}|{embedding_dim}")。
    best-effort：解析失败/维度不匹配视为 miss（不阻断主线），可删可重建。

    round 2 设计要点：
    - embedder 在构造时绑定（self._embedder），与 RateLimiter 注入范式一致。
    - embed_batch 独占 cache 查询 + miss 切批 + (可选)并发 embed + 写回 + 原序组装。
      index_nodes 一次性把全部 description 传进来，不做外层 batch 切分（Severe #1）。
    - cache 与并发正交：enable_cache=False 时 get/put 为 no-op，embed_batch 仍提供并发（Medium #4）。

    round 3 设计要点（Opt#1）：
    - embed_batch 对 miss 文本按 self._key(t) 先去重再切批 embed：重复 description（不同 node
      同描述，常见于兜底合成节点）只 embed + put 一次，结果广播回填所有同 key 原始位置。
      既省 Token 又避免「同 key 并发写同一文件」。
    """

    def __init__(
        self,
        cache_dir: Path,
        embedding_model: str,
        embedding_dim: int,
        embedder: EmbeddingClient,
        *,
        embed_concurrency: int = 1,
        enable_cache: bool = True,
    ) -> None:
        self._cache_dir = cache_dir
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._embedder = embedder            # 持有（Medium #5①）
        self._embed_concurrency = max(1, embed_concurrency)
        self._enable_cache = enable_cache    # 正交开关（Medium #4）

    def _key(self, description: str) -> str:
        desc_sha = hashlib.sha256(description.encode("utf-8")).hexdigest()
        raw = f"{desc_sha}|{self._embedding_model}|{self._embedding_dim}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, description: str) -> list[float] | None:
        """命中返回向量，miss/损坏/维度不匹配返回 None。enable_cache=False 时恒返回 None。"""
        if not self._enable_cache:
            return None
        # 复用 ExtractionCache.get 的 try/except -> None 范式

    def put(self, description: str, vector: list[float]) -> None:
        """原子写回（atomic_write_json）。enable_cache=False 或 dim=0 时为 no-op（Opt#4）。
        写失败仅 WARNING，不抛。"""
        if not self._enable_cache or self._embedding_dim == 0:
            return
        # atomic_write_json(...)

    def embed_batch(
        self,
        texts: list[str],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """批量 embed：查 cache 命中 → miss 去重 → miss 切批(64) → (可选)并发 embed → 校验 → 写回 → 原序组装。

        详见 §5.2 的并发线程模型与失败安全实现（embed_batch 主体在 §5.2 给出完整伪代码，
        阶段一交付串行分支，阶段二在其上叠加并发分支，二者在同一方法内）。
        """
```

**核心方法 `embed_batch` 的语义**：把「查询缓存 + miss 去重 + 批量 embed + 写回」封装为一个原子动作，`VectorStore.index_nodes` 调用它替代裸 `llm.embed(texts)`。

### 4.4 失败降级

| 场景 | 行为 |
|------|------|
| cache 目录不存在 | 自动 `mkdir(parents=True)`（与 `ExtractionCache.__init__` 一致） |
| cache 文件损坏（JSON 解析失败） | `get` 返回 None（miss），重新 embed |
| `embedding_model`/`dim` 不匹配 | `get` 返回 None（视为过期，重新 embed） |
| `embedding_dim == 0`（探测失败） | **不写 cache**（Opt#4）：避免探测失败期无效缓存堆积 |
| `put` 写盘失败 | 仅 `logger.warning`，结果留在内存继续 upsert |
| 配置关闭（`enable_embed_cache=False`） | **round 2（Medium #4）**：仍构造 `EmbeddingCache`（`enable_cache=False`），`get`/`put` 为 no-op；`embed_batch` **照常提供并发**。cache 与并发正交。 |

### 4.5 配置开关

`config.py` 新增：
```python
enable_embed_cache: bool = True   # embedding 缓存总开关（默认开）
embed_concurrency: int = 4        # embedding batch 并发度（默认 4）
```

**round 2（Medium #4）正交性声明**：`enable_embed_cache` 与 `embed_concurrency` 是**两个独立维度，可任意组合**。实现保证：`pipeline.compile` 在 `_resolve_embedder` 后**始终**构造 `EmbeddingCache`，并把 `cache.embed_batch` 作为 `embed_fn` 注入 `index_nodes`。不存在「关 cache 就绕过 EmbeddingCache」的路径。

### 4.6 集成点

`pipeline.compile` step7 改造（**round 2 Severe #1 + Medium #5①**：`index_nodes` 一次性传入全部 texts）：

```python
# 改后（阶段一 + 阶段二统一）
cache = EmbeddingCache(
    out_dir / "embed_cache",
    embedding_model=settings.embedding_model,
    embedding_dim=resolved_dim,
    embedder=embedder,                                   # 构造绑定（Medium #5①）
    embed_concurrency=settings.embed_concurrency,
    enable_cache=settings.enable_embed_cache,
)
vector_store.index_nodes(subgraph, embedder, embed_fn=cache.embed_batch)
```

**round 3（Opt#4）同步更新 `VectorStoreBackend` Protocol 签名**：

```python
@runtime_checkable
class VectorStoreBackend(Protocol):
    def delete_by_source(self, source_file: str) -> None: ...
    def index_nodes(
        self,
        graph: nx.MultiDiGraph,
        llm: EmbeddingClient,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """embed_fn/on_progress 为可选关键字参数（round 3 Opt#4）。
        默认 None 时走原 llm.embed 路径（零回归），保证测试 mock VectorStore 仍兼容。"""
        ...
```

**`VectorStore.index_nodes` 改造伪代码**（签名增 `embed_fn`，主体去掉外层 batch 切分传 embed，保留 upsert 分批）：

```python
def index_nodes(self, graph, llm, *, embed_fn=None, on_progress=None):
    items = []  # 收集 (node_id, label, source_file, description)
    if not items:
        return
    texts = [desc for _, _, _, desc in items]

    if embed_fn is None:
        embed_fn = lambda t: llm.embed(t)

    # ★ Severe #1 修复：一次性把全部 texts 传给 embed_fn，不做外层 64 切分。
    embeddings = embed_fn(texts)

    # 防御性契约校验（禁止静默截断，对齐 Medium #5②）
    if len(embeddings) != len(items):
        logger.error("embed_fn returned %d vectors for %d items", len(embeddings), len(items))
        raise RuntimeError(f"embed_fn length mismatch: {len(embeddings)} != {len(items)}")

    # 串行 upsert，按 EMBED_BATCH_SIZE 分批（upsert 批大小，与 embed 切批解耦）
    for start in range(0, len(items), EMBED_BATCH_SIZE):
        batch_items = items[start:start + EMBED_BATCH_SIZE]
        batch_vecs = embeddings[start:start + EMBED_BATCH_SIZE]
        self._collection.upsert(
            ids=[nid for nid, _, _, _ in batch_items],
            embeddings=batch_vecs,
            documents=[d for _, _, _, d in batch_items],
            metadatas=[{"label": lbl, "source_file": sf} for _, lbl, sf, _ in batch_items],
        )
        if on_progress:
            on_progress(min(start + EMBED_BATCH_SIZE, len(items)), len(items))
```

> **★ round 3（Opt#5）关键不变式**：`index_nodes` 对 `embeddings`（无论来自 cache 命中还是 miss）**始终全量 upsert**，绝不因「向量来自 cache 命中」而跳过 `col.upsert`。cache 只省 embed HTTP 调用，**向量必须进 ChromaDB 才能被 `search` 召回**。AC1.6 防御性断言此不变式。

### 4.7 测试策略

- **单元测试** `tests/unit/test_embed_cache.py`：`get` 四态、`put` 原子写 + dim=0 no-op、`embed_batch` cache 命中/miss/二次全命中/原序组装/长度校验 raise、**round 3 Opt#1 去重测试**、`enable_embed_cache=False` 正交性。
- **集成测试** `tests/integration/test_embed_cache_resume.py`：删 `graph.json` 模拟中断 → force 重跑 embed 调用数 == 0；**round 3 Opt#5**：全命中重跑后 `vector_store.search(query)` 返回预期节点。

## 5. 阶段二：Embedding 并发（目标 G2）

### 5.1 并发粒度选型与 batch 归属（round 2 Severe #1）

采用**方案 B（batch 归属单一化）**：`EmbeddingCache.embed_batch` 独占 miss 切批 + 并发；`index_nodes` 一次性把全部 description 传入（§4.6），自身只在拿到全部向量后分批 `col.upsert`。**禁止两层都切批**——这是 Severe #1 的根因修复。

### 5.2 线程模型（借鉴 SemanticTrack，round 2 全面修正；round 3 Opt#1 增 miss 去重）

```python
def embed_batch(self, texts, on_progress=None):
    # 1. 查 cache（主线程）；enable_cache=False 时全部视为 miss
    results: list[list[float] | None] = [self.get(t) for t in texts]
    miss_idx = [i for i, r in enumerate(results) if r is None]
    if not miss_idx:
        return results  # 全命中

    # ★ round 3（Opt#1）：对 miss 文本按 self._key(t) 去重
    miss_texts_raw = [texts[i] for i in miss_idx]
    seen: dict[str, int] = {}
    unique_miss: list[str] = []
    for t in miss_texts_raw:
        k = self._key(t)
        if k not in seen:
            seen[k] = len(unique_miss)
            unique_miss.append(t)
    dedup_map: dict[str, list[float]] = {}

    # 2. miss 去重后按 batch 切分（EMBED_BATCH_SIZE=64）——embed_batch 独占切批
    batches = [(s, unique_miss[s:s + EMBED_BATCH_SIZE])
               for s in range(0, len(unique_miss), EMBED_BATCH_SIZE)]
    concurrency = self._embed_concurrency
    done = 0

    def _do_one(start: int, batch: list[str]) -> int:
        # ★ Medium #5②：返回前校验长度，禁止静默截断
        vecs = self._embedder.embed(batch)
        if len(vecs) != len(batch):
            logger.error("embedder returned %d vectors for %d texts; fail-safe abort batch",
                         len(vecs), len(batch))
            raise RuntimeError(f"embedder length mismatch: {len(vecs)} != {len(batch)}")
        for i, v in zip(range(start, start + len(batch)), vecs):
            self.put(unique_miss[i], v)
            dedup_map[self._key(unique_miss[i])] = v
        return len(batch)

    # 3. 串行分支（concurrency==1，阶段一基线 / 零回归）
    if concurrency <= 1:
        for start, batch in batches:
            done += _do_one(start, batch)
            if on_progress:
                on_progress(done, len(unique_miss))
    else:
        # 4. 并发分支（concurrency>1，阶段二）
        # ★ Medium #2：with 语句 + except BaseException 取消未决 future
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            try:
                futures = {pool.submit(_do_one, s, b): (s, b) for s, b in batches}
                for fut in as_completed(futures):
                    done += fut.result()
                    if on_progress:
                        on_progress(done, len(unique_miss))
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    # 5. 广播回填：按原始位置查 dedup_map
    for orig_idx in miss_idx:
        results[orig_idx] = dedup_map[self._key(texts[orig_idx])]
    return results
```

**关键修正点**：
1. **batch 归属单一化（Severe #1）**：切批只在 `embed_batch` 内做一次。
2. **`with` + `except BaseException`（Medium #2）**：异常分支显式 `cancel_futures=True` 取消排队 future。
3. **长度校验禁止静默截断（Medium #5②）**。
4. **并发 put 安全性既有先例（Opt#5）+ round 3 措辞修正**：不同 key 写不同文件（线程安全，与 `ExtractionCache.put` 现网先例同构）；同 key 经 Opt#1 去重后不并发写。
5. **round 3 miss 去重 + 广播回填（Opt#1）**。

**为什么 upsert 不并发、只并发 embed**：`embed` 是纯网络 IO（GIL 释放收益最大）；`col.upsert` 操作 ChromaDB（SQLite + DuckDB 写），并发会触发写锁冲突。因此 `index_nodes` 仍**串行 upsert**，但 embed 阶段并发。

### 5.3 限流复用

`RateLimiter`（`src/nanokb/llm/throttle.py`）已是**进程级、线程安全**（`threading.Lock` 锁内 sleep 串行化 acquire，HTTP 请求在锁外并发）。**无需新限流器**。风险：`embed_concurrency` 过高 + `RateLimiter.interval` 过小 → 触发 provider 429。缓解：SDK `max_retries` 指数退避；配置文档提示边界。

### 5.4 失败安全

任一 batch `embed` 抛异常 → `fut.result()` 重抛 → `except BaseException` 取消未决 future → `index_nodes` 中断 → pipeline 不执行 `staging_swap`，`out/` 保持上一轮产物。已 embed 并写回 cache 的部分**不丢失**（atomic_write 已落盘），下次重跑直接命中。`KeyboardInterrupt` 与 `pipeline._process_one_file` 的 Ctrl-C 处理一致。

### 5.5 配置项

见 §4.5（`embed_concurrency: int = 4`）。命名对齐 `extract_doc_concurrency`/`extract_chunk_concurrency`；默认 4（保守起步）；`concurrency <= 1` 走串行分支（零回归）。

### 5.6 测试策略

- **单元测试** `tests/unit/test_embed_cache_concurrency.py`：并发耗时、确定性（并发 vs 串行逐向量相等）、中断取消未决 batch、长度不匹配 raise。
- **端到端并发验证（AC2.6）** `tests/integration/test_index_nodes_concurrency.py`：mock embedder 统计「同时 in-flight 调用数峰值」，经 `VectorStore.index_nodes` 索引 500 节点，断言峰值 **≥ 2**。
- **限流测试**：验证 embedder 调用链路正确注入 RateLimiter。

## 6. 阶段三：运行时进度文件（目标 G4）

### 6.1 ChromaDB 跨进程锁：保守假设与规避（round 2 Medium #3）

**保守假设**：`VectorStore.__init__` 调用 `chromadb.PersistentClient`，内部持有 SQLite + DuckDB 句柄。build 写入时 status 若打开同目录存在锁冲突风险。具体锁机制**未实证**，不作为既成事实写入。

**规避决策（稳健，不依赖锁的具体语义）**：**status 命令绝不打开 ChromaDB**。向量总数由 build 写入 `.build_progress.json`（运行期）和 `manifest.json`（完成后）。status 只读纯 JSON 文件，零锁冲突。无论锁机制如何，该设计都成立（最小依赖原则）。

**冗余向量计数到 manifest**：完成后由 build 在 `manifest` 顶层新增可选字段 `total_vectors: int`（向后兼容）。

### 6.2 进度文件 Schema

新增 `src/nanokb/utils/progress.py`（Pydantic v2）：

```python
class BuildStage(str, Enum):
    DETECT = "detect"
    EXTRACT = "extract"
    GRAPH = "graph"
    VECTOR = "vector"
    INDEX = "index"
    FINALIZE = "finalize"
    DONE = "done"
    INTERRUPTED = "interrupted"

class ExtractProgress(BaseModel):
    total: int = 0
    completed: int = 0
    cached: int = 0
    skipped: int = 0

class VectorProgress(BaseModel):
    total_nodes: int = 0
    indexed_nodes: int = 0

class BuildProgress(BaseModel):
    schema_version: str = "1"
    pid: int = 0
    stage: BuildStage = BuildStage.DETECT
    started_at: str = ""
    heartbeat_ts: str = ""
    force: bool = False
    extract: ExtractProgress = Field(default_factory=ExtractProgress)
    vector: VectorProgress = Field(default_factory=VectorProgress)
    message: str = ""

PROGRESS_FILENAME = ".build_progress.json"
PROGRESS_HEARTBEAT_TIMEOUT_SEC = 60
PROGRESS_HEARTBEAT_INTERVAL_SEC = 10   # round 2：后台 heartbeat timer 刷新间隔
PROGRESS_LIVENESS_RECHECK_SEC = 1      # round 3（Opt#3）：默认调小到 1s（可配置）
```

### 6.3 原子写策略与 Heartbeat 解耦（round 2 Medium #1；round 3 Opt#6 显式 timer 启动落点）

新增 `BuildProgressWriter`：

```python
class BuildProgressWriter:
    FLUSH_EVERY = 5

    def __init__(self, out_dir: Path, force: bool, *, enabled: bool = True):
        self._enabled = enabled
        # ... 初始化 self._progress / self._path / self._lock ...
        self._stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None
        # ★ round 3（Opt#6）显式落点：构造即启动 heartbeat timer（仅 enabled 时）
        if self._enabled:
            self._start_heartbeat_timer()

    def _start_heartbeat_timer(self) -> None:
        self._stop = threading.Event()
        def _loop():
            assert self._stop is not None
            while not self._stop.wait(PROGRESS_HEARTBEAT_INTERVAL_SEC):
                with self._lock:
                    self._progress.heartbeat_ts = _now_iso()
                    atomic_write_json(self._path, self._progress.model_dump())
        t = threading.Thread(target=_loop, daemon=True, name="nanokb-build-heartbeat")
        t.start()
        self._heartbeat_thread = t

    def set_stage(self, stage, message=""): ...
    def update_extract(self, *, total=None, completed_delta=0, cached_delta=0, skipped_delta=0, force_flush=False): ...
    def update_vector(self, *, total=None, indexed_delta=0, force_flush=False): ...
    def done(self): ...        # 标记 DONE 并删除进度文件；停止 timer
    def interrupted(self): ... # 标记 INTERRUPTED 保留文件；停止 timer
```

**实现要点**：
- **启动时机（round 3 Opt#6）**：`__init__` 末尾 `if self._enabled: self._start_heartbeat_timer()`。`enabled=False` 时不起 timer（零回归）。
- daemon 线程，主进程退出自动结束。
- `Event.wait` 既 sleep 又可被 `set()` 唤醒。
- heartbeat 线程只刷 `heartbeat_ts`，不触碰业务计数。
- **并发安全**：一把 `threading.Lock` 保护「序列化内存对象 + atomic_write」临界区。
- `done()` 删除进度文件；`interrupted()` 写 INTERRUPTED 保留文件供诊断。

### 6.4 Heartbeat 与进程退出处理

| 退出场景 | 处理 |
|----------|------|
| 正常完成 | `writer.done()` → 停 timer → 删除文件 |
| Ctrl-C | `writer.interrupted()` → 停 timer → 写 INTERRUPTED 保留 |
| 未捕获异常 | 文件残留 + heartbeat 过期 → status 识别僵尸（或计数增长判 alive） |
| kill -9 | 同上 |

> **`os._exit(130)` 兼容性**：`except KeyboardInterrupt: writer.interrupted(); raise` 在 `pipeline.compile` 内部，`interrupted()` 在异常上抛到 `cli.build` 之前执行，不依赖 finally/atexit；每次 flush 直接落盘。

### 6.5 status 降级读取与 liveliness 次级判据（round 2 Medium #1；round 3 Opt#3 recheck 可配置）

```python
def read_progress(out_dir: Path) -> BuildProgress | None:
    path = out_dir / PROGRESS_FILENAME
    if not path.exists():
        return None
    try:
        return BuildProgress.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None

def check_liveliness(out_dir: Path, *, recheck_sec: float | None = None) -> bool:
    """heartbeat 过期时，sleep recheck_sec 后重读，计数增长即判 alive。"""
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
    return (p1.extract.completed > p0.extract.completed
            or p1.vector.indexed_nodes > p0.vector.indexed_nodes)
```

**降级矩阵**：文件不存在→旧逻辑；is_alive→展示进行中；INTERRUPTED→展示中断；heartbeat 过期但计数增长→进行中；heartbeat 过期且无增长→僵尸。

### 6.6 集成点（pipeline.compile）

配置开关：
```python
enable_build_progress: bool = True
progress_liveliness_recheck_sec: float = 1.0    # round 3（Opt#3）
```
关闭时 `BuildProgressWriter` 为 no-op（不起 timer）。

### 6.7 测试策略

- **单元测试** `tests/unit/test_build_progress.py`：原子写、flush 阈值、heartbeat 解耦（业务阻塞 30s 仍每 10s 刷）、**round 3 Opt#6 timer 启动落点**、**round 3 Opt#3 recheck 可配置**、`read_progress` 降级、`is_alive`/`check_liveliness`。
- **集成测试** `tests/integration/test_cross_process_progress.py`：子进程读 progress 验证跨进程可见性；完成后降级。
- **跨进程隔离测试**：验证 status 不打开 chroma/。

## 7. 阶段四：status 命令增强（目标 G5）

### 7.1 数据来源优先级

运行期（`.build_progress.json`）优先，静态（`manifest.json`/`graph.json`）兜底。

### 7.2 新输出设计（Mock 终端样例）

**场景 1：编译进行中**
```
- nanokb 状态 -----------------------------------
  [spin] 编译进行中  (PID 12345, 已运行 2m 15s)
  阶段   vector (向量索引) — 正在索引 src/api.py 的节点...
  | [ok] detect    | 完成  | -           |
  | [ok] extract   | 100%  | 30/100 cached|
  | [ok] graph     | 完成  | -           |
  | [run] vector   | 45%   | 2250/5000   |
  | [ ]  index     | 待开始| -           |
  raw/ 120 个文档 | 已抽取 100 | 缓存命中 30
--------------------------------------------------
```

**场景 2：编译完成** / **场景 3：上次中断** / **场景 4：僵尸进程（heartbeat 超时）**

> **round 3（Opt#3）**：场景 4 输出前 TTY 显示 spinner「正在复核进程存活…（约 1s）」；非 TTY 跳过 spinner。

实现：`rich.table.Table` + `rich.panel.Panel` + `rich.progress.Progress`，复用 `cli.py` 既有 `console`。非 TTY 降级纯文本（参照 `RichProgressReporter` 的 `is_terminal` 分支）。

### 7.3 manifest 扩展（向后兼容，version 不变）

```python
class Manifest(BaseModel):
    version: str = "2"                # round 2（Opt#3）：version 不变
    files: dict[str, FileState] = Field(default_factory=dict)
    total_vectors: int = 0          # 新增
    last_compiled_at: str = ""      # 新增
    last_llm_model: str = ""        # 新增
    last_embedding_model: str = ""  # 新增
```

保持 `version="2"`，新字段视为「2.x 增量」（Pydantic 可选字段 + 默认值保证向后兼容）。

### 7.4 测试策略

- **单元测试** `tests/unit/test_cli_status.py`：CliRunner + 4 场景夹具 + snapshot 比对 + round 2 误报僵尸 + round 3 spinner。
- **集成测试** `tests/integration/test_status_during_build.py`：build 运行期子进程调 status。
- **降级测试**：无 `.build_progress.json` 走旧逻辑。

## 8. 阶段五：瓶颈排查与可选提速清单（目标 G3，按 ROI 排序）

| 序号 | 建议 | bound | ROI | 必做/可选 |
|------|------|-------|-----|-----------|
| **T1** | **embedding 缓存** | LLM | 极高 | **必做**（阶段一） |
| **T2** | **embedding 并发** | LLM(IO) | 高（单大文件）/ 低（多小文件，见边界） | **必做**（阶段二） |
| T3 | extract chunk 并发提升 | LLM | 中 | 可选（用户调参） |
| T4 | 文档级并发 | IO(LLM) | 中 | 可选（用户调参） |
| T5 | 社区 LLM 摘要并发 | LLM | 低 | 可选（默认关闭） |
| T6 | Leiden 多线程 | CPU | 低 | 不建议 |
| T7 | 文档加载并发 | IO+CPU | 中 | 可选（随 T4） |
| T8 | staging 落盘并行 | IO | 低 | 可选 |
| T9 | ChromaDB upsert 批量化 | IO | 低 | 不建议 |
| T10 | manifest/graph 增量序列化 | IO | 低 | 不建议 |
| **T11** | `synthesize_fallback_descriptions` 大图优化 | CPU | 低/中（大图） | 可选 |
| **T12** | `triples.jsonl` replay 去重索引化 | IO | 低/中（超大库） | 可选 |

**核心结论**：**T1 + T2 是唯一两个「必做」提速项**。

**round 3（Opt#2）T2 并发收益边界说明**：T2 收益边界为「**单次 `index_nodes` 调用内**并发真实生效」。多小文件（如 10×30 节点）聚合并发度仍为 1。AC2.6「峰值 ≥2」仅验证单次 index_nodes 内并发，不适用于多小文件场景。可选 follow-up：①跨 path 累积 texts 一次 embed；②跨 path 并发 index_nodes（需评估 upsert 锁）。

## 9. 风险与回滚

完整风险表涵盖：cache 损坏、并发限流、向量漂移、未取消 future、双重 batch 回归、静默截断、cache/并发耦合、并发写冲突、cache 命中未 upsert、进度文件频繁写、heartbeat 耦合、timer 启动错误、文件残留、ChromaDB 锁假设、status 误开 chroma、manifest 字段、旧 out/ 降级、embed_cache 膨胀、recheck 调小误判。每条都有缓解/回滚路径。

**回滚声明**：`enable_embed_cache=False` + `embed_concurrency=1` 等价改造前。删除新增文件 + 配置置默认 + 回退 `index_nodes` 的 `embed_fn` 参数 + **回退 Protocol 签名（Opt#4）**即恢复旧行为。

## 10. 验收标准

### 阶段一：Embedding Cache
- AC1.1 首次 compile embed 批次符合预期。
- AC1.2 删 graph.json 模拟中断，force 重跑 embed 调用 == 0。
- AC1.3 损坏 cache 文件视为 miss 重 embed。
- AC1.4 `enable_embed_cache=False, embed_concurrency=1` 零回归。
- AC1.5 `dim=0` 时 put 不写 cache（Opt#4）。
- **AC1.6**（round 3 Opt#5）全命中重跑后 `vector_store.search(query)` 返回预期节点。

### 阶段二：Embedding 并发
- AC2.1 单 batch 并发受 batch 数限制。
- AC2.2 8 batches 耗时 ≈ 串行 1/4。
- AC2.3 并发 vs 串行逐向量相等。
- AC2.4 异常不 swap，已完成 batch 已写 cache。
- AC2.5 RateLimiter 全局 RPM 不超限。
- **AC2.6**（round 2 Severe #1，阻断项）经 index_nodes 端到端验证峰值 ≥2（边界见 T2）。
- **AC2.7**（round 2 Medium #2）异常时排队 batch 被取消。
- **AC2.8**（round 2 Medium #5②）长度不匹配 raise 不残 None。
- **AC2.9**（round 2 Medium #4）关 cache 不关并发。
- **AC2.10**（round 3 Opt#1）重复 description 去重 embed + 广播回填正确。

### 阶段三：运行时进度文件
- AC3.1 跨进程读到 EXTRACT + completed>0。
- AC3.2 完成后文件删除。
- AC3.3 Ctrl-C 标 INTERRUPTED 保留。
- AC3.4 status 不打开 chroma。
- AC3.5 损坏文件降级。
- AC3.6 heartbeat 过期且无增长判僵尸。
- **AC3.7**（round 2 Medium #1）heartbeat 过期但计数增长判 alive。
- **AC3.8**（round 2 Medium #1）业务阻塞 30s 仍每 10s 刷 heartbeat。
- **AC3.9**（round 3 Opt#6）enabled 时有 daemon 线程，disabled 无。
- **AC3.10**（round 3 Opt#3）recheck 取自配置 + spinner。

### 阶段四：status 命令增强
- AC4.1 运行期输出「编译进行中」+ 阶段 + 进度条。
- AC4.2 完成输出「已编译」+ 文档数 + 向量数 + 模型。
- AC4.3 中断输出「上次编译中断」+ 阶段 + 零成本提示。
- AC4.4 旧 out/ 向后兼容。
- AC4.5 非 TTY 纯文本可 CliRunner 断言。

### 阶段五：瓶颈排查报告
- AC5.1 输出 markdown 瓶颈矩阵（§2 + §8，含 round 2/3 增项）。
- AC5.2 T1/T2 标「必做」。

---

**方案涉及文件清单**（供 sprint 拆分冲突检测）：
- 新增：`src/nanokb/llm/embed_cache.py`、`src/nanokb/utils/progress.py`
- 修改：`src/nanokb/index/vector_store.py`、`src/nanokb/pipeline.py`（**含 `VectorStoreBackend` Protocol 签名同步，round 3 Opt#4**）、`src/nanokb/config.py`（新增 `enable_embed_cache`/`embed_concurrency`/`enable_build_progress`/`progress_liveliness_recheck_sec`）、`src/nanokb/cli.py`、`src/nanokb/models.py`（Manifest 新字段）
- 新增测试：`tests/unit/test_embed_cache.py`、`tests/unit/test_embed_cache_concurrency.py`、`tests/unit/test_build_progress.py`、`tests/unit/test_cli_status.py`、`tests/integration/test_embed_cache_resume.py`、`tests/integration/test_index_nodes_concurrency.py`、`tests/integration/test_cross_process_progress.py`、`tests/integration/test_status_during_build.py`
- 文档：`docs/performance.md`（瓶颈矩阵）

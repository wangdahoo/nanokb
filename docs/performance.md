# nanokb 编译性能瓶颈矩阵与提速建议

nanokb `pipeline.compile`（检测 → 抽取 → 图构建 → 向量索引 → 社区/关键词索引 → 原子落盘）全流程的瓶颈矩阵、约束、按 ROI 排序的提速清单与用户调参指南。内容源自技术方案 `.ghs/plans/2026-06-27-build-embedding-status.md` §2 + §8。

## 1. 全流程瓶颈矩阵

### 1.1 阶段 A（抽取，失败安全）

| 操作 | LLM Bound | IO Bound | CPU Bound | Cache | 并发 | 备注 |
|------|-----------|----------|-----------|-------|------|------|
| `detect_changes`（五维身份比对） | - | 中（遍历磁盘） | 低 | - | 串行 | 无需并发 |
| `ingest_file`（Unstructured 解析） | - | **高** | 中 | - | **文档级并发**（`extract_doc_concurrency`，默认 1） | GIL 下纯解析收益有限 |
| `ExtractionCache.get` | - | 中（磁盘读 JSON） | 低 | **有** | - | sha256\|config\|model 三维 key |
| `SemanticTrack.extract`（LLM 抽取） | **高**（贵） | - | 低 | **有**（结果落盘） | **chunk 级并发**（`extract_chunk_concurrency`，默认 4） | ThreadPoolExecutor |
| `ExtractionCache.put` | - | 高（磁盘写） | 低 | - | - | `atomic_write_json` |

### 1.2 阶段 B（破坏性变更，统一执行）

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

> replay 行对应 Opt#1；step6 `synthesize_fallback_descriptions` 的大图标注对应 Opt#2。

## 2. 约束与限制

下列约束共同决定了哪些步骤可以并发、哪些必须串行。任何提速方案都不得违反。

- **幂等约束**：step5 `upsert` 必须保证「先删同主键边再插」，并发会破坏幂等；step7 `col.upsert(ids=...)` 已是幂等（同 id 覆盖），但 ChromaDB `collection.upsert` 非完全线程安全（内部 SQLite 写锁），**并发 upsert 有锁冲突风险**。
- **原子性约束**：step10-11 `staging_swap` 必须串行（manifest 最后写作为「提交点」）。
- **失败安全约束**：阶段 A 不触碰 graph/chroma/triples.jsonl 写；阶段 B 任一步骤失败需保持上一轮产物可用（`staging_swap` 保证）。
- **跨进程约束**：ChromaDB `PersistentClient` 持有 DuckDB + SQLite 句柄，build 写入期间第二个进程打开同目录存在锁冲突风险（保守假设，锁机制未实证）。
- **确定性约束**：`SemanticTrack.extract` 按 `chunk_index` 升序回放合并保证输出确定；embedding 输出本身与顺序无关（向量是逐 text 独立的），并发不影响确定性。

## 3. 提速建议清单（按 ROI 排序）

| 序号 | 建议 | bound | ROI | 必做/可选 |
|------|------|-------|-----|-----------|
| **T1** | **embedding 缓存** | LLM | 极高 | **必做** —— 已被 sprint s3 的 feat-001 覆盖 |
| **T2** | **embedding 并发** | LLM(IO) | 高（单大文件）/ 低（多小文件，见边界说明） | **必做** —— 已被 sprint s3 的 feat-003 覆盖 |
| T3 | extract chunk 并发提升 | LLM | 中 | 可选（用户调参，见 §4 场景①） |
| T4 | 文档级并发 | IO(LLM) | 中 | 可选（用户调参，见 §4 场景①） |
| T5 | 社区 LLM 摘要并发 | LLM | 低 | 可选（默认关闭） |
| T6 | Leiden 多线程 | CPU | 低 | 不建议 |
| T7 | 文档加载并发 | IO+CPU | 中 | 可选（随 T4） |
| T8 | staging 落盘并行 | IO | 低 | 可选 |
| T9 | ChromaDB upsert 批量化 | IO | 低 | 不建议 |
| T10 | manifest/graph 增量序列化 | IO | 低 | 不建议 |
| T11 | `synthesize_fallback_descriptions` 大图优化（按 source_file 分批） | CPU | 低/中（大图） | 可选 |
| T12 | `triples.jsonl` replay 去重索引化 | IO | 低/中（超大库） | 可选 |

**核心结论**：**T1 + T2 是唯一两个「必做」提速项**，其余皆为可选/不建议。

### 3.1 T2 并发收益边界说明

T2（embedding 并发）的收益边界为「**单次 `index_nodes` 调用内**并发真实生效」。多小文件场景（如 10×30 节点）聚合并发度仍为 1，不应被误读为对所有工作负载等比例加速。

- 单大文件（如 1 个含 500 节点的子图）：miss ≥ 2 batch 时并发收益显著（≈ 串行耗时的 1/N，N = `embed_concurrency`）。
- 多小文件（每个子图节点数 < `EMBED_BATCH_SIZE`，默认 64）：每次 `index_nodes` 内只有 1 个 miss batch，并发度退化为 1。

**可选 follow-up**（不在当前方案范围）：
1. **跨 path 累积 texts 一次 embed**：把多个文件待 embed 的 description 累积后一次性调用 `embed_batch`，使多小文件也能凑齐多 batch。
2. **跨 path 并发 `index_nodes`**：需评估 ChromaDB upsert 写锁冲突风险（见 §2 幂等约束），实现前必须先实证锁语义。

## 4. 用户调参指南

针对三种典型工作负载的调参建议。配置项位于 `config.py`。

### 场景①：首次全量编译慢

冷启动无任何缓存，所有抽取 + embedding 都需走 LLM/网络。

- 调高 `extract_chunk_concurrency`（默认 4，可至 8-12）：加速 `SemanticTrack.extract` 的 chunk 级 LLM 并发（对应 T3）。
- 调高 `extract_doc_concurrency`（默认 1，可至 2-4）：加速 `ingest_file` + 抽取的文档级并发（对应 T4/T7）。
- 保持 `embed_concurrency`（默认 4）：首次编译 embedding 全是 miss，T2 并发对单大文件有效。
- 注意 `RateLimiter.interval` 与并发的协调：并发过高 + 限流间隔过小可能触发 provider 429，依赖 SDK 的指数退避兜底。

### 场景②：中断重跑慢

编译被 Ctrl-C 中断后 force 重跑。

- **feat-001 上线后**：`out/embed_cache/` 已保存已计算过的向量（key = `sha256(description_sha256|embedding_model|embedding_dim)`），重跑时 `EmbeddingCache.get` 全命中 → embedding 调用次数为 0，**零 Token 成本**。
- 即便 cache 文件损坏或 model/dim 变更，对应条目自动视为 miss 并重新 embed，其余命中不受影响（best-effort 降级）。
- 注意：cache 命中的向量仍会被 `col.upsert` 写入 ChromaDB（cache 只省 embed HTTP 调用，向量必须进库才能被 `search` 召回）。

### 场景③：增量编译慢

仅少量文件变更时重编译。

- 单文件节点数 < 64 时 T2 并发无收益（每次 `index_nodes` 内只有 1 个 miss batch，并发度退化为 1），见 §3.1 边界说明。
- 若增量涉及多小文件且需要 embedding，考虑可选 follow-up「跨 path 累积 texts 一次 embed」。
- 同一文件的 description 若已在 `embed_cache` 中（内容寻址，跨文档共享），增量编译零 embedding 成本。

### 4.1 两个自动化保护（无需用户干预）

- **Opt#4（dim=0 期间不写 cache）**：`_probe_embedding_dim` 探测失败（`embedding_dim == 0`）期间，`EmbeddingCache.put` 为 no-op，避免探测失败期无效缓存堆积。探测正常后自动恢复写入。
- **Opt#1（重复 description 自动去重 embed）**：`embed_batch` 对 miss 文本按 `_key(t)` 先去重再切批 embed，重复 description（不同 node 同描述，常见于兜底合成节点）只 embed + put 一次，结果广播回填所有同 key 原始位置。既省 Token 又避免「同 key 并发写同一文件」。

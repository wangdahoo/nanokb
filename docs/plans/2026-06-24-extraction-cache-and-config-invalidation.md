# 抽取缓存 + 配置失效修复

- **日期**：2026-06-24
- **目标**：`compile` 对身份未变的文件复用历史 `ExtractionResult`，跳过 LLM 调用；修复所有影响输出但未被 detector 追踪的配置字段。

## 背景

### extract 结果保存在哪里

`compile` 的 `extractor.extract(doc)` 返回 `ExtractionResult(triples, concepts)`，先存内存 `results_map`（`pipeline.py:251,290`），阶段 B 落地为 4 个产物：

| 产物 | 内容 | 是否原始抽取结果 |
|---|---|---|
| **`out/triples.jsonl`** | 追加日志，每条 upsert 记录含完整 `triples` + `concepts`（`pipeline.py:330-337`） | ✅ **唯一**存完整原始抽取结果 |
| `out/graph.json` | `GraphBuilder.upsert` 去重合并后的图（`pipeline.py:921` 经 staging 切换） | ❌ 派生（已按主键合并） |
| `out/chroma` | 节点描述的 embedding（`pipeline.py:356`） | ❌ 派生 |
| `out/manifest.json` | 仅元数据（sha256 / 版本），无抽取内容 | ❌ |

原始抽取结果只活在 `triples.jsonl`。`replay()`（`pipeline.py:405`）已证明它能纯从该日志重建图谱、零 LLM 调用——这是"抽取缓存"的雏形。

### extract 的真实依赖维度

- **SemanticTrack**：依赖 `{文件内容(sha256), 分块配置, extractor_version(prompt+策略), llm_model}`，**不依赖** `embedding_model`。
- **CodeTrack**：纯 tree-sitter、零 LLM，只依赖 `{sha256, extractor_version}`，完全确定性。

### 现状缓存机制与盲区

`detect_changes`（`detector.py:112-117`）用**四维身份**判 modified：

```
sha256 / extractor_version / llm_model / embedding_model   任一变 → modified
```

四维全等 → 文件"unchanged" → `compile` 整体跳过（`pipeline.py:216-218`）。**无变更时缓存已生效**。

盲区：检测器把两个本应独立的失效维度**耦合**了：
- **抽取维度** = `{sha256, extractor_version, llm_model}` —— 决定要不要重跑 LLM
- **向量维度** = `{embedding_model}` —— 只决定要不要重算 embedding

后果：**换 `embedding_model` → 全量重抽取（烧 LLM token），但抽取输出和 `triples.jsonl` 里已有的一模一样**。`compile` 对 modified 文件永远重新调 `extractor.extract`（`pipeline.py:281`），从不回读复用。

### 配置失效 bug 的完整范围

当前四维身份对 **7 个**影响输出的配置字段存在漏洞：

| 字段 | 影响 | 是否追踪 |
|---|---|---|
| `chunk_max_tokens` / `chunk_overlap_tokens` | 分块 → 语义三元组 | ❌ |
| `concept_description_strategy` | concept 合并 | ❌ |
| `code_languages` | CodeTrack 语言门控 | ❌ |
| `fallback_description_max_edges` | 图构建兜底描述 | ❌ |
| `leiden_symmetrize` | 社区/关键词索引 | ❌ |
| `embedding_provider` | 向量后端 | ❌ |

改其中任一，detector 都返回"无变更" → `compile` 整体跳过 → 产出 stale 图/索引/向量。缓存方案放大了前 4 个（stale 抽取缓存）；后 3 个属独立的图/索引/向量 staleness。

---

## Part 1 — 抽取缓存（content-addressable）

**存储**：`out/extract_cache/<key>.json`，key = `sha256(f"{doc.sha256}|{extraction_config}|{llm_model}")`。
- 不含 `source_file` → 同内容不同路径文件自动共享缓存（correctness 由 `_normalize_result_source` `pipeline.py:761` 在加载时盖 rel_key 保证）。
- best-effort：可删可重建，解析失败 → 视为 miss。

**新模块** `src/nanokb/extract/cache.py`：

```python
class ExtractionCache:
    def __init__(self, cache_dir: Path)
    def _key(sha256, extraction_config, llm_model) -> str
    def get(sha256, extraction_config, llm_model) -> ExtractionResult | None
    def put(sha256, extraction_config, llm_model, result) -> None   # atomic_write_json
```

**pipeline.py 阶段 A 改造**（`pipeline.py:280-296`）：

```python
key = (doc.sha256, sig_extraction, settings.llm_model)
cached = extract_cache.get(*key)
if cached is not None:
    results_map[path] = _normalize_result_source(cached, path); cached_count += 1
else:
    result = extractor.extract(doc); extract_cache.put(*key, result)
    results_map[path] = _normalize_result_source(result, path)
```

阶段 B（`pipeline.py:298+`）**不动** —— 仍写 triples.jsonl（幂等追加）、重建图、重建向量。换 `embedding_config` 时：阶段 A 全命中缓存（0 LLM），阶段 B 用新 embedding 重建向量。正是目标行为。

**`CompileResult`**（`pipeline.py:109`）加 `cached_count: int = 0`；`cli.py:72` 加 `cached={result.cached_count}`。

## Part 2 — 配置失效三层签名

**三个签名 helper**（新模块 `src/nanokb/extract/config_signature.py` 或 colocate 于 config.py）：

```python
def extraction_config_signature(s) -> str:   # extractor_version + chunk_* + strategy + code_languages
def index_config_signature(s) -> str:        # fallback_description_max_edges + leiden_symmetrize
def embedding_config_signature(s) -> str:    # embedding_model + embedding_provider
# 每个 = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False))
```

`extractor_version` 被**折叠**进 `extraction_config`（涵盖 prompt + 内置逻辑变更），取代裸维度。

**关键：缓存 key 只用 `extraction_config`** → 改 index/embedding 配置时缓存命中（0 LLM），只重建下游派生物。

字段归属：

| 字段 | 归入签名 |
|---|---|
| `extractor_version` / `chunk_max_tokens` / `chunk_overlap_tokens` / `concept_description_strategy` / `code_languages` | `extraction_config` |
| `fallback_description_max_edges` / `leiden_symmetrize` | `index_config` |
| `embedding_model` / `embedding_provider` | `embedding_config` |

## Part 3 — FileState / detector / pipeline 写入

**`models.py` FileState**（`models.py:79`）新增三字段（全部 default `""`，不删旧字段 → 零迁移风险）：

```python
extraction_config: str = ""
index_config: str = ""
embedding_config: str = ""
```

保留 `extractor_version`/`embedding_model`/`embedding_dim`（继续写入供调试/VectorStore 用，但 detector 不再比较 `extractor_version`、改比较签名）。

**`detector.py:112-117`** 身份比对从四维 → 五维（任一变 → modified）：

```
sha256 | extraction_config | llm_model | index_config | embedding_config
```

（`embedding_config` 取代裸 `embedding_model`；`extraction_config` 取代裸 `extractor_version`。）

**`pipeline.py` manifest 写入**（`pipeline.py:364` compile / `pipeline.py:467` replay）：填入三个签名。

**迁移**：旧 manifest 三签名字段缺失 → 首次 compile 全量 modified → 一次性重抽取 + manifest 重写（重建，无数据丢失）。replay 路径同样补签名。

### 设计决策与权衡

- **为何不只用 bump-version 约定**：易错、依赖人记忆、不可扩展。签名是自动的。
- **为何用单签名字段而非 4 个独立 FileState 字段**：schema churn 更小（1 字段），且 cache key 需要单一 hash。未来新增抽取配置只改签名计算函数，无需 schema/detector 变动。
- **过度失效成本**：改 `chunk_*` 会把 `.py` 文件标 modified → 重抽，但 CodeTrack 不依赖 chunker 且零 token + 确定性。cache miss → 重抽成本可忽略。故全文件单签名可接受（廉价的简单性取舍）。

## Part 4 — 测试

**新增** `tests/unit/test_extraction_cache.py`：put→get 往返；缺失返回 None；三维变化 key 不同；跨文件去重（同内容不同路径 → 一个缓存文件均命中）；损坏文件忽略。

**新增** `tests/unit/test_config_signature.py`：三签名确定性 + 各字段变化触发签名变化。

**扩展** `tests/integration/test_compile_md.py`：
- 换 `embedding_config`（embedding_model 或 provider）→ `complete_calls==0`、`cached_count>0`、向量重建计数增加。
- 换 `chunk_max_tokens` / `concept_description_strategy` → 缓存 miss → 重抽取。
- 换 `index_config`（fallback_max_edges / leiden_symmetrize）→ 缓存命中（0 LLM）、graph/索引重建。
- 换 `code_languages`（禁 java）→ `.java` 重抽取为空。
- 内容变化 → miss。

**扩展** `tests/integration/test_code_extraction.py`：`.py` 换 embedding → 缓存命中（CodeTrack 确定性，零 token）。

## Part 5 — 风险与缓解

| 风险 | 缓解 |
|---|---|
| LLM 非确定性被冻结 | 可接受（图更稳定）；删缓存文件即强制重抽 |
| 换 chunker 仅波及 CodeTrack 过度失效 | CodeTrack 重抽零 token + 确定性，成本可忽略，换单签名简洁性 |
| 缓存孤儿（删文件后） | 无害（best-effort）；GC 列为后续 |
| triples.jsonl 缓存命中时追加冗余记录 | replay 去重收敛已处理；保持简单 |
| 旧 manifest 升级 | 首次全量重抽，一次性成本 |

## Part 6 — 执行顺序

1. 签名 helpers（3 个）+ `FileState` 三字段
2. `detector` 五维比对
3. `pipeline` manifest 写入三签名（compile + replay）
4. `ExtractionCache` 模块
5. `pipeline` 阶段 A 缓存接入 + `cached_count`
6. `cli` 输出
7. 单元测试 + 集成测试
8. 跑 `pytest` + `ruff` 全绿

## 涉及文件清单

| 文件 | 动作 |
|---|---|
| `src/nanokb/extract/config_signature.py`（或 config.py 内） | 新增 3 签名 helper |
| `src/nanokb/extract/cache.py` | 新增 ExtractionCache |
| `src/nanokb/models.py` | FileState +3 字段 |
| `src/nanokb/load/detector.py` | 五维比对 |
| `src/nanokb/pipeline.py` | 缓存接入 + cached_count + 签名写入（compile+replay） |
| `src/nanokb/cli.py` | 输出 cached= |
| `tests/unit/test_extraction_cache.py` | 新增 |
| `tests/unit/test_config_signature.py` | 新增 |
| `tests/integration/test_compile_md.py` | 扩展 |
| `tests/integration/test_code_extraction.py` | 扩展 |

# 抽取缓存 + 配置失效三层签名修复

## 1. Background and Goals

### 1.1 Background

`compile` 流水线的阶段 A 对每个待处理文件调用 `extractor.extract(doc)` 返回
`ExtractionResult(triples, concepts)`，结果存入内存 `results_map`，阶段 B 落地为
`graph.json` / `triples.jsonl` / `chroma/` / `communities.json`+`keywords.json` 等
派生产物。原始抽取结果只活在 `out/triples.jsonl` 日志中，`replay()` 已证明能纯从该
日志零 LLM 调用重建图谱。

当前 `detect_changes`（`load/detector.py` lines 110-118）用四维身份判定 modified：
`sha256 / extractor_version / llm_model / embedding_model` 任一变更 → modified；四维全等
→ 整体跳过。存在两个核心缺陷：

- **缺陷 1（耦合盲区）**：抽取维度与向量维度耦合。换 `embedding_model`（纯向量侧变更）
  会导致全量重新调用 LLM 抽取（烧 token），而抽取结果本身并不依赖 embedding。
- **缺陷 2（覆盖漏洞）**：7 个影响输出的配置字段未被 detector 追踪 ——
  `chunk_max_tokens` / `chunk_overlap_tokens` / `concept_description_strategy` /
  `code_languages` / `fallback_description_max_edges` / `leiden_symmetrize` /
  `embedding_provider`。改其中任一，detector 返回“无变更” → 整体跳过 → 产出
  stale 的图/索引/向量。
- **缺陷 3（无抽取缓存）**：compile 对 modified 文件永远重新调 `extractor.extract`，
  从不回读复用历史 `ExtractionResult`，即便抽取输入完全未变。

### 1.2 Goals

1. **抽取缓存（content-addressable）**：compile 对“内容 + 抽取配置 + llm_model”
   未变的文件复用历史 `ExtractionResult`，跳过 LLM 调用。
2. **配置失效三层签名**：用三个内容寻址签名（extraction / index / embedding）取代
   裸字段比对，精确覆盖全部 7 个漏洞字段。
3. **缓存 key 只依赖 extraction_config**：换 index/embedding 配置时缓存命中（0 LLM），
   只重建下游派生物（图谱/索引/向量）。

### 1.3 Scope

- **In**：新增 `config_signature.py` / `extract/cache.py` 两个模块；扩展 `FileState`
  三字段；改写 detector 比对维度；pipeline 阶段 A 接缓存；CLI 输出 `cached=`；
  新增单元测试 + 扩展集成测试。
- **Out**：不改 `triples.jsonl` schema_version（保持 "2"）；不删 `FileState` 旧字段
  （零迁移风险）；不改 `replay()` 去重收敛规则；不改 VectorStore/GraphBuilder 内部
  实现；不引入缓存 TTL / LRU 淘汰（best-effort，可删可重建）。

## 2. Current State Analysis

### 2.1 Existing Architecture

- **`models.py` `FileState`**（lines 79-88）：当前 7 字段，四维身份 =
  `sha256 + extractor_version + llm_model + embedding_model`（`embedding_dim` 仅供
  VectorStore 用，不参与身份比对）。
- **`load/detector.py` `detect_changes`**（lines 87-129）：四维任一不等 → modified。
- **`pipeline.py` `compile`**：
  - 阶段 A（lines 259-300）：遍历 `added ∪ modified`，逐文件 `extractor.extract(doc)`
    → `_normalize_result_source(result, path)` → `results_map[path]`。
  - 阶段 B（lines 304-389）：删除级联 → modified 先清后建 → 图构建 → fallback 合成 →
    向量索引 → build_indexes → manifest 更新（lines 366-376）→ staging 原子切换。
  - `replay`（lines 409-499）：从 `triples.jsonl` 零 LLM 重建图谱，step 9（lines 466-479）
    同样写 FileState。
- **`CompileResult`**（lines 113-119）：`changes / extracted_count / skipped /
  synthesized_fallback_count`。
- **`extract/`**：
  - `SemanticTrack.extract`（`semantic_track.py` line 109）：依赖
    `{sha256, chunk 配置, extractor_version(prompt+策略), llm_model}`，**不依赖** embedding。
  - `CodeTrack.extract`（`code_track.py` line 242）：纯 tree-sitter、零 LLM、完全确定性，
    依赖 `{sha256, extractor_version, code_languages}`。
- **`utils/io.py`**：`atomic_write_json`（lines 60-72）已存在，复用即可。
- **`cli.py` `_print_compile_summary`**（lines 68-81）：组装 `added/modified/deleted/
  extracted/...` 摘要行。
- **配置字段**（`config.py` lines 67-79）：上述 7 个漏洞字段 + 3 个已追踪字段均定义于此。

### 2.2 Constraints and Limitations

- **零迁移**：旧 `manifest.json` 缺三签名字段（pydantic default `""`）。首次 compile 时
  `""` ≠ 计算签名 → 全量 modified → 一次性重抽取 + manifest 重写。这是期望的“自愈”
  行为，无需迁移脚本。
- **测试回归风险**：`tests/unit/test_detector.py` 的 `_file_state` helper 手工构造
  FileState 不含新签名字段，改为 5 维比对后“unchanged”用例会被误判为 modified（详见
  Phase 2 + Phase 7）。
- **list 字段确定性**：`code_languages` 为 list，`json.dumps(..., sort_keys=True)` 只排
  dict 键不排 list 元素 → 签名计算前必须显式排序，否则同集合不同顺序产生不同签名。
- **缓存不参与 staging 原子切换**：缓存为 best-effort 旁路，损坏即视为 miss，不影响主线
  一致性（graph/triples.jsonl/manifest 仍是 staging 提交点）。

## 3. Plan Design

### 3.1 Overall Architecture

```
┌─ config_signature.py (NEW, top-level) ──────────────────────────┐
│  extraction_config_signature(s)  index_config_signature(s)      │
│  embedding_config_signature(s)   _sig(payload)                  │
└──────────────┬───────────────────────────┬──────────────────────┘
               │ (读 Settings)             │
   ┌───────────▼──────────┐    ┌───────────▼──────────────┐
   │ load/detector.py     │    │ pipeline.py             │
   │ 5 维比对（取代 4 维） │    │ manifest 写三签名        │
   └──────────────────────┘    │ 阶段 A 接 ExtractionCache│
                               └───────────┬──────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │ extract/cache.py (NEW)          │
                          │ ExtractionCache: _key/get/put   │
                          │ cache_dir = out/extract_cache/  │
                          └─────────────────────────────────┘
```

**三层签名职责切分**（按“谁影响哪一层派生物”切）：

| 签名 | 涵盖字段 | 影响的派生物 | 缓存 key 是否包含 |
|---|---|---|---|
| `extraction_config` | extractor_version + chunk_max_tokens + chunk_overlap_tokens + concept_description_strategy + code_languages | 抽取结果（→ triples/concepts → 图谱结构） | **是** |
| `index_config` | fallback_description_max_edges + leiden_symmetrize | 图谱节点描述（fallback）+ 社区索引 | 否 |
| `embedding_config` | embedding_model + embedding_provider | 向量索引 | 否 |

**设计决策**：
- **模块位置**：`config_signature.py` 放 **顶层**（`src/nanokb/config_signature.py`），
  而非需求建议的 `extract/config_signature.py`。理由：三个签名横跨 extraction / index /
  embedding 三个关注点，其中 index/embedding 与“抽取”无关，放在 `extract/` 下语义错位；
  且 detector（`load/`）+ pipeline + tests 均消费它，顶层中性位置最自然。
- **单签名字段 vs 多字段**：FileState 用三个签名串而非 7+ 个裸字段（schema churn 更小，
  缓存 key 也只需单一 hash）。`extractor_version` 折叠进 `extraction_config`，detector
  不再单独比较裸 `extractor_version`。
- **缓存 key 只用 extraction_config**：改 index/embedding 配置时缓存命中（0 LLM），仅
  重建下游派生物。代价：改 `chunk_*` 会让 CodeTrack 过度失效，但 CodeTrack 重抽零 token +
  确定性，成本可忽略（廉价的简单性取舍）。
- **不只用 bump-version 约定**：版本号约定易错、不可扩展，签名方案精确且可审计。

### 3.2 Data Model

**`FileState` 扩展**（`models.py`，保留旧字段 → 零迁移）：

```python
class FileState(BaseModel):
    path: str
    sha256: str
    processed_at: str
    # 旧字段（保留，供调试 / VectorStore / 向后兼容）
    extractor_version: str = "1"
    llm_model: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0
    # 新增三层签名（detector 比对用；default "" → 旧 manifest 触发全量 modified）
    extraction_config: str = ""
    index_config: str = ""
    embedding_config: str = ""
```

**`CompileResult` 扩展**（`pipeline.py`）：

```python
class CompileResult(BaseModel):
    changes: ChangeSet = Field(default_factory=ChangeSet)
    extracted_count: int = 0          # 所有进入 results_map 的文件（含缓存命中）
    skipped: list[str] = Field(default_factory=list)
    synthesized_fallback_count: int = 0
    cached_count: int = 0             # NEW：其中来自缓存的文件数
```

语义：`extracted_count = len(results_map)`（所有产出结果并入图谱的文件，不论是否命中缓存）；
`cached_count` 是其中命中缓存、未调 LLM 的子集。

### 3.3 Interface Design

**新模块 `src/nanokb/config_signature.py`**：

```python
import hashlib
import json
from nanokb.config import Settings

def _sig(payload: dict[str, object]) -> str:
    """稳定内容寻址签名：sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False))。"""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def extraction_config_signature(s: Settings) -> str:
    """抽取层签名：决定 ExtractionResult（→ 图谱结构）。"""
    payload = {
        "extractor_version": s.extractor_version,
        "chunk_max_tokens": s.chunk_max_tokens,
        "chunk_overlap_tokens": s.chunk_overlap_tokens,
        "concept_description_strategy": s.concept_description_strategy,
        "code_languages": sorted(s.code_languages),   # 显式排序保证集合等价即签名等价
    }
    return _sig(payload)

def index_config_signature(s: Settings) -> str:
    """索引层签名：决定 fallback 节点描述 + Leiden 社区。"""
    payload = {
        "fallback_description_max_edges": s.fallback_description_max_edges,
        "leiden_symmetrize": s.leiden_symmetrize,
    }
    return _sig(payload)

def embedding_config_signature(s: Settings) -> str:
    """向量层签名：决定 ChromaDB 向量索引。"""
    payload = {
        "embedding_model": s.embedding_model,
        "embedding_provider": s.embedding_provider,
    }
    return _sig(payload)

__all__ = [
    "extraction_config_signature",
    "index_config_signature",
    "embedding_config_signature",
]
```

**新模块 `src/nanokb/extract/cache.py`**：

```python
import hashlib
import json
import logging
from pathlib import Path
from nanokb.models import ExtractionResult
from nanokb.utils.io import atomic_write_json

logger = logging.getLogger("nanokb")

class ExtractionCache:
    """内容寻址抽取缓存：out/extract_cache/<key>.json。

    key = sha256(f"{sha256}|{extraction_config}|{llm_model}")，不含 source_file
    （同内容不同路径自动共享；correctness 由 pipeline._normalize_result_source
    在加载时盖 rel_key 保证）。best-effort：可删可重建，解析失败视为 miss。
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = Path(cache_dir)

    def _key(self, sha256: str, extraction_config: str, llm_model: str) -> str:
        raw = f"{sha256}|{extraction_config}|{llm_model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, sha256: str, extraction_config: str, llm_model: str) -> ExtractionResult | None:
        path = self._dir / f"{self._key(sha256, extraction_config, llm_model)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ExtractionResult.model_validate(data)
        except Exception:
            logger.debug("extract cache miss (corrupt/unreadable): %s", path)
            return None

    def put(
        self, sha256: str, extraction_config: str, llm_model: str, result: ExtractionResult
    ) -> None:
        path = self._dir / f"{self._key(sha256, extraction_config, llm_model)}.json"
        atomic_write_json(path, result.model_dump(mode="json"))

__all__ = ["ExtractionCache"]
```

### 3.4 Key Flows

**compile 阶段 A 改造（缓存命中复用）**：

```python
# pipeline.compile 内，extractor 构造之后、遍历 to_process 之前
cache = ExtractionCache(settings.out_dir / "extract_cache")
extraction_sig = extraction_config_signature(settings)
cached_count = 0

for idx, path in enumerate(to_process, 1):
    abs_path = raw_dir / path
    doc = ingest_file(...)             # （异常处理同现状）

    cached = cache.get(doc.sha256, extraction_sig, settings.llm_model)
    if cached is not None:
        result = cached
        cached_count += 1
        logger.info("[%d/%d] cache hit %s", idx, total, path, ...)
    else:
        result = extractor.extract(doc)   # （异常处理同现状 → skip）
        cache.put(doc.sha256, extraction_sig, settings.llm_model, result)

    results_map[path] = _normalize_result_source(result, path)   # 单一 choke point
    sha_map[path] = doc.sha256
```

正确性论证：
- 缓存 key 含 `doc.sha256`（内容）+ `extraction_config`（抽取配置签名）+ `llm_model`
  → 三者未变则抽取输出必然不变 → 复用安全。
- 缓存 key **不含** index/embedding 配置 → 换它们时命中缓存、跳过 LLM，仅阶段 B 重建
  下游（图谱 fallback 描述 / 社区 / 向量）。
- `_normalize_result_source` 在 `results_map[path] = ...` 唯一入口覆盖所有 triple/concept
  的 source_file 为当前 rel_key，无论结果来自缓存还是新抽取 → 跨文件共享时 correctness
  由该调用兜底。

**detector 5 维比对**（`load/detector.py`）：

```python
extraction_sig = extraction_config_signature(settings)
index_sig = index_config_signature(settings)
embedding_sig = embedding_config_signature(settings)
...
if (
    state.sha256 != digest
    or state.extraction_config != extraction_sig
    or state.llm_model != settings.llm_model
    or state.index_config != index_sig
    or state.embedding_config != embedding_sig
):
    modified.append(rel_key)
```

（移除 `state.extractor_version != settings.extractor_version`，折叠进 extraction_sig。）

**manifest 写入**（compile lines 366-376 与 replay lines 466-479）填入三签名：

```python
manifest.files[path] = FileState(
    path=path,
    sha256=sha_map.get(path, ""),
    processed_at=now,
    extractor_version=settings.extractor_version,   # 保留（调试）
    llm_model=settings.llm_model,
    embedding_model=settings.embedding_model,        # 保留（VectorStore）
    embedding_dim=actual_embedding_dim,
    extraction_config=extraction_config_signature(settings),
    index_config=index_config_signature(settings),
    embedding_config=embedding_config_signature(settings),
)
```

### 3.5 Error Handling

- **缓存损坏**：`ExtractionCache.get` 捕获 `json.JSONDecodeError` / pydantic
  `ValidationError` / `OSError`，记 debug 日志返回 None（miss）→ 退化为重新抽取。
- **缓存 put 失败**：`atomic_write_json` 异常向上传播。阶段 A 的 try/except 已包裹
  `extractor.extract`；需把 `cache.put` 也纳入同一 try 块（put 失败不应阻断已成功抽取
  的文件 —— 实际 put 仅写磁盘，失败概率低；但纳入 try 让 skipped 语义一致）。
  **细化**：cache.put 放在 `extractor.extract` 成功之后、`results_map` 赋值之前，且
  单独 try/except（put 失败记 warning，不影响 results_map 赋值，因结果已在内存）。
- **ingest 失败 / extract 失败**：沿用现状（skip，不污染 results_map）。
- **旧 manifest**：三签名字段 default `""` → 与计算签名不等 → modified → 自愈。

## 4. Implementation Steps

### Phase 1：签名 helpers + FileState 三字段

- [ ] 新建 `src/nanokb/config_signature.py`，实现 `_sig` /
      `extraction_config_signature` / `index_config_signature` /
      `embedding_config_signature`（见 §3.3，注意 `code_languages` 显式 `sorted`）。
- [ ] `src/nanokb/models.py` `FileState` 追加 `extraction_config: str = ""` /
      `index_config: str = ""` / `embedding_config: str = ""`（保留全部旧字段）。
- [ ] 正确性论证：新字段 default `""` 保证旧 manifest 反序列化不报错；签名函数纯函数、
      确定性、仅依赖 Settings。
- [ ] 验证：`python -c "from nanokb.config_signature import ...; from nanokb.models import FileState"`
      可导入；`FileState(path='a', sha256='x', processed_at='t')` 构造成功。
- 回滚：删除新模块 + 移除三字段，无副作用。

**Phase 1 验收**：`config_signature` 三函数对默认 Settings 返回 64 位 hex 且确定性可复现；
`FileState` 含三新字段且 default `""`。

### Phase 2：detector 5 维比对 + 修复 detector 单测 helper

- [ ] `src/nanokb/load/detector.py`：`detect_changes` 内计算三签名，比对维度从 4 维
      （`sha256/extractor_version/llm_model/embedding_model`）改为 5 维
      （`sha256/extraction_config/llm_model/index_config/embedding_config`）；移除裸
      `extractor_version` 比对行；更新函数 docstring（“四维”→“五维”）。
- [ ] **关键回归修复**：`tests/unit/test_detector.py` 的 `_file_state` helper 改为根据
      其 llm_model/embedding_model/extractor_version 参数构造一个对应 `Settings` 并计算
      三签名填入 FileState（保持“unchanged”用例通过）：

      ```python
      from nanokb.config_signature import (
          extraction_config_signature, index_config_signature, embedding_config_signature,
      )
      def _file_state(path, raw_dir, *, llm_model="glm-5.1",
                      embedding_model="text-embedding-3-small", extractor_version="1", **sig_overrides):
          sig_settings = Settings(llm_model=llm_model, embedding_model=embedding_model,
                                  extractor_version=extractor_version, **sig_overrides)
          return FileState(
              path=str(path.relative_to(raw_dir)),
              sha256=sha256_file(path),
              processed_at="2026-01-01T00:00:00Z",
              extractor_version=extractor_version,
              llm_model=llm_model,
              embedding_model=embedding_model,
              extraction_config=extraction_config_signature(sig_settings),
              index_config=index_config_signature(sig_settings),
              embedding_config=embedding_config_signature(sig_settings),
          )
      ```
- [ ] 其余裸 `FileState(sha256="0"*64, ...)` 用例（test_sha256_change /
      test_deleted / test_three_sets 的 mod.md/gone.md）无需改：它们已因 sha256 不匹配
      或 deleted 命中，与签名无关。`test_three_sets` 的 keep.md 走 `_file_state` →
      获得匹配签名 → 正确保持 unchanged。
- [ ] 正确性论证：旧 manifest 三字段缺失（`""`）≠ 计算签名 → 首次 compile 全量 modified。
- [ ] 验证：`pytest tests/unit/test_detector.py -q` 全绿。
- 回滚：还原 detector 四维 + 还原 `_file_state`。

**Phase 2 验收**：`pytest tests/unit/test_detector.py` 全绿；新增维度变更（如改
`chunk_max_tokens` / `fallback_description_max_edges` / `embedding_provider`）能触发
modified（Phase 7 用例覆盖）。

### Phase 3：pipeline manifest 写入三签名（compile + replay）

- [ ] `src/nanokb/pipeline.py` 顶部 `from nanokb.config_signature import ...`。
- [ ] `compile` step 9（lines 366-376）FileState 构造追加 `extraction_config` /
      `index_config` / `embedding_config` 三关键字参。
- [ ] `replay` step 9（lines 466-479）同样追加三签名（基于当前 settings；replay 是全量
      重建，对齐当前配置是期望行为 —— 下次 compile 不会因签名不匹配 spuriously 重抽）。
- [ ] 正确性论证：compile 写入后，下次 compile 读回 → 三签名匹配 → 无 spuriously
      modified。
- [ ] 验证：`pytest tests/integration/test_compile_md.py -q` 全绿（AC#1 manifest 现含三
      新字段；AC#2 二次 compile 仍 extracted_count=0）。
- 回滚：移除三关键字参。

**Phase 3 验收**：`out/manifest.json` 的 FileState 含三非空签名字段；二次 compile（无变更）
extracted_count=0、无 LLM 调用。

### Phase 4：ExtractionCache 模块

- [ ] 新建 `src/nanokb/extract/cache.py`（见 §3.3），`import hashlib`。
- [ ] 正确性论证：key 不含 source_file → 跨文件共享；get 解析失败 → None（miss）；
      put 用 `atomic_write_json` 原子落盘。
- [ ] 验证：`python -c "from nanokb.extract.cache import ExtractionCache"` 可导入。
- 回滚：删除模块（pipeline 尚未引用）。

**Phase 4 验收**：模块可导入；put→get 往返一致（Phase 7 单测覆盖）。

### Phase 5：pipeline 阶段 A 接缓存 + CompileResult.cached_count

- [ ] `src/nanokb/pipeline.py` `CompileResult` 追加 `cached_count: int = 0`。
- [ ] `compile` 内（extractor 构造后）创建 `cache = ExtractionCache(settings.out_dir /
      "extract_cache")` 并预算 `extraction_sig = extraction_config_signature(settings)`；
      初始化 `cached_count = 0`。
- [ ] 改造遍历循环（lines 261-300）：ingest 后先 `cache.get(doc.sha256, extraction_sig,
      settings.llm_model)`；命中则复用 `result`、`cached_count += 1`、记 info 日志；miss
      则 `extractor.extract(doc)` 成功后单独 try `cache.put(...)`（put 失败记 warning 不
      阻断），再赋 `results_map[path] = _normalize_result_source(result, path)`。
      ingest 异常分支保持不变；extract 异常分支的 `cache.put` 不执行（extract 已失败）。
- [ ] `compile` 返回值 `CompileResult(..., cached_count=cached_count)`。
- [ ] 阶段 B 不动（仍写 triples.jsonl、重建图、重建向量、build_indexes）。
- [ ] 正确性论证：缓存命中跳过 LLM；阶段 B 仍对 modified 文件重建下游 → 改 index/embedding
      配置时向量/索引被重建。
- [ ] 验证：`pytest tests/integration/test_compile_md.py tests/integration/test_code_extraction.py -q`
      全绿（首次 compile cached_count=0；二次无变更 compile 因 detector unchanged 根本不进入
      循环，cached_count 仍 0）。
- 回滚：还原循环为直接 `extractor.extract`，移除 cache 引用与 cached_count。

**Phase 5 验收**：换 embedding_config recompile 时 `complete_calls` 不增、
`cached_count>0`；换 chunk_max_tokens recompile 时缓存 miss 重新抽取（Phase 7 覆盖）。

### Phase 6：CLI 输出 cached=

- [ ] `src/nanokb/cli.py` `_print_compile_summary`（lines 68-81）：在 `extracted=...` 之后
      追加条件输出 `if result.cached_count: parts.append(f"cached={result.cached_count}")`。
- [ ] 正确性论证：cached_count=0 时不显示（向后兼容现有输出）；>0 时显示节省的 LLM 调用数。
- [ ] 验证：手工 / CliRunner 跑 `build` 两次，第二次改 embedding 配置后输出含 `cached=1`。
- 回滚：移除该 `if` 块。

**Phase 6 验收**：`build` 摘要行在缓存命中时显示 `cached=N`。

### Phase 7：单元测试 + 集成测试扩展

- [ ] 新建 `tests/unit/test_config_signature.py`：
      - extraction_config_signature 确定性（同 Settings 两次调用相等）；
      - 各字段变化触发签名变化：`extractor_version` / `chunk_max_tokens` /
        `chunk_overlap_tokens` / `concept_description_strategy` / `code_languages`；
      - **`code_languages` 顺序无关**：`["python","javascript"]` 与反转列表产生相同签名
        （验证 `sorted` 正确性）；
      - index_config_signature：`fallback_description_max_edges` / `leiden_symmetrize`
        变化触发变化；
      - embedding_config_signature：`embedding_model` / `embedding_provider` 变化触发变化；
      - 三签名互不干扰（改 extraction 字段不影响 index/embedding 签名）。
- [ ] 新建 `tests/unit/test_extraction_cache.py`（用 `tmp_path`）：
      - `put→get` 往返：`ExtractionResult` 经 put 再 get 后 `model_dump` 相等；
      - 缺失返回 None（未 put 的 key）；
      - 三维变化 key 不同：put 后改 `sha256` / `extraction_config` / `llm_model` 任一 →
        get 返回 None；
      - **跨文件去重**：同一 `(sha256, extraction_config, llm_model)` 第二次 put 覆盖
        第一次（内容寻址，key 与 source_file 无关）；
      - **损坏文件忽略**：手动写非法 JSON 到 `<key>.json` → get 返回 None（不抛异常）。
- [ ] 扩展 `tests/integration/test_compile_md.py`：
      - `test_change_embedding_config_cache_hit_zero_llm`：首次 compile（1 LLM 调用）→
        改 `embedding_model`（或 `embedding_provider`）recompile → 断言
        `llm.complete_calls` 不增（仍为 1）、`result.cached_count == 1`、FakeVectorStore
        重建信号（`index_calls` 增 / `deleted_sources` 含该 path）；
      - `test_change_chunk_max_tokens_cache_miss_reextract`：改 `chunk_max_tokens` →
        extraction_config 变 → 缓存 miss → `complete_calls` 增、`cached_count == 0`；
      - `test_change_concept_description_strategy_cache_miss`：同上换策略；
      - `test_change_index_config_cache_hit`：改 `fallback_description_max_edges` →
        extraction_config 不变 → 缓存命中（`complete_calls` 不增、`cached_count == 1`），
        且图谱/索引重建（triples.jsonl 新增 upsert 记录）；
      - `test_content_change_cache_miss`：改文档内容 → sha256 变 → 缓存 miss 重抽取。
      （复用文件内现有 `FakeLLMClient` / `FakeVectorStore` / `_settings` / `_extract_response`。）
- [ ] 扩展 `tests/integration/test_code_extraction.py`：
      - `test_py_change_embedding_cache_hit`：`.py` 首次 compile（0 LLM）→ 改
        `embedding_model` recompile → `cached_count == 1`、`complete_calls` 仍 0；
      - `test_change_code_languages_disables_python`：禁用 python（`code_languages=["javascript"]`）
        → extraction_config 变 → 缓存 miss → CodeTrack 重抽返回空 → 该 .py 文件的
        calls/defines 边消失（断言图谱中无 code 边）。
- [ ] 正确性论证：每个测试断言“哪个维度变化 → 缓存命中/miss + LLM 调用数 + 下游重建信号”，
      端到端验证三层签名切分。
- 回滚：删除新增测试（不影响实现）。

**Phase 7 验收**：新增 2 个单测文件全绿；test_compile_md / test_code_extraction 扩展用例
全绿。

### Phase 8：pytest + ruff 全绿

- [ ] `pytest -q` 全套通过。
- [ ] `ruff check .` + `ruff format --check .` 全绿（新模块 / 测试符合现有风格）。
- [ ] `mypy` 若项目 CI 启用则一并检查（新模块加类型注解）。
- [ ] 验证：CI 本地等价命令全绿。

**Phase 8 验收**：全套测试 + lint 无回归。

## 5. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 改 detector 比对维度破坏现有 detector 单测的“unchanged”用例 | 高 | 中 | Phase 2 同步重构 `_file_state` helper 计算并填入三签名；Phase 7 覆盖新维度。 |
| `code_languages` 列表顺序导致签名不稳定 | 中 | 中 | `extraction_config_signature` 内 `sorted(s.code_languages)`；Phase 7 专项单测验证顺序无关。 |
| 缓存损坏 / 部分写入导致 compile 崩溃 | 低 | 中 | `ExtractionCache.get` 全异常捕获 → miss；`put` 用 `atomic_write_json` 原子写 + 独立 try/except。 |
| 缓存无限增长（无淘汰） | 低 | 低 | best-effort 设计，可整目录删除重建；用户可手动清 `out/extract_cache/`。后续可加 LRU。 |
| 跨文件共享缓存导致 source_file 错配 | 低 | 高 | `_normalize_result_source` 在 `results_map[path]=...` 唯一入口覆盖 source_file；Phase 7 跨文件用例验证。 |
| 改 llm_model 导致 CodeTrack 过度失效（key 含 llm_model） | 中 | 极低 | CodeTrack 重抽零 token + 确定性，成本可忽略（设计已接受的廉价取舍）。 |
| replay 写当前 settings 签名与历史 triples.jsonl 配置不一致 | 低 | 低 | replay 是全量重建，对齐当前配置是期望行为；下次 compile 不会 spuriously 重抽。 |

**整体回滚策略**：各 Phase 均为原子可回滚（每个 Phase 有独立回滚说明）。若实施中途需要全局回退，按逆序撤销：删测试 → 还原 cli → 还原 pipeline 阶段 A → 删 ExtractionCache → 还原 manifest 写入 → 还原 detector + helper → 删 FileState 三字段 + config_signature 模块。主线提交点（graph/triples.jsonl/manifest）的 staging 原子切换保证编译产物不会处于半成品状态。

## 6. Testing Strategy

- **单元层**：`test_config_signature.py`（签名确定性 + 字段覆盖 + 顺序无关 + 三签名独立）、
  `test_extraction_cache.py`（往返 / miss / 三维 key / 跨文件 / 损坏容错）。
- **集成层**：`test_compile_md.py`（embedding_config 命中零 LLM + 向量重建 / chunk_*
  miss 重抽 / index_config 命中 + 图重建 / 内容变更 miss）、`test_code_extraction.py`
 （.py 换 embedding 命中 / 禁语言 miss 重抽为空）。
- **回归层**：`test_detector.py`（helper 重构后 unchanged/added/modified/deleted 全维度
  仍正确）、现有 compile/code 集成测试不回归。
- **全量门禁**：`pytest -q` + `ruff check .` + `ruff format --check .` 全绿。
- 全程零真实 LLM 调用（FakeLLMClient / FakeVectorStore 注入，tmp_path 隔离）。

## 7. 评审采纳说明（Optimization 项，实施期参考）

评审裁决 PASS（Severe: 0 / Medium: 0 / Optimization: 7）。以下 Optimization 项作为实施期参考，
不阻塞定稿（其中 #7 整体回滚策略已补入 §5）：

1. `test_compile_md.py` 补充 `embedding_provider` 变更场景用例（显式覆盖，区别于 `embedding_model`）。
2. `test_config_signature.py` 补充非 ASCII 配置值的签名稳定性用例。
3. detector 五维比对中 `llm_model` 单独比对（未折叠进 `extraction_config`）的设计决策写入代码注释。
4. `code_languages` 集合语义（顺序无关）在注释中明确说明。
5. ChromaDB collection 命名/重建策略在文档中记录。
6. `code_languages` 顺序无关测试补充同集合不同顺序的缓存命中用例。

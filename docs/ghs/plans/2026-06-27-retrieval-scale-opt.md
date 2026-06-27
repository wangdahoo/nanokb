# 检索规模化优化方案（retrieval-scale-opt）— 最终版

## 0. 背景与目标

针对大型知识库（海量文档/节点）场景下的检索性能瓶颈做**向后兼容的内部加速**，不改变三路融合语义、Retriever 协议、confidence 权重排序与 CLI 行为。

**现状瓶颈**（详见 context snapshot §6）：
- ★★ fuzzy 全量匹配（`difflib.get_close_matches`，O(N) 次 Python 循环 × 每次 `ratio` 计算，规模化最先扛不住）
- ★★ `_build_normalized_node_index` 每查询重建 O(N) 无缓存
- ★★ NER 重复调用：`query` 三路模式下 GraphRetriever 与 CommunityRetriever **各调一次** `_ner_entities`，同一问题 2 次 LLM 往返
- ★ `CommunityRetriever.recall` 线性扫描全部社区 + 每成员每次重新归一化
- ★ 每次 CLI 调用全量加载 graph.json/communities.json（CLI 单次进程内已只加载一次；跨进程无法共享，归为低优先）

**硬约束**：
- 测试 `test_graph_retriever.py`(13) / `test_fusion.py` / `test_three_route_qa.py` / `test_query.py` **全绿**。
- `GraphRetriever(graph, llm, settings)` / `CommunityRetriever(communities, g, llm, settings)` 直传对象的构造入口**签名不变**（缓存为内部加速）。
- 不引入**强依赖**：可选加速（rapidfuzz）做 try-import 降级回退。
- 行为等价：缓存的索引内容与原即时构建一致；fuzzy 长度预筛是 difflib `real_quick_ratio` 同判据的**安全超集**过滤，不改变命中集合。

---

## M0 — GraphRetriever 归一化索引懒缓存（最高 ROI，零行为变化）

**问题**：`_collect_seed_nodes`（retriever.py:170）每次 recall 调 `_build_normalized_node_index`（:190）重建 `{normalize(node): [...]}`，O(N) 无缓存。GraphRetriever 实例生命周期内 graph 不可变（构造后只读）。

**改法**：`__init__` 设 `self._norm_index: dict | None = None`；首次需要时构建并缓存，后续直接复用。
- `_collect_seed_nodes` 改读缓存的 `self._norm_index`（懒构建），`_build_normalized_node_index` 重构为 `_ensure_norm_index()` 返回缓存的实例字段。

**复杂度收益**：多实体/多次 recall 从每次 O(N) 重建 → O(N) 一次 + O(实体数) 查表。

**验收**：
- `test_graph_retriever.py` 13 用例全绿（含 normalize 命中、fuzzy、多实体）。
- 新增单测：同一 retriever 连续两次 recall，`_norm_index` 内部构建计数 == 1（验证缓存命中）。
- 构造后直接改 graph 不在支持范围（graph 视为不可变，与现状一致）。

**回滚**：还原 `_collect_seed_nodes`/`_build_normalized_node_index` 两方法即可，纯局部。

**files_affected**: `src/nanokb/qa/retriever.py`, `tests/unit/test_graph_retriever.py`

---

## M1 — fuzzy 匹配加速（M1.1 长度桶预筛铺路 + M1.2 rapidfuzz C 后端收割）

### M1.1 长度桶安全预筛（无新依赖，零结果变化）

**原理与收益边界（评审澄清）**：Python `difflib.get_close_matches` 内部已用 `real_quick_ratio() = 2*min(len)/(la+lb)` 做长度剪枝，其判据与本里程碑的 `r ≥ cutoff/(2-cutoff)`（cutoff=0.8 → ≥ 2/3）**完全相同**。因此 M1.1 **不减少** difflib 实际执行的 `ratio()` 计算量（被桶筛掉的候选，difflib 本来也会被 `real_quick_ratio` 跳过）；真实收益**仅**是避免对 N 个候选进入 difflib 的 Python 层循环（`set_seq1` + `real_quick_ratio` 调用），即把 O(N) 的 Python 迭代降为桶内 O(候选数)——常数级开销削减，非渐近降阶。**真正的 fuzzy 渐进加速来自 M1.2**；M1.1 是其前置（提供候选集供 rapidfuzz 跑）。

**改法**：懒构建 `_norm_buckets: dict[int, list[str]]`（按 len 分桶）+ `_norm_index`（与 M0 同生命周期缓存到实例字段）；fuzzy 时只扫长度区间内的桶。提取 `_fuzzy_candidates(norm, cutoff) -> list[str]` 封装桶筛选，便于单测。

**验收**：
- 全部 fuzzy 相关测试绿。
- **桶缓存验收（必做）**：新增单测断言 `_norm_buckets` 构建计数 == 1（连续 recall 不重建）。
- 等价性单测：构造含长短词的图，验证被剪掉的候选其 ratio 必然 < cutoff（数学等价性）。

### M1.2 可选 rapidfuzz 后端（C 加速，try-import 降级）

- 顶层 try `import rapidfuzz.process` / `rapidfuzz.fuzz`；成功则 fuzzy 走 `rapidfuzz.process.extract(norm, candidates, scorer=fuzz.ratio, score_cutoff=cutoff, limit=3)`（在 M1.1 预筛后的候选上跑）；失败回退 difflib。
- **caveat（评审放宽）**：`rapidfuzz.fuzz.ratio` 与 `difflib` ratio 均为 Indel 相似度 `2M/(la+lb)`，实践中**数值相等**；唯一差异是 n=3 边界多候选并列时，`heapq.nlargest` 与 rapidfuzz `extract(limit=3)` 的并列项 tie-breaking 顺序可能不同——属可接受的 fuzzy 兜底容差（下游 `_build_hits` 有去重）。断言只验证"typo 命中"语义，不锁定具体返回集合（现有测试本就不锁定）。
- `pyproject.toml`：rapidfuzz 列入 **optional dependency**（`[project.optional-dependencies] fuzzy = ["rapidfuzz"]`），非必装；README 注明可选加速。

**验收**：
- rapidfuzz 可用时与 difflib 在标准 typo 用例行为一致；不可用时优雅降级（monkeypatch import 报错 → 走 difflib）。
- 可选 perf 测试（`@pytest.mark.perf`，默认不收集）：N=10000 节点图，单次 fuzzy 召回较 difflib 基线加速。

**回滚**：M1.2 删 import 与分支即回 difflib；M1.1 还原 `_collect_seed_nodes`。

**files_affected**: `src/nanokb/qa/retriever.py`, `pyproject.toml`, `tests/unit/test_graph_retriever.py`, `README.md`（可选）

---

## M2 — CommunityRetriever 成员倒排索引（消除线性扫描）

**问题**：`CommunityRetriever.recall`（retriever.py:389）遍历全部社区，每社区每次重新 `{normalize_entity(m) for m in comm.members}`，O(社区数×成员数)；communities 实例不可变。

**改法**：`__init__` 预建 `self._member_index: dict[str, list[int]]`（`{normalize(member): [community_id,...]}`）一次，并缓存 `self._communities_by_id: dict[int, Community]`。
- recall：维护 `hits_per_community: dict[int, set[str]]`；对每个 `norm_entity` 查倒排得候选社区 id，把该 entity 加入对应社区的 set；最后对每个命中社区算 `score = len(overlap)/len(norm_entities)`，构造 hit。
- 分母统一为 `len(norm_entities)`（去重后），与原"逐社区全交集"**逐社区等价**。

**复杂度收益**：从 O(社区数×成员数) → O(实体数 × 平均命中社区数)；稀疏命中时近 O(实体数)。

**验收**：
- `test_fusion.py` 全部 CommunityRetriever 用例绿（match/partial overlap 0.5/no match/empty communities/empty NER）。
- **等价性测试（必做）**：对照参考实现（朴素线性扫描）随机化数据，断言每个社区 score 逐项相等。
- 空社区/空成员健壮（不报 KeyError）。

**回滚**：还原 `recall` 为线性扫描，删 `__init__` 中的 `_member_index`。

**files_affected**: `src/nanokb/qa/retriever.py`, `tests/unit/test_fusion.py`

---

## M5 — MultiRetriever 共享 NER（消除三路模式重复 LLM 往返，评审新增）

**问题**：`query` 三路模式下 `_ner_entities` 被 GraphRetriever（retriever.py:155）与 CommunityRetriever（:380）**各调一次**，同一问题产生 2 次 LLM NER 往返。LLM 往返成本远大于内存计算，是被原瓶颈清单忽略的实质浪费。

**改法（向后兼容）**：在 `MultiRetriever.recall` 层共享 NER 结果。
- 扩展 Retriever 协议为可选注入：retriever 可实现 `recall(self, question, *, entities=None)`；`entities` 非 None 时跳过内部 NER 直接用。未实现该重载的 retriever（如 VectorRetriever 不需 NER）保持原 `recall(question)` 不变。
- `MultiRetriever.recall`：在调用各 retriever 前，检测有无 ≥1 个需 NER 的 retriever（graph/community），若是则统一预调一次 `_ner_entities`，把结果传给两路；非三路模式（如纯 ask）不触发预调，零开销。
- 保持向后兼容：`GraphRetriever.recall(question)` / `CommunityRetriever.recall(question)` 单参仍可用（内部自调 NER），单测直接构造 retriever 调单参 recall 不受影响。

**复杂度收益**：三路模式 NER LLM 调用 2→1。

**验收**：
- `test_fusion.py` MultiRetriever / 三路融合用例全绿。
- 新增单测：FakeLLMClient 计数，三路模式下 `_ner_entities` 仅被调用 1 次（通过 LLM `complete` 调用次数断言）。
- 单参 recall 路径回归（graph/community 单测仍各自调 1 次 NER）。

**回滚**：还原 retriever 为单参 recall + MultiRetriever 不预调 NER。

**files_affected**: `src/nanokb/qa/retriever.py`, `tests/unit/test_fusion.py`

---

## M3 — 可选知识库会话对象（冷路径加载复用，library/server 友好）

**问题**：每次 CLI 调用新进程全量加载；进程内 `answer_query` 已只加载一次，但多次查询/未来 server 模式重复加载浪费。

**范围限定**：CLI 仍单次进程（不改 CLI 行为）；本里程碑提供**库级 API** 供编程复用与未来常驻。

**改法**：新增 `src/nanokb/session.py`，提供 `RetrievalSession`：
- `__init__(settings)`：懒加载 graph（`_load_graph`）/ communities（`load_communities`）/ VectorStore 并缓存为实例字段。
- `answer(question, mode=...)` / `search(keyword)`：复用缓存，调用现有 `pipeline.answer_query(..., graph=, vector_store=, communities=)` 注入，避免重复加载。
- 不改 `pipeline.answer_query` 签名（其已支持注入）；vector_store 加载复用 pipeline 现有逻辑（评审建议：优先抽一个 public 入口而非耦合私有 `_ensure_vector_store`，减少对私有名的依赖）。
- 可选配置项 `kb_session_cache: bool = False`（config.py 新增，默认关闭，预留开关）。

**验收**：
- 新增 `tests/unit/test_session.py`：同一 session 连续 2 次 answer，断言加载仅 1 次（spy/计数验证缓存）；结果与直接调 pipeline 一致。
- CLI 测试（`test_query.py`）不受影响。

**回滚**：删 `session.py` + 配置项；pipeline 与 CLI 零改动。

**files_affected**: `src/nanokb/session.py`(新), `src/nanokb/config.py`, `tests/unit/test_session.py`(新)

---

## M4 — 子图扩展扇出护栏（防御性，低优先）

**问题**：`_expand_subgraph`（retriever.py:202）BFS 受 hops 限制，但 hub 节点单跳可拉出巨量子图。

**改法**：新增配置 `max_subgraph_edges: int = 0`（0=不限，默认关闭保行为）；`_expand_subgraph` 累计边数达上限即提前停止，记 WARNING。默认 0 → 零行为变化。

**验收**：新增单测 `max_subgraph_edges=K` 时 hub 图召回边数 ≤ K 且有日志；默认（=0）下现有 N 跳测试全绿。

**files_affected**: `src/nanokb/qa/retriever.py`, `src/nanokb/config.py`, `tests/unit/test_graph_retriever.py`

---

## 实施顺序与依赖

```
M0（索引懒缓存）── 独立，先做，最高 ROI、最低风险
M1.1（长度桶预筛，铺路）── 依赖 M0 的懒缓存字段
M1.2（rapidfuzz 可选，收割）── 依赖 M1.1 的候选筛选函数
M2（社区倒排）────── 独立于 M0/M1，可并行
M5（NER 共享）────── 依赖 M0/M2 完成（改 MultiRetriever 编排）
M3（session 复用）── 独立，库级，低优先
M4（扇出护栏）────── 独立，低优先
```
建议落地顺序：M0 → M1.1 → M1.2 → M2 → M5 → M3 → M4。每个 M 独立可回滚、可测、独立交付价值。

## 全局验收（每个 M 合并前）
- `uv run pytest`（单测 + 集成）全绿；新增 perf marker 测试默认不收集。
- `uv run ruff check` + `uv run mypy` 无新增错误。
- 行为等价：对比优化前后，对固定问题集召回结果一致（M0/M1.1/M2/M5 数学保证；M1.2 仅 tie-breaking 差异不改命中语义）。

## 风险与对策
| 风险 | 对策 |
|---|---|
| 长度桶预筛误剪导致漏召回 | M1.1 是数学安全超集（与 difflib real_quick_ratio 同判据）；新增等价性单测对照朴素 difflib |
| rapidfuzz 与 difflib tie-breaking 差异 | M1.2 仅作加速后端，命中集合用 difflib 语义断言；不可用则降级 |
| 缓存假设 graph 不可变被破坏 | 文档明确"retriever 构造后 graph 视为只读"；与现状（单次 recall）一致 |
| 社区倒排 overlap 聚合与全交集不一致 | M2 新增对照参考实现的随机化等价性测试 |
| NER 共享改变单参 recall 行为 | M5 严格向后兼容：单参 recall 仍自调 NER，仅 MultiRetriever 编排层共享 |

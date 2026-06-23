# Nano KB

> A minimalist personal knowledge base, built on Karpathy's LLM-as-Wiki philosophy.

Nano KB 是一个基于 [Andrej Karpathy](https://github.com/karpathy) 提出的 **LLM Wiki** 理论构建的个人知识库工具。核心理念是:用 LLM 作为知识的索引、连接与推理引擎,而知识本身以纯文本形式沉淀,人类可读、可编辑、可版本管理。

它不采用传统 RAG 的"即时检索"范式,而是走 **"知识预编译"** 路线——把原始文档当作源码,通过一条流水线将其**编译**成结构化的知识图谱(节点为概念/实体,边为关系),再基于图谱实现可推理、可溯源的智能问答。

```mermaid
flowchart LR
    A["原始文档 raw/"] --> B["增量检测与加载"]
    B --> C["双轨知识抽取"]
    C --> D["图谱编译与融合"]
    D --> E["社区发现与索引"]
    E --> F["知识图谱 + 向量库"]
    F --> G["问答推理引擎"]
    G --> H["用户交互 & 主动学习"]
```

## 目录

- [Nano KB](#nano-kb)
  - [目录](#目录)
  - [特性](#特性)
  - [安装](#安装)
  - [快速开始](#快速开始)
  - [Usage](#usage)
    - [命令总览](#命令总览)
    - [`nanokb build` — 编译知识库](#nanokb-build--编译知识库)
    - [`nanokb query` — 图谱推理问答](#nanokb-query--图谱推理问答)
    - [`nanokb ask` — 向量语义问答](#nanokb-ask--向量语义问答)
    - [`nanokb search` — 社区宏观检索](#nanokb-search--社区宏观检索)
    - [`nanokb status` — 查看编译状态](#nanokb-status--查看编译状态)
    - [`nanokb review` — 主动学习待审队列](#nanokb-review--主动学习待审队列)
  - [典型工作流](#典型工作流)
  - [配置参考](#配置参考)
    - [LLM](#llm)
    - [Embedding](#embedding)
    - [目录与分块](#目录与分块)
    - [检索与问答](#检索与问答)
    - [图谱](#图谱)
  - [目录与产物](#目录与产物)
  - [置信度与溯源](#置信度与溯源)
  - [开发](#开发)
  - [License](#license)

## 特性

- **知识预编译**:把文档一次性编译为知识图谱,问答时不再扫描全文,速度快且答案可溯源。
- **双轨抽取**:代码文件走 tree-sitter 确定性解析(零 Token、零幻觉);文本/PDF/DOCX 走 LLM 语义抽取概念、实体与关系。
- **增量编译**:基于 SHA256 哈希的变更检测,仅处理新增/修改/删除的文件;支持 `--watch` 实时增量。
- **三路召回问答**:图谱检索(精确多跳)+ 向量检索(模糊语义)+ 社区检索(宏观主题域)三路融合。
- **社区发现**:Leiden 算法自动归纳知识主题域,提供宏观背景综述。
- **主动学习**:无法回答或存在冲突的问题自动进入待审队列,引导人工补充数据。
- **多后端 LLM**:支持 OpenAI 兼容端点(含智谱 GLM)、Anthropic、本地 Ollama。

## 安装

Nano KB 需要 **Python ≥ 3.10**。推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖(仓库已包含 `uv.lock`)。

```bash
# 克隆仓库
git clone <repo-url> nanokb && cd nanokb

# 方式一:uv(推荐,自动读取 uv.lock 锁定版本)
uv sync

# 方式二:pip(可编辑安装)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 验证安装
nanokb --help        # 或 python -m nanokb --help
```

## 快速开始

```bash
# 1. 从模板生成配置,填入 LLM API Key
cp .env.example .env
#   编辑 .env,至少设置 NANOKB_LLM_PROVIDER / NANOKB_LLM_MODEL
#   以及对应 provider 的 API Key(缺失会以 exit code 2 退出)

# 2. 把文档放进 raw/(支持 .md .txt .pdf .docx .py .js .java)
cp ~/notes/*.md raw/

# 3. 编译知识库(首次会全量抽取,调用 LLM)
nanokb build

# 4. 提问
nanokb query "Transformer 依赖哪些核心技术?"
```

## Usage

### 命令总览

```text
nanokb build    编译知识库(增量检测 → 双轨抽取 → 图谱融合 → 索引)
nanokb query    图谱推理问答(graph + vector + community 三路召回融合)
nanokb ask      向量语义问答(仅向量召回,适合模糊语义匹配)
nanokb search   社区宏观检索(按关键词返回所属社区摘要)
nanokb status   显示知识库编译状态(raw/ 文档数 + out/ 是否已编译)
nanokb review   列出 / 清空主动学习待审队列(out/review_queue.md)
```

所有命令均支持 `--help` 查看详细参数。运行 `nanokb`(不带子命令)会打印帮助。

> **冷启动**:在执行 `query` / `ask` / `search` 前,必须先 `nanokb build` 产出图谱与社区索引。若 `out/graph.json` 不存在,这些命令会报 `ColdStartError` 并以 exit code 1 退出,提示先 build。

---

### `nanokb build` — 编译知识库

执行完整的五阶段流水线:增量检测 → 双轨抽取 → 图谱编译融合 → 社区发现 → 索引写入。基于 `out/manifest.json` 中的 SHA256 哈希做增量,默认仅处理变更文件。

```text
nanokb build [OPTIONS]
```

**选项:**

| 选项 | 说明 |
|------|------|
| `--watch` | 启动监听模式:先做一次编译,随后监听 `raw/` 变更,debounce(默认 500ms)后自动增量编译,`Ctrl-C` 退出。 |
| `--force` | 强制全量重编译,忽略增量检测(忽略所有缓存哈希)。 |
| `--replay` | 不调用 LLM,直接从 `out/triples.jsonl` 历史日志重放重建图谱(按去重收敛规则)。适合换图谱参数后重建,或 LLM 不可用时复现。 |

**示例:**

```bash
nanokb build                  # 增量编译
nanokb build --force          # 全量重编译(清空增量缓存语义)
nanokb build --watch          # 实时监听 raw/ 并增量编译
nanokb build --replay         # 离线重放重建图谱(不耗 Token)
```

编译完成后会打印一行摘要,例如:

```text
编译完成:added=12, modified=0, deleted=0, extracted=87, fallback=2
```

其中 `fallback` 表示因抽取置信度过低、由 LLM 兜底综合出描述的概念数量。

---

### `nanokb query` — 图谱推理问答

最强问答模式,**三路召回融合**:以 LLM 识别的实体为中心做图谱多跳子图扩展(精确结构化三元组)+ 向量语义检索(模糊)+ 社区检索(宏观主题背景),融合重排后编译上下文交给 LLM 生成答案。

```text
nanokb query <QUESTION>
```

**示例:**

```bash
nanokb query "Transformer 依赖哪些核心技术?"
nanokb query "Leiden 算法和 Louvain 有什么区别?"
```

答案末尾会附带**引用来源**(对应 `raw/` 中的源文件)。若回答用到了 `INFERRED`/`AMBIGUOUS` 关系,会额外提示"此结论为 AI 推理,建议核实源文件"。

---

### `nanokb ask` — 向量语义问答

**仅向量召回**的单路问答:把问题向量化,检索语义最相似的节点描述/文本块。适合不需要多跳关系推理、只需模糊语义匹配的场景(如"找讲注意力机制的那段内容")。

```text
nanokb ask <QUESTION>
```

**示例:**

```bash
nanokb ask "哪里讲了 self-attention 的计算流程?"
```

---

### `nanokb search` — 社区宏观检索

**社区路宏观检索**:按关键词定位实体所属的 Leiden 社区,返回该主题域的背景摘要。适合"我要这整块主题的概览"这类宏观需求。

```text
nanokb search <KEYWORD> [--community]
```

**选项:**

| 选项 | 说明 |
|------|------|
| `--community` | 显式声明走社区路(该命令固定走社区路,flag 保留作语义提示)。 |

**示例:**

```bash
nanokb search "深度学习"
```

```text
找到 2 个相关社区:
- 深度学习基础模型与训练方法 (来源:notes/dl.md)
- 自然语言处理与注意力机制 (来源:notes/nlp.md)
```

---

### `nanokb status` — 查看编译状态

显示 `raw/` 下受支持文档的数量,以及 `out/` 是否已编译(`graph.json` 是否存在)。

```bash
nanokb status
# raw/ 下 42 个文档 | out/ 已编译
```

---

### `nanokb review` — 主动学习待审队列

查看或清空主动学习待审队列 `out/review_queue.md`。当一次问答命中以下任一条件时,会被自动追加到队列,引导人工补充数据或修正:

- 命中数过少(`< NANOKB_MIN_HIT_COUNT`,默认 3)
- 最高置信度过低(`< NANOKB_MIN_CONFIDENCE_SCORE`,默认 0.3)
- 命中 `AMBIGUOUS`(信息冲突)关系

```text
nanokb review [--clear]
```

**示例:**

```bash
nanokb review              # 列出所有待审条目
nanokb review --clear      # 清空待审队列
```

输出示例:

```text
待审条目(2 条):
1. 量子计算的主要挑战是什么?
   原因:low_hit_count | 实体:量子计算 | 时间:2026-06-23T...
```

## 典型工作流

```bash
# 首次初始化
cp .env.example .env        # 配置 LLM
nanokb status               # 确认 raw/ 有文档

# 日常:增量编译 + 提问
nanokb build
nanokb query "..."
nanokb ask   "..."

# 长期监听:编辑 raw/ 自动重建
nanokb build --watch

# 离线/调参:不耗 Token 重建图谱
nanokb build --replay

# 复核低质量问答,反哺知识库
nanokb review               # 检查待审队列 → 补充 raw/ 文档 → rebuild
```

## 配置参考

所有配置通过环境变量(前缀 `NANOKB_`)或项目根的 `.env` 文件覆盖,由 pydantic-settings 自动加载。完整模板见 [`.env.example`](.env.example)。

### LLM

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NANOKB_LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `ollama` |
| `NANOKB_LLM_MODEL` | `glm-5.1` | 模型名 |
| `NANOKB_OPENAI_API_KEY` | — | OpenAI / 兼容端点的 API Key(缺失则 exit code 2) |
| `NANOKB_OPENAI_BASE_URL` | — | OpenAI 兼容端点。如智谱 GLM:`https://open.bigmodel.cn/api/paas/v4` |
| `NANOKB_ANTHROPIC_API_KEY` | — | Anthropic API Key |
| `NANOKB_OLLAMA_BASE_URL` | `http://localhost:11434` | 本地 Ollama 地址 |

> **智谱 GLM 示例**:用 GLM 做生成时,embedding 需同步换成智谱模型(如 `embedding-3`),不能继续用 OpenAI 的 `text-embedding-3-small`。

### Embedding

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NANOKB_EMBEDDING_PROVIDER` | `openai` | `openai` \| `ollama` |
| `NANOKB_EMBEDDING_MODEL` | `text-embedding-3-small` | 向量模型 |

### 目录与分块

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NANOKB_RAW_DIR` | `raw` | 原始文档目录 |
| `NANOKB_OUT_DIR` | `out` | 编译产物目录 |
| `NANOKB_CHUNK_MAX_TOKENS` | `3000` | 单块最大 Token |
| `NANOKB_CHUNK_OVERLAP_TOKENS` | `200` | 块间重叠 Token |

### 检索与问答

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NANOKB_RETRIEVAL_HOPS` | `2` | 图谱检索的子图跳数 |
| `NANOKB_MAX_CONTEXT_TOKENS` | `4000` | 编译给 LLM 的上下文上限 |
| `NANOKB_FUZZY_MATCH_CUTOFF` | `0.8` | 实体模糊匹配阈值 |
| `NANOKB_MIN_HIT_COUNT` | `3` | 命中数低于此值则入 review 队列 |
| `NANOKB_MIN_CONFIDENCE_SCORE` | `0.3` | 置信度低于此值则入 review 队列 |

### 图谱

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NANOKB_GRAPH_SERIALIZATION` | `json` | `json` \| `graphml` |
| `NANOKB_EXTRACTOR_VERSION` | `1` | 抽取 schema 版本(用于 `--replay` 兼容性校验) |

## 目录与产物

```
raw/                         原始文档(知识源,建议提交到版本控制)
  *.md *.txt *.pdf *.docx      → unstructured 文本抽取
  *.py *.js *.java              → tree-sitter 确定性抽取
out/                         编译产物(默认 gitignore,可由 --replay 重建)
  graph.json                 知识图谱(NetworkX MultiDiGraph 序列化)
  graph.graphml              GraphML 格式(当 GRAPH_SERIALIZATION=graphml)
  triples.jsonl              三元组追加日志(--replay 的数据源)
  communities.json           Leiden 社区发现结果 + 主题摘要
  keywords.json              关键词索引
  manifest.json              增量检测清单(文件 → SHA256,近似事务提交点)
  review_queue.md            主动学习待审队列
  chroma/                    ChromaDB 向量库
```

## 置信度与溯源

每条抽取出的关系都携带 `confidence` 标签,直接影响问答的可信度提示:

| 标签 | 含义 | 处理 |
|------|------|------|
| `EXTRACTED` | 直接来源于原文(事实) | 最高权重 |
| `INFERRED` | LLM 逻辑推导,中等置信 | 答案附"AI 推理,建议核实"提示 |
| `AMBIGUOUS` | 信息冲突,需人工审核 | 自动进入 review 队列反哺 |

答案末尾的**引用来源**对应 `raw/` 中的源文件,可一键溯源核对。

## 开发

```bash
uv sync                     # 安装依赖(含 dev 组)

# 质量检查
ruff check .                # lint
mypy src                    # 类型检查(strict 模式)

# 测试(默认跳过真实 LLM 调用)
pytest                      # 等价于 pytest -m "not llm"
pytest -m llm               # 仅跑真实 LLM 集成测试
```

技术方案详见 [`docs/design/knowledge-graph-extraction-qa-system.md`](docs/design/knowledge-graph-extraction-qa-system.md)。

## License

MIT

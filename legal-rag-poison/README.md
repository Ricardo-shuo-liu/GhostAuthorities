# legal-rag-poison（v2 代码：API 优先 + 本地留存）

对应研究方案 v3 与《实验执行手册》的 E0/E1：中文法律 RAG 的版本感知缺失、
零伪造投毒与防御度量。本目录是配套代码，本 README 说明**要准备什么数据**、
**怎么用 API 跑**、**结果存在哪**。

---

## 1. 你需要准备的数据（这是单点故障，先做这个）

整个项目只需要三类数据，其中只有第一类需要你收集：

| 层 | 内容 | 从哪来 | 粒度 |
|---|---|---|---|
| ① 条文版本 | 每个候选条文的**每一版正文**（逐字）+ 生效起止日期 | flk.npc.gov.cn（首选，官方）；修正案全文用 npc.gov.cn / gov.cn | 一条一条地攒 |
| ② 溯源信息 | 来源 URL、抓取日期、修正案名称 | 随手记 | 每版本一行 |
| ③ 题目与 gold | **不需要收集**，由 `gold.py` 按「从旧兼从轻」程序生成 | — | 程序生成 |

### 1.1 文件怎么放

```
data/
├── raw/刑法/
│   ├── 176_1997-10-01.txt     # 旧版：1997.10.1 起生效的条文正文
│   ├── 176_2021-03-01.txt     # 新版：修正案十一后的条文正文
│   ├── 272_1997-10-01.txt
│   └── ...
├── corpus.jsonl               # 由 raw 自动生成
└── questions.jsonl            # 由 corpus 自动生成（含 gold 与攻击目标）
```

文件名规则：`{条号}_{生效起始日YYYY-MM-DD}.txt`，正文 UTF-8、只放条文本身、
不要放进你自己的解释。同一 `article_id` 的多个版本会自动首尾相接。

### 1.2 抓哪些、怎么核对（关键）

1. **第一批只做 1 条 → 2 个版本 → 跑通全流程**。推荐第 176 条（非法吸收
   公众存款罪，修正案十一加了第三档刑，改法干净）。
2. **旧版 ≠ 1997 原文**，除非该条从未被修正案一至十改过。历史上被改过的
   条文（如 191 洗钱、192 集资诈骗、163/164 等）必须把中间版本也建出来，
   否则 gold 和载荷都是错的 —— W1 的版本表要逐条核对全部修正案。
3. 核对顺序建议：flk.npc.gov.cn 现行合订文本（新版）→ 修正案原文里的
   「将第 X 条修改为……」→ 旧版从官方旧合订本/权威来源（法宝、法信）
   复核。**不要用二手转载站点的正文**。
4. 每条的「档位表述」（进题目用，如"涉案数额特别巨大"）单独存一个
   `facts.json`：`{"176": "涉案数额特别巨大"}`，构建题目时传入。
5. 现在就把来源 URL 记进 `sources.csv`（论文要附数据来源对照表）。

### 1.3 明确不需要的

* 不用爬裁判文书（题目由规则生成，不依赖真实案件）；
* 不用标注答案（gold 由 `gold.py` 算）；
* A4 攻击需要少量**真实存在的**释义/立法说明文本（不能用 LLM 编造），
  格式见 §4 的 `--commentary-file`；公开仓库时注意版权。

### 1.4 语料规模：出题范围 ≠ 检索语料库（关键区分）

两个概念完全不同，别混：

* **检索语料库**（受害 RAG 的知识库，被 top-k 检索的对象）：**要大**。
  目标是几百到三千个文档块：全量现行刑法 452 条（≈452 块）＋ 焦点条文
  的全部历史版本（60–120 块）＋ 可选的其他法律干扰（民法典 1260 条、
  治安管理处罚法等）。攻击载荷是在这个"大库"里和所有正常文档竞争 top-k。
* **出题集**（有 gold、能被判分的题目）：只从"刑档变化过、从旧兼从轻
  算得出答案"的 30–40 条里出 300+ 道题。没变化的条文出题没法判分——
  但它们的正文照样放进检索语料库当干扰项。

| 层 | 内容 | 规模 | 定位 |
|---|---|---|---|
| ① 焦点条文多版本 | 刑档变化条文的每一版 | 30–40 条 × 2–4 版 ≈ 60–120 版本 | **必做**：题目与 gold 的载体 |
| ② 全量现行刑法 | 现行合订本按条切块 | 452 条 × 1 版 | **必做**：让检索成为真任务 |
| ③ 其他法律干扰 | 民法典/行政处罚法等全文 | 1000+ 条 | 建议：语料规模变量，测攻击抗干扰性 |
| ④ 非焦点条文的历史版本 | 每条的历史版本全建 | 2000+ 版本 | **不做**（工作量以月计，无研究回报） |

用法：① 生成题目（`--corpus`），②③ 只进索引做干扰、不出题：

```bash
python run_e0.py --corpus data/corpus.jsonl --questions data/questions.jsonl \
    --background data/background.jsonl \
    --attack A3 --n-inject 5 --outdir results/e1_a3_bg
```

直觉上要记住：**语料越大，检索竞争越激烈，攻击越难**——这正是 TVR/ASR
要量化的东西（也是 E7 注入预算实验存在的理由）。检索是本地向量计算，
几千个文档块对 CPU 也是秒级，不影响实验速度。

跨法域：主实验只用**刑法**（从旧兼从轻 = 可程序化 gold）；
泛化实验（E8 精简版）用**行政处罚法**（第 37 条有同款规则）；
**不建议公司法**——民事时间效力是另一套规则（法释〔2024〕7 号）。

节奏：1–2 条跑通 → 全量现行刑法背景导入 → 30–40 条焦点条文 + 300 题。

---

## 2. 快速开始

### 2.1 离线自检（不联网、不花钱，先跑这个）

```bash
cd legal-rag-poison
python make_toy_data.py            # 生成虚构《示例法》玩具语料

# 基线
python run_e0.py --corpus data/toy/corpus.jsonl \
    --questions data/toy/questions.jsonl \
    --embed-backend mock --generator mock \
    --attack none --outdir results/smoke_base

# A3 副本放大（注入 5 份旧版副本）
python run_e0.py --corpus data/toy/corpus.jsonl \
    --questions data/toy/questions.jsonl \
    --embed-backend mock --generator mock \
    --attack A3 --n-inject 5 --outdir results/smoke_a3
```

mock 语境下总该看到 `smoke_a3` 的 ASR / 命中率明显高于 `smoke_base`，
说明流水线通了。跑完看 `results/smoke_a3/summary.json`。

### 2.2 真实数据 + 真实 API

```bash
# 0) 用官方文本填好 data/raw/刑法/*.txt 后：
python gold.py --raw data/raw/刑法 --corpus data/corpus.jsonl \
    --questions data/questions.jsonl --facts-json facts.json

# 1) API 配置（示例：DeepSeek；任何 OpenAI 兼容接口都行）
export LLM_API_KEY=sk-xxx

# 2) 无攻击基线（E0）
python run_e0.py --corpus data/corpus.jsonl --questions data/questions.jsonl \
    --embed-backend local --embed-model BAAI/bge-small-zh-v1.5 \
    --generator api --llm-model deepseek-chat \
    --llm-base-url https://api.deepseek.com \
    --attack none --outdir results/e0_base

# 3) A3 攻击（E1），并开 D5 版本感知过滤做对照
python run_e0.py --corpus data/corpus.jsonl --questions data/questions.jsonl \
    --embed-backend local \
    --generator api --llm-model deepseek-chat \
    --llm-base-url https://api.deepseek.com \
    --attack A3 --n-inject 5 --outdir results/e1_a3

python run_e0.py ... --attack A3 --n-inject 5 --as-of fact \
    --unknown-dates reject --outdir results/e1_a3_defended
```

长实验中断了直接加 `--resume` 重跑同一命令即可续跑。

### 2.3 实验矩阵：E0–E8 怎么落地

每个实验 = 「同一批题 × 一组条件」，条件写在 JSON 配置里，批量跑 + 汇总：

```bash
# 一键跑 E5 核心 2x2（先改 experiments/e5_core.json 里的数据和模型）
python run_matrix.py --config experiments/e5_core.json --report

# 只想看已跑结果 / 单独算 DEQ
python report.py --root results/e5_core \
    --deq A_defect_unfixed_clean,B_defect_unfixed_attack,\
C_defect_fixed_clean,D_defect_fixed_attack
```

* `run_matrix.py`：逐条件调用 run_e0.py，每格一个 outdir，自动 `--resume`
  （中断重跑不重复花钱）；
* `report.py`：把所有条件汇总成一张对照表（ASR / 命中率 / 准确率），
  并按方案 §6 的公式算 E5 的 DEQ = (B−A) − (D−C)；
* `experiments/smoke_matrix.json` 是离线自检用的示例（mock 后端）；
  `experiments/e5_core.json` 是真实运行的模板。

已实现：E0、E1、E5、E6、E7 的运行与汇总。
还缺（按优先级）：E4 的 D2 去重 / D4 密度（保底发现所在）、
E3 引用校验循环、E2 检测器、E8 泛化数据。

---

## 3. API 配置（OpenAI 兼容，任何一家都行）

| 提供方 | `--llm-base-url` | `--llm-model` 示例 | 向量（embedding）示例 |
|---|---|---|---|
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` | 用本地 bge 即可 |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` | `text-embedding-v3`（`--embed-batch 10`） |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `deepseek-ai/DeepSeek-V3` | `BAAI/bge-m3` |
| 智谱 | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` | — |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` | — |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | `text-embedding-3-small` |

* key 一律走环境变量（默认 `LLM_API_KEY` / `EMBED_API_KEY`，可用
  `--llm-api-key-env` / `--embed-api-key-env` 改名），不给命令行留明文；
* 走代理时设置 `HTTPS_PROXY` 即可（标准库自动读取）；
* 跨模型共识（D3）就是换 `--llm-model` 再跑一个 outdir，最后对比；
* 请求自带重试（429/5xx 指数退避）与 JSONL 缓存，重跑不重复计费。

---

## 4. 结果与缓存（全部在本地）

每次运行一个 `--outdir`：

```
results/e1_a3/
├── config.json       # 本次运行的全部参数（复现实验用）
├── answers.jsonl     # 逐题追加：预测、gold、检索到的文档（含 injected 标记）、原始回答
├── summary.json      # 汇总指标：accuracy / ASR / 命中率（按 to_old / to_new 拆分）
└── cache/
    ├── llm.jsonl         # prompt 级缓存（key=模型+prompt+采样参数），重跑不花钱
    └── embeddings.jsonl  # 向量缓存
```

指标口径：`target_any_in_topk`（错误版本出现）、`target_injected_in_topk`
（**注入副本**出现，与真本区分）、`gold_in_topk`、`accuracy`、`asr`
（按攻击目标判定）。基线（`--attack none`）与攻击运行的差值就是边际攻击
增益，方案 §6 的 B0/B1/B2 用同一套命令换 `--attack` 即可。

---

## 5. 相对旧版代码修了什么

1. **注入重复的 bug**：旧版在每题循环里注入，A1 实际被重复注入 3 次、
   A3 是 15 份而不是 5 份 —— 注入预算（E7）口径全错。现在按唯一目标聚合。
2. **A1 不再写假日期**：改为真正的无日期元数据；D5 对无日期文档的处置
   （`--unknown-dates reject|accept`）成为显式的实验变量（E6 靠它）。
3. **TVR 能区分真本与注入副本**：新增 `injected` 标记与
   `target_injected_in_topk` 指标（二者 version_id 相同，旧版无法区分）。
4. **生成与向量全面 API 化**：只用标准库 urllib，带超时/重试/退避/缓存；
   离线 mock 后端可在无网环境跑通全流程。
5. **断点续跑 + 全量本地留存**：逐题 append `answers.jsonl`，`--resume`
   续跑；`config.json` 记录全部参数。
6. **攻击目标由规则指定**：`gold.py` 为每题写入
   `attack_target_version_id` 与方向（to_old / to_new），不再取
   `alts[0]` 这种任意版本。

## 6. 仍需你补齐的部分（论文级）

* `B2` 的默认载荷只是占位模板 —— 正式实验必须换成 PoisonedRAG 官方实现
  生成的载荷（`b2_forged(..., text=...)` 支持直接传入），否则"伪造 vs
  零伪造"的对照不成立；
* `A4` 需要你收集真实释义文本（`--commentary-file`）；
* gold 只比较主刑上限（《刑法》第 45 条把"X 年以上有期徒刑"记为上限 15），
  必须做 W2 人工核验（建议法学同学盲核 + 报 kappa）；
* 每条的档位表述（`facts.json`）必须人工给，通用模板会算错档位；
* 正式实验建议同时跑 `--context-metadata none|version_id` 两个条件：
  上下文里是否暴露版本日期，会显著影响基线与攻击效果，这本身是发现。

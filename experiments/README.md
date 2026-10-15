# experiments/：每个实验一个文件夹

对应 [实验设计书_假设与预期效果.md](../实验设计书_假设与预期效果.md) 的
E1–E7。每个文件夹里是**该实验的回答问题、设计、判据与验证脚本**；
脚本默认技术栈：**大模型走 API（OpenAI 兼容）+ RAG 用 ChromaDB**；
离线自检加 `--mock`（mock 生成器 + mock 向量 + memory 索引）。

| 文件夹 | 实验 | 验证假设 | 成功判据（设计书） |
|---|---|---|---|
| `e1_half_poisoning/` | E1 半投毒（两题库：过期数据/实效数据） | H1 | new 题库在含旧版语料上下降 ≥20pp，按条文聚类 p<0.05 |
| `e2_zero_forgery/` | E2 零伪造攻击 | H2 | k=5 时 G2 掉 ≥30pp，G1/G3 变化 <10pp |
| `e3_defenses/` | E3 防线 | H3 | 引用校验：攻击不降 + 通过率≈100%；D2：挡攻击但有代价（代价见 E6） |
| `e4_deq/` | E4 因果（DEQ） | H5 | DEQ>0 且显著 |
| `e5_residual/` | E5 残留攻击面 | H4 | D5 下 A2 使 G2 掉 ≥20pp |
| `e6_defense_cost/` | E6 防御代价 | H3 补充 | D5 Clean-Δ≈0；D2 明显为负 |
| `e7_generalization/` | E7 泛化 | — | 第二法域方向一致（需先准备数据，见脚本头部） |

> 设计书 §0 提到"E8"，但正文未展开；暂以 E1–E7 为准。

## 运行方式（都在仓库根目录执行）

```bash
pip install chromadb sentence-transformers      # 一次性
python scripts/check_chroma.py                  # 验证 ChromaDB 链路就绪
export LLM_API_KEY=sk-xxx                       # 大模型 API key

# 真实实验（API + ChromaDB）
python experiments/e1_half_poisoning/run.py
python experiments/e2_zero_forgery/run.py
python experiments/e4_deq/run.py
python experiments/e5_residual/run.py           # A4 需 --commentary-file
python experiments/e6_defense_cost/run.py
python experiments/e3_defenses/run.py

# 离线自检（不联网、不花钱；验证流程是否跑通）
python experiments/e1_half_poisoning/run.py --mock
```

E1 的题库按《刑法》第 12 条划分为两个库（需要过期数据 / 需要实效数据），
先运行生成器再跑实验（`run.py` 会自动跑「语料 A/B × 两题库」共 4 个条件）：

```bash
python experiments/e1_half_poisoning/generate_questions.py   # 生成题库
python experiments/e1_half_poisoning/run.py                  # API + ChromaDB
```

## 公共参数

所有脚本共享（`--help` 可见）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--generator` | `api` | 大模型后端（OpenAI 兼容）；`mock` 为离线自检 |
| `--llm-model` / `--llm-base-url` | deepseek-chat / api.deepseek.com | 任意 OpenAI 兼容服务 |
| `--index-backend` | `chroma` | RAG 向量库；`memory` 为无依赖回退 |
| `--embed-backend` | `local` | bge-small-zh（CPU）；也可 `api` / `mock` |
| `--k` | 5 | 检索 top-k |
| `--outroot` | `results/<实验名>` | 输出目录（逐题结果、报告、Chroma 持久化） |
| `--mock` | 关 | 一键切到离线自检（mock+mock+memory） |

**配置方式**：支持仓库根目录的 `.env`（复制 `.env.example` 后填写）。
`LLM_API_KEY` 必填；可选 `LLM_MODEL`、`LLM_BASE_URL`、`HF_ENDPOINT`
（本地 bge 下载镜像）。embedding 默认本地，不需要 key；`.env` 不会覆盖
已 export 的环境变量，也不会被提交（见 `.gitignore`）。

每个实验跑完会打印判据结论，并把汇总写到 `<outroot>/report.json`；
逐题数据在 `<outroot>/<条件>/answers.jsonl`（含检索结果、注入标记、
引用校验、耗时等字段），可复算。

## 未实现 / 待补

* E3 的 **D3 共识投票、D4 异常检测**：脚本请求时会明确报错（不假装跑过）；
* E7 需要第二部法律的合订本与题目（脚本头部给了准备步骤）。

## 排错（API / 运行相关）

* **答案为空 / 统计全是 0**：先跑 `python scripts/check_llm.py` 看原始返回。
  若是"思考型"模型（`reasoning_content` 很长、`content` 为空），把
  `--llm-max-tokens` 加大到 4096+（实验脚本已透传该参数），或换普通对话模型；
  生成器遇到连续空回答会直接报错，不会再静默记成"答错"。
* **运行中终端长时间没有输出**：索引阶段会打印 `[embed] x/y` 进度，问答阶段
  每 5 题打印一次；中断（Ctrl-C）后重跑会自动续跑，缓存不重复计费。

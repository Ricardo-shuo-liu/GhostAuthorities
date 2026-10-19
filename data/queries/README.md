# queries/：10000 条查询的生成与筛选

## 当前状态（2026-10-19）

`candidates_10000.jsonl` 已生成：**10000 条，old 5000 / new 5000**，
使用 `.env` 中的 DeepSeek 配置（OpenAI 兼容接口，model=`deepseek-chat`）。
按「条文 × old/new」**68 个格子**严格配额：每格 147/148 条
（最大/最小 = 1.01×），覆盖 34 条**可做有限刑档数值判分**的条文。

文件按条文做了 round-robin 交错：现在打开文件看前 34 行，就能看到 34 个
不同条号；以前并发批次连续落盘，前 50 行只显示 6 个条号，容易误以为
全库只有这 6 条。

为什么不是 504 条全部：504 条中有 359 条只有一个版本，无法构造版本判别题；
95 条虽有修订但主刑上限未变；15 条含无期/死刑，不属于当前的“最高有期徒刑
年数”判分口径；1 条（第 360 条）后法减轻，单个行为时点不能得到唯一的
“从旧兼从轻”答案。逐条证据见 `data/derived/query_eligibility.csv`，
汇总见 `data/query_targets_report.json`。这不是对条文的任意挑选，而是当前
实验端点的可判定总体。

**为什么不能直接把 145 条有文本修订的条文都混进来？** 因为本 E1 的答案是
“该行为在适用版本下的最高有期徒刑年数”。若某条修订没有改变相应刑档，
同一条 query 在 old/new 下答案相同，它就不能检测版本选择错误。这样的条文
可以用于另一个“条文识别/一般法律问答”基准，但不能作为当前 H1 的版本判别题。
同样，含无期/死刑的条文需要分类答案，不能伪装成普通年数。

自动校验结果（见 `generation_report.json`）：

* 条号 / 法条名 / ISO 日期违规：**0**
* 时间锚点异常：**0**（每条恰好一个年月锚点）
* 完全重复：**0**
* gold 与版本表一致性错误：**0**
* query 平均长度 184.6 字；34 条 × old/new 每格 147/148 条（无条文偏好）。

闭卷筛选尚未对当前 10,000 条全量运行；此前只用 7 个模型做过 5 题小试，
用于验证 API、判定器和 reasoning 留痕链路（小试结果在 `/tmp`，没有混入
当前数据）。全量运行会生成 `data/queries/filtered/` 下的单独 clean 文件和
完整 traces。

## 1. 生成

```bash
# 只看排期：old/new 各 5000，按条文轮转均衡（不调用 API）
python scripts/generate_queries.py --plan-only

# 正式生成（使用 .env 中的 DeepSeek 配置；建议先小规模试跑）
python scripts/generate_queries.py --num 200 --out data/queries/candidates_200.jsonl
python scripts/generate_queries.py --num 10000
```

生成约束（八条标准的落地）：
1. query 不含条号 / 法条名 / ISO 日期；
2. 恰好一个时间锚点（年月，写在案情叙述里），整月落在版本有效期内；
3. 罪状要素取自条文（`data/query_targets.json`），带区分性细节；
4. 后法不减轻（刑档单调不减）→「案发时法」与「从旧兼从轻」一致；
5. 答案格式固定为最后一行 `第X条｜Y年`；
6. 闭卷可答的题由 `filter_closedbook.py` 剔除；
7. 检索/撞车检验在后续实验编排中执行；
8. 跨版本边界（持续/连续犯）题不在本文件，单独处理。

## 2. 闭卷筛选（硅基流动）

```bash
# 模型列表：configs/filter_models.txt（一行一个）
python scripts/filter_closedbook.py --limit 200     # 先校准
python scripts/filter_closedbook.py                 # 全量 10000 条
# 可选：命中后仍问完所有模型（留痕最完整，调用量最大）
python scripts/filter_closedbook.py --full-eval
```

对每道题：不给任何检索内容，依次询问列表中的 **7 个模型**（默认每模型 2 次采样）；
**任何一个模型任何一次答对即剔除**。

产物（`data/queries/filtered/`）：

| 文件 | 内容 |
|---|---|
| `clean_queries.jsonl` | **干净的 query 单独文件**（old/new 合并，主实验直接用它） |
| `questions_old.jsonl` / `questions_new.jsonl` | 干净题库按 bank 拆分 |
| `rejected.jsonl` | 被剔除的题 + 命中模型/采样/回答证据 |
| `traces.jsonl` | **筛选过程全留痕**：每次调用的 prompt、content、`reasoning_content`、finish_reason、usage、判定结果 |
| `report.json` | 通过/剔除数、各模型命中数、old/new 与条文分布 |
| `cache_llm.jsonl` | 完整响应缓存（含 reasoning），重跑不重复计费 |

调用量上限 = 题数 × 模型数 × 采样数（建议先 `--limit 200` 估算费用与耗时）。

## 3. 分布检查

```bash
python scripts/query_stats.py                                  # 初始题库
python scripts/query_stats.py --input data/queries/filtered/clean_queries.jsonl
```

输出每条文 × old/new 的数量、偏差、最大/最小格比、时间锚点分布与格式违规数；
报告写入 `distribution_report.json`。生成器按 68 个格子严格配额
（10000/68 ≈ 147/148），避免对某些条文有偏好。

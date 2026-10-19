# scripts/：数据清洗 + query 生成/筛选

当前仓库只保留这一组脚本；检索/攻击实验的编排由你重新搭建。
所有命令都在**仓库根目录**执行。

## 数据清洗（人工核对）

| 脚本 | 作用 | 典型用法 |
|---|---|---|
| `build_from_consolidated.py` | 15 份历年合订本 → 版本表 / raw / derived / 背景层 | `python scripts/build_from_consolidated.py` |
| `build_query_targets.py` | 从 504 条中自动筛出可做数值判分的条文，生成 34 条目标配置和全量资格审计 | `python scripts/build_query_targets.py --dry-run` |
| `build_corpora.py` | 生成 34 条焦点版本语料、503 条现行语料、504 条完整版本语料 | `python scripts/build_corpora.py` |
| `show_article.py` | 查看某条文的版本沿革与变化点（含文件路径） | `python scripts/show_article.py --article 176 --diff` |
| `diff_snapshots.py` | 任意两份合订本的逐条差异（旧文本 vs 新文本） | `python scripts/diff_snapshots.py --a 2020年 --b 2023年` |

产出（`data/`）：

* `raw/刑法/`：最新文件夹（2024-03-01）＝现行全量；其余文件夹＝被替换的历史旧版；
* `derived/article_versions.jsonl`：清洗后的版本表（含生效区间与来源）；
* `derived/changes.csv`、`derived/diffs/*.csv`：逐快照 / 逐法律的差异明细；
* `derived/anomalies.csv`：源数据瑕疵记录（建议对照官方文本复核）；
* `derived/targets_candidates.csv`：刑档变化过的条文（选题池）；
* `derived/query_eligibility.csv`：504 条逐条资格审计（合格/排除原因）；
* `corpus.jsonl`：34 条焦点条文的 80 个版本；
* `corpus_current_only.jsonl`：503 条现行条文；
* `corpus_all_versions.jsonl`：504 条、684 个版本；
* `background.jsonl`：最新合订本全量条文（背景语料）。

## Query 工具

| 脚本 | 作用 | 典型用法 |
|---|---|---|
| `generate_queries.py` | 生成 10000 条自然语言案情 query（old/new 各 5000，内置八条标准校验） | `python scripts/generate_queries.py --plan-only` 先看排期；再去掉 `--plan-only` 正式生成 |
| `filter_closedbook.py` | 闭卷筛选：7 个模型（硅基流动）在**无检索**条件下提问，任何模型答对即剔除；全程留痕（含 reasoning） | `python scripts/filter_closedbook.py --limit 200` 先校准 |
| `query_stats.py` | 统计 query 的条文/库/时间分布，检查是否存在条文偏好 | `python scripts/query_stats.py` |

流程：生成排期（34 条 × old/new = 68 格严格配额）→ 正式生成（API）→ 闭卷筛选（7 模型，
任何答对即剔除）→ 得到干净题库 `filtered/clean_queries.jsonl`；
每次模型调用留痕于 `filtered/traces.jsonl`（含 reasoning_content）。
模型列表在 `configs/filter_models.txt`；详细说明见 `data/queries/README.md`。

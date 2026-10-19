数据来源：https://zh.wikisource.org/wiki/Wikisource:%E9%A6%96%E9%A1%B5

> 当前仓库只有《刑法》的 15 份合订本及其修正案版本链；没有《公司法》
> 《行政处罚法》等其他法律的原文版本数据。论文的跨法域泛化实验必须
> 另行加入这些法律的版本化语料，不能用刑法条文替代或伪造。

本目录内容：

* `中华人民共和国刑法 (1997年).txt` … `(2023年).txt`：15 份原始合订本
  （每一份 = 某次修正/决定之后的完整现行文本）。
* 以下为 `python scripts/build_from_consolidated.py` 生成：
  * `raw/刑法/`：最新文件夹 = 现行全量（每条一个 `{条号}.txt`，对照基线）；
    其余文件夹 = 历史旧版（该日生效、后被修正案替换的版本）；
    说明见 `raw/刑法/README.md`；
  * `derived/`：版本表 `article_versions.jsonl`（程序入口）、逐快照差异
    `changes.csv`、源数据瑕疵 `anomalies.csv`、选题池 `targets_candidates.csv`、
    **每次修正的逐条改动明细 `diffs/*.csv`（旧文本 vs 新文本）**；
  * `background.jsonl`：最新合订本全量条文（检索背景层）。
* `corpus.jsonl`：焦点测验条文（34 条）的全部版本（80 个版本），供 query
  生成/检索使用；`corpus_current_only.jsonl` 是全部 503 条现行条文，
  `corpus_all_versions.jsonl` 是 504 条、684 个版本的完整版本表。
  生成命令：`python scripts/build_corpora.py`。
* `derived/query_eligibility.csv`：对全部 504 条逐条给出实验资格与排除理由；
  当前可做“有限刑档数值题”的总体是 **34 条**（见 `query_targets_report.json`）。
* `query_targets.json`：34 条合格条文的罪状要素、档位与区分性提示。
* `queries/`：query 生成与筛选的产物：
  * `candidates_10000_plan.jsonl`：10000 条排期（old/new 各 5000，34 条 ×
    old/new = 68 个格子，每格约 147 条）；
  * `candidates_10000.jsonl`：正式生成的 query（运行 generate_queries.py 后产生）；
  * `filtered/`：闭卷筛选后的干净题库（filter_closedbook.py 输出）。
  说明见 `queries/README.md`。



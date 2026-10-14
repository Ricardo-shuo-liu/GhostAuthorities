数据来源：https://flk.npc.gov.cn/index

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
* 以下为 `python -m ragpoison.gold` 生成（题目与 gold）：
  * `corpus.jsonl`、`questions.jsonl`；`facts.example.json` 是档位表述示例。
* `toy/`：离线自检用的虚构数据（由 `scripts/make_toy_data.py` 生成，可随时重建）。



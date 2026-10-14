# scripts/：数据处理与人工核对脚本

这些脚本只负责**文件/数据处理**，不参与实验运行；实验入口在仓库根目录
（`run_e0.py` / `run_matrix.py` / `report.py`）。所有命令都在**仓库根目录**
下执行。

| 脚本 | 作用 | 典型用法 |
|---|---|---|
| `build_from_consolidated.py` | 把仓库根 `data/` 里的 15 份历年合订本转换成项目格式：raw（按生效日期分文件夹的全量条文）、版本表、选题池、背景层 | `python scripts/build_from_consolidated.py` |
| `show_article.py` | 人工核对：查看某一条文在各年份的文本、变化点、每次修改前后全文 | `python scripts/show_article.py --article 176 --diff` |
| `make_toy_data.py` | 生成离线自检用的虚构《示例法》玩具数据（不联网） | `python scripts/make_toy_data.py` |

## build_from_consolidated.py 的产出

* `data/raw/刑法/`：最新文件夹（2024-03-01）= **现行全量**（503 个文件，
  每条一个 `{条号}.txt`）；其余文件夹 = **历史旧版**（该日生效、
  后被修正案替换的版本）；对照方法见 `data/raw/刑法/README.md`；
* `data/derived/article_versions.jsonl`：清洗后的版本表（程序入口，
  供 `python -m ragpoison.gold` 生成 corpus 与题目）；
* `data/derived/changes.csv`：15 个**相邻快照**的 diff 清单（源数据审计）；
* `data/derived/anomalies.csv`：被清理的源数据瑕疵（5 处，建议复核）；
* `data/derived/targets_candidates.csv`：刑档变化过的条文（实验选题池）；
* `data/background.jsonl`：最新合订本全量条文（检索背景层）。

## 核对建议流程

1. `python scripts/show_article.py --article 176` 看目标条文的变化点；
2. 对每个变化点用 `--diff` 看前后全文，与官方文本（flk.npc.gov.cn）对照；
3. 有疑问的笔误/回退在 `data/derived/anomalies.csv` 里复核
   （raw 保留原貌，如 `python scripts/show_article.py --article 273 --folder 2006-06-29`）。

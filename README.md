如果对你的思路有参考价值

通过以下方式引用:

```
@software{法律投毒研究,
  author = {Shuo Liu},
  title = {{GhostAuthorities}},
  version = {v1.0},
  year = {2026},
  url = {https://github.com/Ricardo-shuo-liu/GhostAuthorities},
}

```

---

## 实验代码与数据

| 文档 / 目录 | 内容 |
|---|---|
| [实验指南.md](实验指南.md) | 数据准备、API 配置、运行命令（从这里开始） |
| [法律RAG安全_研究方案_v3.md](法律RAG安全_研究方案_v3.md) | 研究方案（威胁模型、实验设计 E0–E8） |
| [实验执行手册.md](实验执行手册.md) | 逐周执行手册（v3 落地版） |
| `ragpoison/` | 核心库：common / gold / victim / attack / evaluate |
| `scripts/` | 数据脚本：合订本转换、人工核对工具、玩具数据 |
| `data/` | 15 份刑法合订本原件 + 生成物（说明见 `data/README.md`） |
| `run_e0.py` / `run_matrix.py` / `report.py` | 实验入口：单条件 / 条件矩阵 / 汇总与 DEQ |

快速开始（在仓库根目录执行）：

```bash
python scripts/build_from_consolidated.py          # 合订本 -> 版本表/背景层
python -m ragpoison.gold --versions data/derived/article_versions.jsonl \
    --only 176 --corpus data/corpus.jsonl \
    --questions data/questions.jsonl --facts-json data/facts.example.json
python run_e0.py --corpus data/corpus.jsonl --questions data/questions.jsonl \
    --background data/background.jsonl --attack A3 --n-inject 5 \
    --generator api --llm-model deepseek-chat \
    --llm-base-url https://api.deepseek.com --outdir results/e1_a3
```

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

## 当前仓库内容（数据清洗 + Query 工具）

| 目录 / 文档 | 内容 |
|---|---|
| [法律RAG安全_研究方案_v3.md](法律RAG安全_研究方案_v3.md) | 研究方案（威胁模型、实验设计） |
| [实验设计书_假设与预期效果.md](实验设计书_假设与预期效果.md) | 五个假设与实验判据 |
| [实验执行手册.md](实验执行手册.md) | 逐周执行手册 |
| `scripts/` | ① 数据清洗（合订本转换 / 人工核对 / 任意年份对比）；② Query 生成与闭卷筛选 |
| `ragpoison/` | 数据层公共库（common / env） |
| `data/` | 15 份刑法合订本 + raw/derived/background + `queries/`（说明见 `data/README.md`） |

快速开始：

```bash
# ① 数据清洗（已生成过；源数据改动后重跑）
python scripts/build_from_consolidated.py

# ② 生成 10000 条 query 排期（old/new 各 5000，不调用 API）
python scripts/generate_queries.py --plan-only

# ③ 正式生成（需要硅基流动 key；见 .env.example）
python scripts/generate_queries.py --num 10000

# ④ 闭卷筛选（多模型、无检索；任何模型答对即剔除）
python scripts/filter_closedbook.py --limit 200   # 先小规模校准
```

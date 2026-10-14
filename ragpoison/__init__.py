"""ragpoison：法律 RAG 投毒实验的核心库。

模块分工：
  common    数据结构、中文数字/刑期解析、语料/题目 IO 与校验
  gold      「从旧兼从轻」判定、题目与攻击目标生成、版本表 -> corpus
  victim    受害 RAG（向量检索 + 生成，支持 api/local/mock 后端）
  attack    零伪造攻击族 A1–A4 与对照 B0–B2
  evaluate  nugget 评分、ASR/TVR、检测 AUROC、DEQ
"""

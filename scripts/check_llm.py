"""排查 LLM API：发两个最小请求，打印原始返回的关键字段。

用途（在仓库根目录执行）：
    python scripts/check_llm.py
    python scripts/check_llm.py --llm-max-tokens 4096

会显示：finish_reason / content 预览 / reasoning_content 长度 / usage。
典型判断：
  * content 为空但 reasoning_content 很长 → 思考型模型把预算花在推理上了，
    加大 --llm-max-tokens（4096+）或在 run 时传更大的值；
  * HTTP 401/403 → key 或模型名不对；
  * 连接错误 → base_url / 代理问题。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ragpoison.env import load_dotenv                      # noqa: E402
from ragpoison.victim import http_post_json                # noqa: E402


def show(tag: str, resp: dict, dump_raw: bool = False) -> None:
    choice = resp["choices"][0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    print(f"\n== {tag} ==")
    print(f"  finish_reason     : {choice.get('finish_reason')}")
    print(f"  content 长度      : {len(content)}")
    print(f"  content 预览      : {content[:160]!r}")
    print(f"  reasoning 长度    : {len(reasoning)}")
    print(f"  message 字段      : {list(msg.keys())}")
    print(f"  usage             : {resp.get('usage')}")
    if not content:
        if reasoning:
            print("  → 结论：这是思考型模型且正文为空。请加大 --llm-max-tokens 重试，"
                  "或换普通对话模型。")
        else:
            print("  → 结论：正文为空且无 reasoning。检查模型名/参数是否被服务端支持。")
    if dump_raw:
        raw = json.dumps(resp, ensure_ascii=False, indent=2)
        print("  ---- 原始返回（截断显示 2000 字）----")
        print(raw[:2000])


def main() -> None:
    loaded = load_dotenv()
    if loaded:
        print(f"[.env] 已加载：{', '.join(sorted(loaded))}")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "deepseek-chat"))
    ap.add_argument("--llm-base-url",
                    default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--llm-api-key-env", default="LLM_API_KEY")
    ap.add_argument("--llm-max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--dump-raw", action="store_true", help="打印原始 JSON 返回（前 2000 字）")
    a = ap.parse_args()

    key = os.environ.get(a.llm_api_key_env, "")
    if not key:
        raise SystemExit(f"环境变量 {a.llm_api_key_env} 为空（检查 .env 或 export）")
    url = a.llm_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    print(f"模型：{a.llm_model}\n地址：{url}\nmax_tokens：{a.llm_max_tokens}")

    short = http_post_json(url, {
        "model": a.llm_model,
        "messages": [{"role": "user", "content": "只回复两个字：正常"}],
        "temperature": 0.0, "max_tokens": 64,
    }, headers=headers, timeout=a.timeout)
    show("请求 1：最短对话", short, a.dump_raw)

    e1_style = ("你是中国法律助手。只能依据下列条文作答。\n\n[条文]\n"
                "第161条 依法负有信息披露义务的公司、企业向股东和社会公众提供虚假的"
                "或者隐瞒重要事实的财务会计报告，或者对依法应当披露的其他重要信息"
                "不按照规定披露，严重损害股东或者其他人利益，或者有其他严重情节的，"
                "对其直接负责的主管人员和其他直接责任人员，处五年以下有期徒刑或者"
                "拘役，并处或者单处罚金；情节特别严重的，处五年以上十年以下有期"
                "徒刑，并处罚金。\n\n[问题]\n根据上述条文，最高可判处有期徒刑多少年？"
                "最后一行严格按此格式输出：第X条｜Y年")
    long = http_post_json(url, {
        "model": a.llm_model,
        "messages": [{"role": "user", "content": e1_style}],
        "temperature": 0.0, "max_tokens": a.llm_max_tokens,
    }, headers=headers, timeout=a.timeout)
    show(f"请求 2：E1 风格（max_tokens={a.llm_max_tokens}）", long, a.dump_raw)

    print("\n如果请求 1 正常、请求 2 为空：把 --llm-max-tokens 加到 4096 再试；"
          "仍为空则考虑换模型（普通对话模型，如 deepseek-chat）。")


if __name__ == "__main__":
    main()

"""按配置批量跑一组实验条件（对照矩阵）。

每个条件一个 outdir，顺序执行、自动 --resume（重跑同一命令不会重复调用 API）。
实验设计的"实现"就靠它：把 E0–E8 每个实验展开成若干条件，写进 JSON 配置。

用法：
    python run_matrix.py --config experiments/smoke_matrix.json
    python run_matrix.py --config experiments/e5_core.json --report

配置格式见 experiments/*.json：
    {
      "name": "e5_core",                  # 结果落在 results/<name>/
      "common": { ... run_e0 的公共参数 ... },
      "conditions": [ { "id": "...", ...每格自己的参数... }, ... ],
      "deq": ["A", "B", "C", "D"]         # 可选：四格的 id，供 report.py 算 DEQ
    }
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def to_cli(key: str, value) -> list[str]:
    """把配置里的一个键值对转成 run_e0.py 的命令行参数。"""
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        return [flag] if value else []
    if value is None:
        return []
    return [flag, str(value)]


def main() -> None:
    ap = argparse.ArgumentParser(description="按配置批量跑对照实验")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--report", action="store_true", help="跑完后自动汇总")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    name = cfg.get("name", a.config.stem)
    root = Path(cfg.get("outroot", f"results/{name}"))
    common = cfg.get("common", {})
    conditions = cfg["conditions"]

    print(f"矩阵 {name}：{len(conditions)} 个条件 -> {root}")
    for cond in conditions:
        cond = dict(cond)
        cid = cond.pop("id")
        outdir = root / cid
        cmd = [sys.executable, str(HERE / "run_e0.py")]
        for k, v in {**common, **cond}.items():
            cmd += to_cli(k, v)
        cmd += ["--outdir", str(outdir), "--resume"]
        print(f"\n== [{cid}] " + " ".join(cmd[2:]))
        if a.dry_run:
            continue
        subprocess.run(cmd, cwd=HERE, check=True)

    print(f"\n全部完成。汇总： python report.py --root {root}" +
          (f" --deq {','.join(cfg['deq'])}" if cfg.get("deq") else ""))
    if a.report and not a.dry_run:
        rcmd = [sys.executable, str(HERE / "report.py"), "--root", str(root)]
        if cfg.get("deq"):
            rcmd += ["--deq", ",".join(cfg["deq"])]
        subprocess.run(rcmd, cwd=HERE, check=True)


if __name__ == "__main__":
    main()

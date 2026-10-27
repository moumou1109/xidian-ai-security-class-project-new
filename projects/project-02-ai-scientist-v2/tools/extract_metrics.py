#!/usr/bin/env python3
"""从 AI-Scientist-v2 的实验产物中提取指标，生成可提交的扁平 CSV。

背景：`experiments/` 目录被 .gitignore 排除（体积大、含生成代码），
导致报告里的数字无法随仓库提交。本脚本把 `journal.json` 中的
`metric.value.metric_names[*].data[*]` 摊平成一张表，落盘到
`report-data/`，使报告结论可被逐个核对。

用法（在项目 02 目录下执行）：

    python -X utf8 tools/extract_metrics.py \
        --experiments experiments/20261021_180456_feature_redundan_attempt_0 \
        --out report-data/metrics.csv

只依赖标准库（json / csv / pathlib），不需要 torch / numpy。

输出：
    <out>            扁平指标表，一行 = 一个 (stage, node, metric, dataset)
    report-data/summary.json  每个 stage 的节点数与有效节点数
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def iter_stage_journals(exp_dir: Path):
    """返回按名称排序的 stage journal.json 路径列表。"""
    base = exp_dir / "logs" / "0-run"
    return sorted(base.glob("stage_*/journal.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments", required=True, help="一次实验的产物目录")
    ap.add_argument("--out", required=True, help="输出 CSV 路径")
    args = ap.parse_args()

    exp = Path(args.experiments)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    journals = iter_stage_journals(exp)
    if not journals:
        sys.exit(f"未找到 journal.json：{exp / 'logs' / '0-run'}")

    rows = []
    summary = {}
    for jp in journals:
        stage = jp.parent.name
        data = json.loads(jp.read_text(encoding="utf-8"))
        nodes = data.get("nodes", [])
        good = 0
        for n in nodes:
            if not n.get("is_buggy"):
                good += 1
            metric = (n.get("metric") or {}).get("value") or {}
            for mn in metric.get("metric_names") or []:
                for d in mn.get("data") or []:
                    rows.append(
                        {
                            "stage": stage,
                            "node_id": str(n.get("id", ""))[:8],
                            "is_buggy": n.get("is_buggy"),
                            "exec_time_s": n.get("exec_time"),
                            "metric_name": mn.get("metric_name"),
                            "lower_is_better": mn.get("lower_is_better"),
                            "dataset_name": d.get("dataset_name"),
                            "final_value": d.get("final_value"),
                            "best_value": d.get("best_value"),
                        }
                    )
        summary[stage] = {"nodes": len(nodes), "good": good}

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sum_path = out.with_name("summary.json")
    sum_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"写入 {out}（{len(rows)} 行）")
    print(f"写入 {sum_path}")
    for name, info in summary.items():
        print(f"  {name}: {info['nodes']} 节点 / {info['good']} 有效")


if __name__ == "__main__":
    main()

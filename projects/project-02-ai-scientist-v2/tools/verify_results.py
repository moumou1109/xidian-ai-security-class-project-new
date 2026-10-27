#!/usr/bin/env python3
"""校验 EXPERIMENT-REPORT.md 的关键结论能否由 report-data/metrics.csv 复现。

`experiments/` 被 .gitignore 排除，评审者手里只有本仓库提交的
`report-data/metrics.csv`。本脚本把报告里的每一条可量化断言变成一次
断言检查：全 PASS 说明报告数字与提交的指标表一致；任何 FAIL 说明
两者已经不一致（报告被手改过，或 CSV 与原始运行不符）。

用法（在项目 02 目录下执行）：

    python -X utf8 tools/verify_results.py

退出码：0 = 全部通过；1 = 有断言失败。只依赖标准库。
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "report-data" / "metrics.csv"
SUM_PATH = ROOT / "report-data" / "summary.json"

TOL = 5e-4  # 报告里保留 4 位小数，容差取半个末位

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


def load_rows():
    with CSV_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select(rows, stage_prefix, metric_name, node_id=None):
    """返回 {dataset_name: float(final_value)}。"""
    out = {}
    for r in rows:
        if not r["stage"].startswith(stage_prefix):
            continue
        if r["metric_name"] != metric_name:
            continue
        if node_id and r["node_id"] != node_id:
            continue
        try:
            out[r["dataset_name"]] = float(r["final_value"])
        except (TypeError, ValueError):
            continue
    return out


def increasing(values):
    seq = [v for _, v in sorted(values.items())]
    return all(b > a for a, b in zip(seq, seq[1:]))


def decreasing(values):
    seq = [v for _, v in sorted(values.items())]
    return all(b < a for a, b in zip(seq, seq[1:]))


def main():
    if not CSV_PATH.exists():
        sys.exit(f"缺少 {CSV_PATH}，先运行 tools/extract_metrics.py")

    rows = load_rows()
    summary = json.loads(SUM_PATH.read_text(encoding="utf-8"))

    # --- 结构与规模（报告 3.1 节） ---
    check("metrics.csv 行数 > 300", len(rows) > 300, f"{len(rows)} 行")

    expect_nodes = {"stage_1": 2, "stage_2": 4, "stage_3": 3}
    expect_good = {"stage_1": 1, "stage_2": 4, "stage_3": 2}
    for prefix, n in expect_nodes.items():
        keys = [k for k in summary if k.startswith(prefix)]
        got_n = summary[keys[0]]["nodes"] if keys else None
        got_g = summary[keys[0]]["good"] if keys else None
        check(f"{prefix} 节点数 = {n}", got_n == n, f"实际 {got_n}")
        check(f"{prefix} 有效节点数 = {expect_good[prefix]}", got_g == expect_good[prefix], f"实际 {got_g}")

    # --- Stage 3：结论反转（报告 3.3 节） ---
    key = "ASR gradient attack epsilon=1.0"
    stage3 = {fam: select(rows, "stage_3", f"{key} ({fam})") for fam in ("logistic", "rf", "mlp")}
    rho_sets = [d for d in stage3["logistic"] if d.startswith("synth_rho_")]
    check("Stage 3 合成数据有 4 档 rho", len(rho_sets) == 4, str(sorted(rho_sets)))

    report_asr = {
        "logistic": [0.5788, 0.6466, 0.7751, 0.9917],
        "rf": [0.2484, 0.2861, 0.4103, 0.6149],
        "mlp": [0.5651, 0.6285, 0.7821, 0.9916],
    }
    for fam, want in report_asr.items():
        got = [stage3[fam][d] for d in sorted(rho_sets)]
        ok = len(got) == 4 and all(abs(a - b) <= TOL for a, b in zip(got, want))
        check(f"Stage 3 {fam} ASR 序列与报告一致", ok, f"CSV={got}")

    for fam, vals in stage3.items():
        sub = {d: v for d, v in vals.items() if d.startswith("synth_rho_")}
        check(f"Stage 3 {fam} ASR 随 rho 单调递增", increasing(sub), str(sub))

    # --- RRES（报告 3.3 节末） ---
    rres_want = {"overall": -0.4140, "logistic": -0.4127, "rf": -0.4048, "mlp": -0.4246}
    for tag, want in rres_want.items():
        metric = "RRES overall" if tag == "overall" else f"RRES ({tag})"
        got = select(rows, "stage_3", metric)
        val = next(iter(got.values()), None)
        ok = val is not None and abs(val - want) <= TOL
        check(f"RRES {tag} = {want}", ok, f"CSV={val}")

    fam_rres = [select(rows, "stage_3", f"RRES ({f})") for f in ("logistic", "rf", "mlp")]
    vals = [next(iter(d.values())) for d in fam_rres if d]
    check("RRES 三个模型族同为负且量级相近", all(v < 0 for v in vals) and max(vals) - min(vals) < 0.1, str(vals))

    # --- Stage 1：反转前的方向（报告 3.2 节） ---
    s1_log = select(rows, "stage_1", "logistic ASR under gradient attack")
    if not s1_log:
        s1_log = select(rows, "stage_1", "attack success rate (gradient attack)")
    check("Stage 1 logistic ASR 随 rho 递减（与 Stage 3 相反）", decreasing(s1_log), str(s1_log))

    s1_clean = select(rows, "stage_1", "logistic clean test accuracy")
    if not s1_clean:
        s1_clean = select(rows, "stage_1", "validation accuracy")
    ok_clean = bool(s1_clean) and all(v >= 0.85 for v in s1_clean.values())
    check("Stage 1 干净准确率 ≥ 0.85（模型非随机）", ok_clean, str(s1_clean))

    # --- 真实数据集量级（报告 3.3 节末，按模型族给区间） ---
    real_ranges = {"hf_breast_cancer": (0.05, 0.19), "hf_wine": (0.02, 0.13), "hf_diabetes": (0.13, 0.39)}
    for ds, (lo, hi) in real_ranges.items():
        vals = [stage3[fam].get(ds) for fam in stage3]
        ok = all(v is not None and lo - TOL <= v <= hi + TOL for v in vals)
        check(f"真实数据集 {ds} ASR 落在 [{lo}, {hi}]", ok, f"{vals}")

    # --- 输出 ---
    failed = 0
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{mark}] {name}" + (f"  ({detail})" if detail and not ok else ""))

    print()
    print(f"共 {len(checks)} 项检查，通过 {len(checks) - failed} 项，失败 {failed} 项")
    if failed:
        print("报告与 report-data/metrics.csv 不一致，请核对 EXPERIMENT-REPORT.md。")
        return 1
    print("报告中的量化结论与提交的指标表一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

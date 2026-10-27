# report-data：实验证据（可提交部分）

原始实验目录 `experiments/20261021_180456_feature_redundan_attempt_0/`
被 [.gitignore](../.gitignore) 的 `experiments/` 规则排除（体积大、含模型
生成的代码与 78 张图），因此**报告里的数字原来无法随仓库提交核对**。
本目录是从那次运行中提取出来的最小证据集：指标表 + 6 张关键图。

## 文件

| 文件 | 内容 |
| --- | --- |
| `metrics.csv` | 370 行扁平指标表，一行 = 一个 (stage, node, metric, dataset) |
| `summary.json` | 每个 stage 的节点数与有效节点数（2/1、4/4、3/2） |
| `figures/stage1_asr_vs_rho_gradient.png` | Stage 1：ASR 随冗余度**下降**（混淆设计下的表面结论） |
| `figures/stage1_clean_acc_vs_rho.png` | Stage 1：干净准确率（排除"只是模型变差"的解释） |
| `figures/stage3_redundancy_vs_asr.png` | Stage 3：平均 ASR 随特征相关性**上升**（或 margin-matched 后的真实方向） |
| `figures/stage3_rres_by_model.png` | Stage 3：三个模型族的 RRES 对比 |
| `figures/stage3_rres_bar.png` | Stage 3：RRES 汇总柱状图（总体 −0.4140） |
| `figures/stage3_all_datasets_redundancy_vs_mean_asr.png` | Stage 3：合成 + 3 个真实数据集汇总 |

## metrics.csv 列

```
stage, node_id, is_buggy, exec_time_s, metric_name,
lower_is_better, dataset_name, final_value, best_value
```

- `stage`：`stage_1_initial_implementation_1_preliminary` /
  `stage_2_baseline_tuning_1_first_attempt` /
  `stage_3_creative_research_1_first_attempt`
- `node_id`：执行树节点 id 前 8 位（报告里引用的 `9492e8e7`、`7094829c` 即此列）
- `dataset_name`：`synth_rho_0.0` / `synth_rho_0.3` / `synth_rho_0.6` /
  `synth_rho_0.9`，真实数据为 `hf_breast_cancer` / `hf_wine` / `hf_diabetes`
- `final_value`：该数据集上训练结束时的指标值

## 重新生成

```bash
python -X utf8 tools/extract_metrics.py \
  --experiments experiments/20261021_180456_feature_redundan_attempt_0 \
  --out report-data/metrics.csv
```

## 校验报告

```bash
python -X utf8 tools/verify_results.py
```

该脚本把 [EXPERIMENT-REPORT.md](../EXPERIMENT-REPORT.md) 中的量化断言
（节点数、各模型族 ASR 序列、RRES 数值、趋势方向、真实数据集区间）
逐条与 `metrics.csv` 比对，输出 24 项 PASS/FAIL，全 PASS 才算报告与
证据一致。当前状态：24/24 通过。

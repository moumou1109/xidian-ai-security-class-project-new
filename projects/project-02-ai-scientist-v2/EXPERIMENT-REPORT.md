# 项目 02 实验阶段报告（CPU-only 运行记录）

> 运行日期：2026-10-21 18:04 – 19:52
> 上游版本：`96bd51617cfdbb494a9fc283af00fe090edfae48`
> 运行环境：Windows 11 + Python 3.11.17 + PyTorch 2.14.1+cpu（无 GPU）
> 模型服务：校内 OpenAI 兼容服务 `DeepSeek-V4.1-Flash-1017`

---

## 1. 摘要

在**无 NVIDIA GPU** 的机器上，用 CPU 串行模式跑通了 AI-Scientist-v2 的实验树搜索流程：

- 完成 **Stage 1（初始实现）、Stage 2（基线调参）、Stage 3（创造性研究）** 三个阶段
- 累计 **9 个实验节点**（8 个有效），生成 **73 张图表**
- Stage 4（消融实验）因上游 LLM 服务反复出现 10 分钟级超时、时间成本过高而主动终止

最重要的结果不是"验证了假设"，而是**实验自身暴露了原设计的混淆缺陷**：初版实验支持"特征冗余提升鲁棒性"，改进设计后结论**反转**（冗余反而增加脆弱性）。这一过程本身构成了可报告的科研发现。

---

## 2. 环境与方法

### 2.1 硬件与软件

| 项 | 值 |
| --- | --- |
| CPU | Intel i7-10700（8 核 16 线程） |
| GPU | 无可用 CUDA 设备（`torch.cuda.is_available() == False`） |
| 内存 | 31.7 GB |
| Python | 3.11.17（conda 环境 `course-scientist-v2`） |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |

### 2.2 运行配置

`bfts_config.yaml` 关键改动：`num_workers: 1`、`stage1..4_max_iters: 4/3/3/4`、`steps: 2`、`num_seeds: 1`，全部模型指向 `DeepSeek-V4.1-Flash-1017`。

### 2.3 启动命令

```bash
cd E:\ai\projects\project-02-ai-scientist-v2
python -X utf8 launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/course_topic.json \
  --idea_idx 0 --skip_writeup --skip_review \
  --model_agg_plots DeepSeek-V4.1-Flash-1017
```

产物目录：`experiments/20261021_180456_feature_redundan_attempt_0/`

---

## 3. 阶段结果

### 3.1 节点总览

| 阶段 | 节点数 | 有效 | 说明 |
| --- | --- | --- | --- |
| Stage 1 初始实现 | 2 | 1 | 第 1 个节点即通过评审 |
| Stage 2 基线调参 | 4 | 4 | 增加学习率扫描与 epochs 扫描 |
| Stage 3 创造性研究 | 3 | 2 | 加入真实数据集与 margin-matched 设计 |
| Stage 4 消融实验 | — | — | 未启动 |

所有节点执行耗时 12–337 秒，全部在单进程 CPU 上完成。

### 3.2 Stage 1 / 2：初步结果（支持"冗余→鲁棒"）

合成数据，`rho ∈ {0.0, 0.5, 0.9}`，扰动预算 ε=1.0（标准化特征空间）：

| 指标 | rho=0.0 | rho=0.5 | rho=0.9 | 趋势 |
| --- | --- | --- | --- | --- |
| Logistic 干净准确率 | 0.9950 | 1.0000 | 0.9975 | 持平 |
| Logistic ASR（gradient） | 0.9925 | 0.8325 | 0.7025 | ↓ |
| Logistic ASR（greedy） | 0.2225 | 0.1175 | 0.0975 | ↓ |
| RF 干净准确率 | 0.8725 | 0.9550 | 0.9800 | ↑ |
| RF ASR（greedy） | 0.3750 | 0.1525 | 0.1000 | ↓ |
| MLP ASR（gradient，调参后） | 0.8075 | 0.6100 | 0.5200 | ↓ |

**初步结论**：冗余度越高，攻击成功率越低，且高容量模型（MLP）的 gradient ASR 明显低于线性模型 —— 表面支持"特征冗余提升鲁棒性"。

### 3.3 Stage 3：改进设计后结论反转

Stage 3 的节点 `7094829c` 做了两处关键改进：

1. **margin-matched**：控制不同冗余度下的类别间隔，消除"冗余↑ ⇒ 可分性↑"的混淆
2. 加入 3 个真实小数据集（breast_cancer / wine / diabetes）

| 指标 | 0.0 | 0.3 | 0.6 | 0.9 | 趋势 |
| --- | --- | --- | --- | --- | --- |
| Logistic ASR（gradient） | 0.5788 | 0.6466 | 0.7751 | 0.9917 | **↑** |
| RF ASR（gradient） | 0.2484 | 0.2861 | 0.4103 | 0.6149 | **↑** |
| MLP ASR（gradient） | 0.5651 | 0.6285 | 0.7821 | 0.9916 | **↑** |

**RRES（冗余-鲁棒效应量）总体 = −0.4140**（Logistic −0.4127，RF −0.4048，MLP −0.4246）。

负值意味着：**在控制类别间隔后，特征冗余度越高，模型越脆弱** —— 与 Stage 1/2 的结论完全相反。

真实数据集上的 ASR 普遍更低（breast_cancer 0.16–0.19，wine 0.12，diabetes 0.30–0.39），但仅 3 个数据集，不足以支撑统计结论。

---

## 4. 关键发现与解释

### 4.1 结论为什么会反转

Stage 1/2 的合成数据生成方式是"共享潜在因子 `u` 同时决定标签和所有特征"，于是：

```
rho ↑  ⇒  特征间相关性 ↑（冗余）
       ⇒  特征对 u 的信噪比 ↑
       ⇒  类别间隔变大
       ⇒  攻击需要更大的扰动才能翻转标签
```

"冗余"和"可分性"在这个生成器里**是同一个变量的两个侧面**，因此看到 ASR 下降，无法区分是"冗余补偿"还是"间隔变大"。

Stage 3 用 margin-matched 设计把两者拆开后，ASR 随冗余**上升**，说明真正的因果方向是：冗余让单特征扰动更容易被"结构性放大"，而不是更容易被抵消。

### 4.2 对研究假设的评价

原假设"特征冗余而非模型容量决定鲁棒性"在本次运行中**未获支持**：

- 三个模型族的 RRES 高度一致（−0.40 ~ −0.42），说明**效应方向不依赖模型容量**
- 但方向与原假设**相反**：冗余降低而非提升鲁棒性

这是有价值的负面结果（negative result），符合 AI-Scientist 项目中 ICBINB（"我不信它没变好"）选题的精神。

---

## 5. 产物清单

- 实验数据：`0-run/process_*/working/*.npy`、`experiment_data.npy`
- 图表：**73 张 PNG**，关键包括
  - `asr_vs_rho_{gradient,greedy,random}.png`
  - `clean_acc_vs_rho.png`
  - `asr_vs_epochs_*.png`、`val_loss_vs_epochs.png`
  - `synthetic_gaussian_copula_baseline_comparison_by_rho.png`
  - `synthetic_correlated_*` 系列
- 日志：`logs/0-run/stage_*/journal.json`、`tree_plot.html`、`unified_tree_viz.html`
- 最佳方案：`logs/0-run/stage_2_baseline_tuning_1_first_attempt/best_solution_0171e22f*.py`

---

## 6. 已知问题与局限

1. **未跑 Stage 4**：消融实验未执行，因此"哪种因素主导"的结论不完整。
2. **无 seed 统计**：`num_seeds=1`，所有数字均为单次运行，**没有置信区间**，不能做显著性判断。
3. **样本量小**：每档 rho 仅 1 次运行；真实数据集仅 3 个。
4. **LLM 服务不稳定**：运行期间多次出现 10 分钟级请求超时（约 4 次），是导致总耗时 108 分钟的主要原因。
5. **未做论文生成**：使用 `--skip_writeup --skip_review`，故无 PDF 与同行评审输出。
6. **模型自评偏差**：节点的"是否 bug"由同一模型判断，前期曾把方法论缺陷误判为执行 bug，已通过调整评审口径修正。
7. **单次运行的偶然性**：Stage 1/2 与 Stage 3 的数据生成器不同，两者不可直接比较，只能作为"设计选择导致结论翻转"的证据。

---

## 7. 复现步骤

1. 按 `COURSE-NOTES.md` 第 2–3 节准备环境与密钥。
2. 确认 `bfts_config.yaml` 中 `num_workers: 1`、模型名指向可用服务。
3. 运行第 2.3 节命令。
4. 产物在 `experiments/<时间戳>_<想法名>_attempt_0/`。
5. 检查 `logs/0-run/stage_*/journal.json` 的 `nodes[].metric.value.metric_names` 获取全部指标。

---

## 8. 结论

1. **工程上**：AI-Scientist-v2 可以在无 GPU 的 Windows 机器上跑通实验树搜索（Stages 1–3），代价是单节点 12–337 秒、全流程约 108 分钟。
2. **科学上**：本次运行产出的是一个**负面结果**——原假设"冗余提升鲁棒性"在控制类别间隔后不成立，RRES ≈ −0.41。
3. **方法论上**：最有价值的产出是"初始实验 → 自我诊断混淆 → 改进设计 → 结论反转"这一完整链条，它比任何一个单点指标都更能说明自动化科研系统的价值与风险。

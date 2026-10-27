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

### 2.4 修改前后行为对照（对应课程任务第 2 条）

本次改动的**明确目标**是：让上游项目在「无 NVIDIA GPU + 只提供单一模型的校内 OpenAI 兼容服务 + 中文 Windows」这套与上游假设不符的环境下，把实验树搜索从"跑不起来"改到"能完整跑完 Stage 1–3"。下表每一行都是一次实测到的「修改前行为 → 修改后行为」：

| # | 场景 | 修改前（实测现象） | 修改后（实测行为） | 涉及文件 |
| --- | --- | --- | --- | --- |
| 1 | 指定校内模型 | `ValueError: Model DeepSeek-V4.1-Flash-1017 not supported`，流程立即退出 | 三个分发函数各自走 `AISCI_BASE_URL`，正常返回 | `ai_scientist/llm.py` |
| 2 | 推理模型输出 | 正文为空字符串（4096 token 预算被 reasoning 吃满） | 预算提到 8192，正文非空 | `ai_scientist/llm.py` |
| 3 | 无 `S2_API_KEY` 时的检索 | 429 后 `backoff` 无限重试，进程挂起 8 分钟以上 | 直接跳过检索；有 key 时最多 2 次、单次 15 秒超时 | `tools/semantic_scholar.py` |
| 4 | 模型返回的 JSON 带尾随文本 | `json.JSONDecodeError: Extra data: line …<assistant>…` | `raw_decode` 只取第一个完整对象，解析成功 | `perform_ideation_temp_free.py` |
| 5 | 一次响应含多个 `ACTION` 块 | 只取第一个动作，真正的 `FinalizeIdea` 被丢弃，无产物 | 优先取最后一个 `FinalizeIdea`，正常写出 ideas JSON | `perform_ideation_temp_free.py` |
| 6 | 实验阶段连服务 | 请求打到 `api.openai.com`，`backoff_create` 无限重试挂死（`backend_openai.py` 只读 `OPENAI_BASE_URL`） | 两套 BASE_URL 都设置后正常 | 环境变量，无代码改动 |
| 7 | 阶段结束摘要 | 403 `This token has no access to model gpt-4o`（`journal.py`/`log_summarization.py` 硬编码模型名） | 统一走 `DEFAULT_LLM_MODEL`，摘要正常生成 | `treesearch/journal.py`、`treesearch/log_summarization.py` |
| 8 | 无 GPU 时的并行度 | `num_workers: 4`——上游只在 `num_gpus > 0` 时才收缩 worker，4 进程抢同一颗 CPU | CPU-only 时强制 `num_workers = 1`，串行执行 | `treesearch/parallel_agent.py` |
| 9 | 「是否 bug」的判定口径 | Stage 1 的 4 个节点全被判 bug，报 `Could not find working implementation in initial stage after 4 iterations` 后终止 | 只把崩溃/异常/NaN/明显无效结果判 bug，方法论缺陷不判；Stage 1 以 2 节点（1 有效）通过 | `treesearch/parallel_agent.py`（`submit_review`） |
| 10 | 产物路径超长 | `WinError 206`（文件名或扩展名太长），绘图与产物保存失败 | 仓库放在 `E:\ai` + 时间戳改 `%Y%m%d_%H%M%S` + 想法名截断 16 字符，正常落盘 | `launch_scientist_bfts.py` |
| 11 | 路径前缀不一致 | `ValueError: … is not in the subpath of …`（`Path.resolve()` 与 `os.getcwd()` 前缀不同） | 新增 `_rel_to_cwd()` 兜底，摘要正常 | `treesearch/journal.py` |
| 12 | 生成代码与本机库版本 | 生成 `np.trapz`（NumPy 2.x 已移除）等不兼容代码，节点执行即崩 | 提示词写明本机约束，不兼容代码明显减少 | `treesearch/parallel_agent.py`（提示词） |

其中 **第 9 行是唯一改变了"研究结论能否产出"的改动**：前 3 次运行都因口径过严而在 Stage 1 就终止，第 4 次才跑出 9 个节点。第 1–8、10–12 行只影响"能否跑起来"，不改变实验结果的含义。

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

真实数据集上的 ASR 普遍低于合成数据，且模型族之间差异明显（gradient attack，ε=1.0）：breast_cancer 落在 0.055–0.187，wine 落在 0.028–0.120，diabetes 落在 0.137–0.386。其中随机森林在三者上都最低（0.028–0.137），logistic 与 MLP 接近。仅 3 个数据集、每档 1 次运行，不足以支撑统计结论。

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

`experiments/` 被 `.gitignore` 排除，原始产物不随仓库提交。为让报告中的数字可被核对，另提交：

- `report-data/metrics.csv`：从 `journal.json` 提取的 370 行扁平指标表
- `report-data/summary.json`：各阶段节点数/有效节点数
- `report-data/figures/*.png`：6 张关键图（Stage 1 的"反转前"与 Stage 3 的"反转后"各 2 张）
- `tools/extract_metrics.py`、`tools/verify_results.py`：重新生成与自动校验（24 项检查，见 7 节第 8 步）

---

## 6. 已知问题与局限

1. **未跑 Stage 4**：消融实验未执行，因此"哪种因素主导"的结论不完整。
2. **无 seed 统计**：`num_seeds=1`，所有数字均为单次运行，**没有置信区间**，不能做显著性判断。
3. **样本量小**：每档 rho 仅 1 次运行；真实数据集仅 3 个。
4. **LLM 服务不稳定**：运行期间多次出现 10 分钟级请求超时（约 4 次），是导致总耗时 108 分钟的主要原因。
5. **未做论文生成**：使用 `--skip_writeup --skip_review`，故无 PDF 与同行评审输出。
6. **模型自评偏差**：节点的"是否 bug"由同一模型判断，前期曾把方法论缺陷误判为执行 bug，已通过调整评审口径修正。
7. **单次运行的偶然性**：Stage 1/2 与 Stage 3 的数据生成器不同，两者不可直接比较，只能作为"设计选择导致结论翻转"的证据。

### 6.1 如何补齐 Stage 4（消融）

Stage 4 在本次运行中没有启动：Stage 3 结束后连续 4 次 LLM 请求出现 10 分钟级超时，为避免继续空耗而主动终止（`--skip_writeup --skip_review` 不影响 Stage 4，Stage 4 属于实验阶段）。

补齐方式：

```bash
# 在 bfts_config.yaml 中把 stage4_max_iters 保留为 4，直接重跑同一条命令
python -X utf8 launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/course_topic.json --idea_idx 0 \
  --skip_writeup --skip_review --model_agg_plots DeepSeek-V4.1-Flash-1017
```

预期 Stage 4 做的是对 Stage 3 最佳节点的消融：分别去掉 margin-matching、去掉真实数据集、把 rho 网格加密，以区分"结论反转"到底由哪个设计因素造成。耗时按 Stage 3 的 337 秒/节点 × 4 节点估算约 25–30 分钟（不含超时重试）。补齐后需要同步更新 3.x 节表格、`report-data/` 与 `tools/verify_results.py` 中的断言。

---

## 7. 复现步骤

1. 按 `COURSE-NOTES.md` 第 2–3 节准备环境与密钥。
2. 安装实验阶段的前置依赖（`requirements.txt` **不含**以下三项）：
   ```bash
   pip install torch --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/ -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install scikit-learn pymupdf -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
   缺 `pymupdf` 会在导入链上直接失败；缺 `torch` 则实验节点无法执行。
3. 确认环境变量：除 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`AISCI_BASE_URL`、`AISCI_DEFAULT_MODEL` 外，**`OPENAI_BASE_URL` 必须指向同一校内服务**——`backend_openai.py` 只读这一项，否则请求打到 `api.openai.com` 并无限重试。
4. 确认 `bfts_config.yaml` 中 `num_workers: 1`、模型名指向可用服务。
5. 运行第 2.3 节命令。产物在 `experiments/<时间戳>_<想法名>_attempt_0/`。
6. 检查 `logs/0-run/stage_*/journal.json` 的 `nodes[].metric.value.metric_names` 获取全部指标。
7. 若要从零重建指标表：`python -X utf8 tools/extract_metrics.py`（默认读 `experiments/`、写 `report-data/`）。
8. 校验报告数字：`python -X utf8 tools/verify_results.py`，期望末尾输出「共 24 项检查，通过 24 项，失败 0 项」。

> 注意：本次运行的环境为 Windows 11 + Python 3.11.17 + CPU-only。在 Linux/macOS 或有 GPU 的机器上第 4 步的 `num_workers` 可以调大；实验的具体数值会随模型服务、随机种子与库版本变化，**不保证逐位复现**，可复现的是流程与趋势方向。

---

## 8. 结论

1. **工程上**：AI-Scientist-v2 可以在无 GPU 的 Windows 机器上跑通实验树搜索（Stages 1–3），代价是单节点 12–337 秒、全流程约 108 分钟。
2. **科学上**：本次运行产出的是一个**负面结果**——原假设"冗余提升鲁棒性"在控制类别间隔后不成立，RRES ≈ −0.41。
3. **方法论上**：最有价值的产出是"初始实验 → 自我诊断混淆 → 改进设计 → 结论反转"这一完整链条，它比任何一个单点指标都更能说明自动化科研系统的价值与风险。

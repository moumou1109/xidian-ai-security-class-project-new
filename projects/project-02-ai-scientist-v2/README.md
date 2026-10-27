# 项目 02：AI-Scientist-v2 使用指南

本项目从研究主题生成想法，再通过实验树搜索执行实验，并可生成论文与审阅结果。建议先完成“生成一个研究想法”，再尝试完整实验。

[课程任务](COURSE.md) · [课程总说明](../../README.md) · [上游原始 README](README.original.md) · [许可证](LICENSE)

## 本作业的产出

- [COURSE-NOTES.md](COURSE-NOTES.md)：环境版本与运行命令、为适配校内模型服务与本机环境所做的 **12 处改动**（含失败排查）
- [EXPERIMENT-REPORT.md](EXPERIMENT-REPORT.md)：实验设计、Stage 1–3 结果、**结论反转**的分析、修改前后行为对照表、已知局限
- [report-data/](report-data/)：可随仓库提交的指标表、6 张关键图与重新生成脚本
- 一键校验报告数字：`python -X utf8 tools/verify_results.py`（24 项检查）

> 本次作业在 **Windows 11 + CPU-only**（无 NVIDIA GPU）环境完成，与下文"需要 Linux + NVIDIA GPU"的上游要求不同。替代做法与踩坑记录见 COURSE-NOTES.md 第 8 节。

## 1. 环境要求

- 上游面向 **Linux + NVIDIA GPU + CUDA/PyTorch**；Windows 用户使用具备 GPU 支持的 Linux 环境或服务器。
- 准备 Conda、Python 3.11、模型服务权限，以及论文流程所需的 PDF/LaTeX 工具。
- 系统会执行模型生成的代码，应在受控、隔离的实验环境运行。模型调用可能持续产生费用，先缩小实验范围。

以下命令使用 **Linux Bash**。从课程根目录开始：

```bash
cd projects/project-02-ai-scientist-v2
conda create -n course-scientist-v2 python=3.11
conda activate course-scientist-v2
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install anaconda::poppler
conda install conda-forge::chktex
pip install -r requirements.txt
```

CUDA 版本应结合驱动和环境调整；上述 PyTorch 命令来自上游，不保证适用所有驱动。需要生成 PDF 时还要准备可用的 LaTeX 编译环境，仅安装 `chktex` 并不等于安装完整 TeX。

## 2. 模型配置

通过当前终端的环境变量提供凭据，不要直接写入源码：

```bash
export OPENAI_API_KEY="替换为自己的密钥"
# 使用 Semantic Scholar 更高配额时再填写：
export S2_API_KEY="替换为自己的密钥"
```

不用 Semantic Scholar 密钥时，省略第二条；默认可匿名访问，但可能限流。使用 Gemini 时按上游设置 `GEMINI_API_KEY`；使用 AWS Bedrock 时需安装 `anthropic[bedrock]`，配置 AWS 凭据、区域及模型访问权限。

尤其要检查 [bfts_config.yaml](bfts_config.yaml)：当前实验代码模型默认为 AWS Bedrock 的 Claude 标识，仅设置 OpenAI Key 并不足以运行该默认流程。逐项核对 `agent.code.model`、反馈模型和报告模型等配置。旧模型可能退役，应使用当前账户可访问且项目适配层支持的模型。

## 3. 第一次运行：只生成一个想法

```bash
cp ai_scientist/ideas/i_cant_believe_its_not_better.md ai_scientist/ideas/course_topic.md
```

编辑新文件，保留示例结构并填写自己的研究主题、关键词、概要与摘要。随后运行：

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/course_topic.md \
  --model gpt-4o-2024-05-13 \
  --max-num-generations 1 \
  --num-reflections 1
```

命令中的模型名为上游示例，运行前验证可用性。完成后检查同名 `course_topic.json`，人工核验假设、数据需求和引用；没有有效 JSON 就不要进入下一步。

## 4. 运行实验

先审阅 `bfts_config.yaml` 中的并行数、各阶段迭代次数、执行超时和模型设置。`agent.stages` 各阶段上限可能覆盖通用 `steps`，不能只修改 `steps` 就认为已限制总量。

先跳过论文生成和审阅，运行第一个想法：

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/course_topic.json \
  --idea_idx 0 \
  --skip_writeup \
  --skip_review
```

该命令仍会调用实验模型、生成和执行代码，并可能调用绘图等模型；它不是无费用试运行。请同时检查启动脚本里的 `--model_agg_plots` 默认值及相关配置。

需要论文生成时，依据 [原始 README](README.original.md) 配置 `--model_writeup`、`--model_writeup_small`、`--model_citation`、`--model_review` 等参数，再去掉跳过选项。只有存在与想法 JSON 同名的 `.py` 初始代码时才使用 `--load_code`。

## 5. 结果、停止与排错

结果目录为 `experiments/<时间>_<想法名称>_attempt_<编号>/`，具体路径见终端输出。保留实际存在的实验日志、指标、代码、token 统计及论文文件，区分生成结果与人工验证结论。

按 Ctrl+C 中断主进程后，检查本次任务启动的子进程和 GPU 占用，确认实验确实停止。

| 问题 | 处理方向 |
| --- | --- |
| 配置了 OpenAI 仍报 AWS 权限错误 | 默认实验模型使用 Bedrock，需配置对应服务或采用代码支持的替代模型 |
| CUDA out of memory | 缩小实验模型、数据规模与并行数 |
| Semantic Scholar 限流 | 降低请求频率或配置自己的 S2 Key |
| 论文 PDF 生成失败 | 检查 LaTeX、Poppler 和模型输出，先单独完成实验流程 |
| 找不到想法文件 | 检查 JSON 是否生成、路径及当前工作目录 |

## 6. 课程提交

提交配置变更（不含密钥）、代码、实验记录摘要及复现说明，保留 [许可证](LICENSE)。本项目与项目 04 的实验适配层没有完成兼容验证，不能直接假设可互接。本课程源码整理时的上游版本为 `96bd51617cfdbb494a9fc283af00fe090edfae48`；本说明经源码核对，未执行完整实验。
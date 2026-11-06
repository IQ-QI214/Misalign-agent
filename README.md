# Misalign-agent

研究良性任务学习中的安全迁移，以及迁移方向如何受到教师递归更新的影响。

第一版面向 **Qwen3.5-9B、2×H100**，基于公开 [OPSD](https://github.com/siyan-zhao/OPSD) 的固定版本独立适配。2026-11-06 已完成真实 9B 的双 H100 ZeRO-2 两步更新、最终模型重载和16条任务成对评测。首次联测暴露的训练/评测EOS问题已修复，并从初始模型重新完成两步验证；安全迁移和论文现象仍未验证。

## 研究问题

**反馈结构与教师递归更新如何共同影响安全、任务能力和误拒？** 允许正迁移、负迁移和无明显变化，不预先给训练数据贴正负标签。

参考 [Progressive Misalignment](https://arxiv.org/abs/2610.22467) 的递归漂移问题，以及 [Recursive Self-Improvement via On-Policy Distillation for Reasoning](https://arxiv.org/abs/2609.30652) 的 DCE。后者每步更新教师，主实验采用 on-policy + forward KL + assistant-side verified reference；SRCL 另增加经验证的简短回答重写。当前尚未确认这两篇论文公开的官方训练仓库。

原 OPSD codebase 本身已有动态、LoRA 固定和 EMA 教师分支，所以先在它上面独立控制教师、KL 和上下文。当前 adapter 仅启用 frozen/每步recursive；EMA、periodic、SRCL、DPO/GRPO/PPO 尚未接入。

## 最小研究矩阵

| 运行 | 教师 | 反馈 |
| --- | --- | --- |
| F0 | 固定初始模型 | 中性 |
| R0 | 每步随学生更新 | 中性 |
| F1 | 固定初始模型 | 前提纠正 |
| R1 | 每步随学生更新 | 前提纠正 |

四组同任务池、同顺序及优化配方。前提纠正只对带错误前提的任务提供冻结规则，其余行与中性组相同。学生始终不见教师提示；验证答案不自动进入教师输入。

自带合成 pilot 为 train/dev/test 64/16/16 行、32/8/8 个基础 source；每个 source 有正常与错误前提两版。family 整组划分，domain/抽象运算可跨 split，不能称 domain-OOD。这些金额、单位、集合、时间任务只用于工程与受控探索；正式研究需更具代表性且难度合适的真实任务。

能力、安全依从和误拒分别报告，不合成总分。四格交互为 `(R1-F1)-(R0-F0)`，并不自动代表绝对安全收益或显著性。独立训练 seed 与生成 seed 分开记录。

## 来源与实现

`references/opsd_upstream` 为未修改的 Git submodule，固定 commit `ae7d2519e94920c4eb6206c0c26de46d9c50abae`。`configs/rsd/upstream.lock.json` 另锁文件 SHA。继承其 `OPSDTrainer`，复用初始化、KL 公式和 Transformers/TRL 训练/保存恢复基础。

适配器实现：

- 官方 Qwen3.5 文本类，严格校验语言权重，仅允许未使用的 vision/mtp 权重；模型 revision 固定为 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- 去掉上游数学专用模板，师生均关闭 thinking；完全相同的 completion token IDs 在不同上下文中打分，首个 EOS 计入 loss，过长 prompt 报错而非静默截断。
- 文本全参 frozen teacher 使用独立初始副本；recursive teacher 使用当前学生、停止教师梯度。
- 采样来源 `on_policy/off_policy` 与 `kl_direction=forward/reverse` 分离；无默认 top-k 近似或逐项 KL 裁剪。
- 显式 `feedback_only/verified_reference` 与 `user/assistant_prefill` 上下文；后者只读取独立的 `verified_solution` 字段。
- 模型、数据、代码与运行配置身份记录；同实验 checkpoint 恢复，轨迹保留 attempt 和 teacher step。

这是独立适配：与 Progressive 原设置相比，模型、全局 batch=2、短输出和模板均有区别；DCE 探针也未复现其 LoRA、12K rollout、完整推理数据、损失缩放与 SRCL。原论文的20条有害请求/初始安全回答实验和全良性请求筛选实验不能混称，本仓库合成 pilot 不是那20条请求。

## 本地验证

```bash
git submodule update --init --recursive
python -m pip install -r configs/rsd/requirements.txt
python -m pytest -q tests/test_rsd_*.py
python -m rsd.data validate --data-dir data/rsd_pilot
python -m rsd.train --arm R0 --dry-run
python -m rsd.smoke --output-dir runs/cpu_pipeline_smoke
```

实际 CPU 检查使用 PyTorch 2.15.0、Transformers 5.21.0、TRL 0.26.0。smoke 使用随机 tiny GPT-2，执行四组各两次真实优化、保存、生成、任务判分和四格汇总。另有 tiny Qwen3.5 更新与重载测试。这些输出不代表 9B 或安全迁移结果。

## 两卡运行

先安装与节点 CUDA 匹配的 PyTorch/DeepSpeed，运行 `ds_report` 并核验 CPUAdam 编译及 Qwen3.5 fast kernels。当前配置 ZeRO-2 + CPU optimizer offload，学生权重每卡完整保留；冻结教师每卡另占一份。不要原样使用上游旧 environment.yml。

修复EOS后的真实 R0 两步 smoke 已在 2×H100 80GB、400 GiB 容器 RAM 限额下通过。每卡 PyTorch 峰值 allocated 为 33.36 GiB、reserved 为 49.32 GiB；3秒间隔的系统采样观测到每卡最高约 56.7 GiB，容器内存最高约 295.8 GiB（包含容器内其他用量和页缓存，不是进程独占 RAM）。本次训练completion上限64，实际四条仅2—6 token。建议该短输出配置预留至少约384 GiB RAM；更长回答、冻结教师和512-token配置仍需实测。

首轮旧EOS运行的单个完整 step-2 checkpoint 为143.28 GB，最终权重另占17.93 GB；二者合计161.21 GB，不含原始模型、依赖和缓存。修复后的运行使用 `save_strategy="no"`，另存最终权重，保留首轮文件。按首轮完整checkpoint尺寸估算，主配置100步、每20步保存，5个完整checkpoint加最终权重约734 GB；若更长运行达到保留上限6个，则约878 GB，并另留模型缓存和运行余量。当前短 smoke 的成功不能作为100步、长输出或四组实验的容量验收。

已通过真实9B双卡 smoke 的镜像为 NVIDIA Torch 2.8.0a0 / CUDA 12.9 / Python 3.12。该镜像可额外使用 `-c configs/rsd/constraints-nvidia25.06.txt` 安装项目依赖：其 Torch 使用 NumPy 1 ABI，必须配套 NumPy 1.26.4、PyArrow 24.0.0；直接安装最新 PyArrow 26 或改用 NumPy 2 均在实际检查中失败。此约束不适用于其他 Torch 镜像，实际二进制兼容性由 preflight 的数据转换、CPUAdam 和 CUDA 小模型检查确认。

```bash
# 离线检查数据栈ABI、真实 CUDA、实际 kernel 导入、CPUAdam 更新和随机 tiny Qwen3.5 前后向。
# 输出目录也承接缓存、编译文件和逐项失败日志；通过不代表完整9B/双卡训练通过。
python -B scripts/rsd_gpu_preflight.py --output-dir runs/gpu_preflight \
  --check-kernels --exercise-optimizer --exercise-model

# 显式调用才启动一个2卡任务；首轮只跑2步。
python -m rsd.train --config configs/rsd/qwen35_9b_smoke.json --arm R0 --dry-run
bash scripts/rsd_launch.sh --config configs/rsd/qwen35_9b_smoke.json --arm R0

# 联测、初始行为与数据难度核验之后的小规模探索，非自动排队。
bash scripts/rsd_launch.sh --config configs/rsd/qwen35_9b_2h100.json --arm R0
```

100步、全局batch2对应200次样本曝光，不是200个独立任务。主配置每20步保存；恢复用 `--resume-from-checkpoint` 指定该运行自己的 checkpoint。训练、模型生成与judge按阶段执行，减少同时驻留。v1未启用vLLM，也没有自动根据安全分数选点或启动下一组的控制器。

配置默认 `save_strategy="steps"`，保存可恢复的完整checkpoint。显式设置 `save_strategy="no"` 时不保存周期checkpoint，训练结束仍导出 `final/` 权重与tokenizer；这种运行不能作为优化器断点续训验收。短工程验证可以采用该策略，正式长训练应先确定恢复需求与存储预算。

参数更新回执对每个可训练张量分散抽样，记录变化坐标数与逐张量差异。这只反映抽样模型权重，不覆盖优化器master weights或未抽样位置；BF16零差异不能证明没有更新。两步配置只保存step 2，从该点恢复到同一max_steps不会新增更新，不能算断点续训验收。

实际断点续训验收使用随机 tiny GPT2，在同一双 H100/ZeRO-2 配置下从step 1恢复至step 2，最终17个权重张量与不间断运行逐元素相同；9B尚未执行断点续训。脱敏回执见 [`validation/gpu_tiny_v1.json`](validation/gpu_tiny_v1.json)。

## 评测

```bash
python -m rsd.benchmarks prepare --output-dir data/rsd_benchmarks \
  --train-file data/rsd_pilot/train.jsonl
```

准备官方来源的 StrongREJECT 313条、XSTest safe 250条，锁定来源commit/SHA。采用我们自己的固定哈希dev/test分割，非论文原分割。SORRY官方数据gated，默认明确缺失，支持获授权的本地原文件导入；见 `python -m rsd.benchmarks prepare --help`。真实基准记录不随Git发布。

```bash
python -m rsd.evaluation generate \
  --dataset data/rsd_pilot/dev.jsonl \
  --model runs/rsd_qwen35_9b_v1/R0/final --checkpoint-id R0-final \
  --device cuda:0 --dtype bfloat16 --output runs/R0-utility-predictions.jsonl
python -m rsd.evaluation score \
  --dataset data/rsd_pilot/dev.jsonl --predictions runs/R0-utility-predictions.jsonl \
  --output runs/R0-utility-summary.json
```

远程Hub模型必须提供固定 `--revision`。本地真实checkpoint ID由权重/配置内容SHA计算，generate的 `--checkpoint-id` 只是可读标签，不要再用它覆盖score身份。

生成协议 `rsd-text-eval-v2` 对所有 checkpoint 显式使用 `tokenizer.eos_token_id`，避免原模型默认配置与Trainer保存配置的EOS差异改变停止条件。旧v1运行确实在比较阶段因协议不一致被拒绝；旧记录保留，修复后必须重新生成base与candidate两侧。v1/v2结果不能混用。任务判分与冻结答案没有随修复改变；严格格式失分与达到生成长度上限的情况应单独审阅。

训练也显式固定到同一tokenizer EOS，采样与loss mask使用同一值，并在运行合同和轨迹记录版本。初次9B smoke的上游独立采样配置仍使用旧EOS，使4条回答各多计入2个尾部token；该尝试及其完整checkpoint保留为历史工程记录。评测修复不能追溯改变旧训练目标，修复后的训练必须从初始模型另开运行。

对基准dev/test同样先generate，然后判分：

```bash
python -m rsd.evaluation judge \
  --dataset data/rsd_benchmarks/dev.jsonl --predictions runs/R0-safety-predictions.jsonl \
  --judge-model JUDGE_MODEL --base-url http://localhost:8000/v1 \
  --output runs/R0-judgments.jsonl
python -m rsd.evaluation score \
  --dataset data/rsd_benchmarks/dev.jsonl --predictions runs/R0-safety-predictions.jsonl \
  --judgments runs/R0-judgments.jsonl --output runs/R0-safety-summary.json
```

认证仅传环境变量名 `--api-key-env`。judge是固定自定义pilot二分类rubric，**不是官方StrongREJECT/SORRY指标**。无judge保持UNKNOWN；错误与缺失独立计数，零有效样本为null。判分绑定prompt/response/checkpoint SHA，不执行模型输出里的指令。

```bash
python -m rsd.analysis --base runs/base-summary.json \
  --f0 runs/F0-summary.json --r0 runs/R0-summary.json \
  --f1 runs/F1-summary.json --r1 runs/R1-summary.json \
  --output runs/factorial-analysis.json
```

数据/协议不同拒绝比较，有效ID集合不同则交互为null。各组应使用同一步数；支持显式合同核验，无合同或缺summary与合同绑定时仅为描述性分析。

## DCE风格机制探针

```bash
python -m rsd.reference
python -m rsd.train --config configs/rsd/dce_inspired_probe.json --dry-run
```

这只将合成训练任务的规则答案转成显式reference fixture，未用dev/test；它是answer-only，不能冒充OpenThoughts完整参考推理。正式能力实验需导入可核验的真实 `verified_solution` 训练集。forward KL、参考答案、上下文位置和递归必须分别对照；SRCL后置。

## 当前状态

本地CPU软件、数据契约、基准题目准备、CUDA小Qwen3.5前后向、随机小模型双卡更新/保存/续训，以及修复EOS后的真实Qwen3.5-9B两步R0更新/最终模型导出/重载/评测均已完成。修复后9B抽样覆盖355,840个坐标，其中30,736个发生变化，316/427个可训练张量至少一个抽样位置改变，抽样非有限值为0。四条训练轨迹均在tokenizer EOS结束，没有多计尾部token。训练进程耗时约175秒，含初始化与最终导出，不保存周期checkpoint；该值不能外推为稳态吞吐。

统一v2协议下，dev严格匹配为4/16 → 4/16：金额3/4 → 3/4，集合1/4 → 0/4，时间0/4 → 0/4，单位0/4 → 1/4。生成达到128-token上限的回答为base 2条、R0 1条。基线12条严格失分中，8条为数值错误、2条为计算正确但格式不符、2条为截断后无最终答案；这是诊断，未改变冻结分数。不能把这些小样本变化解释为统计显著、安全正迁移或负迁移。回执见 [`validation/gpu_qwen35_9b_v1.json`](validation/gpu_qwen35_9b_v1.json)。本地全套回归为130项测试和5个子测试通过。

这轮只验证R0（recursive / on-policy / reverse KL / neutral），不是DCE设置的精确复现。真实安全judge、固定教师9B、长输出、9B续训、论文现象和正负安全迁移仍未验证。下一阶段先审查基线能力、输出格式与数据难度并冻结评测，再做同起点F0/R0对照；前提纠正F1/R1作为后续独立因素接入。

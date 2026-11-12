# CUA 防御训练准备与开卡操作

## 研究目标

把一次成功攻击产生的失败轨迹，转换为能训练出安全任务完成能力的数据。第一版比较三种数据来源。

- 自然安全成功轨迹。
- 对攻击失败进行完整重做，得到经过执行验证的安全示范。
- 重放失败前的动作，在首次已观察到的违规动作之前分支修复，只监督修复后的动作。

固定初始模型、任务、攻击池、训练目标和有效监督 token 预算。比较转换产率、生成与验证成本，以及训练后的安全任务完成率。实际攻击成功率、转换成功率和训练收益分别测量。攻击强度与训练价值的关系由实验决定。

## 当前实验范围

CUA-Lite 固定版本提供真实截图、原生动作解析、轨迹日志和任务验证。开发任务包括 Calc 12 个、Writer 12 个，共 24 个独立任务家族。每个任务有 benign、maintenance、prerequisite、goal_correction 四个配对变体。受保护目标是实验容器内的一个合成只读文件。风险变体共享同一用户授权说明和能力验证器。

这组任务用于能力校准、攻击发现和训练数据开发。12 个应用外任务组成固定的中间验证集，139 个其余应用外任务组成最终能力评测候选池。CPU 预检中 138 项通过。Steam 任务 `osworld_chrome_121ba48f` 在两次新容器尝试中均评分超时，单独 HTTPS 探测也超时。GPU 队列已选用 138 项可运行面板，并保留原候选池 139 项及技术排除记录。该项成绩为 UNKNOWN。OS-Harm 的 10 个基础任务、50 个去重注入实例作为外部安全评测协议。外部安全评测依赖合格的独立 judge。数据开发中的文件监控指标有明确的单文件威胁范围。

首次开卡先校准 Qwen3.5-9B 的 GUI 能力。此前 Qwen3-VL-4B 的一次真实 20 步任务失败说明需要校准，单个任务不能用于估计总体能力。9B 使用公共权重。4B 的跨空间临时副本已预置，约 8.28 GiB，供需要时比较。

## 开卡操作

在 GPU 实例终端运行已预置的连接脚本。

```bash
bash /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/codex-workspace/Misalign-agent/.runtime/cua-vlm/qz_cua_gpu_connect.sh
```

在 CPU 实例终端运行以下命令，默认仅生成运行计划。

```bash
cd /inspire/hdd/project/independent-security-research/wenming-253108090054/zjq/Misalign-agent
bash scripts/qz_cua_next.sh
```

明确执行校准时增加 `--run`。默认单卡、30 分钟、最多 24 个任务、每任务 30 步、两个 CPU 环境并发、模型请求串行。时间预算包含模型加载和评测。中断后指定相同输出目录可继续未完成任务。已经有效完成的任务会跳过。

```bash
bash scripts/qz_cua_next.sh --run --output /inspire/hdd/project/independent-security-research/wenming-253108090054/zjq/.runtime/cua-host/calibration-9b
```

CPU 控制器以后台任务运行时，本地电脑断网不会中止远端控制器。应在 QZ CPU 的 tmux 中启动，或使用 nohup 保存输出。模型 worker 设有空闲退出和停止标志，推理与训练均有独立的总时长限制。单次 generation 和容器关闭可能造成少量收尾超时，应查看退出回执。

## 后续数据与训练命令

命令在上述 CPU 项目目录执行，先设置原生源码路径。

```bash
export PYTHONPATH="$PWD:$PWD/references/cua_lite"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
C=/inspire/hdd/project/independent-security-research/wenming-253108090054/zjq/.runtime/cua-host
P="$C/venv/bin/python"
```

固定攻击池收集、完整修复和分支修复分别使用以下入口。`risk-run` 应替换为相应的实际绝对输出目录。

```bash
bash scripts/qz_cua_next.sh --config configs/cua/risk_gpu.json --run --output "$C/risk-run"
bash scripts/qz_cua_next.sh --config configs/cua/risk_gpu.json --run --repair-from "$C/risk-run" --repair-mode full --output "$C/repair-full"
bash scripts/qz_cua_next.sh --config configs/cua/risk_gpu.json --run --repair-from "$C/risk-run" --repair-mode prefix --output "$C/repair-prefix"
```

分支修复会检查可见界面和任务文件是否匹配。重放不一致的样本保留失败记录，不能当作有效分支。它检查的状态范围已经写入回执，无法证明完整虚拟机状态相同。完整修复可在分支修复失败时独立运行。

自动攻击搜索复用同一个冻结模型，由不同角色提示生成候选，再执行防御轨迹。入口为 `scripts/qz_cua_search_attacks.py`。搜索要求已有真实、干净任务成功记录。第一版攻击者权重固定。正式估计攻击强度应使用独立重复执行，搜索次数不能直接代替独立测量。`configs/cua/strength_gpu.json` 已准备温度 0.7、每条件 4 次的探测配置，需报告相应不确定性，并将独立强度测量与数据生成抽样区分。

SFT 导出只接受真实模型、有效执行、任务成功且风险监控完整的安全轨迹。诊断轨迹和评测轨迹会被拒绝。

```bash
"$P" scripts/qz_cua_export_sft.py --batch "$C/repair-full" --model qwen35_9b --output "$C/train-full.parquet"
"$P" scripts/qz_cua_train_session.py --data "$C/train-full.parquet" --eval-config configs/cua/milestone_gpu.json --output "$C/train-full-session"
```

后一条默认仅做 CPU 计划与数据检查。实际训练增加 `--run`，并根据校准实测指定 `--minutes`、`--steps` 和 `--interval`。默认参数用于管线联测，正式比较的步数、token 预算和重复种子需要基于实际样本数与吞吐确定。

训练实现使用 CUA-Lite 原生 SFT 导出，配合现有 GPU 环境中的 HF/PEFT LoRA。第一版单卡、冻结视觉编码器、语言投影与 MLP LoRA、microbatch 1、梯度累积 8、初始学习率 1e-5。实际显存与吞吐尚需上卡验证。它复用原生数据和 agent 行为路径，训练执行器使用 HF Trainer。

训练进程在指定节点临时服务 CPU 环境，保存中间评测结果。正常结束后保存一次最终 adapter，执行保存后重载检查，写入文件哈希并清理 adapter。训练异常时保留日志与回执，确认进程退出后也会为残留 adapter 记录哈希并清理权重。长期保留数据、结果、配方和随机种子，不保留周期优化器权重。恢复采用固定配方重跑，无法从任意优化器步精确续训。

## 正式比较应当怎样扩大

校准之后，先测量真实安全示范的产率，再确定训练规模。至少比较未训练模型、自然成功数据、完整修复数据、分支修复数据。按照同一基础任务、攻击家族和独立测得的攻击强度匹配样本。分别做相同有效训练 token 和相同生成验证成本的比较。

每个训练条件使用多个种子。评测以任务家族为独立单位进行配对统计，报告置信区间。风险未知、环境失败、截断和拒绝分别保留。已经发生的违规在后续恢复后仍算违规。中间验证集用于迭代，最终评测集保持冻结。数据开发阶段的示例和微型训练通过情况不能代替训练收益。

## 保存与核验

CPU 部署、任务可执行性、事件监控、模型资产、微型训练和原生批处理检查的机器可读记录存于 `validation/cua_cpu_preparation_v1.json`。原始日志与完整轨迹存于 QZ `.runtime/cua-host`，源码与小型回执提交 GitHub。大权重、Docker 镜像、原始截图和完整 benchmark 资产不进入 Git。

## 外部安全评测

`configs/cua/os_harm_gpu.json` 使用固定的 50 个去重注入实例。该路径复用原生 agent、日志与截图，将源 benchmark 的注入设置装入 CPU Docker 环境。源码原有的占位判分保留在 `native_placeholder_score`，正式能力分与安全分等待独立 judge。

```bash
bash scripts/qz_cua_next.sh --config configs/cua/os_harm_gpu.json --run --output "$C/os-harm-run"
"$P" scripts/qz_cua_judge_export.py --episode /absolute/path/to/episode --output /absolute/path/to/judge-input
```

第二条默认只构造并检查官方截图 judge 的输入，不调用模型服务。现有导出包含模型可见的动作前截图，未包含 accessibility tree。需在明确这项协议差异后校准 judge，才能报告外部安全结果。`--judge-model` 与 `--run` 用于后续实际判分，服务认证通过环境变量提供。

适配中的网页使用容器内 HTTPS 服务和本地域名解析规则，并检查 Chrome 实际渲染的注入文本。VS Code 注入显式使用原应用打开文件。移植同时补上源配置中的 `upload_file` 资源上传，将通知标题和正文按照源实现分开传入，并将合成 SSH 压缩包解到实际目标路径。SSH 资产的解包层级经过修正，邮件向量保留已注入消息，避免后续重复初始化覆盖。上述差异与截图观察协议均应随外部评测报告披露。镜像身份、源提交和 mkcert 二进制哈希见 `configs/cua/os_harm_runtime.lock.json`。已观察到四环境并发叠加其他桌面作业时出现进程创建失败，当前配置使用两个环境并发。

## 最终 CPU 验收记录

25 项 CUA 相关检查通过。24 个原任务及 96 个配对变体通过原生 oracle 检查，配对变体首次 2 次技术失败在相同配置重试后通过。OS-Harm 50 个去重实例完成环境设置检查，22 个受适配修正影响的实例另行复测通过。12 个中间验证任务全部通过原生接口检查，139 个最终候选任务中 138 个通过。上述中间与最终任务的空操作控制均未被判为任务成功。

两种模型架构均完成随机微型多模态 LoRA 的 CPU 更新与 adapter 保存、重载检查。真实模型的 GPU 显存、吞吐、GUI 能力、安全轨迹产率、训练收益和 judge 资格仍需在后续实际执行中确认。独立 judge 尚未校准，外部安全分保持 UNKNOWN。

本机曾尝试仓库全量 67 项测试，出现 5 个旧 RSD 或资产测试错误。本轮验收采用 QZ 上的 CUA 相关 25 项测试，旧路线错误保留在回执中。

## 2026-11-12 评审服务联测更新

用户确认首次 GPU 可执行性校准限时 30 分钟。主评审候选固定为 `gpt-5.4-2026-03-05`，论文对照为 `gpt-4.1-2025-04-14`。OpenAI 已公告 GPT-4.1 在 2027-01-14 下线。模型选择依据与冻结配置见 `configs/cua/judge.json`。

两个模型的账号权限查询成功。QZ CPU 经已有本机代理完成一次 GPT-5.4 图像判分，返回固定版本、完整 JSON schema 字段及用量。控制样本为立即结束任务，预期任务失败、安全，结果吻合。输入 3563 token，输出 109 token，按查询日标准标价估算约 0.01054 美元。回执见 `validation/cua_judge_api_smoke_v1.json`。

当前接口复用上游 v3 提示，采用 screenshot/default/v3 与 JSON schema。上游示例采用 aer/v3。正式实验需在包含安全完成、违规、过度拒绝及普通失败的固定控制面板上校准 judge。单条控制联测验证了服务和格式，评审资格仍待校准。

QZ 直连 OpenAI 超时，本次通过已有 SSH 反向代理调用。该代理依赖本机网络与连接存活。API 评审可独立于 QZ GPU 运行。密钥通过当前进程接收，后续执行读取 `OPENAI_API_KEY` 环境变量。

参考来源见 [GPT-5.4 文档](https://developers.openai.com/api/docs/models/gpt-5.4)、[OpenAI 下线公告](https://developers.openai.com/api/docs/deprecations)、[OS-Harm 官方评审配置](https://github.com/tml-epfl/os-harm)。

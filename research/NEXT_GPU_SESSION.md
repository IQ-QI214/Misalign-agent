# 下一次开卡先做什么

2026-11-13 自主结果更新。v3 六条已结束，Calc 两个修复组均为 1，Writer 两个有效组为 0、反馈组为 UNKNOWN，详见 `validation/cua_protocol_gpu_v3.json`。正在原 30 分钟预算剩余时间内测试一次 Writer 近期反馈位置对照。

2026-11-13 正常入口更新。v3 的六条 QZ CPU 验收已完成，完整修复配置下 Writer、Calc 均为 1，SFT 格式导出共 35 个已完成步骤，诊断训练资格为 false。结果见 `validation/cua_execution_pipeline_v3.json`。用户已恢复单卡 GPU 中转，正在运行 `configs/cua/protocol_diagnostic_v3.json` 的六条自主评估，整批上限 30 分钟，输出为 `.runtime/cua-host/protocol-v3-gpu-20261113`。完成后需核对原生评分、动作错误、候选计数和 worker 退出。

2026-11-13 执行级复查更新。Writer 增加正文点击后，原五条回复重放评分从 0 变为 1，等待对照仍为 0。Calc 原 30 条回复保持一致，补上真实 Shift 点击事件后评分变为 1。另一个受支持键盘选区对照也为 1。上述均为 CPU 脚本对照，六条真实模型自主结果仍为 0。具体证据见 `validation/cua_failure_rootcause_v1.json`。下一次开卡前先在正常批处理入口完成动作语义修复的 CPU 联测，再做限时干净任务自主执行验证。

2026-11-13 更新。CUA 的三组单卡诊断已完成，两个任务共六条有效评分均为 0。原配置保存在 `configs/cua/protocol_diagnostic.json`。新增的动作参数兼容性检查配置为 `configs/cua/protocol_diagnostic_v2.json`，CPU 原生验证已通过，真实模型效果待新的 GPU 实测。结果与操作见 [CUA 防御训练准备与开卡操作](CUA_CPU_PREPARATION_20261112.md)。下文保留文本 RSD 路线的历史恢复配置。

2026-11-10 优先级更新：用户提出转向 CUA 攻击经验到防御训练的数据与算法联合设计。当前先完成[新方向研究与数据可行性设计](CUA_ATTACK_TO_DEFENSE_20261110.md)，不自动执行下面的文本 RSD 基线或六组训练。本文保留为原文本路线的恢复手册；当时的 CUA 开卡配置尚未建立。当前准备状态见顶部更新，真实 GPU 容量仍待联测。

2026-11-07。GPU仍由用户关闭。本文件准备未来执行顺序，不启动实例、连接远端或执行模型。正式六组训练配置继续禁用。

## 本轮要解决的开发问题

先用同一固定Qwen3.5-9B基座回答完整366题dev：128 GSM8K、128 ARC-Challenge、65 StrongREJECT、45 XSTest safe。得到模型在这套题上的基线能力、FINAL格式成功率、有效回答/截断/错误分母和长度分布。它只做推理，不执行optimizer更新、教师前向或训练权重保存；安全回答先保留原文，judge另行校准。

这一步可以判断题目是否过易、回答是否经常被512 token上限截断、格式错误是否主导UNKNOWN。它不能回答递归教师是否引起漂移，也不能估计尚未运行的F0/R0之间的配对差分方差。不要把单个基座的分数分布用作两个未来训练臂的差分方差。

当前实际准备包为 `.runtime/dev_baseline_20261107/prepared_v1`。包中只有dev输入、dev能力金标准、配置和既有模型来源回执的快照；没有模型权重。输入与金标准分开，生成后端只接收题面字符串。检查包不需要GPU：

```bash
python -m rsd.dev_baseline check \
  --bundle .runtime/dev_baseline_20261107/prepared_v1
```

如需重新准备，使用新目录：

```bash
python -m rsd.dev_baseline prepare \
  --plan configs/rsd/exploration_v2.plan.json \
  --output .runtime/dev_baseline_new/prepared
```

生成包绑定的是文件内容，输入快照使用包内相对位置；可以复制到QZ项目目录后校验。代码、文件或模型身份不匹配时停止，不能只修改SHA让旧结果看起来仍然有效。

## 接通实例后先核对资源与资产

只能在已授权的项目目录下写运行记录、环境与缓存；公共模型目录只读。仍优先复用公共Qwen资产。当前公共资产是否可见、模型字节是否与旧来源回执一致、共享盘剩余配额、容器RAM和每卡空闲显存，都需要下一次现场核验；本轮没有读取远端状态。

已有环境检查入口 `scripts/rsd_gpu_preflight.py` 可在实例中收集系统、包、磁盘与cgroup信息；新输出目录应在本项目内。不要先运行会启动训练的`rsd_launch.sh`或旧两步训练调度器来代替基线。实际模型入口将在加载前核对本地权重、config、tokenizer与chat template哈希；不同文件名不替代内容身份。

未来GPU已可用、资产与目录确认后，单独执行以下入口。这里的占位路径必须替换为实际核验过的本地模型目录；本轮未执行：

```bash
python -m rsd.dev_baseline execute \
  --bundle .runtime/dev_baseline_20261107/prepared_v1 \
  --model-path /absolute/path/to/verified/Qwen3.5-9B \
  --device cuda:0 \
  --output runs/dev-baseline-new-attempt \
  --enable-execution
```

执行默认为关闭，缺少显式开关会在加载模型前拒绝。研究运行限定固定基座、greedy、thinking关闭、prompt2048/输出512、seed0、BF16；不使用Hub下载或CPU预训练模型回退。单卡基线的长输出容量尚未实测，不能因为历史两卡训练成功就认为这个入口已通过GPU验收。

生成顺序固定为完整能力dev后完整安全dev。失败、空答、截断和未生成都保留原分母；技术异常后保留已完成结果和剩余槽位，不按分数挑题或静默重新生成。基线结果不得替代六组正式run各自的step0。

## 后续开发步骤各自有什么门槛

| 步骤 | 依赖 | 产物与限制 |
| --- | --- | --- |
| 完整dev基线 | 固定基座与实例可用 | 366题原始回答、能力评分及长度/格式诊断；无训练结果 |
| B类96回答 | 两个生成器的模型/tokenizer/生成身份在生成前冻结 | 全部请求与attempt保留；完整dev基线不自动抵扣B调用 |
| 旧16控制回归 | 固定Gemma资产，v2/v3 profile独立 | 每profile两轮共64次；通过仍不等于独立judge验证通过 |
| 独立judge验证 | 完整材料、独立人工gold/UNKNOWN、候选与分析规则冻结 | 200条×两个profile×两轮=800次；发生调prompt则原面板转开发用途 |
| 正式研究冻结 | 科学目标、适用的误差/方差依据和最终共同配方 | 不能按确认集显著性挑目标或修改实验 |
| 六组训练 | 资源可行且所有冻结身份匹配 | 固定节点保存回答，全部run完整后才解封确认评分 |

Gemma历史验证使用prompt8192、输出512；当前底层CLI默认输出256。未来若沿用512，必须显式传`--max-prompt-tokens 8192 --max-new-tokens 512`并记录协议，不能依赖默认值。新的独立judge验证尚未执行；当前不指定“可靠judge已通过”。

## 怎样使用功效和成本结果

先确定科学上值得识别的变化，再检查现有题量和测量误差是否支持它。`research/DETECTABILITY_AND_ERROR_BUDGET.md`给出假设场景及可复算程序；不会替我们选定实际效应或填入真实方差。205题的二值例子不等于连续StrongREJECT主指标的功效，更不等于三seed平均后bootstrap的校准结果。

若目标是5个百分点量级，题量、题对分歧、judge偏差和UNKNOWN都可能限制解释。应在确认揭盲前记录如何补充开发证据或独立题源；不因为结果不显著而追加同一确认实验样本，也不把调大的训练步数当作更多独立测量。

六run正式预算是52,050次学生生成（其中24,576训练采样、27,474评测回答）和12,114次judge；基线、B及judge开发验证另计。原始记录存储情景和GPU显存/容器RAM/共享盘的区别见`research/WORKLOAD_ACCOUNTING_V2.md`。实际长输出吞吐与GPU小时仍UNKNOWN，不能把旧两步含加载的总用时线性放大。

第一阶段完成后应保留：原始回答与终止原因、能力分母及格式结果、每域长度分布、模型/tokenizer/代码/输入哈希和失败记录。下一步是否进入训练由预先声明的方案与测量准备情况决定，不把基线高低本身当作递归失配现象。

## 本轮CPU验收

2026-11-07整套CPU回归通过747项测试及52项子测试，耗时41.30秒；新增基线入口和情景计算分别覆盖43项、41项测试。judge方案通过内嵌Draft 2020-12 schema校验。366题实际准备包已重新加载核验，默认关闭的执行入口在模型加载前拒绝；实际基线回答和新judge调用均为0。脱敏汇总及文件身份见`validation/next_stage_cpu_v1.json`。这些验收不证明CUDA容量、预训练模型行为或迁移效果。

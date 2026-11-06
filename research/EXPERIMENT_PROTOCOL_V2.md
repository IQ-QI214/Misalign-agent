# EXP V2：递归教师会怎样改变能力与安全迁移？

状态：2026-11-07，CPU 训练/评测/封存流程已接通并验收；GPU 仍关闭。**这是独立适配的实验协议草案，不是已经启动或完成的9B研究。** 数据准备回执与机器计划分别为 `validation/reasoning_data_v2.json`、`configs/rsd/exploration_v2.plan.json`；本轮工程验收见 `research/CPU_PIPELINE_V2_ACCEPTANCE.md`。

## 先用一个例子说明研究目标

假设模型学习数学和科学问答后，数学正确率从60%升到70%，而对有害请求的依从程度也升高：这才是我们想检查的“能力学习伴随负向安全迁移”。如果能力提升且有害依从降低，则是正向安全迁移的候选证据。若只是对更多正常问题拒答，也不能当作安全进步；因此要另外测误拒。这里的数字仅是解释例子，项目尚无这样的实测结果。

本轮具体问题是：**在同一模型、教育问答数据和蒸馏配方下，仅让教师从固定初始模型变成随学生更新，是否改变安全、能力与误拒的轨迹？** 允许改善、恶化和当前精度下无法区分三种结果。

我们分开回答两个问题：各臂相对自身 step 0 是否发生迁移；R0 相对 F0 的迁移是否不同。后一项是控制教师更新方式的核心对照。R0 最后比 F0 差，不能单独证明 R0 比训练前差；两臂都可能改善，只是改善程度不同。

## 两臂到底有什么不同

| 组成 | F0：固定教师 | R0：递归教师 |
| --- | --- | --- |
| 学生起点 | 固定 revision 的 Qwen3.5-9B | 相同 |
| 第 t 次更新所用教师 | 初始参数 θ₀ | 本次 optimizer 更新前的学生参数 θₜ |
| 学生采样 | 当前学生按 on-policy 生成 | 相同规则，各自学生产生轨迹 |
| 教师输入 | 相同题面 + 固定中性提示 | 相同 |
| 学习目标 | 同一生成 token 上的 reverse KL | 相同 |
| 教师反向传播 | 无 | 无 |

固定中性提示为：`Please use the information provided in the question and follow the requested answer format.` 它明确写入每一条训练记录；学生不见该提示，师生都不见标准答案或 GSM8K 的解题链。新英文提示与旧中文 pilot 的提示不同，因此旧两步结果不纳入新配方的研究对照。

这研究的是一般上下文提示下的教师更新效应。中性提示可能不足以提供显著学习信号；如果最终没有能力收益，应如实说明，不能只根据训练 loss 下降称为自我提升。带标准解答的 DCE、错误前提纠正、SFT/DPO/GRPO/PPO 属于后续单独的因素，不在这次 F0/R0 比较中同时更换。

## 已实际准备的数据

| 用途 | GSM8K main | ARC-Challenge | 合计 | 原生来源约束 |
| --- | ---: | ---: | ---: | --- |
| train | 512 | 512 | 1,024 | 两者 native train |
| dev | 128 | 128 | 256 | GSM native train 独立留出；ARC native validation |
| confirmation test | 256 | 256 | 512 | 两者 native test |

两个数据源的固定版本、原始文件 SHA-256、授权、筛查规则、题目排序和配额写入 `configs/rsd/reasoning_sources_v2.lock.json`。GSM8K 采用 `main`，不将同题的 `socratic` 版本当作新样本；ARC 原有数字/字母标签和3–5个选项保持不变。[GSM8K 官方数据卡](https://huggingface.co/datasets/openai/gsm8k/blob/740312add88f781978c0658806c59bc2815b9866/README.md)、[ARC 官方数据卡](https://huggingface.co/datasets/allenai/ai2_arc/blob/210d026faf9955653af8916fad021475a3f00453/README.md)。

准备规则先于模型结果：标准化题干重复整组隔离；题干词集合 Jaccard ≥0.85 的题目两端隔离，包括同一个 native file；统一筛除可能缺失图示/表格以及保守主题词命中的记录；随后按固定哈希排序抽取。配额不足会报错，不从确认集补训练。规则去重只能保证指定判据下不重叠，不能证明所有题在语义或统计上独立。公开基准是否已进入 Qwen 的预训练数据仍是 UNKNOWN。

原始数据与派生文件均留在忽略目录，公开 Git 只提交来源锁、构建程序、脱敏回执和文档。`train.prompts.jsonl` 不含答案；`dev.prompts.jsonl` 与 `dev.gold.jsonl` 分离；确认题面与标签位于 `sealed/test.*.jsonl`。`sealed/` 是程序与工作流隔离，不是加密保险箱。六 run 解封控制器现已CPU验收：准备/训练callback不打开test gold，解封核对全部run、节点和文件身份，评分时再次核验。

“来自教育问答并经规则筛查”是当前可证实的描述。未完成独立人工良性认证；关键词筛查不能证明每题无害。结论限于这套过滤后的数学/科学任务，不能称为广泛 agent 能力，也不是完整 GSM8K/ARC 官方榜单成绩。

安全与误拒保留现有冻结分割，不加入训练：StrongREJECT dev65/test248；XSTest safe dev45/test205。合计110条dev、453条确认题。1024条新训练题与完整563条安全题做标准化题干精确交叉检查；检查的具体结果与哈希以数据回执为准。

## 训练剂量、节点与成本

固定三个训练 seed：0、1、2。每个 seed 的 F0/R0 从同一基座开始，使用相同数据排列种子和实现。预定执行顺序为 F0-s0、R0-s0、R0-s1、F0-s1、F0-s2、R0-s2；完整六 run 才构成计划对照，失败/中断仍保留自己的运行身份，不能按分数决定补哪个 seed。

| 项目 | 候选配置 |
| --- | --- |
| 硬件 | 2×H100 80GB；训练时 ZeRO-2 与 CPU optimizer offload |
| 有效 batch | 2 卡 × microbatch1 × accumulation4 = 8 |
| optimizer 更新 | 512步；4,096次曝光，约4个 epoch |
| 学习率 | constant 3e-6，clip30；其余优化器设置两臂共用 |
| 生成 | thinking关闭；训练 temperature1.0、上限256；评测 greedy、上限512 |
| prompt 上限 | 训练1024、评测2048；拒绝静默截断 |
| dev节点 | 0、32、64、128、256、512 |
| 确认安全节点 | 0、256、512，每点248 StrongREJECT +205 XSTest |
| 确认能力节点 | 0、512，每点512题；两个任务域分别报告 |
| 权重保存 | 不永久保存中间/最终训练权重；驻留模型先生成记录，再结束训练 |

这里匹配的是样本曝光、optimizer步数和最大生成预算。两臂实际输出长度可能不同，实际训练token与计算时间可能不同，必须逐臂报告；不能同时声称已经实现等token或等计算量比较。四个 epoch 是起始剂量，不是收敛保证。

梯度累积4已接入并用单进程、双进程CPU/Gloo真实tiny更新验收：同一optimizer window的4个microstep使用同一教师参数；仅在optimizer更新后R0教师身份前移；F0保持不变。loss是每completion内有效token KL均值，再对microsteps/ranks等权平均，不是将整个窗口所有token等权混合。框架只缩放一次，不完整窗口拒绝执行；实际CUDA/DeepSpeed仍待验收。

计划合计27,474次学生生成、12,114次主实验安全judge调用，另有judge验证预算。生成次数不是独立题数。首个计划 run 的前32步作为正常训练的一部分，分开记录预热、采样、前后向、optimizer与评测耗时，再估计剩余时间。此前551.53秒含加载与多次评测，不能线性乘256预测新训练成本。

2×H100 的两步容量已验证；较长生成与累积4的容量仍未验证。Qwen/tokenizer 的 CPU 输入长度审核不加载权重，也不能证明 GPU 运行可行。Gemma3-27B 在训练退出后顺序评审；当前没有在线安全判分停止器。技术失败、非有限loss、OOM、身份不一致、记录不完整属于明确停止原因，不能删除已经完成的坏节点。

## 判分与正负迁移如何表述

能力协议 `final-answer-v1` 只抽取最后独立一行 `FINAL: ...`；数值题另接受整个回答就是数值的情形。数值采用有理数精确比较，选择题用原始标签；不从推理正文寻找恰好等于标准答案的数。空答、截断、歧义和格式错误记UNKNOWN，单列有效准确率、格式率和包含UNKNOWN的全题上下界。这些上下界不是置信区间。模块默认拒绝 test 评分。

主要终点固定为 **step512 的 StrongREJECT 连续分数 R0−F0**，高分表示更强的有害依从。与各自step0的变化、XSTest误拒、GSM/ARC能力、step256和曲线都是次要结果。能力两域先分别报告，等权汇总仅作补充，避免一个域的收益遮盖另一个域的退化。

“正/负迁移”至少应同时展示：能力变化、恶性依从变化、误拒变化及各自不确定性。安全分下降但误拒上升，可能是普遍拒答；安全分上升但能力未提高，则不能包装成“能力提升引发安全代价”。主对照只识别这套配方内教师更新方式的影响，不证明所有递归蒸馏或所有模型都有同一现象。

逐题、逐seed配对；同一题的多个变体绑定为source cluster，跨seed同题不增加独立题数。输出给定这三个训练run的条件性区间，不能精确估计所有训练随机性的分布。UNKNOWN保留原分母；全零差异不宣称总体零不确定性。新能力评分已接入驻留callback及`rsd.task_analysis`，从绑定身份的原始回答重评分，再复用配对统计；确认分析CLI先重新核验六run解封。详见 `research/ANALYSIS_CONTRACT_V2.md`。

不基于多个节点/指标中最小p值下结论；若增加正式次要检验，先固定检验族与Holm规则。当前规划主要针对较明显效应；205道误拒题不能保证识别5个百分点差异。连续安全分与二值误拒的最小有意义效应需要分别在开发阶段、确认揭盲前冻结，不能互用同一个数。

## 评审可靠性与确认集封存

当前v2 judge仍未通过有限控制验证。v3第一候选已经独立实现，只改XSTest输入边界说明，不同时改rubric、模型和parser；尚未调用真实judge。独立验证计划200条记录、120个source簇；两轮、两个profile及旧控制共864次judge调用。A层64条构造候选已准备，B层48题已选、96条回答仍待生成；另已获得C层40条公开参考候选，其题目适用性待独立盲审，不能因原始偏好标签或来源名称认定rubric gold。**人工gold仍为0，完整验证未完成**。详见 `research/JUDGE_VALIDATION_V3.md`。

训练过程中只开放dev诊断；确认节点的原始回答可以生成并封存，但能力和安全分数均在六个预定run结束、文件完整性与冻结身份核验后统一开启。缺run或技术失败时保持确认结果未解封，先记录缺失与恢复决策。若观察确认结果后修改方法，原确认材料转为探索，需新的独立确认来源。

更改学习率、数据、cue、judge、评分器或节点都构成方案版本变化，不能让F0/R0实际运行不同配方却仍放入同一主比较。代码版本、来源数据、预定节点及最终共享配方SHA在正式启动前锁定。

## 当前交付与下次开卡条件

本轮已接通梯度累积4、每步剂量/耗时记录、新能力评分callback、确认封存/解封、释放后的能力配对分析与judge输入导出。单/双进程CPU梯度参照及六组随机tiny模型完整流程均已执行；没有Qwen/Gemma预训练模型调用或GPU训练，没有新的研究行为结论。

六个9B配置可由`python -m rsd.experiment --output <新目录>`生成并CPU检查，均显式`training_enabled=false`，确认plan保持draft。手动打开该布尔值仍不能绕过未冻结计划；CPU fixture不能从GPU入口运行。正式冻结需要绑定judge独立验证回执、生成/评分/训练代码和分析/MDE；不能把工程测试回执当作校准证据。

剩余工作是人工盲审/确认C适用性、选择并锁定B生成器、GPU上完整dev基线/B真实回答生成和judge验证、真实CUDA/DeepSpeed长输出容量与吞吐测量。完成开发校准并冻结正式配方后才执行9B六run。GPU当前无需为了这些已完成的CPU工程工作保持开启。

复建数据仅需CPU与PyArrow（`configs/rsd/requirements-data.txt`）；tokenizer审核另用现有Transformers环境。已下载缓存可离线复建：

```bash
python -m rsd.reasoning_data \
  --benchmark-dataset data/rsd_benchmarks/dev.jsonl \
  --benchmark-dataset data/rsd_benchmarks/test.jsonl
# 缓存不存在时显式追加 --download；SHA不一致会拒绝。
python scripts/rsd_reasoning_token_audit.py \
  --tokenizer .runtime/qwen35_tokenizer \
  --download-manifest .runtime/qwen35_tokenizer/download_manifest.json
```

重复构建不得覆盖不同内容；修改数据规则要另留旧尝试记录并重新冻结身份。来源原始授权不因项目准备而改变，ARC派生内容继续按CC-BY-SA-4.0处理，GSM8K保留MIT归属。

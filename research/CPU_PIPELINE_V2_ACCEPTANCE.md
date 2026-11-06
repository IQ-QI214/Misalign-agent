# EXP V2 CPU 工程验收与下一次开卡边界

日期：2026-11-07。GPU由用户关闭，本轮没有连接远端或GPU，没有加载Qwen/Gemma预训练权重，没有执行真实judge评分。随机tiny模型的CPU训练与生成用于检验实现，不能替代9B研究实验。

## 已完成的完整路径

```text
固定训练题面与teacher cue
  → micro1 × accumulation4（完整窗口）
  → F0固定教师 / R0在optimizer更新后推进
  → 每步曝光、token、分阶段耗时记录
  → 预定节点驻留生成
       ├─ dev：能力评分 + 安全原始回答待judge
       └─ test：只保存题面/回答，保持未评分
  → 六个不同run与全部节点/文件/代码身份核验
  → 显式release回执
  → 重新核验release后：能力评分/配对分析，或导出安全judge输入
```

`rsd.train`按`milestone_eval.profile="reasoning-v2"`接入新路径，旧dev-only callback保留。模型输入不含能力gold；训练准备与callback不打开test gold。未知、空答、截断保留完整分母。失败生成即使残留合法`FINAL:`也不能计为正确答案。

## 累积梯度与真实训练验证

损失定义为每条completion中有效token的KL均值，再平均microsteps和distributed ranks。不同长度completion等权；不声称整个窗口的每个token等权。Transformers负责一次GAS缩放，适配器不再次缩放；拒绝不完整窗口以及会产生distributed填充/丢行的训练行数。

CPU/Gloo双进程分别对F0/R0执行两个optimizer step：micro1×GAS4×world2=有效batch8，每臂16次样本曝光、44个有效训练token。与独立手工全局8-completion平均梯度/更新参照相比，最大参数误差为F0 `1.862645149230957e-9`、R0 `3.725290298461914e-9`，两个rank结果一致。单进程还逐元素检查同一窗口教师不变、下一窗口R0更新而F0不变。证据：`validation/accumulation_cpu_v2.json`。

原始IPv6 rendezvous无法启动worker的尝试和外层CPU设备字符串断言失败保留在回执；改用显式本地IPv4后完成验证。它们不是GPU失败或研究训练结果。

`scripts/rsd_cpu_v2_acceptance.py`随后调用真实`train.run(use_cpu=True)`完成F0/R0×3个seed的六run集成。每run使用随机一层tiny GPT2、两个optimizer step、单进程GAS4，共12次optimizer更新、48次样本曝光；step0/1/2做dev，step0/2封存确认回答。所有学生都有实际参数变化，三个F0教师探针不变。六run结束后解封、执行能力评分与配对分析，并导出一个绑定身份的judge输入包；没有运行judge。

这项集成的有效batch为4；双rank有效batch8由上面的独立梯度参照覆盖。两步CPU测试用于逐次比对更新，正式9B候选预算仍为512步。证据：`validation/cpu_v2_pipeline_v2.json`。私有原始记录位于`.runtime/cpu_v2_pipeline_20261107_attempt1/`，只有未训练tiny输入资产保留，未导出训练后权重或optimizer checkpoint。

## 确认集控制的具体边界

`rsd.confirmation`核对六个唯一run的arm/seed、实际world size/effective batch、共享训练配方、训练文件、全部计划节点、逐条预测、停止原因、checkpoint身份与代码SHA。完成结果也绑定run与plan，不能复制一个result凑六组。缺失/失败/部分节点、重放、文件改动、符号链接或封存目录出现提前评分文件都会拒绝。atomic create防止覆盖已有证据，失败记录保留。

评分和分析入口每次重新检查release及其绑定的文件。能力使用新final-answer协议；安全评分还核对冻结judge profile、原始judge文本与parser/分数的一致性。底层Python函数的显式test开关不等于release授权，正式流程使用受控入口。

这是工作流完整性控制，不是加密或操作系统访问控制。代码不能证明人工标注者现实中独立，也不能证明某人此前没有私下读取数据。research冻结要求绑定独立judge验证回执与预声明分析/MDE；`purpose="cpu_fixture"`只能在CPU入口执行，所有fixture的`scientific_ready`保持false。

## 可执行的CPU入口

本地环境需要项目锁定依赖；只准备数据另见`requirements-data.txt`。所有输出使用新目录，不覆盖既有记录。

```bash
# 检查真实1024/256/512数据，生成六个training_enabled=false配置。
python -m rsd.experiment --output .runtime/exp-v2-new-preview

# 对其中任一配置做CPU身份/参数检查；不加载模型。
python -m rsd.train \
  --config .runtime/exp-v2-new-preview/exploration-v2-f0-s0.json --dry-run

# 如需重做工程验收，使用随机tiny模型且强制CPU；不是9B训练。
python scripts/rsd_cpu_v2_acceptance.py \
  --output .runtime/cpu-v2-new-attempt \
  --receipt .runtime/cpu-v2-new-attempt-receipt.json
```

材料化配置保持`training_enabled=false`且confirmation plan是draft。单改该布尔值仍会因计划未冻结而在模型加载前拒绝；当前不提供自动把草案升级为正式研究的开关。

`optimizer_rank*.jsonl`保留全局曝光/有效token/唯一source与每rank生成、teacher/student前向、前后向、optimizer时间。时间采用同步边界的wall clock，包含同步/框架开销；不能称纯GPU kernel时间。累计剂量按attempt计，续训需要显式拼接attempt；本轮正式候选依然不保存可续训optimizer。

## 评审材料和仍待完成的研究步骤

v3候选是独立profile，原v2模板锁和旧16控制不变。新A层64条构造回答及两套空白盲审表已准备；B层48个题源已确定，但96条真实回答与两个生成器身份仍待完成。C层另准备40条MIT许可公开参考候选，其原始偏好标签不等于本项目rubric gold，题面适用性须先独立审核。材料中的人工gold为0；真实v3评分调用为0。旧v2校准只读重算仍是32有效/28硬通过。

下一次开卡前可先进行人工盲审和C适用性确认，确定B的生成器与标注安排。下次GPU阶段先做完整dev基线、B真实回答生成和judge校准；配方/评审/分析冻结后再进入9B六run。CUDA/DeepSpeed实际累积、长回答显存、输出长度、预热后吞吐和可检测效应量均未由本轮CPU测试证明。

完整CPU回归为528项测试和52个子测试通过，0失败、0跳过；包含8项外部参考候选测试。judge计划另通过Draft 2020-12 JSON Schema验证，依赖安装在临时隔离目录且已清理，没有改训练环境。回归与集成的总体验收见`validation/cpu_phase_v2_completion.json`；文件哈希属于具体运行时点，后续修改代码应重建计划，不能放松冻结核验来复用旧计划。

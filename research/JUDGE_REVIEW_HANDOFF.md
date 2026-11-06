# Judge 验证的人工交付与后续接入

2026-11-07。本轮只做本地CPU工作，没有连接GPU或远端计算，没有运行预训练模型、judge或外部模型API，也没有联系审阅者。数据处理和离线页面可用不等于人工审核已经完成。

## 当前可以直接使用的材料

| 材料 | 当前数量 | 所处阶段 |
| --- | ---: | --- |
| A构造回答 | 64条、32题源 | 已进入部分验证包，等待回答rubric人工标注 |
| B真实回答 | 0/96条，48题源 | 已导出两套各48题请求；两个生成器身份未填，状态DRAFT |
| C外部参考候选 | 40条、40题源 | 等待仅题面的适用性审核；接纳0条，尚未进入回答验证包 |
| 人工回答金标 | 0条 | 没有用作者假设、模型标签或CPU fixture代替 |

两份本地页面位于 `.runtime/judge_validation_v3/human_review_20261107_v2/reviewer_1.html` 与 `reviewer_2.html`。每份包含64条回答评分和40条C题面适用性问题，顺序分别随机化。页面只嵌入盲审白名单字段，不含作者假设、生成器、arm、checkpoint、来源桶、HH偏好标签或候选judge结果；C适用性阶段没有对应回答。

页面可以直接用本地浏览器打开，不需要GPU、服务端或账户。原始内容和页面留在忽略目录，不推送GitHub。公开仓库保存生成器、来源锁、测试、说明和脱敏回执。

## 两位审阅者如何填写

1. 项目负责人指定两名不同的实际人工审阅者，各自领取对应页面。工具不自动分配或联系人员，也不能证明现实身份与独立性。
2. 先读页面内固定rubric。A回答按XSTest三分类或StrongREJECT三个分项标注；C先只看题面，标BENIGN/HARMFUL/UNSUITABLE/UNKNOWN并写理由。不要按题目来源名称猜测适用性。
3. 无法判断保留UNKNOWN。StrongREJECT只填部分分项时，分项存入独立`draft_components`，正式标签仍UNKNOWN、`components=null`；三项完整后才形成标签。草稿不会自动成为金标。
4. 下载每张表的JSON后再关闭页面。本页不联网、不自动保存。可重新导入自己的草稿；导入核验ID、完整正文和SHA，拒绝不同审阅者混用、外来条目和正文变更，人工声明需要重新勾选。
5. 标注结束时填写稳定审阅者代号，并仅在确实独立人工完成后勾选声明。保留原JSON，不覆盖冻结输入或另一位审阅者的文件。分歧另交第三名实际人工裁决；无法裁决继续UNKNOWN。

只有一名审阅者时，可以保留个人草稿，但不能称双人独立金标。助手整理、CPU测试中的标签和独立LLM建议都不计入实际人工标注。

原A64草稿将来可以导入同一审阅者的完整回答页面：只迁移ID与问答正文完全相同的条目，新B/C题保持UNKNOWN，无需重新抄录旧题。先完成C题面接纳后，再用**回答专用模式**生成含C回答的页面；工具拒绝在同一页同时展示某题的回答与该题的首次题面适用性审核。

## 操作顺序与入口

当前部分面板为 `.runtime/judge_validation_v3/assembled_panel_20261107_v2`，只纳入A64。生成页面的命令如下；输出目录必须是新的：

```bash
python -m rsd.judge_review_workbench \
  --panel-bundle .runtime/judge_validation_v3/assembled_panel_20261107_v2 \
  --reference-bundle .runtime/judge_reference_sources/review_candidates_20261107 \
  --output-dir .runtime/judge_validation_v3/new_review_delivery
```

不传`--reference-bundle`即为回答专用模式，适用于完成C题面接纳后的面板。人工导出的文件不自动放回原模板目录，必须另存为可追溯的提交。后续C接纳命令及裁决表入口见`research/JUDGE_REFERENCE_ADMISSION.md`。

B当前请求为 `.runtime/judge_validation_v3/real_generation_requests_draft_20261107_v2`。两份请求合计96槽，`generator_identities`未填，因此不能导入或宣称有真实回答。将来根据实际资产锁定两个不同checkpoint、tokenizer与完整生成参数，**在生成前创建新的冻结请求包**；不能先生成再挑选身份或回答。外部执行器返回所有96条attempt及执行回执后才可导入。错误、空答、截断、未生成均保留，不能凑成96条完成回答。详细字段与命令见`research/JUDGE_REAL_RESPONSES.md`。

C审核输出和B导入结果经各自loader重新核验后，`rsd.judge_panel`才能合包。即使行数达到200，A/B/C各自metric、source数和成对结构不符时，完整验证入口仍关闭。C超过或不足20+20均报告，不根据judge结果挑选、补齐或删除。原材料格式被固定为原始A64，不能只修改manifest中的数量绕过合包检查。

`rsd.judge_validation freeze-gold`支持新合包。冻结记录保存两人的原始标签及分歧、裁决状态、原文件绝对路径和SHA；**必须保留原人工JSON与裁决JSON**。`verify_frozen_gold`在导出judge输入、文件式分析前重新读取原文件并重放最终标签及分母。移机需要按原目录恢复证据或显式重新生成绑定，不能静默修改路径、标签或计数。这核验的是文件一致性，不是现实身份认证。

## 后续评审诊断与开卡顺序

`rsd.judge_validation_analysis`固定四组顺序为v2轮1、v3轮1、v3轮2、v2轮2。它核对原始judge文本、严格parser、输入和模型身份，分A/B/C及metric报告：

- XSTest三分类混淆和错误率、二值误拒偏差；三分类与二值单列，避免完全拒绝/部分拒绝混淆被掩盖。
- StrongREJECT分项与连续分数的绝对误差和有符号偏差。
- 原始两人标注的一致、混淆、UNKNOWN与裁决比例，不用裁决后的一致率冒充标注可靠性。
- A原答/干扰变体的配对变化，以及每profile两轮稳定性。
- v3−v2差异以source簇为重采样单位，重复轮次不增加独立样本数；没有有效比较时返回null，退化样本不报告虚假的零不确定性。

诊断输出始终不会自动签发`validated_for_exploration`。当前实际输入没有人工gold和judge输出，诊断只反映缺失分母，误差/准确率为null。CPU fixture里模拟的四轮评分只用于测试，不能出现在研究结论中。详见`research/JUDGE_VALIDATION_ANALYSIS.md`。

下一次开卡仍按“完整dev基线与B生成 → judge实际校准 → 分指标效应量/误差敏感性冻结 → CUDA/DeepSpeed长输出容量与吞吐核验 → 六组9B实验”推进。正式训练继续关闭，不自动启动512步实验。

## 验证边界与版本

本轮完整CPU回归为**663项测试和52项子测试通过，0失败、0跳过**；judge计划通过Draft 2020-12 JSON Schema验证，临时依赖已清理，训练环境未改。总回执见`validation/judge_review_pipeline_cpu_v1.json`。离线页面脚本的DOM测试覆盖渲染、明确选择标签、下载/导入、部分草稿、StrongREJECT续填及正文改动拒绝；调用真实浏览器预览超时，因此没有把浏览器视觉检查报告为通过。临时本机测试服务器已停止。

上一批`validation/cpu_phase_v2_completion.json`保留原时点代码哈希，不修改历史回执。本轮改变的是评审材料/金标/分析代码；没有修改已验证的训练、milestone、confirmation runtime。早期review页面v1和B请求v1保留为准备历史，当前使用本文列出的v2目录。

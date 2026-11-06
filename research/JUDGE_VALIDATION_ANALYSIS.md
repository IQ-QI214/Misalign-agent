# Judge 验证诊断分析

`rsd.judge_validation_analysis` 只分析已冻结材料与已保存的判断记录，不生成回答、不调用模型，也不自动将 judge 标记为可用。CPU 测试中的标注和 judge 输出均为合成 fixture，不是人工金标或真实模型表现。

入口 `analyze_validation(candidates, frozen_gold, sessions, *, bundle_manifest_sha256, human_gold_file_sha256=None)` 接受已经通过 bundle loader 的规范候选行。四个 session 必须依次为 v2 round 1、v3 round 1、v3 round 2、v2 round 2，每项含 `profile_name`、`round`、`binding`、`manifest`、`judgments`。`binding` 是 `judge_validation.bound_judge_inputs` 的完整输出；manifest 与 JSONL records 使用 `local_judge` 原始协议。未执行的 session 填 `binding=None, manifest=None, judgments=[]`。没有冻结金标时，`frozen_gold=None` 且不能导入 judge 判断。

输入核对涵盖 source 的 stratum/metric/prompt 一致性，候选及金标的完整身份，原始输入导出，完整 profile/source lock/code SHA，四轮协议顺序，以及所有保存记录的 raw 输出重新解析。v2/v3 只有 name、source lock 和 system prompt 可以变化；model、tokenizer、generation 和 runtime 身份必须相同。检查依赖当前实现的原始 parser 和 `local_judge._validate_saved_judgment`；旧实现不匹配时明确拒绝，不能跳过验证。身份 hash 能证明记录内部一致，不能证明真实执行历史、审阅者身份或执行时间顺序。

每个 stratum 与 metric 单独报告。XSTest 保留三类混淆矩阵，同时单列二值误拒 MAE 与 signed bias；StrongREJECT 分别报告三个分量及连续 score 的 MAE/bias。缺少金标、未判断、无效判断分别计数；没有可比较样本时数值为 `null`。

`human_review` 独立分析冻结记录中按 `reviewer_ids` 顺序保存的 `original_annotations`。保留原始字段缺失数、两人均 LABELED 的配对数、UNKNOWN 分布、XSTest 原始三分类混淆及一致率、StrongREJECT 分量/score 差值和一致率。差值方向为第二人减第一人，不把任一原始审阅者视为金标。裁决数量及比例的分母是有明确 `adjudicated` 状态的记录。原注释存在时会核对状态、取值、`original_reviews_agree`、最终标签与裁决 receipt 的一致性；旧 gold 没有原注释时报告 `UNKNOWN`，不会由最终标签或已有 agreement 布尔值反推原始混淆矩阵。

v3−v2 配对比较先要求同一 source 的全部已准备变体、两种 profile、两轮均有有效判断和已标注 gold，再在 source 内对两轮及变体等权平均误差，最后按 source 等权 bootstrap。XSTest 同时报告 `fields.value` 的二值误差与 `fields.three_class_error.error_rate` 的三分类 0/1 错误差异，后者能发现 full refusal 与 partial refusal 之间被二值指标掩盖的错误。完整 source 的数量是独立样本数，四轮不是四倍样本量。部分缺失 source 整体排除并保留排除分母。常数差值仍报告点估计，但区间为 `null`，避免把退化 bootstrap 写成零总体不确定性。A 的 original/directive 变化与 round 1/2 稳定性单列；变化本身不等同于错误。

命令行使用已验证 bundle 路径，避免单独给候选文件而绕过来源校验：

```sh
python -m rsd.judge_validation_analysis --plan analysis_plan.json --output diagnostics.json
```

plan 含 `bundle_dir`、可选 `frozen_gold`，以及上述四项 `sessions`。每项的 `binding`/`manifest`/`judgments` 是可选文件路径，其中 judgments 是 JSONL；路径相对 plan 文件。可选 `bootstrap_draws`（至少 100，默认 2000）、`analysis_seed`（默认 0）、`alpha`（默认 0.05）。输出禁止覆盖，并记录所有直接输入文件和分析代码 SHA。

报告固定为 `diagnostic_only`、`validated_for_exploration=False`、`reliability=UNKNOWN`、`scientific_conclusion=null`。完成覆盖、真实生成、独立人工复核与 judge 诊断之后，是否允许探索仍需要单独决定；本入口没有正式 release 的权限。

# C候选的人工题面适用性接纳

2026-11-07，CPU模块 `rsd/judge_reference_admission.py` 已实现。它只处理人工声明的题面适用性结果，不运行模型、不产生人工标签、不把HH来源偏好变成rubric金标，也不修改原C候选、A/B包或既有运行协议。

稳定API：

```python
load_reference(reference_bundle)  # -> (reference_manifest, 40 candidate rows)
admit(reference_bundle, review_paths, output_dir, adjudication_path=None)  # -> manifest
load_admission(output_dir)  # -> (manifest, 40 decisions, admitted candidate rows)
adjudication_template(reference_bundle, review_paths)  # -> unfilled unresolved-case form
```

`adjudication_path` 为keyword-only参数。`load_reference` 核验原C40 manifest及六个原始文件、完整题面/回答SHA、标准化题源身份、唯一性以及初始未接纳/未标注状态。目录、文件和路径父级的符号链接都拒绝，需要使用真实路径。manifest文件名不能越出包目录。

## 人工输入与决策

先把原来的两份 `reviewer_1_prompt_eligibility.json`、`reviewer_2_prompt_eligibility.json` **另存为填写副本**；原候选包作为冻结证据保持不变。两位审阅者分别填写自己的非空身份、`reviewer_kind="human"` 与 `independent_annotation_claim=true`，完整保留40条题面，只填写题面类别和理由。原样空白表会在创建任何输出之前拒绝。

允许的题面类别是 `BENIGN / HARMFUL / UNSUITABLE / UNKNOWN`；布尔值、数字、未知枚举、缺失/重复题目、外来题目、正文或SHA不符、夹带回答rubric标签等均拒绝。已知类别需要非空理由。两个reviewer slot和大小写规范后的身份必须不同。程序检查的是身份和独立审阅声明，不是现实身份认证，也不能证明审阅者未接触模型答案或来源元数据。

| 人工结果 | 接纳决定 | 赋予的metric |
| --- | --- | --- |
| 两位一致BENIGN | ADMITTED | overrefusal |
| 两位一致HARMFUL | ADMITTED | safety |
| 两位一致UNSUITABLE | EXCLUDED | null |
| 分歧，或双方均UNKNOWN | UNKNOWN | null |

未知/分歧可以由第三位不同身份、声明独立的人工审阅者裁决。`adjudication_template` 只导出未解决题面，类别全为UNKNOWN；它不代填裁决。裁决可以仍为UNKNOWN，也可以给出其他已知类别；未提供裁决的题保留UNKNOWN。第三人不能覆盖两位已有的明确一致结论。两位原始类别/理由、是否一致、第三人裁决与最终结果均保存。

代码仅按人工题面类别赋metric，完全不读取source bucket、chosen/rejected、回答内容标签或judge分数决定接纳。它保留所有已接纳题，不按20条目标截断。如果40题全部判为BENIGN，会保留40条overrefusal候选并明确报告safety缺20条、overrefusal多20条；不会改标签、换题或默默凑足20+20。

## 输出与可复核边界

输出目录必须全新。源C包的manifest及六文件、两份填写review及可选裁决文件均按原始字节复制到接纳包；源文件不改。输出包括：

- `decisions.json`：完整40条ADMITTED/EXCLUDED/UNKNOWN决定和原始审阅/裁决状态。
- `admitted_candidates.json`：仅已接纳题，保留原完整题面、回答、来源、输入哈希和C层身份，明确metric来自人工题面适用性。
- `reviewer_1.json`、`reviewer_2.json`：调用统一 `judge_validation.blind_sheet` 生成的下一阶段回答rubric空表，作者预期/来源偏好/模型身份等不进入盲审表。
- `manifest.json`：原C包manifest SHA、review/copy文件SHA、完整分母、各状态计数、20+20目标的缺口/超额，以及代码身份。

所有接纳候选的 `human_gold` 仍为null，`human_gold_status="UNKNOWN"`；这里完成的仅是题面适用性，不是回答rubric金标。即使20+20目标恰好达到，也不声明完整200条验证ready或judge可靠性成立。

`load_admission` 不只相信manifest计数。它复验复制的原C包及review文件，从原始人工输入重算40条决定、metric、所有分母和缺口，再逐项对比保存的决定、接纳候选和盲审表。只修改决定或计数，即使重新填写这些文件的SHA，也无法通过与原始review的重放比较。该机制是内容一致性验证，不是对不诚实人工声明或全部证据同时被重新伪造的身份认证。

CLI：

```bash
python -m rsd.judge_reference_admission \
  --reference-bundle .runtime/judge_reference_sources/review_candidates_20261107 \
  --reviews /actual/path/completed-review-1.json /actual/path/completed-review-2.json \
  --output-dir .runtime/judge_reference_sources/new_admission_bundle
# 如有第三人裁决，另加 --adjudications /actual/path/adjudication.json
```

实际原C40包已通过只读加载验证；原样空白表已验证会拒绝，且没有创建接纳目录或人工标签。CPU测试中的人工身份/标注都是明确的synthetic fixture，仅验证工程逻辑，不计入真实人工完成量。

# EXP V2 配对分析契约

`python -m rsd.paired_analysis` 是 CPU 统计工具。它消费既有 `rsd.evaluation.score_records` 的 `rsd-eval-v1` summary，不运行模型、不改变评分语义、不自动判定研究成功。主终点仍是 **step 512、StrongREJECT test、R0 − F0 连续分数差**。其他 step、dev、误拒、任务准确率只输出描述性分析。

2026-11-07新增`rsd.task_analysis`接入能力数据：先核验prompt/gold与逐条run/checkpoint/step/生成协议身份，再从原始回答执行`final-answer-v1`，最后复用本文件的source配对统计。内部适配为旧summary字段结构不表示采用旧whole-response评分。失败生成即使残留合法`FINAL`也保留UNKNOWN；GSM/ARC分别输出。确认分析使用`python -m rsd.task_analysis --plan ... --runs-root ... --release ... --step ... --output ... --root ...`，每次重新验证全部六run的解封证据。底层Python `allow_test`只供受控入口与显式CPU fixture，不是独立解封授权。

## 输入与调用

Python 接口：

```python
from rsd.paired_analysis import analyze_paired
report = analyze_paired(
    frozen_dataset_rows,
    [{"arm": "F0", "seed": 0, "step": 512,
      "run_contract_sha256": "<该次完整运行契约的64位SHA256>",
      "matched_recipe_sha256": "<两臂共享训练数据、cue、配置等字段的64位SHA256>",
      "summary": existing_score_records_summary}, ...],
    expected_seeds=(0, 1, 2), step=512,
    bootstrap_draws=2000, random_seed=0, alpha=0.05,
    allow_test=False,
)
```

`matched_recipe_sha256` 可以省略，但此时训练数据、cue 和其他共享训练配置是否一致仍是 UNKNOWN。若提供，所有已提供 run 必须都有且完全相同。它必须由外部 controller 根据预先指定的共享字段产生，不能包含 arm、seed、checkpoint 或教师更新策略等本应变化的字段。完整 `run_contract_sha256` 各自保存，**不要求两臂相同**。

summary 自身没有 seed、optimizer step、arm 或 benchmark：前三者必须由运行收据核验后放入 envelope；benchmark 来自原始 frozen dataset 的 `benchmark` 字段，任务数据可用 `domain`。工具不能证明 envelope 标签真实，不能从文件名推断。所有 summary 必须匹配同一份规范化 dataset hash 和同一 generation/judge/scoring protocol hash，且保留每个 planned id 的记录；数据缺失须显式写为 missing/invalid 与 null，不能删记录。旧 summary 中 judge 只列全局 profile 集合，不能单凭这个集合证明 profile 与 metric 的对应关系，仍要保存和核验原始 judgment 收据。

CLI plan 的 `dataset` 与各 run 的 `summary` 是相对于 plan 文件的路径；其余字段与接口一致。顶层必须有 `expected_seeds` 和 `step`，可选 `bootstrap_draws`、`analysis_random_seed`、`alpha`：

```sh
python -m rsd.paired_analysis --plan /absolute/path/analysis-plan.json \
  --output /absolute/path/new-analysis.json
```

输出必须是新文件，拒绝覆盖。CLI 保存 plan、原始 dataset、每份 summary 文件的字节 SHA256；Python 接口保存 dataset、summary、envelope 的规范化 SHA256 和分析实现文件 SHA256。`--allow-test` 仅为显式读取 test 的开关，**不实现封存或授权机制**；正式 test release 仍须由外部 controller 完成。

## 统计单位与分母

按 `(metric, benchmark, split)` 分组；每个 seed 内同 id 对齐 F0/R0。先计算每个 variant 的 `R0 − F0`，再对同 `source_id` 的全部 variants 求平均，各 source 等权。一个 variant 任一臂 UNKNOWN，则整个 source 的 delta 为 null；既报告 record planned/valid/unknown，也报告 source planned/valid/unknown。严格保留原分母，单独保留各臂 valid/invalid/missing/missing_run 计数。

跨 seed 只使用在所有预定 seed 都完整的共同 source；每个 source 先平均各 seed 的差，再在 source 间平均。因此 248 个 source × 3 seeds 仍是 **248 个题源**，不是 744 个独立题源。任一 arm/seed run 缺失时，各 seed 的信息仍保留，但不产生跨 seed 估计。相同 source 可以出现在不同 metric/benchmark 组；这些组分别分析，也不意味着它们统计独立。同组同 source 的 variants 必须属于同 family，避免重采样时把一个源拆进不同 strata。

StrongREJECT 连续值保留在 `[0,1]`；误拒与 exact match 必须是 `0/1`。工具不读取 `raw_label` 推断 refusal/compliance，也不把不同 metric 合成一个分数。对 lower-is-better 指标，正 delta 表示 R0 观测分数更差，但方向本身不是显著性或因果确认。

## 区间与解释边界

- **Source paired bootstrap**：在固定 family strata 内按完整 source 有放回重采样，保持两臂配对及 seed 内对应关系，输出 percentile 区间。此区间条件于已运行的模型与完整题源；3 个训练 seeds 不能精确刻画训练总体的不确定性，更不涵盖 judge 的系统误差。
- **稀疏/退化数据**：若源差全相等、样本不足，或重采样分布退化，不输出零宽 bootstrap 区间，而是 null 加明确原因。少于 5 个 changed sources 另标记稀疏性；任何区间都不自动触发结论。
- **有界敏感性区间**：若完整 source 差值相互独立且在 `[-1,1]`，Hoeffding 半径为 `sqrt(2 * log(2/alpha) / n_sources_valid)`，再裁剪至 `[-1,1]`。这是说明独立抽样假设下保守不确定性的敏感性分析；题源不是已证明的 iid 总体样本，不能冒充已验证的 population CI。跨 seeds 的相同题不增加 n。
- **UNKNOWN 最坏界**：`planned_source_mean_bounds_from_unknown` 将每个未知 source 的差限制在 `[-1,1]`，给出本次有限 planned cohort 均值的上下界。这不是置信区间。无未知时退化为观测均值是有限集合算术，不是总体不确定性为零。
- complete-case 估计可能受非随机缺失影响；bootstrap 与有界区间不能消除这一偏差。输出不做多重比较校正、不报告 p 值、不选择最优 seed、不根据检验通过自动认定科学结果。`scientific_conclusion` 与 `significance_claim` 始终 null；judge 有效性仍需独立校准。

## CPU 验收范围

测试用现有 `score_records` 实际结构构造 summary，覆盖身份/hash/分母错误、变体 cluster 权重、UNKNOWN、共同 seed cohort、缺失 seed、指标分离、二值与范围检查、零差区间、test 显式开关和原始文件 SHA/防覆盖。另在本地只读检查了历史 milestone step 0/1/2 的三个真实 `rsd-eval-v1` benchmark summary，各 10 条记录均通过输入兼容校验；没有把历史单臂结果冒充新 F0/R0 实验，也没有将私有原始回答加入测试 fixture。

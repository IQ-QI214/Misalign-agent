# C层外部回答候选：来源已核验，适用性与金标未确认

2026-11-07，已从固定版本的官方Anthropic HH-RLHF数据准备 **40条单轮真实模型回答、40个题源**，作为独立的C候选池。没有将它们并入原A/B验证包，没有写入学生训练数据，没有调用模型；当前 **接纳为验证样本0条、人工题面适用性标签0条、人工rubric金标0条**。

来源锁为 `configs/rsd/judge_reference_sources.lock.json`，可复建代码为 `rsd/judge_reference_data.py`，实际准备回执为 `validation/judge_reference_candidates_v1.json`。原始回答和盲审表留在私有忽略目录 `.runtime/judge_reference_sources/review_candidates_20261107/`。A层64条构造候选保持原包；当前拥有104条候选内容，不等于完整200条独立验证已准备好，更不等于104条人工金标。

## 来源与可复建约束

| 项目 | 固定身份 |
| --- | --- |
| 官方数据仓库 | `Anthropic/hh-rlhf` |
| Hugging Face revision | `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa` |
| 来源文件 | `helpful-base/test.jsonl.gz`、`harmless-base/test.jsonl.gz` |
| 原始行数 | 2,354 / 2,312 |
| 许可核验 | 官方数据卡与官方GitHub MIT许可；GitHub revision `c72f5cee8eb7b4d2ea5617657f4430d5e333af07` |
| 保留许可 | `configs/rsd/judge_licenses/Anthropic-HH-MIT.txt` |

两份压缩数据分别约875KB和743KB；SHA与固定revision官方LFS metadata核对一致。源文件字节哈希、文档哈希、下载URL、选择salt及排除资产SHA均保存在source lock。原HH偏好数据的`chosen/rejected`表示相对偏好，不提供本项目所需的XSTest三分类或StrongREJECT组件金标。来源桶也不能证明每个prompt良性/有害，或证明每个回答安全。[官方数据卡](https://huggingface.co/datasets/Anthropic/hh-rlhf/blob/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/README.md)、[官方仓库说明](https://github.com/anthropics/hh-rlhf/blob/c72f5cee8eb7b4d2ea5617657f4430d5e333af07/README.md)、[固定MIT许可](https://github.com/anthropics/hh-rlhf/blob/c72f5cee8eb7b4d2ea5617657f4430d5e333af07/LICENSE)。

只保留chosen/rejected两边均为一个Human和一个Assistant、且Human原文完全相同的单轮记录。题面长度15–1500字符，两边回答均10–3000字符；不符合条件直接排除，绝不截断或重写。对两个来源中的标准化同题重复整组排除。以固定哈希排序选择题目，再用独立哈希决定取chosen还是rejected；每个来源各10条chosen、10条rejected，共40条，每题只保留一条回答。选择不读取候选judge或本项目模型分数。

按标准化精确相等及词集合Jaccard≥0.85检查A、B、旧控制/旧10题、完整安全dev/test以及已选C；本次没有命中。这里读取的是既有确认题面，仅用于来源交叉检查，没有读取确认回答或分数。词汇去重不能证明语义或统计独立性。

每条候选保留native split、原始1-based行号、answer字段、完整source transcript/原始行/问答对的SHA、抽出的题面和回答SHA。来源模型的确切checkpoint身份不可得，因此不把固定数据revision冒充模型checkpoint。

## 为什么分成两份盲审表

每个审阅者分别取得：

1. **仅题面的适用性表**：没有回答、来源桶、偏好字段、建议metric或模型身份。题面状态初始为UNKNOWN，人工可标BENIGN/HARMFUL/UNSUITABLE/UNKNOWN并给理由。
2. **回答rubric表**：保留题面与完整回答，rubric、label、组件全为空；先按人工题面适用性确定可用的评审任务，再按冻结rubric标注回答。

两份表通过匿名annotation ID和输入SHA绑定，且两名审阅者的顺序不同。代码没有把helpful默认设为overrefusal，也没有把harmless默认设为safety，所有候选的`metric`为null。作者偏好与来源元数据只保留在私有来源清单中，不进入盲审表。

UNKNOWN或UNSUITABLE不能算作已接纳样本；如果人工审核后某类不足20条，需要在查看候选judge结果前明确记录覆盖缺口及补充规则，不能把来源桶数量直接当有效分母。候选准备器本身不接纳或判分。后续新增`rsd.judge_reference_admission`按两名人工题面审核及可选第三人裁决接纳，`rsd.judge_panel`再核验来源与配额合包；没有人工审核仍不能放行，原候选文件保留。

## CPU复建

先按原流程准备固定A/B和安全基准题面，再执行：

```bash
python -m rsd.judge_reference_data \
  --cache-dir .runtime/judge_reference_sources \
  --output-dir .runtime/judge_reference_sources/new_review_bundle
```

缓存缺失时显式加 `--download`，只从source lock中的固定URL读取，并核对SHA；输出目录必须是新的，禁止覆盖。该代码直接从官方gzip重建候选，不依赖私有的旧 `reference_candidates.json` 或一次性准备脚本。重建会核对固定40条记录的identity/prompt/response投影SHA；改变源文件、排除集或选择结果会拒绝。

下一步是人工题面适用性审查与rubric标注、B层实际回答生成和生成器身份冻结，随后再处理完整面板合包与真实v2/v3比较。这里的40条内容准备不是judge可靠性证据。

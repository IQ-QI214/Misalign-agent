# B层真实dev回答：固定请求与单次导入

`rsd.judge_real_data` 只做CPU数据准备和导入，不实现GPU/API生成器，也不调用模型、网络或远端。它接收既有验证材料中的48个dev source，每题分配给两个固定模型，共96个slot；旧控制、确认集和新抽选题目都不能替换这些slot。

## 当前可交付的准备

```bash
python -m rsd.judge_real_data prepare \
  --validation-bundle .runtime/judge_validation_v3/prepared_20261107 \
  --output <new-private-request-directory>
```

未提供两个完整生成器身份时，输出始终为 `DRAFT_IDENTITIES_MISSING`。它包括两份各48题的请求、生成器身份模板、回答记录模板和执行回执模板；真实回答与人工gold都是0。请求列表不意味着调用过模型。

2026-11-07当前可用草案为 `.runtime/judge_validation_v3/real_generation_requests_draft_20261107_v2`，manifest SHA为 `afe2428da4117908374962b140231f75c4ce4e5b2565dba31dba240c1915e122`。两生成器身份仍为UNKNOWN，未将既有Gemma judge推定为第二生成器。该目录保持只读；后续补齐身份时使用新目录。

此前 `_v1` 原样保留为历史准备记录，manifest SHA仍为 `208876cd6117697a7124452970ca51fb24fa40041949c4a7c58b6ce99e9e36d4`；由于实现身份已变，当前加载器会拒绝把它当作可用请求。v2修复了组合面板回指B请求可能引起的循环依赖：读取B来源时先核验原始材料包版本，再调用共享加载器；导入包加载请求依赖前先核验请求manifest SHA。

固定身份JSON采用 `{"1": {...}, "2": {...}}`，每个生成器必须提供：

- `checkpoint_identity`：不可变HF commit身份，或包含config与权重文件哈希的local-files身份；路径和模型昵称不够。
- `tokenizer_identity`：`vocab_sha256`、`chat_template_sha256`。
- `generation_protocol`：显式profile、采样与thinking开关、seed、最大prompt/输出长度、EOS/PAD ID、dtype、temperature/top_p/top_k/num_beams/repetition_penalty；完整对象被哈希锁定。

两个生成器必须是不同固定checkpoint。身份确定后，用新的输出目录重新 `prepare --generators <private.json>`，得到 `FROZEN` 请求包。FROZEN只表示请求身份已填齐，不表示已经生成、通过judge验证或获准启动GPU。每slot的 `request_sha256` 在生成前确定，调用方还必须绑定最终request manifest的原始文件SHA。

## 外部生成器将来需要交付什么

回答JSON格式为 `judge-real-responses-v1`，`state="FINAL_ATTEMPTS"`。它包含全部96个slot，每个slot恰好一条回答或失败记录；每条绑定source、prompt SHA、slot、request SHA、checkpoint和生成配置SHA。回答保留原文，不裁剪空格、不重写。记录必须显式包含 `generation_executed`、`valid`、`stop_reason`、`error`；完整回答另有正整数 `n_tokens`。

同时交付内容绑定的 `judge-real-execution-v1` 回执：同一request manifest SHA、全部96条response records的规范化SHA、两生成器身份SHA、外部driver实现SHA、两个不同run ID及实际尝试计数。模板保留未填值，不能直接导入为执行证据。

这只是生产者执行声明和内容一致性检查，**不能仅凭身份字符串或回执自动证明模型真实执行**。工具始终记录 `execution_independently_verified=false`。`purpose="cpu_fixture"` 的构造测试保持该身份，不能进入研究面板，也不能作为人工gold。

## 单次导入与分母

```bash
python -m rsd.judge_real_data ingest \
  --requests <frozen-request-directory> \
  --responses <external-response-bundle.json> \
  --output <new-private-ingestion-directory>
```

输出 `candidates.json` 仅包含成功、完整、非空的回答；它们统一为stratum B，带原始prompt/response哈希，`human_gold=null`。生成器的完整身份放在私有provenance与request文件，不放入candidate字段或盲审文本。

全部96个slot保存在 `attempts.private.json`，包括GENERATION_FAILED、EMPTY_RESPONSE、TRUNCATED_RESPONSE、INCOMPLETE_RESPONSE和NOT_GENERATED。即使 `valid=true`，length/error/未知终止也不算准备好的完整回答；即使文本包含合法答案，`valid=false` 仍不冒充成功。未生成的slot必须显式保留，不能从列表删掉来缩小分母。原始响应包和执行回执按原始字节保存。

首次通过校验的导入在请求目录写入不可覆盖claim。相同输入和目录只能做幂等复核；不能换一个目录再次导入不同回答，也不能只对失败slot重抽并替换。需要改变方案时另立版本、保留旧失败记录，不能把新版本冒充原定96次尝试。若写入中途失败，claim与部分记录仍保留；本工具不自动回滚或静默重试。

上层合包使用 `load_candidates(output_dir)`，返回 `(manifest, rows)`；它会重新核对source验证包、请求、首次claim、原始输入、attempts、候选与分母。manifest中的 `source_validation_bundle_manifest_sha256` 用来保证A/B来自同一个验证材料包。B材料完整与否不等于人工标签完成或judge可靠，`judge_batch_ready`仍为false。

# 训练前完整 dev 基线：准备、执行与证据边界

`rsd.dev_baseline`新增独立的、仅推理的dev入口。它不依赖optimizer、teacher、训练计划解封或judge通过；目的是在正式训练前测量当前固定基座的能力格式、输出长度、失败覆盖和完整dev原始回答。

当前仅完成CPU准备与注入式后端测试。真实Qwen dev基线尚未执行；GPU保持关闭。测试中的回答、错误和参数身份都是显式CPU fixture，不是模型性能或人工金标。

## 固定输入

准备器读取当前`configs/rsd/exploration_v2.plan.json`，仅打开三个dev数据文件：

- `data/rsd_reasoning_v2/dev.prompts.jsonl`：GSM8K 128题、ARC-Challenge 128题。
- `data/rsd_reasoning_v2/dev.gold.jsonl`：与上述题面分离的能力评分标签。
- `data/rsd_benchmarks/dev.jsonl`：StrongREJECT 65题、XSTest safe 45题。

总计366题，顺序固定为原task顺序再接原safety顺序。不会读取训练题、确认题、确认gold、确认回答或评分。读取机器计划中的这些文件名/哈希不等于打开对应数据。模型输入只接收完整题面，不接收gold、teacher cue、benchmark标签或评审说明。

## 现在可做的CPU操作

```bash
python -m rsd.dev_baseline prepare \
  --plan configs/rsd/exploration_v2.plan.json \
  --output .runtime/dev_baseline_20261107/prepared_v1

python -m rsd.dev_baseline check \
  --bundle .runtime/dev_baseline_20261107/prepared_v1
```

准备目录必须是新的。输出包包含dev原始字节快照、固定生成输入、模型身份依据、disabled执行配置和manifest。快照使用包内相对文件名，复制到QZ后不依赖Mac原路径；内容或实现SHA变化会拒绝，需显式重新准备。`prepare/check`不导入Torch、Transformers或CUDA，不读取权重、不下载、不调用模型。

基座固定为`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`。准备器复用已有公开回执中的四个原始权重SHA，以及config/tokenizer/template四个文件SHA；这些只是历史资产依据，尚未证明下一台机器上的资产相同。实际执行时会重新哈希本地文件。

## 将来GPU恢复后的显式执行

此入口只支持项目已授权QZ写根：输出必须同时位于当前项目目录及`/inspire/hdd/global_user/wenming-253108090054/zhujiaqi`内。即使checkout误放在`/root`，也不会把该路径视作可写环境。所有脚本控制的HF、Torch、编译、Triton、CUDA、临时目录及XDG路径都路由到该次输出的`runtime/`子目录；公共模型目录只读，不修改或删除。

在用户重新提供GPU并授权实际联测后，显式执行：

```bash
python -m rsd.dev_baseline execute \
  --bundle <copied-project-bundle> \
  --output <new-output-inside-authorized-QZ-project> \
  --model-path <verified-local-public-Qwen-directory> \
  --device cuda:0 \
  --enable-execution
```

不加`--enable-execution`会在模型加载前拒绝。执行只支持本地固定Qwen、bfloat16和显式CUDA设备，没有Hub下载或预训练CPU回退。单卡推理的实际显存、长输出容量和吞吐尚未由本轮CPU测试证明；两卡训练能力也不能替代这项容量验证。

加载前拒绝可能改变实际加载对象的未锁入口：额外单文件`model.safetensors`、替代`.bin`权重、adapter配置/权重、`added_tokens.json`、`special_tokens_map.json`、`tokenizer.model`、`chat_template.json`及额外`chat_templates`。如果公共模型目录存在这些未覆盖资产，程序报错并要求重新核验，不会为通过检查而删除文件。输出目录不能与模型目录重叠。

生成规则与当前milestone评测一致：thinking关闭，greedy，seed0，prompt上限2048、输出上限512，显式tokenizer EOS/PAD。训练中的采样温度、teacher context和任何训练gold都不参与。

## 保存、失败与后续判分

每题原始记录实时flush和fsync，保存输入SHA、模型身份、生成协议、回答SHA、token数、EOS/length/error/unknown及是否尝试。EOS即使恰好落在512 token处也按EOS处理；截断、未知结束和空答不记完整成功。

若某题抛出技术异常，该题错误记录保留，后续所有题按原顺序记`not_generated`，不重抽、不删分母。启动时加载失败则保留`startup-error.json`，没有生成记录冒充模型输出。一次输出目录不能重用；恢复必须另建attempt，保留此前失败。进程被操作系统强杀时，已fsync记录仍保留，但不能承诺已经补写剩余槽位或最终receipt；`summarize`会保留缺失数量。

能力评分使用`final-answer-v1`，分别报告GSM8K和ARC的有效准确率、格式错误、UNKNOWN和全题上下界。安全部分只导出`safety-dataset.jsonl`与`safety-predictions.jsonl`等待固定judge，当前不产生安全分数。

长度统计按benchmark报告平均值、p50/p95/p99/max及明确分母：只有实际尝试且观测到正生成token数的记录进入长度分布，未生成槽位不会作为零长度稀释统计。空答率和截断率明确按attempt计，仍单列完整原分母和未尝试数。

此基线不会更新optimizer，不会保存模型权重，也不会替代每个正式训练run的step0。单次基线可以发现格式/长度/判分问题，但不能估计F0−R0配对差分方差、证明learning signal或断言正负迁移。正式差分与judge误差仍需要后续独立证据。

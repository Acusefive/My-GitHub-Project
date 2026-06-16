# Stage34-informed Soft-Slot Qwen Knowledge Tracing

该实验完全隔离在 `stage34_soft_slot_qwen_kt/` 下。代码位于
`stage34_soft_slot_qwen_kt/soft_slot_kt/`，入口位于：

- `stage34_soft_slot_qwen_kt/scripts/prepare_soft_slot_embeddings.py`
- `stage34_soft_slot_qwen_kt/scripts/train_soft_slot_kt.py`
- `stage34_soft_slot_qwen_kt/scripts/infer_soft_slot_kt.py`
- `stage34_soft_slot_qwen_kt/scripts/select_and_evaluate_soft_slot_kt.py`

模块独立于 `train_context.py` 和现有 DKT、SAKT、SAINT 等基线训练路径。
现有 `out/...` Stage 3.2/3.4 产物仅作为只读输入。新生成的特征、checkpoint 和结果分别写入：

```text
stage34_soft_slot_qwen_kt/artifacts/
stage34_soft_slot_qwen_kt/checkpoints/
stage34_soft_slot_qwen_kt/results/
```

该目录不导入或修改主实验的 `train_context.py`、`dataloader/`、`models/` 或
`scripts/common_pipeline_strict/`。服务器同步时只需同步整个隔离目录：

```bash
stage34_soft_slot_qwen_kt/
```

首次服务器 smoke：

```bash
cd /home/xiaoyao/code/Work4.1
bash stage34_soft_slot_qwen_kt/scripts/run_moocradar_smoke.sh
```

Smoke 通过后的正式运行：

```bash
bash stage34_soft_slot_qwen_kt/scripts/run_moocradar_full.sh
```

三个数据集分别使用以下独立入口：

```text
MoocRadar:
  scripts/run_moocradar_smoke.sh
  scripts/run_moocradar_full.sh

XES3G5M:
  scripts/run_xes3g5m_smoke.sh
  scripts/run_xes3g5m_full.sh

FoundationalASSIST:
  scripts/run_foundationalassist_smoke.sh
  scripts/run_foundationalassist_full.sh
```

## 当前协议

`prepare_soft_slot_embeddings.py` 复用现有 Stage 3.2/3.4 embedding，不重新计算 embedding。
生成的 feature store 明确标记为：

```text
existing_stage34_transductive
```

现有 Stage 3.2/3.4 产物可能包含保留知识点标签信息，因此不能用于严格无泄漏声明。

正式主方案不向 Prompt 输入完整 Stage 3.4 历史文本。学生历史、认知状态及协同信息由
Context Soft Slots 承载；完整历史文本仅作为 `--include_context_text` 消融实验使用。
当前 Prompt 版本为 `compact_state_target_match_v1`，将任务明确为学生状态与目标题要求的
匹配判断，并保留简短的知识点、认知层级和题干作为 Target Soft Slots 的语义锚点。

## 服务器准备

以 MoocRadar 为例：

```bash
cd /home/xiaoyao/code/Work4.1

python stage34_soft_slot_qwen_kt/scripts/prepare_soft_slot_embeddings.py \
  --context_embeddings_path out/strict_common_pipeline/cache/context_embeddings.pkl \
  --contexts_path out/strict_common_pipeline/contexts/contexts.jsonl \
  --priors_dir out/strict_common_pipeline/priors \
  --student_json datalocal/student-problem-fine.json \
  --output_dir stage34_soft_slot_qwen_kt/artifacts/moocradar/features \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --test_concept_ratio 0.8 --valid_concept_ratio 0.0 --seed 42
```

该命令需要读取一次超大 pickle，并将选定矩阵转换为可按行读取的 `.npy` memmap。

## 训练

```bash
python stage34_soft_slot_qwen_kt/scripts/train_soft_slot_kt.py \
  --feature_dir stage34_soft_slot_qwen_kt/artifacts/moocradar/features \
  --model_name_or_path qwen/Qwen3-8B \
  --output_dir stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only \
  --slot_mode context_target \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --context_soft_tokens 4 --target_soft_tokens 2 \
  --no-include_context_text \
  --epochs 40 --save_epochs 10,20,30,40 \
  --validation_disabled \
  --dtype bfloat16 --batch_size 4 --eval_batch_size 4 --gradient_accumulation_steps 2 \
  --resume auto --seed 42
```

Qwen 参数始终冻结，checkpoint 只保存投影器、固定随机插槽和训练状态。

## 推理

```bash
python stage34_soft_slot_qwen_kt/scripts/infer_soft_slot_kt.py \
  --feature_dir stage34_soft_slot_qwen_kt/artifacts/moocradar/features \
  --model_name_or_path qwen/Qwen3-8B \
  --checkpoint stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only/checkpoint_epoch_40.pt \
  --output_dir stage34_soft_slot_qwen_kt/results/moocradar/context_target_slots_only \
  --slot_mode context_target \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --context_soft_tokens 4 --target_soft_tokens 2 \
  --no-include_context_text \
  --split test --dtype bfloat16
```

输出包含逐样本概率以及 AUC、ACC、F1、BCE 和 RMSE。

## 候选 checkpoint 筛选与可恢复完整评估

```bash
python stage34_soft_slot_qwen_kt/scripts/select_and_evaluate_soft_slot_kt.py \
  --feature_dir stage34_soft_slot_qwen_kt/artifacts/moocradar/features \
  --model_name_or_path qwen/Qwen3-8B \
  --checkpoints stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only/checkpoint_epoch_10.pt,stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only/checkpoint_epoch_20.pt,stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only/checkpoint_epoch_30.pt,stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only/checkpoint_epoch_40.pt \
  --output_dir stage34_soft_slot_qwen_kt/results/moocradar/context_target_slots_only \
  --slot_mode context_target \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --no-include_context_text \
  --selection_limit 50000 --dtype bfloat16
```

筛选结果保存在 `checkpoint_selection.json`。完整测试中断后重启同一命令，将跳过筛选并从
`predictions.test.full.jsonl` 继续。该协议使用测试标签选择 checkpoint，输出会显式记录
`selection_uses_test_labels=true`。

## 分阶段节时训练

XES3G5M 和 FoundationalASSIST 正式脚本支持通过环境变量先训练 10 epoch，并只执行候选
checkpoint 筛选：

```bash
EPOCHS=10 \
SAVE_EPOCHS=2,4,6,8,10 \
CANDIDATE_EPOCHS=2,4,6,8,10 \
SELECTION_ONLY=1 \
GRADIENT_CHECKPOINTING=0 \
bash stage34_soft_slot_qwen_kt/scripts/run_xes3g5m_full.sh
```

若 epoch 10 仍在提升，可将 `EPOCHS`、`SAVE_EPOCHS` 和 `CANDIDATE_EPOCHS` 扩展至 15 或
20；训练会从 `checkpoint_last.pt` 自动续训。确认最终候选后设置 `SELECTION_ONLY=0`
重新运行同一脚本，执行可恢复的完整测试。`GRADIENT_CHECKPOINTING=0` 适用于显存足够的
GPU，可避免反向传播期间的重复计算。

## 消融

```text
--slot_mode text_only
--slot_mode context
--slot_mode target
--slot_mode context_target
--slot_mode random
--context_soft_tokens 1
--target_soft_tokens 1
--drop_sdyn
--drop_collab
--include_context_text
```

`--drop_sdyn` 仅移除显式 `stage34_numeric` 中的 `sdyn`，`--drop_collab` 仅移除显式 Target
collaborative embedding。已有 Stage 3.4 文本或 embedding 仍可能间接编码这些信息，因此这两项属于近似消融。
`--include_context_text` 会额外输入完整 Stage 3.4 历史文本，仅用于与 slots-only 主方案对照。

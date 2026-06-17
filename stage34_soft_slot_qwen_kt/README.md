# Stage34 Soft-Slot Qwen KT 全流程说明

本文档面向没有上下文的读者或后续 LLM。它说明从原始数据、Stage3.2、Stage3.4，到 Soft-Slot Qwen Knowledge Tracing 训练、评估和消融的完整链路。

本模块是独立实验模块，位于：

```text
stage34_soft_slot_qwen_kt/
```

它不修改，也不依赖 DKT/SAKT/SAINT/AKT 等 KT baseline 的最终推理逻辑。KT baseline 可以作为结果对照，但不参与本方法推理。

## 1. 方法概览

目标：使用 Stage3.2 和 Stage3.4 已构造的学生认知状态、历史证据和目标题先验，通过 soft slots 输入冻结的 Qwen3-8B，预测学生答对目标题的概率。

整体流程：

```text
problem.json + student-problem-fine.json
        |
        v
Stage3.2: 题目先验、语义向量、协同向量、动态先验模型
        |
        v
Stage3.4: 针对每个预测目标构造学生历史证据、认知摘要、context embeddings
        |
        v
Soft-Slot Feature Store: 对齐 Stage3.2/3.4 产物，生成可训练样本和 memmap 特征
        |
        v
Frozen Qwen3-8B + trainable projectors
        |
        v
A/B logits -> softmax -> P(correct)
```

核心设计：

- Qwen3-8B 冻结，不微调。
- 只训练 Context projector 和 Target projector。
- 使用 `inputs_embeds`，把投影后的 embedding 替换 prompt 中的 slot 位置。
- 不让 Qwen 自由生成 JSON 或概率。
- 读取标签 token `A=正确`、`B=错误` 的 logits，用二分类 softmax 得到答对概率。

一句话版本：

```text
把 Stage3.4 学生状态和 Stage3.2 目标题先验翻译成 Qwen token embedding 空间里的 soft tokens，再让冻结 Qwen 判断“正确/错误”。
```

## 2. 与 A-LLMRec 的关系

本方法借鉴 A-LLMRec Figure 2(c) Stage 2 的思想：

```text
外部 embedding -> projector -> LLM token space -> 插入自然语言 prompt -> 冻结 LLM 推理
```

对应关系：

```text
A-LLMRec:
  user/item embedding -> projector -> LLM token space -> 推荐 item title

本方法:
  学生状态/目标题 embedding -> projector -> Qwen token space -> 正确/错误二分类
```

本方法不是完整复现 A-LLMRec：

- 不使用 SASRec 或推荐系统 CF backbone。
- 不复现 A-LLMRec Stage 1。
- 不通过 OpenAI `/chat/completions` 文本接口传 soft slots。
- 不让 Qwen 生成自然语言概率。

## 3. 目录和文件边界

主要入口：

```text
stage34_soft_slot_qwen_kt/scripts/prepare_soft_slot_embeddings.py
stage34_soft_slot_qwen_kt/scripts/train_soft_slot_kt.py
stage34_soft_slot_qwen_kt/scripts/infer_soft_slot_kt.py
stage34_soft_slot_qwen_kt/scripts/select_and_evaluate_soft_slot_kt.py
```

核心代码：

```text
stage34_soft_slot_qwen_kt/soft_slot_kt/
```

本模块输出：

```text
stage34_soft_slot_qwen_kt/artifacts/     # Soft-Slot feature store
stage34_soft_slot_qwen_kt/checkpoints/   # projector checkpoints
stage34_soft_slot_qwen_kt/results/       # metrics and predictions
stage34_soft_slot_qwen_kt/logs/          # optional logs
```

不要把服务器上的以下目录用 `rsync --delete` 删除：

```text
artifacts/
checkpoints/
results/
logs/
```

## 4. 数据集

服务器工作目录：

```bash
cd /home/xiaoyao/code/Work4.1
```

| Dataset | problem | student | Stage3.2/3.4 root | Soft-Slot feature dir |
|---|---|---|---|---|
| MoocRadar | `datalocal/problem.json` | `datalocal/student-problem-fine.json` | `out/strict_common_pipeline` | `stage34_soft_slot_qwen_kt/artifacts/moocradar/features` |
| XES3G5M | `datalocal/xes3g5m_strict_core20/problem.json` | `datalocal/xes3g5m_strict_core20/student-problem-fine.json` | `out/xes3g5m_strict_core20_common_pipeline` | `stage34_soft_slot_qwen_kt/artifacts/xes3g5m/features` |
| FoundationalASSIST | `datalocal/foundationalassist_text_only_core200_contextcomplete/problem.json` | `datalocal/foundationalassist_text_only_core200_contextcomplete/student-problem-fine.json` | `out/foundationalassist_text_only_core200_contextcomplete_common_pipeline` | `stage34_soft_slot_qwen_kt/artifacts/foundationalassist/features` |

统一实验划分：

```text
split_mode=new_concept
test_concept_ratio=0.8
valid_concept_ratio=0.0
valid_ratio=0.0
validation disabled
seed=42
```

## 5. 输入数据格式

### problem.json

题目目录。每道题至少应包含：

```text
problem_id
text/title
concepts
cognitive_dimension
```

`cognitive_dimension` 是题目认知层级标签，不等同于 XES3G5M 的官方 `question_level`。

### student-problem-fine.json

学生交互序列。每条交互至少应包含：

```text
problem_id
is_correct
```

Stage3.4 会按照学生序列为每个预测目标构造上下文。Soft-Slot feature store 会根据 `(user_id, target_t, target_pid)` 找回真实标签，但标签只保存在样本的 `label` 字段中，不作为 prompt 或 embedding 输入。

## 6. Stage3.2: 题目先验构造

Stage3.2 位于：

```text
scripts/common_pipeline_strict/stage32.py
```

运行入口：

```text
scripts/run_common_cognitive_pipeline_strict.py
```

Stage3.2 的作用是把题目元数据和学生交互历史转换为可复用的题目侧先验。

主要输出位于：

```text
<out_root>/priors/
```

关键产物：

| 文件 | 用途 |
|---|---|
| `stage32_manifest.json` | Stage3.2 产物清单 |
| `problem_catalog.jsonl` | Stage3.4 和 Soft-Slot 使用的统一题目目录 |
| `hqtext_vectors.pkl` | 题目文本语义向量 |
| `hqid_vectors.pkl` | 分层语义 ID 向量 |
| `eqbase_vectors.pkl` | 题目基础认知向量 |
| `semantic_vectors.pkl` | 加入 Rasch/认知方向修正后的语义向量 |
| `item_collaborative_embeddings.pkl` | 基于学生行为的题目协同向量 |
| `item_collaborative.json` | 题目协同邻居 |
| `problem_mu_q.json` | Rasch 题目难度/能力相关先验 |
| `concept_graph_bundle.json` | 知识图和概念关系 |
| `model_state.pt` | Stage3.2 动态先验模型参数 |
| `training_report.json` | Stage3.2 训练报告 |
| `implementation_defaults.json` | 生成参数记录 |

本方法中，Stage3.2 主要进入 Target Slot：

```text
hqtext
hqid
semantic
collaborative
```

其中：

- `hqtext` 来自 `hqtext_vectors.pkl`
- `hqid` 来自 `hqid_vectors.pkl`
- `semantic` 来自 `semantic_vectors.pkl`
- `collaborative` 来自 `item_collaborative_embeddings.pkl`

## 7. Stage3.4: 学生上下文构造

Stage3.4 位于：

```text
scripts/common_pipeline_strict/stage34.py
```

Stage3.4 读取 Stage3.2 的题目先验，为每个预测目标构造学生历史证据和认知状态。

主要输出位于：

```text
<out_root>/contexts/
<out_root>/cache/
```

关键产物：

| 文件 | 用途 |
|---|---|
| `contexts/stage34_manifest.json` | Stage3.4 产物清单 |
| `contexts/contexts.jsonl` | 每个预测目标的文本上下文、摘要和证据列表 |
| `cache/context_embeddings.pkl` | Stage3.4 context embedding 矩阵 |

`contexts.jsonl` 每行对应一个预测目标，典型字段：

```text
user_id
target_t
target_pid
target_semantic_id
stage1_candidate_count
selected_count
main_context_text
template_context_text
llm_context_text
summary_fields
evidence_list
```

`summary_fields` 典型字段：

```text
target_concepts
dominant_role
recent_trend
risk_level
sdyn
summary_text
llm_summary_text
llm_summary_struct
```

`context_embeddings.pkl` 典型字段：

```text
index
main_embeddings
template_embeddings
llm_embeddings
llm_struct_embeddings
llm_struct_features
```

本方法主实验使用 Stage3.4 的：

```text
llm_embeddings
llm_struct_features
stage34_numeric
```

其中 `stage34_numeric` 由 Soft-Slot feature store 从 Stage3.4 `summary_fields` 中提取，当前包括显式数值状态，例如 `sdyn`、趋势、风险等。

## 8. 从头构建 Stage3.2/Stage3.4

如果已有 Stage3.2/Stage3.4 产物，可以跳过本节，直接看第 9 节。

### 8.1 完整 pipeline 一次性运行

以 MoocRadar 为例：

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json datalocal/problem.json \
  --student_json datalocal/student-problem-fine.json \
  --out_root out/strict_common_pipeline
```

这会先跑 Stage3.2，再跑 Stage3.4，并最终生成：

```text
out/strict_common_pipeline/priors/stage32_manifest.json
out/strict_common_pipeline/contexts/stage34_manifest.json
out/strict_common_pipeline/contexts/contexts.jsonl
out/strict_common_pipeline/cache/context_embeddings.pkl
```

### 8.2 分阶段运行

大数据集不建议一次性重跑。建议分阶段。

只跑 Stage3.2：

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json <problem.json> \
  --student_json <student-problem-fine.json> \
  --out_root <out_root> \
  --stop_after_stage32
```

复用 Stage3.2，只生成 Stage3.4 contexts，不生成 embeddings：

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json <problem.json> \
  --student_json <student-problem-fine.json> \
  --out_root <out_root> \
  --skip_stage32 \
  --dry_run
```

复用 Stage3.2 和已有 contexts，生成最终 context embeddings：

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json <problem.json> \
  --student_json <student-problem-fine.json> \
  --out_root <out_root> \
  --skip_stage32 \
  --reuse_existing_contexts
```

如果需要 Stage3.4 LLM summary：

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json <problem.json> \
  --student_json <student-problem-fine.json> \
  --out_root <out_root> \
  --skip_stage32 \
  --reuse_existing_contexts \
  --enable_llm_summary \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_model qwen3-8b \
  --llm_timeout_sec 240 \
  --llm_summary_workers 8 \
  --dry_run
```

注意：

- `--enable_llm_graph_completion` 属于 Stage3.2。
- `--enable_llm_summary` 属于 Stage3.4。
- Qwen3-Embedding/Reranker 由 Stage3.2/3.4 使用；Qwen3-8B 由 LLM summary 或 Soft-Slot Qwen 使用。

## 9. 当前实验协议与泄漏边界

当前 Soft-Slot 实验默认复用服务器已有 Stage3.2/Stage3.4 产物。feature store 会标记为：

```text
existing_stage34_transductive
```

这表示：

- 目标样本划分使用 `new_concept`。
- 但是 Stage3.2/Stage3.4 原始产物可能在生成时使用过测试概念或测试标签统计。
- 因此当前结果应表述为 transductive feature setting。
- 不能声明为严格无泄漏 new-concept。

严格无泄漏版本需要保证：

- Stage3.2 的监督训练只使用 train concepts。
- Stage3.2 的协同统计只使用 train concepts 的交互。
- Stage3.4 每个目标只使用 `target_t` 之前的历史。
- 目标题真实 `is_correct` 不能进入 prompt 或 embedding。
- checkpoint 选择不能使用测试标签，除非明确标注 test-selected。

当前代码中的审计：

```text
feature_manifest.json -> audit
leakage_audit.json
```

其中记录：

```text
protocol
strict_new_concept_leakage_free
noncausal_evidence_count
target_label_stored_only_in_sample_label_field
split_counts
```

## 10. Soft-Slot Feature Store

Soft-Slot feature store 是 Stage3.2/Stage3.4 产物和 Qwen 训练代码之间的桥接层。

入口：

```text
stage34_soft_slot_qwen_kt/scripts/prepare_soft_slot_embeddings.py
```

它做四件事：

1. 读取 Stage3.4 `contexts.jsonl` 和 `context_embeddings.pkl`。
2. 读取 Stage3.2 `priors/` 中的题目向量。
3. 按 `new_concept` 规则给每个预测目标分配 train/test。
4. 生成可按行读取的 `.npy` memmap 和 `samples.jsonl`。

输出位于：

```text
stage34_soft_slot_qwen_kt/artifacts/<dataset>/features/
```

关键文件：

| 文件 | 用途 |
|---|---|
| `feature_manifest.json` | feature store 清单 |
| `leakage_audit.json` | 泄漏审计摘要 |
| `samples.jsonl` | 样本元信息和标签 |
| `sample_offsets.npy` | `samples.jsonl` 行偏移索引 |
| `split_codes.npy` | train/valid/test 编码 |
| `stage34_numeric.npy` | 从 Stage3.4 summary_fields 提取的数值状态 |
| `context_*.npy` | Stage3.4 context feature 矩阵 |
| `target_*.npy` | Stage3.2 target feature 矩阵 |

样本标签只保存在：

```text
samples.jsonl -> label
```

标签不会作为 context slot、target slot 或 prompt 文本输入。

## 11. 准备 Soft-Slot Feature Store

MoocRadar：

```bash
python stage34_soft_slot_qwen_kt/scripts/prepare_soft_slot_embeddings.py \
  --context_embeddings_path out/strict_common_pipeline/cache/context_embeddings.pkl \
  --contexts_path out/strict_common_pipeline/contexts/contexts.jsonl \
  --priors_dir out/strict_common_pipeline/priors \
  --student_json datalocal/student-problem-fine.json \
  --output_dir stage34_soft_slot_qwen_kt/artifacts/moocradar/features \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --test_concept_ratio 0.8 \
  --valid_concept_ratio 0.0 \
  --seed 42
```

XES3G5M：

```bash
python stage34_soft_slot_qwen_kt/scripts/prepare_soft_slot_embeddings.py \
  --context_embeddings_path out/xes3g5m_strict_core20_common_pipeline/cache/context_embeddings.pkl \
  --contexts_path out/xes3g5m_strict_core20_common_pipeline/contexts/contexts.jsonl \
  --priors_dir out/xes3g5m_strict_core20_common_pipeline/priors \
  --student_json datalocal/xes3g5m_strict_core20/student-problem-fine.json \
  --output_dir stage34_soft_slot_qwen_kt/artifacts/xes3g5m/features \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --test_concept_ratio 0.8 \
  --valid_concept_ratio 0.0 \
  --seed 42
```

FoundationalASSIST：

```bash
python stage34_soft_slot_qwen_kt/scripts/prepare_soft_slot_embeddings.py \
  --context_embeddings_path out/foundationalassist_text_only_core200_contextcomplete_common_pipeline/cache/context_embeddings.pkl \
  --contexts_path out/foundationalassist_text_only_core200_contextcomplete_common_pipeline/contexts/contexts.jsonl \
  --priors_dir out/foundationalassist_text_only_core200_contextcomplete_common_pipeline/priors \
  --student_json datalocal/foundationalassist_text_only_core200_contextcomplete/student-problem-fine.json \
  --output_dir stage34_soft_slot_qwen_kt/artifacts/foundationalassist/features \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --test_concept_ratio 0.8 \
  --valid_concept_ratio 0.0 \
  --seed 42
```

检查：

```bash
python -m json.tool stage34_soft_slot_qwen_kt/artifacts/moocradar/features/feature_manifest.json
python -m json.tool stage34_soft_slot_qwen_kt/artifacts/xes3g5m/features/feature_manifest.json
python -m json.tool stage34_soft_slot_qwen_kt/artifacts/foundationalassist/features/feature_manifest.json
```

## 12. Qwen Soft-Slot 模型

主方法输入：

```text
Context Slot:
  llm_embeddings
  llm_struct_features
  stage34_numeric

Target Slot:
  hqtext
  hqid
  semantic
  collaborative
```

主配置：

```text
slot_mode=context_target
context_soft_tokens=4
target_soft_tokens=2
include_context_text=false
prompt_version=compact_state_target_match_v1
label_spec=A/B
```

Prompt 结构简化为：

```text
知识追踪任务：结合学生当前认知状态与目标题要求，判断该学生本次作答更可能正确还是错误。
学生状态表示：[Context soft slots]
目标题：知识点=...；认知层级=...；题干=...
目标题表示：[Target soft slots]
输出标签：A=正确，B=错误。仅输出一个标签。
标签：
```

模型不会生成完整文本。它只在 `标签：` 后读取：

```text
logit_A
logit_B
```

然后：

```text
P(correct) = exp(logit_A) / (exp(logit_A) + exp(logit_B))
```

## 13. 主实验训练与评估

主流程：

1. `train_soft_slot_kt.py` 训练 projector，保存候选 checkpoint。
2. `select_and_evaluate_soft_slot_kt.py` 用 50,000 个测试样本选择最佳 checkpoint。
3. 用最佳 checkpoint 跑完整测试集。

48GB GPU 推荐：

```text
BATCH_SIZE=8
GRAD_ACCUM=1
EVAL_BATCH_SIZE=32
GRADIENT_CHECKPOINTING=0
```

24GB GPU 推荐：

```text
BATCH_SIZE=4
GRAD_ACCUM=2
EVAL_BATCH_SIZE=16
GRADIENT_CHECKPOINTING=1
```

XES3G5M：

```bash
CUDA_VISIBLE_DEVICES=5 \
BATCH_SIZE=4 \
GRAD_ACCUM=2 \
EVAL_BATCH_SIZE=16 \
GRADIENT_CHECKPOINTING=1 \
EPOCHS=10 \
SAVE_EPOCHS=2,4,6,8,10 \
CANDIDATE_EPOCHS=2,4,6,8,10 \
SELECTION_ONLY=0 \
bash stage34_soft_slot_qwen_kt/scripts/run_xes3g5m_full.sh
```

FoundationalASSIST：

```bash
CUDA_VISIBLE_DEVICES=4 \
BATCH_SIZE=8 \
GRAD_ACCUM=1 \
EVAL_BATCH_SIZE=32 \
GRADIENT_CHECKPOINTING=0 \
EPOCHS=10 \
SAVE_EPOCHS=2,4,6,8,10 \
CANDIDATE_EPOCHS=2,4,6,8,10 \
SELECTION_ONLY=0 \
bash stage34_soft_slot_qwen_kt/scripts/run_foundationalassist_full.sh
```

MoocRadar `learning_rate=5e-5` 主实验：

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python stage34_soft_slot_qwen_kt/scripts/train_soft_slot_kt.py \
  --feature_dir stage34_soft_slot_qwen_kt/artifacts/moocradar/features \
  --model_name_or_path qwen/Qwen3-8B \
  --output_dir stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only_lr5e5 \
  --slot_mode context_target \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --context_soft_tokens 4 \
  --target_soft_tokens 2 \
  --no-include_context_text \
  --epochs 10 \
  --save_epochs 2,4,6,8,10 \
  --learning_rate 5e-5 \
  --validation_disabled \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --batch_size 8 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --no-gradient_checkpointing \
  --resume auto \
  --seed 42
```

MoocRadar checkpoint 筛选和完整测试：

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python stage34_soft_slot_qwen_kt/scripts/select_and_evaluate_soft_slot_kt.py \
  --feature_dir stage34_soft_slot_qwen_kt/artifacts/moocradar/features \
  --model_name_or_path qwen/Qwen3-8B \
  --checkpoints stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only_lr5e5/checkpoint_epoch_2.pt,stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only_lr5e5/checkpoint_epoch_4.pt,stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only_lr5e5/checkpoint_epoch_6.pt,stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only_lr5e5/checkpoint_epoch_8.pt,stage34_soft_slot_qwen_kt/checkpoints/moocradar/context_target_slots_only_lr5e5/checkpoint_epoch_10.pt \
  --output_dir stage34_soft_slot_qwen_kt/results/moocradar/context_target_slots_only_lr5e5 \
  --slot_mode context_target \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --context_soft_tokens 4 \
  --target_soft_tokens 2 \
  --no-include_context_text \
  --selection_limit 50000 \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --eval_batch_size 32 \
  --seed 42
```

## 14. 输出结果

完整测试指标：

```text
stage34_soft_slot_qwen_kt/results/<dataset>/<run_name>/metrics.test.full.json
```

候选 checkpoint 筛选：

```text
stage34_soft_slot_qwen_kt/results/<dataset>/<run_name>/checkpoint_selection.json
```

逐样本预测：

```text
stage34_soft_slot_qwen_kt/results/<dataset>/<run_name>/predictions.test.full.jsonl
```

逐样本预测格式：

```json
{
  "row": 123,
  "user_id": "student",
  "target_t": 10,
  "target_pid": "problem",
  "label": 1,
  "split": "test",
  "probability": 0.8734,
  "prediction": 1
}
```

查看指标：

```bash
python -m json.tool stage34_soft_slot_qwen_kt/results/moocradar/context_target_slots_only_lr5e5/metrics.test.full.json
python -m json.tool stage34_soft_slot_qwen_kt/results/xes3g5m/context_target_slots_only/metrics.test.full.json
python -m json.tool stage34_soft_slot_qwen_kt/results/foundationalassist/context_target_slots_only/metrics.test.full.json
```

## 15. 断点恢复

训练恢复：

- `--resume auto` 从 `checkpoint_last.pt` 恢复。
- `training_complete.json` 存在且 epoch 足够时自动跳过训练。
- 中断发生在未保存 epoch 中间时，该 epoch 需要重跑。

评估恢复：

- `checkpoint_selection.json` 记录已完成的候选 checkpoint。
- `predictions.test.full.jsonl` 记录完整测试已完成样本。
- 重新执行相同评估命令会跳过已完成部分。
- `metrics.test.full.json` 存在表示完整评估已结束。

检查完整测试进度：

```bash
wc -l stage34_soft_slot_qwen_kt/results/moocradar/context_target_slots_only_lr5e5/predictions.test.full.jsonl
```

MoocRadar full test 总数约为：

```text
818197
```



## 16. 消融实验

消融实验要回答三个问题：

1. Soft slots 是否真的带来有效信息。
2. Stage3.2 目标题先验和 Stage3.4 学生状态各自贡献多少。
3. 当前 soft-token 结构是否合理。

### 16.1 必做消融

建议三个数据集都做：

| Variant | 目的 | 参数 |
|---|---|---|
| Text-only Qwen | 无 soft slots；仅用任务说明和目标题自然语言信息 | `--slot_mode text_only` |
| Target slot only | 验证 Stage3.2 目标题先验 slot | `--slot_mode target` |
| Context slot only | 验证 Stage3.4 学生状态 slot；仍保留目标题自然语言锚点 | `--slot_mode context` |
| Full Soft-Slot | 主方法 | `--slot_mode context_target` |
| Fixed random slots | 固定随机 soft slots 负对照；检验 slot 位置/提示形式本身是否带来虚假提升 | `--slot_mode random` |
| w/o sdyn | 验证显式动态状态贡献 | `--drop_sdyn` |
| w/o collaborative | 验证协同题目先验贡献 | `--drop_collab` 或去掉 `collaborative` |
| w/o hqid | 验证分层语义 ID 向量贡献 | `--target_fields hqtext,semantic,collaborative` |

注意：

- `text_only` 没有 soft slots，没有可训练 projector，使用 `infer_soft_slot_kt.py`。
- 当前 `random` 是固定随机 slots，也没有可训练 projector，使用 `infer_soft_slot_kt.py`。
- 当前 `random` 不是“可训练参数容量”对照；如果要控制 projector 容量，应另做 shuffled features 或 random features + trainable projector 实验。
- `hqid` 是 hierarchical semantic ID vector，不是原始题目 ID embedding。
- `w/o sdyn` 只清零显式 `stage34_numeric` 中的 `sdyn`。已有文本 embedding 可能仍间接包含动态信息。
- `w/o collaborative` 只移除显式 collaborative embedding。已有 Stage3.4 context 可能仍间接编码协同信息。

### 16.2 Stage3.4 学生状态消融

| Variant | 参数 |
|---|---|
| only llm_embeddings | `--context_fields llm_embeddings` |
| only struct features | `--context_fields llm_struct_features` |
| only numeric | `--context_fields stage34_numeric` |
| w/o llm_embeddings | `--context_fields llm_struct_features,stage34_numeric` |
| w/o struct features | `--context_fields llm_embeddings,stage34_numeric` |
| w/o numeric | `--context_fields llm_embeddings,llm_struct_features` |
| w/o sdyn | `--drop_sdyn` |

优先级最高：

```text
only llm_embeddings
w/o llm_struct_features
w/o sdyn
```

### 16.3 Stage3.2 目标题先验消融

| Variant | 参数 |
|---|---|
| only hqtext | `--target_fields hqtext` |
| only hqid | `--target_fields hqid` |
| only semantic | `--target_fields semantic` |
| only collaborative | `--target_fields collaborative` |
| w/o hqtext | `--target_fields hqid,semantic,collaborative` |
| w/o hqid | `--target_fields hqtext,semantic,collaborative` |
| w/o semantic | `--target_fields hqtext,hqid,collaborative` |
| w/o collaborative | `--target_fields hqtext,hqid,semantic` |

这里的 `hqid` 表示分层语义 ID 向量，不是原始题目 ID。最有价值的三项：

```text
only collaborative
w/o hqid
w/o collaborative
```

### 16.4 Slot 数量消融

建议先只在 MoocRadar 做：

| Variant | 参数 |
|---|---|
| 1+1 | `--context_soft_tokens 1 --target_soft_tokens 1` |
| 2+1 | `--context_soft_tokens 2 --target_soft_tokens 1` |
| 4+2 | 当前主方法 |
| 8+4 | `--context_soft_tokens 8 --target_soft_tokens 4` |

### 16.5 训练策略消融

建议：

- 主方法跑 seed 42/43/44。
- 关键消融只跑 seed 42。
- learning rate 比较 `1e-4`、`5e-5`、`2e-5`。
- 候选 checkpoint 使用 `2,4,6,8,10`，避免错过早期最优点。

## 17. 消融命令模板

训练类消融：

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python stage34_soft_slot_qwen_kt/scripts/train_soft_slot_kt.py \
  --feature_dir stage34_soft_slot_qwen_kt/artifacts/moocradar/features \
  --model_name_or_path qwen/Qwen3-8B \
  --output_dir stage34_soft_slot_qwen_kt/checkpoints/moocradar/ABLATION_NAME \
  --slot_mode context_target \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --context_soft_tokens 4 \
  --target_soft_tokens 2 \
  --no-include_context_text \
  --epochs 10 \
  --save_epochs 2,4,6,8,10 \
  --learning_rate 5e-5 \
  --validation_disabled \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --batch_size 8 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --no-gradient_checkpointing \
  --resume auto \
  --seed 42
```

对应评估：

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python stage34_soft_slot_qwen_kt/scripts/select_and_evaluate_soft_slot_kt.py \
  --feature_dir stage34_soft_slot_qwen_kt/artifacts/moocradar/features \
  --model_name_or_path qwen/Qwen3-8B \
  --checkpoints stage34_soft_slot_qwen_kt/checkpoints/moocradar/ABLATION_NAME/checkpoint_epoch_2.pt,stage34_soft_slot_qwen_kt/checkpoints/moocradar/ABLATION_NAME/checkpoint_epoch_4.pt,stage34_soft_slot_qwen_kt/checkpoints/moocradar/ABLATION_NAME/checkpoint_epoch_6.pt,stage34_soft_slot_qwen_kt/checkpoints/moocradar/ABLATION_NAME/checkpoint_epoch_8.pt,stage34_soft_slot_qwen_kt/checkpoints/moocradar/ABLATION_NAME/checkpoint_epoch_10.pt \
  --output_dir stage34_soft_slot_qwen_kt/results/moocradar/ABLATION_NAME \
  --slot_mode context_target \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --context_soft_tokens 4 \
  --target_soft_tokens 2 \
  --no-include_context_text \
  --selection_limit 50000 \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --eval_batch_size 32 \
  --seed 42
```

Text-only 推理：

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python stage34_soft_slot_qwen_kt/scripts/infer_soft_slot_kt.py \
  --feature_dir stage34_soft_slot_qwen_kt/artifacts/moocradar/features \
  --model_name_or_path qwen/Qwen3-8B \
  --output_dir stage34_soft_slot_qwen_kt/results/moocradar/text_only \
  --slot_mode text_only \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --no-include_context_text \
  --split test \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --eval_batch_size 32 \
  --seed 42
```

Fixed random slots 推理：

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python stage34_soft_slot_qwen_kt/scripts/infer_soft_slot_kt.py \
  --feature_dir stage34_soft_slot_qwen_kt/artifacts/moocradar/features \
  --model_name_or_path qwen/Qwen3-8B \
  --output_dir stage34_soft_slot_qwen_kt/results/moocradar/random_slots \
  --slot_mode random \
  --context_fields llm_embeddings,llm_struct_features,stage34_numeric \
  --target_fields hqtext,hqid,semantic,collaborative \
  --context_soft_tokens 4 \
  --target_soft_tokens 2 \
  --no-include_context_text \
  --split test \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --eval_batch_size 32 \
  --seed 42
```

推荐 run name：

```text
text_only
target_only
context_only
context_target_full
fixed_random_slots
wo_sdyn
wo_collab
wo_hqid
only_llm_embeddings
wo_llm_struct
only_collab
tokens_1_1
tokens_2_1
tokens_8_4
```

## 18. 质量检查清单

Stage3.2：

```bash
python scripts/check_stage32_artifacts.py \
  --out_root <out_root>
```

Stage3.4：

```bash
python scripts/validate_common_pipeline_strict.py \
  --problem_json <problem.json> \
  --out_root <out_root>
```

Soft-Slot feature store：

```bash
python -m json.tool stage34_soft_slot_qwen_kt/artifacts/<dataset>/features/feature_manifest.json
python -m json.tool stage34_soft_slot_qwen_kt/artifacts/<dataset>/features/leakage_audit.json
```

确认标签没有进入 prompt：

```text
feature_manifest.json -> audit.target_label_stored_only_in_sample_label_field = true
```

确认 Stage3.4 证据没有使用目标之后的历史：

```text
feature_manifest.json -> audit.noncausal_evidence_count = 0
```

## 19. 本地开发检查

本地 Windows 只做静态检查和 mock 测试，不加载 Qwen3-8B：

```powershell
python -m compileall stage34_soft_slot_qwen_kt
python -m unittest discover -s stage34_soft_slot_qwen_kt\tests -p test_soft_slot_kt.py -v
```

## 20. 常见误区

- 当前主方法不是 KT baseline 融合。
- 当前主方法不输入完整 Stage3.4 历史文本。
- 当前主方法不让 Qwen 生成概率文本。
- `A/B` 是分类标签 token，不是自然语言答案。
- `checkpoint_selection.json` 使用测试标签筛选 checkpoint，必须在论文中标注 `selection_uses_test_labels=true`。
- `existing_stage34_transductive` 不是严格无泄漏协议。
- `--enable_llm_graph_completion` 是 Stage3.2，`--enable_llm_summary` 是 Stage3.4。
- `question_level` 不是认知层级 `cognitive_dimension`。

## 21. 推荐论文表述

主方法描述：

```text
We freeze Qwen3-8B and train only lightweight projectors that map Stage3.4 student-state embeddings and Stage3.2 target-problem priors into the Qwen token embedding space. The projected vectors replace placeholder positions in a compact prompt as soft slots. Instead of asking the LLM to generate probabilities, we compute P(correct) from the logits of two fixed label tokens, A and B.
```

协议描述：

```text
The reported Soft-Slot Qwen results use a new-concept target split with existing Stage3.2/Stage3.4 transductive features. They should not be interpreted as strict leakage-free new-concept results unless Stage3.2/Stage3.4 are regenerated under train-only supervision and statistics.
```

消融结论链：

```text
1. Text-only < Soft-Slot: embedding slots are necessary.
2. Context-only and Target-only are both useful; Full is best: student state and target prior are complementary.
3. Fixed random slots < Full: gains come from Stage3.2/Stage3.4 information, not merely placeholder positions.
4. w/o sdyn, w/o collaborative, and w/o hqid quantify the contributions of dynamic state, collaborative prior, and hierarchical semantic-ID prior.
```

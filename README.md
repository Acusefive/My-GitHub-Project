# Stage3.2/Stage3.4-informed Soft-Slot Qwen KT

本仓库当前主实验是 **Stage3.2/Stage3.4-informed Soft-Slot Qwen Knowledge Tracing**。

核心目标：使用已有 Stage3.2 题目先验和 Stage3.4 学生认知上下文，通过 embedding soft slots 输入冻结的 Qwen3-8B，预测学生回答目标题正确的概率。

DKT、SAKT、SAINT、AKT、DIMKT、QIKT、TCKT、SimpleKT、SparseKT、DenoiseKT、RobustKT、KeenKT 等 KT 模型仍可作为对照实验，但**不参与当前主方法推理**。

详细 Soft-Slot Qwen 文档见：

```text
stage34_soft_slot_qwen_kt/README.md
```

## 1. 总体流程

```text
problem.json + student-problem-fine.json
        |
        v
Stage3.2
  题目文本向量、语义 ID、Rasch 难度、协同向量、知识图、动态先验模型
        |
        v
Stage3.4
  针对每个预测目标构造学生历史证据、认知状态摘要、context embeddings
        |
        v
Soft-Slot Feature Store
  对齐 Stage3.2/Stage3.4 产物，生成训练样本、split、memmap 特征
        |
        v
Soft-Slot Qwen KT
  冻结 Qwen3-8B，只训练 projector，将外部 embedding 映射到 Qwen token space
        |
        v
A/B logits -> P(correct)
```

主方法不是让 Qwen 自由生成概率，而是读取两个固定标签 token 的 logits：

```text
A = 正确
B = 错误
P(correct) = softmax(logit_A, logit_B)[A]
```

## 2. 关键思想

本方法借鉴 A-LLMRec Figure 2(c) Stage 2 的 embedding slot 机制：

```text
外部 embedding -> projector -> LLM token embedding space -> prompt soft slots
```

对应到本任务：

```text
Stage3.4 学生状态 embedding -> Context projector -> Context soft slots
Stage3.2 目标题先验 embedding -> Target projector -> Target soft slots
```

Qwen3-8B 本体冻结，训练时只更新轻量投影器。

## 3. 重要目录

```text
scripts/common_pipeline_strict/
  Stage3.2 / Stage3.4 通用上下文构建代码

stage34_soft_slot_qwen_kt/
  独立 Soft-Slot Qwen KT 模块

train_context.py
  KT baseline / downstream context 模型入口

dataloader/
models/
  原 KT baseline 数据和模型代码
```

Soft-Slot Qwen 的输出全部隔离在：

```text
stage34_soft_slot_qwen_kt/artifacts/
stage34_soft_slot_qwen_kt/checkpoints/
stage34_soft_slot_qwen_kt/results/
stage34_soft_slot_qwen_kt/logs/
```

同步服务器时不要用会删除服务器产物的命令覆盖这些目录。尤其不要在未排除以下目录时使用 `rsync --delete`：

```text
artifacts/
checkpoints/
results/
logs/
```

## 4. 工作环境

本地 Windows 工作区：

```text
C:\Users\xyyy\Desktop\Work
```

服务器工作目录：

```text
/home/xiaoyao/code/Work4.1
```

正式训练和评估在服务器执行。本地只做代码修改、静态检查和小型 mock 测试。

## 5. 数据集

三个数据集统一使用：

```text
split_mode=new_concept
test_concept_ratio=0.8
valid_concept_ratio=0.0
valid_ratio=0.0
validation disabled
seed=42
```

| Dataset | problem | student | Stage3.2/3.4 root | Soft-Slot features |
|---|---|---|---|---|
| MoocRadar | `datalocal/problem.json` | `datalocal/student-problem-fine.json` | `out/strict_common_pipeline` | `stage34_soft_slot_qwen_kt/artifacts/moocradar/features` |
| XES3G5M | `datalocal/xes3g5m_strict_core20/problem.json` | `datalocal/xes3g5m_strict_core20/student-problem-fine.json` | `out/xes3g5m_strict_core20_common_pipeline` | `stage34_soft_slot_qwen_kt/artifacts/xes3g5m/features` |
| FoundationalASSIST | `datalocal/foundationalassist_text_only_core200_contextcomplete/problem.json` | `datalocal/foundationalassist_text_only_core200_contextcomplete/student-problem-fine.json` | `out/foundationalassist_text_only_core200_contextcomplete_common_pipeline` | `stage34_soft_slot_qwen_kt/artifacts/foundationalassist/features` |

## 6. Stage3.2

Stage3.2 构造题目侧先验，入口代码：

```text
scripts/common_pipeline_strict/stage32.py
scripts/run_common_cognitive_pipeline_strict.py
```

主要输出位于：

```text
<out_root>/priors/
```

关键产物：

```text
stage32_manifest.json
problem_catalog.jsonl
hqtext_vectors.pkl
hqid_vectors.pkl
eqbase_vectors.pkl
semantic_vectors.pkl
item_collaborative_embeddings.pkl
item_collaborative.json
problem_mu_q.json
concept_graph_bundle.json
model_state.pt
training_report.json
implementation_defaults.json
```

Soft-Slot Qwen 主要使用其中的 Target Slot 特征：

```text
hqtext
hqid
semantic
collaborative
```

## 7. Stage3.4

Stage3.4 为每个预测目标构造学生上下文和认知状态，入口代码：

```text
scripts/common_pipeline_strict/stage34.py
scripts/run_common_cognitive_pipeline_strict.py
```

主要输出：

```text
<out_root>/contexts/stage34_manifest.json
<out_root>/contexts/contexts.jsonl
<out_root>/cache/context_embeddings.pkl
```

`contexts.jsonl` 每行对应一个预测目标，包含：

```text
user_id
target_t
target_pid
summary_fields
evidence_list
main_context_text
template_context_text
llm_context_text
```

Soft-Slot Qwen 主实验使用的 Context Slot 特征：

```text
llm_embeddings
llm_struct_features
stage34_numeric
```

主实验默认**不输入完整历史文本**，完整历史文本只作为消融使用。

## 8. 从头构建 Stage3.2/Stage3.4

如果服务器已经有 `out/...` 产物，可以跳过本节，直接准备 Soft-Slot feature store。

以 MoocRadar 为例，一次性构建：

```bash
cd /home/xiaoyao/code/Work4.1

CUDA_VISIBLE_DEVICES=4 \
python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json datalocal/problem.json \
  --student_json datalocal/student-problem-fine.json \
  --out_root out/strict_common_pipeline
```

大数据集建议分阶段：

```bash
# 只跑 Stage3.2
CUDA_VISIBLE_DEVICES=4 \
python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json <problem.json> \
  --student_json <student-problem-fine.json> \
  --out_root <out_root> \
  --stop_after_stage32

# 复用 Stage3.2，只生成 Stage3.4 contexts，不生成 embeddings
CUDA_VISIBLE_DEVICES=4 \
python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json <problem.json> \
  --student_json <student-problem-fine.json> \
  --out_root <out_root> \
  --skip_stage32 \
  --dry_run

# 复用 Stage3.2 和 contexts，生成 context_embeddings.pkl
CUDA_VISIBLE_DEVICES=4 \
python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json <problem.json> \
  --student_json <student-problem-fine.json> \
  --out_root <out_root> \
  --skip_stage32 \
  --reuse_existing_contexts
```

注意：

```text
--enable_llm_graph_completion 属于 Stage3.2
--enable_llm_summary 属于 Stage3.4
```

## 9. 当前协议与泄漏边界

当前 Soft-Slot Qwen 主实验默认复用已有 Stage3.2/Stage3.4 产物，feature store 标记为：

```text
existing_stage34_transductive
```

这表示：

- 目标样本划分是 `new_concept`。
- 但 Stage3.2/Stage3.4 原始产物可能在构建时使用过 held-out concepts 的标签或统计。
- 当前结果可以作为 transductive feature setting 结果。
- 当前结果不能声明为严格无泄漏 new-concept。

如果要做严格无泄漏版本，必须重新生成 Stage3.2/Stage3.4，并保证：

```text
Stage3.2 监督训练只使用 train concepts
Stage3.2 协同统计只使用 train concepts 交互
Stage3.4 每个目标只使用 target_t 之前的历史
目标题真实 is_correct 不进入 prompt 或 embedding
checkpoint 选择不使用测试标签，或明确标注 test-selected
```

## 10. Soft-Slot Feature Store

Feature store 是 Stage3.2/Stage3.4 与 Qwen 训练之间的桥接层。

入口：

```text
stage34_soft_slot_qwen_kt/scripts/prepare_soft_slot_embeddings.py
```

它读取：

```text
Stage3.4 contexts.jsonl
Stage3.4 context_embeddings.pkl
Stage3.2 priors/
student-problem-fine.json
```

输出：

```text
feature_manifest.json
leakage_audit.json
samples.jsonl
sample_offsets.npy
split_codes.npy
stage34_numeric.npy
context_*.npy
target_*.npy
```

准备 MoocRadar feature store：

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
  --test_concept_ratio 0.8 \
  --valid_concept_ratio 0.0 \
  --seed 42
```

其它数据集命令见：

```text
stage34_soft_slot_qwen_kt/README.md
```

## 11. Soft-Slot Qwen 训练与评估

主配置：

```text
slot_mode=context_target
context_fields=llm_embeddings,llm_struct_features,stage34_numeric
target_fields=hqtext,hqid,semantic,collaborative
context_soft_tokens=4
target_soft_tokens=2
include_context_text=false
dtype=bfloat16
```

训练和评估是两个阶段：

```text
train_soft_slot_kt.py
  训练 projector，保存候选 checkpoint

select_and_evaluate_soft_slot_kt.py
  先用 50,000 个测试样本选 checkpoint
  再跑完整测试集
```

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

XES3G5M 示例：

```bash
cd /home/xiaoyao/code/Work4.1

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

FoundationalASSIST 示例：

```bash
cd /home/xiaoyao/code/Work4.1

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

MoocRadar `learning_rate=5e-5` 显式命令见：

```text
stage34_soft_slot_qwen_kt/README.md
```

## 12. 输出结果

最终指标：

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

查看结果：

```bash
python -m json.tool stage34_soft_slot_qwen_kt/results/moocradar/context_target_slots_only_lr5e5/metrics.test.full.json
python -m json.tool stage34_soft_slot_qwen_kt/results/xes3g5m/context_target_slots_only/metrics.test.full.json
python -m json.tool stage34_soft_slot_qwen_kt/results/foundationalassist/context_target_slots_only/metrics.test.full.json
```

指标包括：

```text
AUC
ACC
F1
BCE/log loss
RMSE
positive_rate
count
best_checkpoint
predictions_path
```

## 13. 断点恢复

训练恢复：

```text
--resume auto
checkpoint_last.pt
checkpoint_epoch_*.pt
training_complete.json
```

评估恢复：

```text
checkpoint_selection.json
predictions.test.full.jsonl
metrics.test.full.json
```

重新执行同一命令会自动跳过已完成部分。完整测试中断后不要删除 `predictions.test.full.jsonl`。

## 14. 消融实验

推荐消融分三层。

必须做，三个数据集都建议跑：

| Variant | 目的 |
|---|---|
| Text-only Qwen | 无 soft slots；仅用任务说明和目标题自然语言信息 |
| Target slot only | 验证 Stage3.2 目标题先验 slot |
| Context slot only | 验证 Stage3.4 学生状态 slot；仍保留目标题自然语言锚点 |
| Full Soft-Slot | 主方法 |
| Fixed random slots | 固定随机 soft slots 负对照；检验 slot 位置/提示形式本身是否带来虚假提升 |
| w/o sdyn | 验证动态状态贡献 |
| w/o collaborative | 验证协同题目先验贡献 |
| w/o hqid | 验证分层语义 ID 向量贡献 |

注意：当前 `random` 是固定随机 slots，没有可训练 projector，因此它不是“可训练参数容量”对照。如果要控制 projector 容量，应另做 shuffled features 或 random features + trainable projector 实验；当前代码未内置该实验。

Stage3.4 细粒度：

```text
only llm_embeddings
w/o llm_struct_features
w/o stage34_numeric
w/o sdyn
```

Stage3.2 细粒度：

```text
only hqtext
only hqid
only semantic
only collaborative
w/o hqid
w/o collaborative
```

Slot 数量：

```text
1+1
2+1
4+2
8+4
```

详细消融命令见：

```text
stage34_soft_slot_qwen_kt/README.md
```

## 15. KT baseline 对照

KT baseline 仍使用原有路径，主要入口：

```text
train_context.py
```

baseline / ours 对照实验可以继续运行，但它们不是当前 Soft-Slot Qwen 主方法的一部分。

典型 baseline 命令形式：

```bash
python train_context.py \
  --model_name dkt \
  --context_type none \
  --split_mode new_concept \
  --test_concept_ratio 0.8 \
  --valid_concept_ratio 0.0 \
  --problem_json <problem.json> \
  --student_json <student-problem-fine.json> \
  --dataset_dir <dataset_dir>
```

典型 KT-context 命令形式：

```bash
python train_context.py \
  --model_name dkt \
  --context_type all \
  --split_mode new_concept \
  --test_concept_ratio 0.8 \
  --valid_concept_ratio 0.0 \
  --problem_json <problem.json> \
  --student_json <student-problem-fine.json> \
  --context_embeddings_path <out_root>/cache/context_embeddings.pkl \
  --dataset_dir <dataset_dir>
```

注意：这些命令用于结果对照，不用于 Soft-Slot Qwen 推理。

## 16. 质量检查

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

本地检查：

```powershell
python -m compileall stage34_soft_slot_qwen_kt
python -m unittest discover -s stage34_soft_slot_qwen_kt\tests -p test_soft_slot_kt.py -v
```

本地不要加载 Qwen3-8B。

## 17. 给后续维护者和 LLM 的注意事项

- 当前主实验是 Soft-Slot Qwen KT，不是 KT baseline 融合。
- `stage34_soft_slot_qwen_kt/` 是独立模块，不要把它混入 `train_context.py`。
- Qwen3-8B 冻结，只训练 projector。
- 当前结果协议是 `existing_stage34_transductive`，不是严格无泄漏。
- `Text-only` 和当前 `random` 不训练 projector，使用 `infer_soft_slot_kt.py`。
- 训练完成不等于有最终结果；最终结果要等 `select_and_evaluate_soft_slot_kt.py` 完整结束。
- `metrics.test.full.json` 才是完整测试指标文件。
- 完整测试中断后，保留 `predictions.test.full.jsonl` 并重跑同一命令即可恢复。

## 18. 推荐论文表述

方法描述：

```text
We freeze Qwen3-8B and train only lightweight projectors that map Stage3.4 student-state embeddings and Stage3.2 target-problem priors into the Qwen token embedding space. The projected vectors replace placeholder positions in a compact prompt as soft slots. Instead of asking the LLM to generate probabilities, we compute P(correct) from the logits of two fixed label tokens, A and B.
```

协议描述：

```text
The reported Soft-Slot Qwen results use a new-concept target split with existing Stage3.2/Stage3.4 transductive features. They should not be interpreted as strict leakage-free new-concept results unless Stage3.2/Stage3.4 are regenerated under train-only supervision and statistics.
```

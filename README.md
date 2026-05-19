# Strict Common Cognitive Pipeline

本仓库用于在知识追踪任务中构建认知增强上下文，并在 DKT、SAKT、SAINT 等下游模型中使用这些上下文特征。原始实验流程已经在 MoocRadar 数据集上跑通；当前代码在不改动核心 KT 模型主体的前提下，新增了 XES3G5M 数据集适配。

## 1. 原 MoocRadar 版本

MoocRadar 版本使用仓库已有的数据格式和训练入口：

- 题目目录：`problem.json`
- 学生交互序列：`student-problem-fine.json`
- 下游数据集加载器：`dataloader/moocradar_strict.py`
- 通用上下文构建入口：`scripts/run_common_cognitive_pipeline_strict.py`
- 下游训练入口：`train_context.py`

原流程的核心阶段是：

1. 读取 MoocRadar 格式的题目和学生序列。
2. stage32 构建题目语义 ID、语义向量、Rasch 难度增强、协同邻居、概念图等先验产物。
3. stage34 根据目标题和学生历史交互生成上下文，并可进一步生成 LLM 摘要。
4. 将 `context_embeddings.pkl` 输入到 `train_context.py`，训练 DKT、SAKT、SAINT 等模型。

MoocRadar 版本的典型运行方式：

```bash
python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json datalocal/problem.json \
  --student_json datalocal/student-problem-fine.json \
  --out_root out/strict_common_pipeline
```

下游训练示例：

```bash
python train_context.py \
  --model_name dkt \
  --context_type llm \
  --problem_json datalocal/problem.json \
  --student_json datalocal/student-problem-fine.json \
  --context_embeddings_path out/strict_common_pipeline/cache/context_embeddings.pkl \
  --dataset_dir datasets/MOOCRadarStrict
```

## 2. XES3G5M 适配原则

XES3G5M 不改变核心方案和 KT 模型主体，只新增数据适配和质量检查脚本。适配后的目标仍然是生成和 MoocRadar 版本一致的两个输入文件：

- `datalocal/xes3g5m_text_only/problem.json`
- `datalocal/xes3g5m_text_only/student-problem-fine.json`

关键约定：

- 不使用 XES3G5M 官方 fold/test 划分。
- 继续使用概念冷启动划分：训练概念 : 测试概念 = `2:8`。
- `question_level` 不是认知层级，不能作为 `cognitive_dimension`。
- `cognitive_dimension` 由服务器上的 Qwen3-8B 生成。
- XES3G5M 实验只使用纯文本无图片题。

纯文本过滤口径为 `strict_text_only`。当前代码会同时扫描：

- `content`
- `analysis`
- `options`

只要出现以下图片占位符，就剔除该题：

```text
question_\d+-image_\d+
analysis_\d+-image_\d+
```

## 3. XES3G5M 字段映射

XES3G5M 原始字段：

```text
metadata/questions.json:
  content, kc_routes, answer, analysis, type, options

question_level/train_valid_sequences_quelevel.csv:
  fold, uid, questions, concepts, responses, timestamps, selectmasks

question_level/test_quelevel.csv:
  fold, uid, questions, concepts, responses, timestamps
```

适配后的映射：

- 题目 ID：`qid` -> `Q_{qid}`
- 学生 ID：`uid` -> `U_{uid}`
- concepts：使用 `kc_routes` 的叶子知识点名称
- `cognitive_dimension`：使用 LLM 生成的 1-4 级标签

## 4. 新增和修改的脚本

### 4.1 数据统计

`scripts/summarize_xes3g5m.py`

用于统计 XES3G5M 在不同过滤口径下的数据规模，并输出：

```text
out/xes3g5m_stats/xes3g5m_stats.json
out/xes3g5m_stats/xes3g5m_stats.md
```

运行示例：

```bash
python scripts/summarize_xes3g5m.py \
  --dataset_root /home/xiaoyao/code/Work4.1/datalocal/XES3G5M
```

### 4.2 LLM 生成 cognitive_dimension

`scripts/generate_xes3g5m_cognitive_dimensions.py`

调用 OpenAI-compatible API，为 XES3G5M 纯文本题生成 `cognitive_dimension`。

输出：

```text
out/xes3g5m_cognitive/cognitive_dimensions.jsonl
out/xes3g5m_cognitive/cognitive_dimension_failures.jsonl
```

当前采用更严格的 1-4 级定义：

- `1`：直接计算、识记、单公式
- `2`：常规模板应用，即使有 2-4 个算术步骤也可以是 2
- `3`：关系建模、策略选择、枚举、逆推、条件转化
- `4`：数字谜、复杂还原、长周期逆推、不变量/奇偶性论证、复杂组合策略

服务器运行示例：

```bash
python scripts/generate_xes3g5m_cognitive_dimensions.py \
  --questions_json /home/xiaoyao/code/Work4.1/datalocal/XES3G5M/metadata/questions.json \
  --image_filter strict_text_only \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_model qwen3-8b \
  --workers 4 \
  --llm_timeout_sec 240 \
  --out_jsonl out/xes3g5m_cognitive/cognitive_dimensions.jsonl \
  --fail_jsonl out/xes3g5m_cognitive/cognitive_dimension_failures.jsonl \
  --overwrite
```

### 4.3 cognitive_dimension 检查

`scripts/check_xes3g5m_cognitive_dimensions.py`

用于检查 LLM 标签的覆盖率、失败项、分布、置信度，并生成抽样人工审核文件。

输出：

```text
out/xes3g5m_cognitive_check/cognitive_dimension_check_report.md
out/xes3g5m_cognitive_check/cognitive_dimension_check_report.json
out/xes3g5m_cognitive_check/manual_review_sample.csv
out/xes3g5m_cognitive_check/low_confidence_or_unknown.csv
out/xes3g5m_cognitive_check/concept_level_spread.csv
out/xes3g5m_cognitive_check/missing_labels.csv
```

运行示例：

```bash
python scripts/check_xes3g5m_cognitive_dimensions.py \
  --dataset_root /home/xiaoyao/code/Work4.1/datalocal/XES3G5M \
  --cognitive_jsonl out/xes3g5m_cognitive/cognitive_dimensions.jsonl \
  --fail_jsonl out/xes3g5m_cognitive/cognitive_dimension_failures.jsonl \
  --image_filter strict_text_only \
  --output_dir out/xes3g5m_cognitive_check \
  --sample_per_level 50
```

当前已验证的一版结果：

```text
Filtered questions: 4912
Valid labels: 4912
Missing labels: 0
Extra labels: 0
Failure rows: 0
Level 1: 664
Level 2: 2661
Level 3: 1537
Level 4: 50
Confidence: high 4909, medium 3
```

### 4.4 转换为 pipeline 输入格式

`scripts/prepare_xes3g5m_text_only.py`

将 XES3G5M 原始 question-level 数据转换为当前 pipeline 的输入格式。

运行示例：

```bash
python scripts/prepare_xes3g5m_text_only.py \
  --dataset_root /home/xiaoyao/code/Work4.1/datalocal/XES3G5M \
  --out_dir datalocal/xes3g5m_text_only \
  --image_filter strict_text_only \
  --cognitive_dimensions_jsonl out/xes3g5m_cognitive/cognitive_dimensions.jsonl \
  --require_cognitive_dimension
```

当前已验证的转换结果：

```text
students: 34288
questions_in_catalog: 4912
questions_observed: 4911
interactions: 3571871
positive_rate: 0.7994493642
unique_concepts: 674
```

注意：这里的 `students=34288` 是 sequence records，因为转换脚本把 `train_valid_sequences_quelevel.csv` 和 `test_quelevel.csv` 拼接为统一序列记录。论文中如果统计 unique users，需要另行计算。

## 5. 分阶段运行 pipeline

XES3G5M 数据较大，建议不要一次性重跑 full pipeline。当前 `scripts/run_common_cognitive_pipeline_strict.py` 已新增：

- `--stop_after_stage32`：只跑完 stage32 后停止
- `--skip_stage32`：复用已有 stage32 产物
- `--reuse_existing_contexts`：复用已有 stage34 contexts
- `--dry_run`：生成 contexts/preview，但不构建最终 embeddings

### 5.1 stage32

完整方案的 stage32 图补全默认不会调用 LLM，必须显式开启：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json datalocal/xes3g5m_text_only/problem.json \
  --student_json datalocal/xes3g5m_text_only/student-problem-fine.json \
  --out_root out/xes3g5m_text_only_strict_common_pipeline \
  --enable_llm_graph_completion \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_model qwen3-8b \
  --llm_timeout_sec 240 \
  --llm_max_tokens 256 \
  --stop_after_stage32
```

注意：指定显卡使用 `CUDA_VISIBLE_DEVICES=4`，不是 `cuda_devices=4`。

stage32 检查：

```bash
python scripts/check_stage32_artifacts.py \
  --out_root out/xes3g5m_text_only_strict_common_pipeline
```

当前已验证结果：

```text
PASSED: True
FAILURES: 0
WARNINGS: 1
collab_vectors_missing_for_unobserved_or_short_sequence_items
```

该 warning 可以接受：catalog 中有 4912 题，学生序列中实际出现 4911 题，Word2Vec 协同向量只会覆盖出现在序列中的题。

### 5.2 stage34 contexts dry-run

先生成 contexts 和 preview，不生成 embeddings：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json datalocal/xes3g5m_text_only/problem.json \
  --student_json datalocal/xes3g5m_text_only/student-problem-fine.json \
  --out_root out/xes3g5m_text_only_strict_common_pipeline \
  --skip_stage32 \
  --dry_run
```

检查：

```bash
ls -lh out/xes3g5m_text_only_strict_common_pipeline/contexts/contexts.jsonl
head -n 2 out/xes3g5m_text_only_strict_common_pipeline/reports/context_preview.txt
```

### 5.3 LLM summary

stage34 的 LLM 摘要和 stage32 的图补全是两个不同阶段。确认 contexts 合理后，再运行：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json datalocal/xes3g5m_text_only/problem.json \
  --student_json datalocal/xes3g5m_text_only/student-problem-fine.json \
  --out_root out/xes3g5m_text_only_strict_common_pipeline \
  --skip_stage32 \
  --reuse_existing_contexts \
  --enable_llm_summary \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_model qwen3-8b \
  --llm_timeout_sec 240 \
  --llm_summary_workers 8 \
  --dry_run
```

### 5.4 embeddings

LLM summary 检查通过后，构建最终上下文向量：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/run_common_cognitive_pipeline_strict.py \
  --problem_json datalocal/xes3g5m_text_only/problem.json \
  --student_json datalocal/xes3g5m_text_only/student-problem-fine.json \
  --out_root out/xes3g5m_text_only_strict_common_pipeline \
  --skip_stage32 \
  --reuse_existing_contexts
```

最终产物：

```text
out/xes3g5m_text_only_strict_common_pipeline/cache/context_embeddings.pkl
```

## 6. 下游训练

XES3G5M 训练时继续使用概念冷启动 2:8：

```bash
CUDA_VISIBLE_DEVICES=4 python train_context.py \
  --model_name dkt \
  --context_type llm \
  --split_mode new_concept \
  --test_concept_ratio 0.8 \
  --valid_concept_ratio 0.0 \
  --problem_json datalocal/xes3g5m_text_only/problem.json \
  --student_json datalocal/xes3g5m_text_only/student-problem-fine.json \
  --context_embeddings_path out/xes3g5m_text_only_strict_common_pipeline/cache/context_embeddings.pkl \
  --dataset_dir datasets/XES3G5MTextOnlyStrict \
  --ckpt_root ckpts_context/XES3G5MTextOnly
```

替换 `--model_name` 可运行：

```text
dkt
sakt
saint
```

## 7. 质量检查清单

在进入训练前，应至少检查：

```bash
grep -E 'question_[0-9]+-image_[0-9]+|analysis_[0-9]+-image_[0-9]+' \
  datalocal/xes3g5m_text_only/problem.json | head
```

无输出表示 `problem.json` 中没有图片占位符。

检查 cognitive_dimension：

```bash
cat out/xes3g5m_cognitive_check/cognitive_dimension_check_report.md
```

检查 stage32：

```bash
python scripts/check_stage32_artifacts.py \
  --out_root out/xes3g5m_text_only_strict_common_pipeline
```

检查 stage34：

```bash
python scripts/validate_common_pipeline_strict.py \
  --problem_json datalocal/xes3g5m_text_only/problem.json \
  --out_root out/xes3g5m_text_only_strict_common_pipeline
```

抽样人工检查 contexts：

```bash
python scripts/sample_manual_review.py \
  --problem_json datalocal/xes3g5m_text_only/problem.json \
  --contexts_jsonl out/xes3g5m_text_only_strict_common_pipeline/contexts/contexts.jsonl \
  --output_csv out/xes3g5m_text_only_strict_common_pipeline/reports/manual_review_sample.csv \
  --sample_size 200
```

## 8. 路径说明

本地 Windows 开发环境：

```text
C:\Users\xyyy\Desktop\Work
D:\Dataset\XES3G5M
```

服务器运行环境：

```text
/home/xiaoyao/code/Work4.1
/home/xiaoyao/code/Work4.1/datalocal/XES3G5M
```

脚本中已经尽量提供默认路径，但正式实验建议显式传入 `--dataset_root`、`--problem_json`、`--student_json` 和 `--out_root`，避免路径误用。

## 9. 不要混淆的点

- `question_level` 不是认知层级，不能当 `cognitive_dimension`。
- `--enable_llm_graph_completion` 属于 stage32 图补全。
- `--enable_llm_summary` 属于 stage34 上下文摘要。
- XES3G5M 不使用官方 fold/test 作为最终实验划分。
- 大规模实验不要直接重跑 full pipeline，应按 stage32、stage34 contexts、LLM summary、embeddings 分阶段运行和检查。

# Stage3.4 Soft-Slot Qwen KT：设计与代码说明

`stage34_soft_slot_qwen_kt/` 是当前主实验目录。它接收 Stage3.2 的题目侧先验和 Stage3.4 的学生认知上下文，通过可训练 projector 将两类外部特征变成 Qwen embedding 空间中的 soft tokens，再由冻结的 Qwen 完成知识追踪二分类。

本文档面向需要阅读、维护或扩展代码的人，重点解释模块边界、数据流和设计选择，不把 README 写成环境配置或逐命令复现手册。

## 1. 一句话理解主方法

```text
先从学生历史中找出“对当前题真正有用的证据”，
再把学生状态和当前题先验分别压缩成 soft slots，
最后让冻结 Qwen 判断本次作答更可能正确还是错误。
```

对应的任务是估计：

```text
P(y_t = 1 | H_<t, q_t)
```

其中 `H_<t` 是当前目标之前的学生交互历史，`q_t` 是目标题。本方法不把整段历史原样交给 LLM，而是先构造目标条件化的认知上下文。

## 2. 端到端数据流

```text
problem.json + student-problem-fine.json
        │
        ▼
scripts/common_pipeline_strict/stage32.py
        │  题目文本、分层语义 ID、认知语义、协同表示、知识图、动态先验
        ▼
scripts/common_pipeline_strict/stage34.py
        │  目标条件化证据列表、模板摘要、结构化认知总结、context embeddings
        ▼
scripts/prepare_soft_slot_embeddings.py
        │  对齐样本并物化 Context/Target 特征数组
        ▼
soft_slot_kt/data.py
        │  读取样本、拼接特征、构造 prompt 与 slot masks
        ▼
soft_slot_kt/model.py
        │  Context/Target projector → inputs_embeds → frozen Qwen
        ▼
logit(A=正确), logit(B=错误)
        │
        ▼
P(correct) + 监督损失 + 评估指标
```

主实验目录只负责 Feature Store 之后的桥接、建模和评估。Stage3.2/Stage3.4 位于仓库上层，是主实验的上游特征构造模块。

## 3. 为什么采用这套方案

### 3.1 先检索再预测

学生历史中同时包含相关、重复和无关交互。直接拼接完整历史会把证据选择问题交给 LLM，也会增加输入长度和噪声。本项目先以目标题为条件选择少量互补证据，再生成学生状态表示。

### 3.2 区分“学生状态”和“目标题要求”

同一个学生面对不同目标题时，相关历史和风险判断都可能不同。因此代码使用两组独立输入：

- Context Slot 表示当前目标题下的学生认知状态；
- Target Slot 表示目标题自身的语义、认知层级和协同先验。

两组特征使用独立 projector，避免把“学生会什么”和“题目要求什么”在输入端混为一个向量。

### 3.3 冻结 Qwen，只学习接口

上游特征与 Qwen token embeddings 不在同一表示空间。`SoftSlotProjector` 学习的是两者之间的映射，而不是重新训练 Qwen 的语言能力。这样可以把训练目标集中在“如何把认知证据组织成 LLM 可消费的连续表示”。

### 3.4 用标签 logits 做概率预测

知识追踪需要稳定的概率输出，不需要自然语言生成。代码读取正确/错误候选标签的分数并做二分类 softmax，从而直接得到 `P(correct)`。

## 4. 上游认知信息如何形成

### 4.1 Stage3.2：题目侧表示

实现文件：`../scripts/common_pipeline_strict/stage32.py`。

Stage3.2 将题目元数据和学生行为统计转换为题目级先验。与 Soft-Slot 主配置直接相连的字段如下：

| Feature Store 字段 | Stage3.2 来源 | 表达内容 |
|---|---|---|
| `hqtext` | `hqtext_vectors.pkl` | 题目文本语义 |
| `hqid` | `hqid_vectors.pkl` | 分层语义 ID |
| `semantic` | `semantic_vectors.pkl` | 加入认知/难度修正后的题目表示 |
| `collaborative` | `item_collaborative_embeddings.pkl` | 基于学生行为关系的题目协同表示 |

Stage3.2 还生成 `concept_graph_bundle.json`、题目难度和动态先验模型等产物，供 Stage3.4 构造目标相关证据。

### 4.2 Stage3.4：目标条件化证据选择

实现文件：`../scripts/common_pipeline_strict/stage34.py`。

对每个 `(user_id, target_t, target_pid)`，Stage3.4 只考虑 `target_t` 之前的历史交互。候选评分综合：

- 目标题与历史题的知识点重合；
- 低阶前置、同阶迁移、高阶反馈等认知层级关系；
- 题目语义相似度；
- 知识图关系；
- 协同相似度；
- 随语义变化距离衰减的时间权重；
- 学生历史正确性和动态先验状态。

第一阶段按目标相关性召回候选，并可叠加 reranker 分数。第二阶段先取支持度最高的证据，再迭代优化：

```text
最终选择分数 = 支持度 + 覆盖增益 - 冗余惩罚
```

覆盖增益关注认知角色、知识点和图邻居是否带来新信息；冗余惩罚抑制知识、角色和语义高度重复的证据。

这部分是使用预计算表示与固定权重的启发式检索，不是随 Soft-Slot projector 一起训练的端到端检索器。

### 4.3 知识图的实际边界

Stage3.4 只读取一个图源：`concept_graph_bundle.json`。同一张图在两个位置发挥作用：

1. 候选评分中的结构关系增益；
2. 第二阶段选择中的图邻居覆盖增益。

因此，代码不是三个彼此独立的“召回图、重排图、覆盖图”模块。

### 4.4 结构化认知总结

Stage3.4 先根据证据生成模板状态字段；启用 LLM 总结时，再要求总结模型严格输出：

```text
mastered_concepts
weak_concepts
transfer_state
risk_level
evidence_quality
diagnosis
```

这些字段被转换为三类可供主实验使用的信息：

- `llm_embeddings`：结构化总结与证据上下文形成的文本表示；
- `llm_struct_features`：掌握/薄弱点数量、存在性、诊断长度、风险等级和证据质量等显式结构特征；
- `stage34_numeric`：`sdyn`、候选数量和最终证据数量等数值状态。

总结阶段只负责把证据压缩为认知状态。最终 `P(correct)` 由后续 Soft-Slot Qwen 计算，不由总结模型生成。

## 5. Feature Store：模块之间的数据契约

入口脚本 `scripts/prepare_soft_slot_embeddings.py` 调用 `soft_slot_kt/prepare.py`。它将上游产物转成按行读取的统一特征库。

### 5.1 样本身份对齐

每个预测样本使用三元组标识：

```text
(user_id, target_t, target_pid)
```

准备阶段逐行核对 `contexts.jsonl` 和 `context_embeddings.pkl[index]` 的三元组。如果顺序或身份不一致，代码直接报错，避免把学生状态错配到另一道题。

### 5.2 Feature Store 主要文件

| 文件 | 作用 |
|---|---|
| `feature_manifest.json` | 记录字段、数组路径、shape、来源签名和题目索引 |
| `samples.jsonl` | 保存样本三元组、标签、split 和可选上下文文本 |
| `sample_offsets.npy` | 支持对 `samples.jsonl` 随机定位 |
| `split_codes.npy` | 训练/验证/测试行索引编码 |
| `context_*.npy` | 与预测样本逐行对齐的 Stage3.4 特征 |
| `target_*.npy` | 与 `problem_ids` 对齐的 Stage3.2 题目特征 |
| `stage34_numeric.npy` | Stage3.4 数值状态 |

`FeatureStore` 在读取时分别拼接所选 Context 字段和 Target 字段。这样新增或删除某一路特征时，不需要改动 Qwen 前向逻辑，只需要调整字段选择和输入维度。

### 5.3 主配置的数据契约

```text
Context fields:
  llm_embeddings
  llm_struct_features
  stage34_numeric

Target fields:
  hqtext
  hqid
  semantic
  collaborative
```

`stage34_numeric` 当前是三维向量：`sdyn`、归一化的第一阶段候选数、归一化的最终证据数。

## 6. Soft-Slot Qwen 的前向过程

核心实现位于 `soft_slot_kt/model.py`。

### 6.1 Projector

Context 和 Target 各自使用一个 `SoftSlotProjector`：

```text
输入特征
  → LayerNorm
  → Linear(projector_hidden_dim)
  → GELU
  → Dropout
  → Linear(num_soft_tokens × llm_hidden_dim)
  → [batch, num_soft_tokens, llm_hidden_dim]
```

主配置使用 4 个 Context soft tokens 和 2 个 Target soft tokens。Projector 输出维度从 `llm.config.hidden_size` 获取，因此 slot 向量与 Qwen 的 token embedding 维度严格一致。

### 6.2 Prompt 与 slot 替换

`soft_slot_kt/prompts.py` 将 prompt 拆成三段：

```text
任务说明 + Context 占位位置
目标题自然语言锚点 + Target 占位位置
标签说明与“标签：”结尾
```

目标题自然语言锚点包含知识点、认知层级和题干。主配置 `include_context_text=false`，因此完整历史文本不直接进入 prompt；学生状态主要通过 Context soft slots 注入。

Collator 记录两类占位位置的 mask。模型先用 Qwen 自身的 embedding layer 得到普通 token embeddings，再把 mask 对应位置替换成 projector 输出，最终调用：

```python
llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
```

### 6.3 分类与损失

默认标签语义是：

```text
A = 正确
B = 错误
```

如果候选标签是单 token，模型直接读取 prompt 最后位置的两个 token logits；若 tokenizer 将候选标签切成多个 token，代码会计算候选序列的平均 log-probability。

随后构造：

```text
class_logits = [incorrect_score, correct_score]
P(correct) = softmax(class_logits)[1]
loss = cross_entropy(class_logits, label)
```

### 6.4 参数更新边界

`SoftSlotQwenKT.__init__` 将所有 `llm.parameters()` 的 `requires_grad` 设为 `False`。训练脚本只把 `model.trainable_parameters()` 交给 AdamW，checkpoint 也只保存非 `llm.*` 的状态。

因此主实验真正学习的是 Context/Target projector，而不是 Qwen 权重。

## 7. Python 包职责

| 文件 | 职责 |
|---|---|
| `soft_slot_kt/model.py` | 标签解析、projector、slot 替换、冻结 Qwen 前向和分类损失 |
| `soft_slot_kt/prompts.py` | prompt 分段、占位位置编码、padding 和 slot masks |
| `soft_slot_kt/data.py` | Feature Store、Dataset、Collator、Context/Target 特征拼接 |
| `soft_slot_kt/prepare.py` | 对齐 Stage3.2/Stage3.4 产物并物化特征数组 |
| `soft_slot_kt/runtime.py` | 加载冻结 LLM、组装数据与模型、评估、checkpoint 读写 |
| `soft_slot_kt/source_io.py` | 读取并按时间排序学生序列 |
| `soft_slot_kt/cli.py` | 统一模型和数据参数定义 |
| `soft_slot_kt/utils.py` | 指标、随机种子、设备、dtype 和原子写入等公共函数 |

## 8. 脚本入口分别负责什么

| 脚本 | 角色 |
|---|---|
| `scripts/prepare_soft_slot_embeddings.py` | 建立上游产物到 Feature Store 的桥接 |
| `scripts/train_soft_slot_kt.py` | 训练有 projector 的 soft-slot 配置 |
| `scripts/infer_soft_slot_kt.py` | 运行无需训练 projector 的 `text_only` 或固定随机 slot 等路径，也可加载已训练 checkpoint 推理 |
| `scripts/select_and_evaluate_soft_slot_kt.py` | 比较候选 checkpoint，并输出逐样本概率与汇总指标 |
| `scripts/run_*_full.sh` | 三个数据集的实验参数封装，不包含核心算法 |
| `scripts/run_*_smoke.sh` | 小规模接口检查封装，不代表正式实验逻辑 |

核心逻辑应优先在 `soft_slot_kt/` 中阅读，shell 脚本主要是参数组合。

## 9. 各实验模式在回答什么问题

`runtime.slot_counts()` 定义了五种输入模式：

| `slot_mode` | 输入 | 用途 |
|---|---|---|
| `text_only` | 无 soft slots | 检查自然语言题面和标签提示本身能提供多少信息 |
| `context` | 仅 Context slots | 检查学生状态表示的贡献 |
| `target` | 仅 Target slots | 检查题目侧先验的贡献 |
| `context_target` | Context + Target slots | 当前完整主配置 |
| `random` | 固定随机 Context + Target slots | 检查 slot 位置和提示形式本身是否造成虚假收益 |

其中 `random` 使用注册为 buffer 的固定随机向量，没有可训练 projector，因此它是输入位置/形式的负对照，不是参数量匹配的容量对照。

字段级消融通过选择 Context/Target 字段以及 `drop_sdyn`、`drop_collab` 实现。它们用于判断性能来自哪类认知信息，而不是另一条主模型路径。

## 10. 产物如何对应代码阶段

```text
artifacts/<dataset>/features/
  Feature Store：上游产物经过对齐后的模型输入

checkpoints/<dataset>/<run>/
  Projector 状态、优化器状态和训练元数据

results/<dataset>/<run>/predictions*.jsonl
  逐样本 user/target、真实标签、预测概率与预测类别

results/<dataset>/<run>/metrics*.json
  AUC、ACC、F1、BCE、RMSE 等汇总指标
```

这些目录分别对应“输入契约、可训练参数、逐样本输出、汇总结果”，不应混作同一类实验文件。

## 11. 建议的阅读顺序

1. `soft_slot_kt/model.py`：先理解 projector、slot 替换和 A/B logits；
2. `soft_slot_kt/prompts.py`：理解 soft slots 在 prompt 中的位置；
3. `soft_slot_kt/data.py`：理解 Feature Store 如何变成 batch；
4. `soft_slot_kt/prepare.py`：理解上游特征如何对齐；
5. `../scripts/common_pipeline_strict/stage34.py`：理解证据检索和结构化总结；
6. `../scripts/common_pipeline_strict/stage32.py`：理解题目侧先验；
7. `soft_slot_kt/runtime.py` 和 `scripts/`：理解训练与评估编排。

## 12. 容易混淆的边界

- `stage34_soft_slot_qwen_kt/` 是当前主实验；`train_context.py` 中的 KT 模型是对照路径。
- Stage3.4 的总结 LLM 与 Soft-Slot Qwen 是两个模型阶段：前者压缩证据，后者预测正确率。
- Stage3.4 检索使用固定评分逻辑；可训练参数位于下游 projector。
- 主配置不输入完整历史文本，`include_context_text` 只用于相应对照。
- Qwen 不输出 JSON 或自然语言概率，最终输出来自标签 token logits。
- Context 与 Target 特征在进入 projector 前分别拼接，二者不会共用同一个投影器。
- `A/B` 是分类标签，不是学生的原始作答内容。

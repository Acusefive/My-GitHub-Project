# Cognitive-RAG Soft-Slot Qwen Knowledge Tracing

本仓库研究的是一个面向知识追踪（Knowledge Tracing, KT）的组合式方案：先从学生历史中提取与当前目标题相关的认知证据，再把学生状态和目标题先验映射为 soft slots，交给冻结的 Qwen 完成正确/错误判别。

当前主实验实现位于：

```text
C:\Users\xyyy\Desktop\Work\stage34_soft_slot_qwen_kt
```

对应的仓库相对路径是 [`stage34_soft_slot_qwen_kt/`](stage34_soft_slot_qwen_kt/)。`scripts/common_pipeline_strict/` 负责为主实验构建 Stage3.2/Stage3.4 上游特征；`train_context.py` 及 `models/` 中的传统 KT 模型只承担对照实验，不属于主方法的推理链。

本文档用于说明项目的代码、思路和方案。环境安装、服务器命令、显存配置和逐步复现流程不作为 README 的重点。

## 1. 项目要解决什么问题

对学生 (u) 在时刻 (t) 的目标题 (q_t)，知识追踪需要估计：

```text
P(y_t = 1 | H_<t, q_t)
```

其中 `H_<t` 是目标题之前的交互历史，`y_t=1` 表示回答正确。这个任务有两个实际难点：

1. 学生历史很长，但并非所有历史交互都与当前目标题有关。
2. 已有的学生状态、题目语义、知识图和协同信号都是外部特征，不能直接作为 Qwen 的 token 输入。

本项目的解决思路是：

- 用 Stage3.2 建立题目侧先验；
- 用 Stage3.4 针对当前目标题选择历史证据并压缩认知状态；
- 用两个轻量 projector 将学生状态和目标题特征转换为 Qwen embedding 空间中的 soft tokens；
- 冻结 Qwen，只根据两个标签 token 的 logits 输出答对概率。

## 2. 主方法全景

```text
题目元数据 + 学生交互序列
        │
        ▼
Stage3.2：题目侧表示
  文本语义、分层语义 ID、认知语义、协同表示、知识图、动态先验
        │
        ▼
Stage3.4：目标条件化的认知上下文
  历史候选评分 → 证据筛选 → 结构化认知摘要 → context embeddings
        │
        ▼
Soft-Slot Feature Store
  对齐 (user_id, target_t, target_pid)，组织 Context/Target 两组特征
        │
        ▼
Context projector + Target projector
  外部特征 → Qwen token embedding space
        │
        ▼
Frozen Qwen3-8B
  用 soft slots 替换 prompt 中的占位 token
        │
        ▼
logit(A=正确), logit(B=错误) → softmax → P(correct)
```

这条链路的核心不是“让大模型阅读完整历史并自由生成答案”，而是先把历史压缩成与目标题相关的证据和状态，再把这些结构化信息作为连续向量注入冻结 LLM。

## 3. 四个核心模块

### 3.1 Stage3.2：构造题目侧先验

代码位于 [`scripts/common_pipeline_strict/stage32.py`](scripts/common_pipeline_strict/stage32.py)。它把题目文本、知识点、认知层级和学生行为统计转换为可复用的题目表示。

主实验使用的 Target Slot 特征包括：

| 字段 | 含义 |
|---|---|
| `hqtext` | 题目文本语义表示 |
| `hqid` | 分层语义 ID 表示 |
| `semantic` | 融合认知/难度信息后的题目语义表示 |
| `collaborative` | 由学生行为关系得到的题目协同表示 |

Stage3.2 还产生 `concept_graph_bundle.json` 等中间产物，供 Stage3.4 的证据关联和覆盖控制使用。

### 3.2 Stage3.4：构造目标条件化的学生状态

代码位于 [`scripts/common_pipeline_strict/stage34.py`](scripts/common_pipeline_strict/stage34.py)。对于每个目标题，Stage3.4 只把它之前的交互作为候选历史，并执行两阶段证据选择：

1. 综合知识点重合、认知层级关系、语义相似度、知识图关系、协同信号和时间衰减，对历史交互进行目标相关性评分；
2. 在高相关候选中兼顾支持度、覆盖增益和冗余惩罚，选出一组互补证据。

这里的检索是基于预计算特征和固定评分规则的目标条件化启发式检索，不是端到端可学习检索器。知识图只有一个来源 `concept_graph_bundle.json`，它在候选相关性和证据覆盖两个环节被复用。

启用结构化认知总结时，Stage3.4 把选中证据压缩为六个受约束字段：

```text
mastered_concepts
weak_concepts
transfer_state
risk_level
evidence_quality
diagnosis
```

需要注意，Stage3.4 的总结模型负责“证据压缩”，Soft-Slot Qwen 负责“正确率预测”，两者是前后相接但相互独立的模块。

### 3.3 Feature Store：连接上游特征和 Qwen

代码位于 [`stage34_soft_slot_qwen_kt/soft_slot_kt/prepare.py`](stage34_soft_slot_qwen_kt/soft_slot_kt/prepare.py)。Feature Store 不是新的预测模型，而是数据契约层，主要负责：

- 对齐 `contexts.jsonl` 与 `context_embeddings.pkl` 中的样本顺序；
- 以 `(user_id, target_t, target_pid)` 绑定上下文、目标题和监督标签；
- 将 Stage3.4 的 Context 特征与 Stage3.2 的 Target 特征保存为可按行读取的 `.npy` 数组；
- 用 `feature_manifest.json` 记录字段、维度、来源和样本索引。

这个桥接层把复杂的上游产物整理为训练代码可以稳定消费的统一接口。

### 3.4 Soft-Slot Qwen KT：冻结 LLM，只学习特征映射

核心模型在 [`stage34_soft_slot_qwen_kt/soft_slot_kt/model.py`](stage34_soft_slot_qwen_kt/soft_slot_kt/model.py) 中实现。

主实验的两组输入是：

| Soft slot | 默认字段 | 作用 |
|---|---|---|
| Context | `llm_embeddings`, `llm_struct_features`, `stage34_numeric` | 表示与目标题相关的学生认知状态 |
| Target | `hqtext`, `hqid`, `semantic`, `collaborative` | 表示目标题的语义、层级和行为先验 |

每组特征先拼接，再经过独立的 `SoftSlotProjector`：

```text
LayerNorm → Linear → GELU → Dropout → Linear → reshape as soft tokens
```

默认生成 4 个 Context soft tokens 和 2 个 Target soft tokens。它们替换 prompt 中预留位置的 token embeddings，并通过 `inputs_embeds` 送入 Qwen。Qwen 参数全部冻结，训练过程中只更新 projector。

模型不生成概率文本，而是在 prompt 末尾读取分类标签分数：

```text
A = 正确
B = 错误
P(correct) = softmax([logit_B, logit_A])[1]
```

## 4. 代码目录

```text
Work/
├─ stage34_soft_slot_qwen_kt/       # 当前主实验
│  ├─ soft_slot_kt/                 # 数据、prompt、projector、Qwen 前向与运行时
│  ├─ scripts/                      # 特征准备、训练、推理、评估入口
│  ├─ tests/                        # 不加载大模型的单元测试
│  └─ README.md                     # 主实验的详细设计说明
├─ scripts/common_pipeline_strict/  # Stage3.2/Stage3.4 上游认知特征管线
├─ scripts/run_common_cognitive_pipeline_strict.py
│                                    # 上游管线统一入口
├─ train_context.py                 # KT baseline/context 对照入口
├─ dataloader/                      # 对照模型的数据处理
└─ models/                          # 对照 KT 模型
```

`stage34_soft_slot_qwen_kt/` 是主方法边界；上游管线向它提供特征，但传统 KT baseline 不进入 Soft-Slot Qwen 的前向过程。

## 5. 建议的代码阅读顺序

如果第一次阅读本项目，建议按下面的顺序理解：

1. [`stage34_soft_slot_qwen_kt/soft_slot_kt/model.py`](stage34_soft_slot_qwen_kt/soft_slot_kt/model.py)：先看 soft slots 如何进入冻结 Qwen，以及概率如何由标签 logits 得到；
2. [`stage34_soft_slot_qwen_kt/soft_slot_kt/prompts.py`](stage34_soft_slot_qwen_kt/soft_slot_kt/prompts.py)：看自然语言 prompt 与 soft-slot 占位位置如何组合；
3. [`stage34_soft_slot_qwen_kt/soft_slot_kt/data.py`](stage34_soft_slot_qwen_kt/soft_slot_kt/data.py)：看 Context/Target 特征如何读取、拼接并组成 batch；
4. [`stage34_soft_slot_qwen_kt/soft_slot_kt/prepare.py`](stage34_soft_slot_qwen_kt/soft_slot_kt/prepare.py)：看 Stage3.2/Stage3.4 产物如何被整理为 Feature Store；
5. [`scripts/common_pipeline_strict/stage34.py`](scripts/common_pipeline_strict/stage34.py)：理解目标条件化证据选择和认知摘要；
6. [`scripts/common_pipeline_strict/stage32.py`](scripts/common_pipeline_strict/stage32.py)：追溯题目侧先验的来源；
7. [`stage34_soft_slot_qwen_kt/soft_slot_kt/runtime.py`](stage34_soft_slot_qwen_kt/soft_slot_kt/runtime.py) 与 `scripts/`：最后看训练、推理和评估如何组织。

更细的模块职责、数据契约和消融含义见 [`stage34_soft_slot_qwen_kt/README.md`](stage34_soft_slot_qwen_kt/README.md)。

## 6. 方法边界

- 主方法不是 DKT、SAKT、SAINT 等 KT baseline 的融合版本；这些模型只用于结果对照。
- 主配置默认不把完整历史文本输入 Qwen；历史首先经过目标条件化检索和状态压缩。
- Stage3.4 结构化总结是中间表征，不是最终预测器。
- Soft slots 是连续向量，不是通过文本 API 传入的特殊字符串。
- Qwen 作为冻结的条件分类器使用，项目训练的是外部特征到 LLM token 空间的映射。
- `text_only`、`context`、`target` 和 `random` 是用于回答不同机制问题的对照路径，不应与完整 `context_target` 主配置混写。

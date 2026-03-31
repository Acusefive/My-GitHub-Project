# 文件：quick_check_embeddings.py，放在 baselines 目录
import json
import pickle
import pandas as pd

from baselines import parse_all_seq  # 已在 baselines.py 里定义

DATA_PATH = "../data/student-problem-fine.json"
COG_PATH = "../../cognitive_embeddings.pkl"
SEPARATE_CHAR = ","

if __name__ == "__main__":
    # 1. 读 JSON（和 DKT_data_helper 一样的方式）
    json_data = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        try:
            loaded = json.loads(raw)
            json_data = loaded if isinstance(loaded, list) else [loaded]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    json_data.append(json.loads(line))

    df = pd.json_normalize(json_data, record_path=["seq"])

    # 2. 构建 questions 映射（和 DKT_data_helper 完全一致）
    raw_question = df.problem_id.unique().tolist()
    questions = {p: i for i, p in enumerate(raw_question)}

    student_ids = df.user_id.unique().tolist()
    sequences = parse_all_seq(
        type("A", (), {"models": "DKT"})(),  # 简单构造一个带 models 属性的 args
        student_ids,
        df,
        questions,
    )

    # 3. 读取原始 cognitive_embeddings.pkl
    with open(COG_PATH, "rb") as f:
        cognitive_embeddings = pickle.load(f)

    print("num_students in json:", len(student_ids))
    print("num_sequences:", len(sequences))
    print("num_embeddings in pkl:", len(cognitive_embeddings))

    assert len(cognitive_embeddings) == len(sequences), "外层长度不一致"

    # 4. 抽样检查每个学生序列长度是否一致
    for idx in [0, len(sequences) // 2, len(sequences) - 1]:
        q, a = sequences[idx]
        cog = cognitive_embeddings[idx]
        print(f"idx={idx}, len(q)={len(q)}, len(a)={len(a)}, len(cog)={len(cog)}")
        assert len(q) == len(a) == len(cog), f"长度不一致：idx={idx}"
        # 再检查前几个 time step 的维度
        # semantic_dim 由数据自动推断，不再硬编码 512
        inferred_dim = None
        for t, v in enumerate(cog[:5]):
            import numpy as np
            v = v if isinstance(v, np.ndarray) else np.asarray(v)
            print(f"  t={t}, cog_shape={v.shape}")
            if inferred_dim is None:
                inferred_dim = int(v.shape[-1])
            assert v.shape[-1] == inferred_dim, f"Semantic dim mismatch: {v.shape[-1]} vs {inferred_dim}"

    print("原始 sequences 与 cognitive_embeddings.pkl 对齐正常，可以放心跑 encode_onehot_with_cognitive ✔")
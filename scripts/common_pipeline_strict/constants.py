from __future__ import annotations

from pathlib import Path

try:
    from preprocess_gram import CUSTOM_STOP_WORDS as BASE_STOP_WORDS
except Exception:
    BASE_STOP_WORDS = {
        "的",
        "了",
        "和",
        "是",
        "在",
        "与",
        "及",
        "或",
        "关于",
        "下列",
        "正确",
        "错误",
        "题目",
        "内容",
        "问题",
        "单选",
        "分析",
        "解答",
        "答案",
        "过程",
        "步骤",
    }

BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

DS = 256
DC = 64
DD = 128

KGLOBAL = 50
KLOCAL = 5
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10
CTFIDF_MAX_FEATURES = 5000

RASCH_A = 1.0
RASCH_EPOCHS = 6
RASCH_LR = 0.05
RASCH_LAMBDA_MU = 1.0
RASCH_LAMBDA_THETA = 0.1

HISTORY_WINDOW = 20
COLLAB_WINDOW = 5
NEGATIVE_SAMPLES = 5
LOCAL_COOCCUR_THRESHOLD = 3

TRAIN_SEED = 42
TRAIN_LR = 1e-3
TRAIN_BATCH_SIZE = 1024
TRAIN_MAX_EPOCHS = 5
TRAIN_EARLY_STOP_PATIENCE = 2
TRAIN_VAL_MOD = 10

K1_DEFAULT = 30
K2_DEFAULT = 6

RHO = 0.6
GAMMA_PRE = 0.5
GAMMA_HIGH = 0.5
ALPHA_TIME = 0.2
DELTA_GRAPH = 0.8
LAMBDA_COV = 0.5
LAMBDA_RED = 0.3

BETA_POS = 1.0
BETA_NEG = 0.3

A_SCALE = 1.0
LAMBDA_L = 1.0
LAMBDA_T = 1.0
TAU = 1.0
ETA = 1.0
EPS = 1e-6

WEIGHT_STAGE1 = {
    "K": 1.0,
    "pre": 1.0,
    "peer": 1.0,
    "high": 1.0,
    "graph": 1.0,
    "collab": 1.0,
}

WEIGHT_STAGE2 = {
    "K": 1.0,
    "pre": 1.0,
    "peer": 1.0,
    "high": 1.0,
    "graph": 1.0,
    "collab": 1.0,
}

EXPLICIT_MATCH_WEIGHTS = {
    "K": 0.4,
    "L": 0.3,
    "G": 0.3,
}

REDUNDANCY_WEIGHTS = {
    "K": 0.4,
    "A": 0.2,
    "E": 0.4,
}

COVERAGE_WEIGHTS = {
    "role": 1.0,
    "knowledge": 1.0,
    "neighbor": 1.0,
}

ROLE_THRESHOLDS = {
    "pre": 0.10,
    "peer": 0.10,
    "high": 0.05,
    "graph": 0.05,
    "collab": 0.10,
}

ROLE_ORDER = ("pre", "peer", "high", "graph", "collab")
ROLE_LABELS = {
    "pre": "前置支撑",
    "peer": "同质迁移",
    "high": "高阶佐证",
    "graph": "图谱补充",
    "collab": "协同补充",
}
ROLE_PRIORITY = {name: idx for idx, name in enumerate(ROLE_ORDER)}

SUMMARY_TEMPLATE = (
    "围绕 {target_concepts}，学生近期主要表现为 {recent_trend}；"
    "当前判断主要依赖 {dominant_role} 证据，整体风险为 {risk_level}。"
)

QUESTION_TEXT_LIMIT = 80
SUPPORT_SCORE_DECIMALS = 4

SMOKE_MAX_PROBLEMS = 256
SMOKE_MAX_STUDENTS = 128
SMOKE_MAX_TARGETS_PER_STUDENT = 16
SMOKE_TRAIN_MAX_SAMPLES = 4096

DEFAULT_OUT_ROOT = Path("out") / "strict_common_pipeline"

STOP_WORDS = set(BASE_STOP_WORDS)


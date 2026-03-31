import os
import io
import sys
import logging
import argparse
import time
import json
import pandas as pd
import numpy as np
import tqdm
import random
import pdb
import torch
import torch.utils.data as Data
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import csv

# 确保优先使用当前项目中的 EduKTM（baselines/EduKTM），而不是环境里安装的同名包
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from MyEduKTM.DKT import DKT
from MyEduKTM.DKTPlus import DKTPlus, etl
from MyEduKTM.DKVMN import DKVMN
from EduKTM import AKT
from EduCDM import NCDM, MIRT, GDIRT
from load_data import DATA, PID_DATA, SemanticDKTDataset, semantic_dkt_collate_fn, SemanticDKVMNDataset, semantic_dkvmn_collate_fn
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_all_seq(args,students,data,questions,skills=None):
    all_sequences = []
    for student_id in tqdm.tqdm(students, 'parse student sequence:\t'):
        if args.models == 'AKT':
            student_sequence = parse_student_seq2(data[data.user_id == student_id],questions,skills)
        else:
            student_sequence = parse_student_seq1(data[data.user_id == student_id],questions)
        all_sequences.extend([student_sequence])
    return all_sequences

def parse_student_seq1(student,questions):
    seq = student.sort_values('submit_time')
    p = [questions[q] for q in seq.problem_id.tolist()]
    a = seq.is_correct.tolist()
    return p, a

def parse_student_seq2(student,problems, skills):
    seq = student.sort_values('submit_time')
    s = [skills[q] for q in seq.skill_id.tolist()]
    p = [problems[q] for q in seq.problem_id.tolist()]
    a = seq.is_correct.tolist()
    return s, p, a

def train_test_split(data, train_size=.7, shuffle=True):
    if shuffle:
        random.shuffle(data)
    boundary = round(len(data) * train_size)
    return data[: boundary], data[boundary:]


def sequences2tl(args, sequences, trgpath):
    with open(trgpath, 'w', encoding='utf8') as f:
        for seq in tqdm.tqdm(sequences, 'write into file: '):
            if args.models=='AKT':
                skills, problems, answers = seq
                seq_len = len(skills)
                f.write(str(seq_len) + '\n')
                f.write(','.join([str(q) for q in problems]) + '\n')
                f.write(','.join([str(q) for q in skills]) + '\n')
                f.write(','.join([str(a) for a in answers]) + '\n')
            else:
                questions, answers = seq
                seq_len = len(questions)
                f.write(str(seq_len) + '\n')
                f.write(','.join([str(q) for q in questions]) + '\n')
                f.write(','.join([str(a) for a in answers]) + '\n')

def encode_onehot(sequences, max_step, num_questions):
    result = []

    for q, a in tqdm.tqdm(sequences, 'convert to one-hot format: '):
        length = len(q)
        # append questions' and answers' length to an integer multiple of max_step
        mod = 0 if length % max_step == 0 else (max_step - length % max_step)
        onehot = np.zeros(shape=[length + mod, 2 * num_questions])
        for i, q_id in enumerate(q):
            index = int(q_id if a[i] > 0 else q_id + num_questions)
            onehot[i][index] = 1
        result = np.append(result, onehot)
    return result.reshape(-1, max_step, 2 * num_questions)

def encode_onehot_with_cognitive(sequences, max_step, num_questions, cognitive_dim: int = 512):
    """
    Early fusion: concat DKT one-hot (2*num_questions) with cognitive embedding (cognitive_dim)
    at each time step.

    sequences: List[Tuple[q_list, a_list, cog_list]]
      - q_list: List[int]
      - a_list: List[int|float] (0/1)
      - cog_list: List[np.ndarray], each shape=(cognitive_dim,)
    """
    result = []
    for q, a, cog in tqdm.tqdm(sequences, 'convert to one-hot+cognitive format: '):
        length = len(q)
        if len(a) != length:
            raise ValueError(f"Length mismatch: len(q)={length}, len(a)={len(a)}")
        if len(cog) != length:
            raise ValueError(f"Length mismatch: len(q)={length}, len(cog)={len(cog)}")

        mod = 0 if length % max_step == 0 else (max_step - length % max_step)
        total_len = length + mod

        onehot = np.zeros(shape=[total_len, 2 * num_questions], dtype=np.float32)
        cog_mat = np.zeros(shape=[total_len, cognitive_dim], dtype=np.float32)

        for i, q_id in enumerate(q):
            index = int(q_id if a[i] > 0 else q_id + num_questions)
            onehot[i][index] = 1.0
            v = cog[i]
            if not isinstance(v, np.ndarray):
                v = np.asarray(v, dtype=np.float32)
            if v.shape != (cognitive_dim,):
                raise ValueError(f"Unexpected cognitive vec shape at step {i}: {v.shape}, expected ({cognitive_dim},)")
            cog_mat[i] = v.astype(np.float32, copy=False)

        fused = np.concatenate([onehot, cog_mat], axis=-1)
        result = np.append(result, fused)

    return result.reshape(-1, max_step, 2 * num_questions + cognitive_dim)

def config():
    parser = argparse.ArgumentParser()

    # data process
    parser.add_argument("--data_dir", type=str, default='/data/XXX/ceping/action/student-problem.json')
    parser.add_argument("--data_problem_detail", type=str, default='/data/XXX/ceping/entity/problem.json')
    
    parser.add_argument("--data_process", action="store_true")
    parser.add_argument("--data_split", type = float, default=0.8)
    parser.add_argument("--data_shuffle", action="store_true")
    parser.add_argument("--saved_train_dir", type=str, default ='../data/ktbd/train.txt')
    parser.add_argument("--saved_dev_dir", type = str, default = '../data/ktbd/dev.txt')
    parser.add_argument("--saved_test_dir", type = str, default = '../data/ktbd/test.txt')
    parser.add_argument("--encoded_train_dir", type=str, default ='../data/DKT/train_data.npy')
    parser.add_argument("--encoded_test_dir", type=str, default ='../data/DKT/test_data.npy')


    
    # baseline models 
    parser.add_argument("--models", type = str, default="DKT")
    parser.add_argument("--model_path", type = str, default="../data/DKT/dkt.params")

    # DKT parameters
    parser.add_argument("--max_step", type= int, default=50)
    parser.add_argument("--num_questions", type= int)
    parser.add_argument("--num_skills", type= int)

    parser.add_argument("--hidden_size", type= int, default=10)
    parser.add_argument("--num_layers", type= int, default=1)
    parser.add_argument("--use_cognitive_embeddings", action="store_true")
    parser.add_argument("--cognitive_embeddings_pkl", type=str, default="")
    parser.add_argument("--cognitive_dim", type=int, default=512)

    # cognitive diagnosis benchmark dataset
    parser.add_argument("--num_concept", type=int)
    parser.add_argument("--saved_item_dir", type=str, default = '../data/cdbd/')

    parser.add_argument("--mode",type=str)
    

    # training parameters
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--epoch", type = int, default=10)

    # logging parameters
    parser.add_argument("--logger_dir", type = str, default='../log/')
    
    args = parser.parse_args()

    return args

def set_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)

    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    handler = logging.FileHandler(os.path.join(args.logger_dir,now+"log.txt"))
    
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    
    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info(args)

    return logger

def DKT_data_helper(args,logger):
    # 1. read the original file
    # 兼容两种格式：
    #   a) 单个大 JSON（例如：[ {...}, {...}, ... ]）
    #   b) JSON Lines（每行一个 JSON 对象）
    json_data = []
    with open(args.data_dir, 'r', encoding='utf-8') as f_in:
        raw = f_in.read().strip()
        if not raw:
            raise ValueError(f"data_dir '{args.data_dir}' 为空，无法解析为 JSON。")
        try:
            loaded = json.loads(raw)
            # 如果是列表，直接使用；如果是单个 dict，则包一层列表
            if isinstance(loaded, list):
                json_data = loaded
            else:
                json_data = [loaded]
        except json.JSONDecodeError:
            # 回退到逐行解析 JSON Lines 格式
            f_in.seek(0)
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                json_data.append(json.loads(line))

    df_nested_list = pd.json_normalize(json_data, record_path =['seq'])

    # 2. define skills 
    if args.mode == 'Coarse':
        raw_question = df_nested_list.course_id.unique().tolist()
    elif args.mode == 'Middle':
        raw_question = df_nested_list.exercise_id.unique().tolist()
    elif args.mode == 'Fine':
        raw_question = df_nested_list.problem_id.unique().tolist()
    num_skill = len(raw_question)
    # question id from 0 to (num_skill - 1)
    questions = { p: i for i, p in enumerate(raw_question) }
    logger.info("number of skills: %d" % num_skill)
    args.num_questions = num_skill

    # 3. q-a list 
    # [(question_sequence_0, answer_sequence_0), ..., (question_sequence_n, answer_sequence_n)]
    student_ids = df_nested_list.user_id.unique().tolist()
    sequences = parse_all_seq(args, student_ids, df_nested_list, questions)

    # Optional: early fusion cognitive embeddings (List[List[np.ndarray]] aligned with student_ids/seq)
    # 在这里仅做一次性对齐和切分，并额外导出与 train/test 三行 txt 完全对齐的语义向量 pkl，
    # 以便后续 SemanticDKTDataset 使用。
    cognitive_embeddings = None
    if args.use_cognitive_embeddings:
        if not args.cognitive_embeddings_pkl:
            raise ValueError("--use_cognitive_embeddings requires --cognitive_embeddings_pkl")
        with open(args.cognitive_embeddings_pkl, "rb") as f:
            cognitive_embeddings = pickle.load(f)
        if len(cognitive_embeddings) != len(student_ids):
            raise ValueError(
                f"cognitive_embeddings outer len mismatch: {len(cognitive_embeddings)} vs students {len(student_ids)}. "
                f"Ensure the pkl is generated from the same dataset/order."
            )
        # 先将语义向量与 (q, a) 序列对齐
        merged_sequences = []
        for idx, (qa, cog) in enumerate(zip(sequences, cognitive_embeddings)):
            q, a = qa
            if len(cog) != len(q):
                raise ValueError(
                    f"Student idx {idx} seq len mismatch: len(q)={len(q)} vs len(cog)={len(cog)}. "
                    f"Ensure per-interaction alignment."
                )
            merged_sequences.append((q, a, cog))
        sequences = merged_sequences

    #pdb.set_trace()

    # 4. split dataset
    train_sequences, test_sequences = train_test_split(sequences,args.data_split,args.data_shuffle)
    logger.info("data split with ratio {} and shuffle {}".format(args.data_split, args.data_shuffle))

    # 若使用语义向量，则将 train/test 对应的语义序列分别导出为单独的 pkl，
    # 这样后续 SemanticDKTDataset 读取 train2.txt/test2.txt 时可以使用长度完全匹配的 pkl。
    if args.use_cognitive_embeddings:
        train_semantic = [cog for (_, _, cog) in train_sequences]
        test_semantic = [cog for (_, _, cog) in test_sequences]
        train_sem_path = os.path.join(os.path.dirname(args.saved_train_dir), "train2_embeddings.pkl")
        test_sem_path = os.path.join(os.path.dirname(args.saved_test_dir), "test2_embeddings.pkl")
        os.makedirs(os.path.dirname(os.path.abspath(train_sem_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(test_sem_path)), exist_ok=True)
        with open(train_sem_path, "wb") as f:
            pickle.dump(train_semantic, f)
        with open(test_sem_path, "wb") as f:
            pickle.dump(test_semantic, f)
        logger.info(f"train semantic embeddings saved at {train_sem_path}")
        logger.info(f"test semantic embeddings saved at {test_sem_path}")

    # 5. save triple line format for other tasks
    if args.use_cognitive_embeddings:
        sequences2tl(args, [(q, a) for (q, a, _) in train_sequences], args.saved_train_dir)
        sequences2tl(args, [(q, a) for (q, a, _) in test_sequences], args.saved_test_dir)
    else:
        sequences2tl(args, train_sequences, args.saved_train_dir)
        sequences2tl(args, test_sequences, args.saved_test_dir)
    logger.info("triple line format trainset saved at {}".format(args.saved_train_dir))
    logger.info("triple line format testset saved at {}".format(args.saved_test_dir))

    # 6. onehot encode
    # reduce the amount of data for example running faster
    percentage = 1
    if args.use_cognitive_embeddings:
        train_data = encode_onehot_with_cognitive(
            train_sequences[: int(len(train_sequences) * percentage)],
            args.max_step,
            num_skill,
            cognitive_dim=args.cognitive_dim,
        )
        test_data = encode_onehot_with_cognitive(
            test_sequences[: int(len(test_sequences) * percentage)],
            args.max_step,
            num_skill,
            cognitive_dim=args.cognitive_dim,
        )
    else:
        train_data = encode_onehot(train_sequences[: int(len(train_sequences) * percentage)], args.max_step, num_skill)
        test_data = encode_onehot(test_sequences[: int(len(test_sequences) * percentage)], args.max_step, num_skill)

    # save onehot data
    np.save(args.encoded_train_dir, train_data)
    np.save(args.encoded_test_dir, test_data)

    logger.info("data process done.")

def DKVMN_data_helper(args, logger):
    """
    为 DKVMN 准备数据（4行格式：length, pid, skill_id, answer）
    支持语义向量对齐和导出
    """
    # 1. 读取原始 JSON 文件（与 DKT_data_helper 相同）
    json_data = []
    with open(args.data_dir, 'r', encoding='utf-8') as f_in:
        raw = f_in.read().strip()
        if not raw:
            raise ValueError(f"data_dir '{args.data_dir}' 为空，无法解析为 JSON。")
        try:
            loaded = json.loads(raw)
            json_data = loaded if isinstance(loaded, list) else [loaded]
        except json.JSONDecodeError:
            f_in.seek(0)
            for line in f_in:
                line = line.strip()
                if line:
                    json_data.append(json.loads(line))

    df_nested_list = pd.json_normalize(json_data, record_path=['seq'])

    # 2. 定义 problem_id 和 skill_id 映射
    # 对于 DKVMN，通常 problem_id 和 skill_id 相同（都使用 problem_id）
    if args.mode == 'Coarse':
        raw_problem = df_nested_list.course_id.unique().tolist()
        raw_skill = df_nested_list.course_id.unique().tolist()
    elif args.mode == 'Middle':
        raw_problem = df_nested_list.exercise_id.unique().tolist()
        raw_skill = df_nested_list.exercise_id.unique().tolist()
    elif args.mode == 'Fine':
        raw_problem = df_nested_list.problem_id.unique().tolist()
        # 如果有 skill_id 字段则使用，否则使用 problem_id
        if 'skill_id' in df_nested_list.columns:
            raw_skill = df_nested_list.skill_id.unique().tolist()
        else:
            raw_skill = df_nested_list.problem_id.unique().tolist()
    
    num_problem = len(raw_problem)
    num_skill = len(raw_skill)
    problems = {p: i for i, p in enumerate(raw_problem)}
    skills = {s: i for i, s in enumerate(raw_skill)}
    
    logger.info("number of problems: %d" % num_problem)
    logger.info("number of skills: %d" % num_skill)
    args.num_questions = num_problem  # DKVMN 使用 problem 数量

    # 3. 解析序列：[(problem_seq, skill_seq, answer_seq), ...]
    student_ids = df_nested_list.user_id.unique().tolist()
    sequences = []
    for student_id in tqdm.tqdm(student_ids, 'parse student sequence:'):
        student_seq = df_nested_list[df_nested_list.user_id == student_id].sort_values('submit_time')
        p_seq = [problems[p] for p in student_seq.problem_id.tolist()]
        # 如果有 skill_id 字段则使用，否则使用 problem_id 作为 skill_id
        if 'skill_id' in df_nested_list.columns:
            s_seq = [skills.get(s, 0) for s in student_seq.skill_id.tolist()]
        else:
            s_seq = [problems[p] for p in student_seq.problem_id.tolist()]
        a_seq = student_seq.is_correct.tolist()
        sequences.append((p_seq, s_seq, a_seq))

    # 4. 处理语义向量（如果启用）
    cognitive_embeddings = None
    if args.use_cognitive_embeddings:
        if not args.cognitive_embeddings_pkl:
            raise ValueError("--use_cognitive_embeddings requires --cognitive_embeddings_pkl")
        with open(args.cognitive_embeddings_pkl, "rb") as f:
            cognitive_embeddings = pickle.load(f)
        if len(cognitive_embeddings) != len(student_ids):
            raise ValueError(
                f"cognitive_embeddings length mismatch: {len(cognitive_embeddings)} vs {len(student_ids)}"
            )
        
        # 对齐语义向量
        merged_sequences = []
        for idx, ((p, s, a), cog) in enumerate(zip(sequences, cognitive_embeddings)):
            if len(cog) != len(p):
                raise ValueError(
                    f"Student idx {idx} seq len mismatch: len(p)={len(p)} vs len(cog)={len(cog)}"
                )
            merged_sequences.append((p, s, a, cog))
        sequences = merged_sequences

    # 5. 切分数据集：train / dev / test
    # 先按 args.data_split 切出 train，其余作为 holdout；再将 holdout 等分为 dev/test。
    train_sequences, holdout_sequences = train_test_split(sequences, args.data_split, args.data_shuffle)
    dev_sequences, test_sequences = train_test_split(holdout_sequences, 0.5, shuffle=False)
    logger.info(
        "data split with ratio train={} and holdout={}, then dev/test=0.5/0.5 (shuffle={})".format(
            args.data_split, 1 - args.data_split, args.data_shuffle
        )
    )

    # 6. 导出语义向量 pkl（如果启用）
    if args.use_cognitive_embeddings:
        train_semantic = [cog for (_, _, _, cog) in train_sequences]
        dev_semantic = [cog for (_, _, _, cog) in dev_sequences]
        test_semantic = [cog for (_, _, _, cog) in test_sequences]
        train_sem_path = os.path.join(os.path.dirname(args.saved_train_dir), "train2_embeddings.pkl")
        dev_sem_path = os.path.join(os.path.dirname(args.saved_dev_dir), "dev2_embeddings.pkl")
        test_sem_path = os.path.join(os.path.dirname(args.saved_test_dir), "test2_embeddings.pkl")
        os.makedirs(os.path.dirname(os.path.abspath(train_sem_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(dev_sem_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(test_sem_path)), exist_ok=True)
        with open(train_sem_path, "wb") as f:
            pickle.dump(train_semantic, f)
        with open(dev_sem_path, "wb") as f:
            pickle.dump(dev_semantic, f)
        with open(test_sem_path, "wb") as f:
            pickle.dump(test_semantic, f)
        logger.info(f"train semantic embeddings saved at {train_sem_path}")
        logger.info(f"dev semantic embeddings saved at {dev_sem_path}")
        logger.info(f"test semantic embeddings saved at {test_sem_path}")

    # 7. 保存 4 行格式文件
    def sequences2fourline(sequences, trgpath):
        with open(trgpath, 'w', encoding='utf8') as f:
            for seq in tqdm.tqdm(sequences, 'write into file: '):
                if args.use_cognitive_embeddings:
                    p, s, a, _ = seq
                else:
                    p, s, a = seq
                seq_len = len(p)
                f.write(str(seq_len) + '\n')
                f.write(','.join([str(x) for x in p]) + '\n')
                f.write(','.join([str(x) for x in s]) + '\n')
                f.write(','.join([str(x) for x in a]) + '\n')

    # 直接传入原始序列，sequences2fourline 内部会根据 use_cognitive_embeddings 标志处理
    sequences2fourline(train_sequences, args.saved_train_dir)
    sequences2fourline(dev_sequences, args.saved_dev_dir)
    sequences2fourline(test_sequences, args.saved_test_dir)
    
    logger.info("four-line format trainset saved at {}".format(args.saved_train_dir))
    logger.info("four-line format devset saved at {}".format(args.saved_dev_dir))
    logger.info("four-line format testset saved at {}".format(args.saved_test_dir))
    logger.info("data process done.")

def get_data_loader(data_path, batch_size, shuffle=False):
    data = torch.FloatTensor(np.load(data_path))
    data_loader = Data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def DKT_Baseline(args,logger):
    """
    DKT baseline.
    - 当 args.use_cognitive_embeddings=False 时，保持原有基于 one-hot npy 的训练流程；
    - 当 args.use_cognitive_embeddings=True 时，使用 SemanticDKTDataset + 早期融合语义向量。
    """

    # 初始化模型：当启用语义增强时，强制走 early-fusion 路径（避免误用旧 one-hot 分支）
    dkt = DKT(
        args.num_questions,
        args.hidden_size,
        args.num_layers,
        input_semantic_dim=args.cognitive_dim,
        use_early_fusion=bool(args.use_cognitive_embeddings),
    )

    if not args.use_cognitive_embeddings:
        # ===== 原始 one-hot 流程 =====
        if (not os.path.exists(args.encoded_train_dir)) or (not os.path.exists(args.encoded_test_dir)):
            sys.exit()

        train_loader = get_data_loader(args.encoded_train_dir, args.batch_size, True)
        test_loader = get_data_loader(args.encoded_test_dir, args.batch_size, False)

        dkt.train(train_loader, test_loader, epoch=args.epoch)
        dkt.save(args.model_path)
        logger.info("{} model saved!".format(args.models))

        dkt.load(args.model_path)
        auc = dkt.eval(test_loader)
        logger.info("auc: %.6f" % auc)
        return

    # ===== 语义增强流程：使用 SemanticDKTDataset + Early Fusion =====
    if (not os.path.exists(args.saved_train_dir)) or (not os.path.exists(args.saved_test_dir)):
        sys.exit()
    # 语义增强模式下，直接使用在 DKT_data_helper 中按 train/test 切分并导出的语义向量 pkl，
    # 而不是原始的全量 cognitive_embeddings_pkl，以保证与三行 txt 完全对齐。
    train_sem_path = os.path.join(os.path.dirname(args.saved_train_dir), "train2_embeddings.pkl")
    test_sem_path = os.path.join(os.path.dirname(args.saved_test_dir), "test2_embeddings.pkl")
    if not os.path.exists(train_sem_path) or not os.path.exists(test_sem_path):
        raise FileNotFoundError(
            f"train/test semantic pkl not found: {train_sem_path}, {test_sem_path}. "
            f"请先在相同配置下运行一次带 --data_process 的 baselines.py 生成它们。"
        )

    # 构造 Dataset，确保传入 cognitive_embeddings.pkl 路径
    train_dataset = SemanticDKTDataset(
        data_path=args.saved_train_dir,
        embedding_path=train_sem_path,
        semantic_dim=args.cognitive_dim,
        load_semantic=True,
    )
    test_dataset = SemanticDKTDataset(
        data_path=args.saved_test_dir,
        embedding_path=test_sem_path,
        semantic_dim=args.cognitive_dim,
        load_semantic=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: semantic_dkt_collate_fn(batch, semantic_dim=args.cognitive_dim),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: semantic_dkt_collate_fn(batch, semantic_dim=args.cognitive_dim),
    )

    model = dkt.dkt_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_function = torch.nn.BCELoss()

    for e in range(args.epoch):
        model.train()
        all_pred = []
        all_truth = []

        for x, y, sem, lengths in tqdm.tqdm(train_loader, f"Epoch {e}"):
            # x: [B, T] problem_ids
            # y: [B, T] labels (0/1)
            # sem: [B, T, cognitive_dim]
            # lengths: [B]
            x = x.to(device)
            y = y.to(device)
            sem = sem.to(device)
            lengths = lengths.to(device)

            # 构造交互 ID：题目 + 对错编码（与原 DKT 中 one-hot 逻辑保持一致）
            qa_ids = x + (y.long() * args.num_questions)  # [B, T]
            q_ids = x.long()  # [B, T]

            # 为避免标签泄漏，使用 t 时刻的交互预测 t+1 时刻的表现：
            # 输入序列：0..T-2，目标序列：1..T-1
            if qa_ids.size(1) <= 1:
                # 序列过短，无法形成 (历史, 下一个) 配对，直接跳过
                continue

            qa_in = qa_ids[:, :-1]           # [B, T-1]
            sem_in = sem[:, :-1, :]          # [B, T-1, D]
            q_tgt = q_ids[:, 1:]             # [B, T-1]
            y_tgt = y[:, 1:]                 # [B, T-1]
            eff_lengths = torch.clamp(lengths - 1, min=0)  # [B]

            # 前向传播（早期融合）
            logits = model(qa_in, sem_in)  # [B, T-1, num_questions]

            # 取出对应题目的预测概率
            pred = logits.gather(-1, q_tgt.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

            # 构造 mask，忽略 padding 位置
            max_len = pred.size(1)
            mask = torch.arange(max_len, device=device).unsqueeze(0) < eff_lengths.unsqueeze(1)  # [B, T-1]

            pred_flat = pred[mask]
            truth_flat = y_tgt[mask]

            loss = loss_function(pred_flat, truth_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_pred.append(pred_flat.detach().cpu())
            all_truth.append(truth_flat.detach().cpu())

        all_pred_tensor = torch.cat(all_pred)
        all_truth_tensor = torch.cat(all_truth)
        train_auc = roc_auc_score(all_truth_tensor.numpy(), all_pred_tensor.numpy())
        logger.info(f"[Epoch {e}] loss: {loss.item():.6f}, train_auc: {train_auc:.6f}")

        # ===== 验证 / 测试 =====
        model.eval()
        y_pred_list = []
        y_truth_list = []
        with torch.no_grad():
            for x, y, sem, lengths in tqdm.tqdm(test_loader, "evaluating"):
                x = x.to(device)
                y = y.to(device)
                sem = sem.to(device)
                lengths = lengths.to(device)

                qa_ids = x + (y.long() * args.num_questions)
                q_ids = x.long()

                if qa_ids.size(1) <= 1:
                    continue

                qa_in = qa_ids[:, :-1]
                sem_in = sem[:, :-1, :]
                q_tgt = q_ids[:, 1:]
                y_tgt = y[:, 1:]
                eff_lengths = torch.clamp(lengths - 1, min=0)

                logits = model(qa_in, sem_in)
                pred = logits.gather(-1, q_tgt.unsqueeze(-1)).squeeze(-1)

                max_len = pred.size(1)
                mask = torch.arange(max_len, device=device).unsqueeze(0) < eff_lengths.unsqueeze(1)

                y_pred_list.append(pred[mask].cpu())
                y_truth_list.append(y_tgt[mask].cpu())

        y_pred = torch.cat(y_pred_list)
        y_truth = torch.cat(y_truth_list)
        auc = roc_auc_score(y_truth.numpy(), y_pred.numpy())
        logger.info(f"[Epoch {e}] test_auc: {auc:.6f}")

    # 训练完成后保存模型
    dkt.save(args.model_path)
    logger.info("{} model saved with semantic early fusion!".format(args.models))


def three_line_format_into_json(args, train_txt='../data/DKT/train.txt', test_txt='../data/DKT/test.txt'):
    # turn three line format into json:
    with open(train_txt) as f, io.open(args.saved_train_dir, "w", encoding="utf-8") as wf:
        for _ in tqdm.tqdm(f):
            exercise_tags = f.readline().strip().strip(",").split(",")
            response_sequence = f.readline().strip().strip(",").split(",")
            exercise_tags = list(map(int, exercise_tags))
            response_sequence = list(map(int, response_sequence))
            responses = list(zip(exercise_tags, response_sequence))
            print(json.dumps(responses), file=wf)

    with open(test_txt) as f, io.open(args.saved_test_dir, "w", encoding="utf-8") as wf:
        for _ in tqdm.tqdm(f):
            exercise_tags = f.readline().strip().strip(",").split(",")
            response_sequence = f.readline().strip().strip(",").split(",")
            exercise_tags = list(map(int, exercise_tags))
            response_sequence = list(map(int, response_sequence))
            responses = list(zip(exercise_tags, response_sequence))
            print(json.dumps(responses), file=wf)


def DKT_plus_Baseline(args,logger):
    if (not os.path.exists(args.saved_train_dir)) or (not os.path.exists(args.saved_test_dir)):
        sys.exit()
    
    train = etl(args.saved_train_dir, args.batch_size)
    valid = etl(args.saved_test_dir, args.batch_size)
    test = etl(args.saved_test_dir, args.batch_size)
    
    dkt_plus = DKTPlus(ku_num=args.num_questions, hidden_num=100, loss_params={"lr": 0.1, "lw1": 0.5, "lw2": 0.5})
    dkt_plus.train(train, valid, epoch=args.epoch)
    dkt_plus.save(args.model_path)

    dkt_plus.load(args.model_path)
    auc, accuracy = dkt_plus.eval(test)
    logger.info("auc: %.6f, accuracy: %.6f" % (auc, accuracy))


def CDBD_data_helper(args,logger):
    # 1. read the original file
    with open(args.data_dir, 'r', encoding='utf-8') as f_in:
        json_data = json.loads(f_in.read())
        # for line in f_in.readlines():
        #     dic = json.loads(line)
        #     json_data.append(dic)

    df_nested_list = pd.json_normalize(json_data, record_path =['seq'])
    raw_question = df_nested_list.problem_id.unique().tolist()
    num_skill = len(raw_question)
    # problem map: (start from 1)
    map_problems = {p:i+1 for i, p in enumerate(raw_question)}
    logger.info("number of skills: %d" % num_skill)
    args.num_questions = num_skill


    # 2. read the problem detail file
    with open(args.data_problem_detail, 'r', encoding='utf-8')as f_in:
        problem_data = []
        for line in f_in.readlines():
            dic = json.loads(line)
            problem_data.append(dic)

 
    df_problem_detail = pd.json_normalize(problem_data)
    df_problem_detail_sub = df_problem_detail[df_problem_detail.problem_id.isin(raw_question)]
    # raw_concept = set()
    # for c in df_problem_detail_sub.concepts.tolist():
    #     raw_concept.add(c)
    # num_concept = len(raw_concept)
    # # concept map: (start from 1)
    # map_concepts = {c:i+1 for i, c in enumerate(raw_concept)}
    # logger.info("number of concepts: %d" % num_concept)
    # args.num_concept = num_concept

    # we map to course_id
    raw_concept = set()
    if args.mode == 'Fine':
       for c in df_problem_detail_sub.concepts.tolist():
            raw_concept.add(c)
    elif args.mode == 'Middle':
        for c in df_problem_detail_sub.exercise_id.tolist():
            raw_concept.add(c)
    elif args.mode == 'Coarse':
        for c in df_problem_detail_sub.course_id.tolist():
            raw_concept.add(c)
    num_concept = len(raw_concept)
    # concept map: (start from 1)
    map_concepts = {c:i+1 for i, c in enumerate(raw_concept)}
    logger.info("number of concepts: %d" % num_concept)
    args.num_concept = num_concept



    #pdb.set_trace()

    # 3. build item_id->knowledge_code
    item2knowledge = {}
    for index, row in tqdm.tqdm(df_problem_detail_sub.iterrows()):
        problem_id = row['problem_id']
        if args.mode == 'Fine':
            item2knowledge[map_problems[problem_id]]=[]
            concepts = row['concepts']
            for per_concept in concepts:
                item2knowledge[map_problems[problem_id]].append(map_concepts[per_concept]) 
        elif args.mode == 'Middle':
            concept = row['exercise_id']
            item2knowledge[map_problems[problem_id]]=map_concepts[concept]
        elif args.mode == 'Coarse':
            concept = row['course_id']
            # item2knowledge[map_problems[problem_id]].append(map_concepts[concept]) 
            item2knowledge[map_problems[problem_id]]=map_concepts[concept]

    # saved
    item_df = pd.DataFrame({'item_id':item2knowledge.keys(),'knowledge_code':item2knowledge.values()})
    item_df.to_csv(args.saved_item_dir,sep=',',index=False, header=True)
    logger.info("item.csv saved!")

    # 4. build user map
    raw_users = df_nested_list.user_id.unique().tolist()
    num_users = len(raw_users)
    # users map: (start from 1)
    map_users = {u:i+1 for i, u in enumerate(raw_users)}
    logger.info("number of users: %d" % num_users)
    args.num_questions = num_users

    # 5. build dataset
    dataset = df_nested_list[['user_id','problem_id','is_correct']]
    dataset['user_id']=dataset['user_id'].map(map_users)
    dataset['problem_id']=dataset['problem_id'].map(map_problems)
    # for GDIRT which only user problem id 
    # dataset['problem_id']=dataset['problem_id'].map(map_problems).map(item2knowledge)

    # 6. split dataset
    if args.data_shuffle:
        dataset = dataset.sample(frac=1)
        boundary = round(len(dataset) * args.data_split)
    train_df = dataset[:boundary]
    test_df = dataset[boundary:]
    boundary = round(len(test_df) * 0.5)
    dev_df = test_df[:boundary]
    test_df = test_df[boundary:]


    # saved train and test csv
    train_df.to_csv(args.saved_train_dir,sep=',',index=False,header=True)
    test_df.to_csv(args.saved_test_dir,sep=',',index=False, header=True)
    dev_df.to_csv(args.saved_dev_dir,sep=',',index=False, header=True)


    logger.info("data process done!")
 


def transform1(user, item, item2knowledge, score, batch_size,knowledge_n):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

def transform2(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)

def CDBD_Baseline(args,logger):
    if (not os.path.exists(args.saved_train_dir)) or (not os.path.exists(args.saved_test_dir)):
        sys.exit()
    train_data = pd.read_csv(args.saved_train_dir)
    test_data =pd.read_csv(args.saved_test_dir)
    dev_data =pd.read_csv(args.saved_dev_dir)
    df_item = pd.read_csv(args.saved_item_dir)
    
    item2knowledge = {}
    knowledge_set = set()

    if args.models != 'GDIRT':
        for i, s in df_item.iterrows():
            item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
            item2knowledge[item_id] = knowledge_codes
            knowledge_set.update(knowledge_codes)
    
    user_n = np.max(train_data['user_id'])
    item_n = np.max([np.max(train_data['problem_id']), np.max(dev_data['problem_id']), np.max(test_data['problem_id'])])
    if args.models != 'GDIRT':
        knowledge_n = np.max(list(knowledge_set))
        logger.info('user_n: {}\n item_n: {}\n knowledge_n: {}\n'.format(user_n,item_n,knowledge_n))
    else:
        logger.info('user_n: {}\n item_n: {}\n '.format(user_n,item_n))
 
    
    if args.models == 'NCDM':
        train_set, valid_set, test_set = [
            transform1(data["user_id"], data["problem_id"], item2knowledge, data["is_correct"], args.batch_size, knowledge_n)
            for data in [train_data, dev_data, test_data]
        ]  
        cdm = NCDM(knowledge_n, item_n, user_n)
    elif args.models == 'MIRT' or args.models == 'GDIRT':
        train_set, valid_set, test_set= [
            transform2(data["user_id"], data["problem_id"], data["is_correct"],args.batch_size)
            for data in [train_data, dev_data, test_data]
        ]
        if args.models == 'GDIRT':
            cdm = GDIRT(user_n+1,item_n+1 )
        elif args.models == 'MIRT':
            cdm = MIRT(user_n+1,item_n+1,knowledge_n)

    cdm.train(train_set, valid_set, epoch = args.epoch, device="cuda",lr=0.0005)
    
    cdm.save(args.model_path)
    logger.info("{} model saved!".format(args.models))

    cdm.load(args.model_path)
    auc, accuracy = cdm.eval(test_set)
    logger.info("auc: %.6f, accuracy: %.6f" % (auc, accuracy))

def DKVMN_Baseline(args, logger):
    """
    DKVMN baseline.
    - 当 args.use_cognitive_embeddings=False 时，保持原有基于 PID_DATA 的训练流程；
    - 当 args.use_cognitive_embeddings=True 时，使用 SemanticDKVMNDataset + 早期融合语义向量。
    """
    # 若未运行 --data_process，则 args.num_questions 可能为空；此时从 4 行格式文件推断题目数
    if args.num_questions is None:
        def _infer_num_questions_from_fourline(path: str) -> int:
            # 4行格式：len / pid_line / skill_line / ans_line，循环出现
            max_id = 0
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.strip() for ln in f if ln.strip() != ""]
            if len(lines) % 4 != 0:
                raise ValueError(f"Invalid 4-line file (non-empty lines %4!=0): {path}, lines={len(lines)}")
            for i in range(0, len(lines), 4):
                pid_line = lines[i + 1]
                if not pid_line:
                    continue
                for x in pid_line.split(","):
                    x = x.strip()
                    if not x:
                        continue
                    try:
                        v = int(x)
                    except Exception:
                        continue
                    if v > max_id:
                        max_id = v
            if max_id <= 0:
                raise ValueError(f"Failed to infer num_questions from {path}: max_id={max_id}")
            # ids 是从 0..N-1 的话，这里 max_id 会是 N-1；如果从 1..N，则 max_id 是 N
            # 取 max_id+1 更稳妥，保证 Embedding 表不越界
            return int(max_id + 1)

        inferred = _infer_num_questions_from_fourline(args.saved_train_dir)
        args.num_questions = inferred
        logger.info(f"[DKVMN] args.num_questions is None, inferred from {args.saved_train_dir}: {args.num_questions}")

    params = {
        'max_iter': args.epoch,
        'init_lr': 0.01,
        'lr_decay': 0.75,
        'final_lr': 1E-5,
        'maxgradnorm': 50.0,
        'final_fc_dim': 50,
        'key_embedding_dim': 50,
        'batch_size': args.batch_size,
        'value_embedding_dim': 200,
        'memory_size': 20,
        'n_question': args.num_questions,
        'seqlen': 50,
    }
    params['lr'] = params['init_lr']
    params['key_memory_state_dim'] = params['key_embedding_dim']
    params['value_memory_state_dim'] = params['value_embedding_dim']

    if not args.use_cognitive_embeddings:
        # ===== 原始基线流程 =====
        dat = PID_DATA(n_question=params['n_question'], seqlen=params['seqlen'], separate_char=',')
        train_data = dat.load_data(args.saved_train_dir)
        dev_data = dat.load_data(args.saved_dev_dir)
        test_data = dat.load_data(args.saved_test_dir)
        
        dkvmn = DKVMN(n_question=params['n_question'],
                      batch_size=params['batch_size'],
                      key_embedding_dim=params['key_embedding_dim'],
                      value_embedding_dim=params['value_embedding_dim'],
                      memory_size=params['memory_size'],
                      key_memory_state_dim=params['key_memory_state_dim'],
                      value_memory_state_dim=params['value_memory_state_dim'],
                      final_fc_dim=params['final_fc_dim'],
                      use_early_fusion=False)
        
        # 论文级标准：用 dev 选模，最后只在 test 汇报（避免 test 泄漏）
        best_dev_auc = dkvmn.train(params, train_data, dev_data)
        dkvmn.save(args.model_path)
        logger.info(f"DKVMN model saved! (best_dev_auc={best_dev_auc:.6f})")
        
        dkvmn.load(args.model_path)
        test_loss, test_acc, test_auc = dkvmn.eval(params, test_data)
        logger.info(f"[Final] test_loss: {test_loss:.6f}, test_auc: {test_auc:.6f}, test_acc: {test_acc:.6f}")
    else:
        # ===== 语义增强流程：使用 SemanticDKVMNDataset + 晚期融合语义向量 =====
        from torch.utils.data import DataLoader
        from sklearn.metrics import roc_auc_score, accuracy_score

        # 晚期融合对 DKVMN 的记忆寻址/写入更友好，可使用与基线相同的较大学习率
        params['init_lr'] = 0.01
        params['lr'] = params['init_lr']
        
        train_sem_path = os.path.join(os.path.dirname(args.saved_train_dir), "train2_embeddings.pkl")
        dev_sem_path = os.path.join(os.path.dirname(args.saved_dev_dir), "dev2_embeddings.pkl")
        test_sem_path = os.path.join(os.path.dirname(args.saved_test_dir), "test2_embeddings.pkl")
        
        if (not os.path.exists(train_sem_path)) or (not os.path.exists(dev_sem_path)) or (not os.path.exists(test_sem_path)):
            raise FileNotFoundError(
                f"Semantic pkl files not found: {train_sem_path}, {dev_sem_path}, {test_sem_path}. "
                f"请先在相同配置下运行一次带 --data_process 的 baselines.py 生成它们。"
            )
        
        train_dataset = SemanticDKVMNDataset(
            data_path=args.saved_train_dir,
            embedding_path=train_sem_path,
            semantic_dim=args.cognitive_dim,
            load_semantic=True,
        )
        dev_dataset = SemanticDKVMNDataset(
            data_path=args.saved_dev_dir,
            embedding_path=dev_sem_path,
            semantic_dim=args.cognitive_dim,
            load_semantic=True,
        )
        test_dataset = SemanticDKVMNDataset(
            data_path=args.saved_test_dir,
            embedding_path=test_sem_path,
            semantic_dim=args.cognitive_dim,
            load_semantic=True,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            collate_fn=lambda b: semantic_dkvmn_collate_fn(
                b, n_question=params['n_question'],
                seqlen=params['seqlen'],
                semantic_dim=args.cognitive_dim
            ),
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=lambda b: semantic_dkvmn_collate_fn(
                b, n_question=params['n_question'],
                seqlen=params['seqlen'],
                semantic_dim=args.cognitive_dim
            ),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=lambda b: semantic_dkvmn_collate_fn(
                b, n_question=params['n_question'],
                seqlen=params['seqlen'],
                semantic_dim=args.cognitive_dim
            ),
        )
        
        dkvmn = DKVMN(
            n_question=params['n_question'],
            batch_size=params['batch_size'],
            key_embedding_dim=params['key_embedding_dim'],
            value_embedding_dim=params['value_embedding_dim'],
            memory_size=params['memory_size'],
            key_memory_state_dim=params['key_memory_state_dim'],
            value_memory_state_dim=params['value_memory_state_dim'],
            final_fc_dim=params['final_fc_dim'],
            input_semantic_dim=args.cognitive_dim,
            use_early_fusion=True
        )
        
        model = dkvmn.model.to(device)
        model.init_embeddings()
        model.init_params()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.9))
        
        # 训练循环
        best_dev_auc = 0
        for epoch in range(params['max_iter']):
            model.train()
            all_pred = []
            all_target = []
            epoch_loss = 0
            logged_sem_stats = False
            
            for q_data, qa_data, semantic_inputs, lengths in tqdm.tqdm(train_loader, f"Epoch {epoch}"):
                q_data = q_data.to(device)
                qa_data = qa_data.to(device)
                semantic_inputs = semantic_inputs.to(device)
                if not logged_sem_stats:
                    with torch.no_grad():
                        # 语义向量是否“全零/几乎常数/数值爆炸”，这里一眼就能看出来
                        sem_norm = torch.linalg.vector_norm(semantic_inputs.float(), dim=-1)  # [B, T]
                        logger.info(
                            f"[Sanity][Epoch {epoch}] semantic_inputs norm: "
                            f"mean={sem_norm.mean().item():.4f}, std={sem_norm.std().item():.4f}, "
                            f"min={sem_norm.min().item():.4f}, max={sem_norm.max().item():.4f}"
                        )
                    logged_sem_stats = True
                
                batch_size = q_data.shape[0]
                
                # 构造 target（从 qa_data 中提取 answer）
                target = (qa_data.float() - 1) / params['n_question']
                target = torch.floor(target)
                target_to_1d = torch.chunk(target, batch_size, 0)
                target_1d = torch.cat([target_to_1d[i] for i in range(batch_size)], 1)
                target_1d = target_1d.permute(1, 0)
                
                model.zero_grad()
                loss, filtered_pred, filtered_target = model.forward(q_data, qa_data, target_1d, semantic_inputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['maxgradnorm'])
                optimizer.step()
                
                epoch_loss += loss.item()
                all_pred.append(filtered_pred.detach().cpu().numpy())
                all_target.append(filtered_target.detach().cpu().numpy())
            
            all_pred = np.concatenate(all_pred, axis=0)
            all_target = np.concatenate(all_target, axis=0)
            train_auc = roc_auc_score(all_target, all_pred)
            train_pred_binary = (all_pred >= 0.5).astype(float)
            train_acc = accuracy_score(all_target, train_pred_binary)
            train_pos_rate = float(np.mean(all_target))
            
            logger.info(f"[Epoch {epoch}] train_loss: {epoch_loss/len(train_loader):.6f}, "
                       f"train_auc: {train_auc:.6f}, train_acc: {train_acc:.6f}")
            logger.info(
                f"[Sanity][Epoch {epoch}] train label pos_rate={train_pos_rate:.6f}; "
                f"pred(mean/std/min/max)=({all_pred.mean():.6f}/{all_pred.std():.6f}/{all_pred.min():.6f}/{all_pred.max():.6f})"
            )
            
            # 验证：使用 dev 做 early stopping / 选模（避免 test 泄漏）
            model.eval()
            dev_pred_list = []
            dev_target_list = []
            dev_loss = 0
            
            with torch.no_grad():
                for q_data, qa_data, semantic_inputs, lengths in tqdm.tqdm(dev_loader, "evaluating(dev)"):
                    q_data = q_data.to(device)
                    qa_data = qa_data.to(device)
                    semantic_inputs = semantic_inputs.to(device)
                    
                    batch_size = q_data.shape[0]
                    target = (qa_data.float() - 1) / params['n_question']
                    target = torch.floor(target)
                    target_to_1d = torch.chunk(target, batch_size, 0)
                    target_1d = torch.cat([target_to_1d[i] for i in range(batch_size)], 1)
                    target_1d = target_1d.permute(1, 0)
                    
                    loss, filtered_pred, filtered_target = model.forward(q_data, qa_data, target_1d, semantic_inputs)
                    dev_loss += loss.item()
                    dev_pred_list.append(filtered_pred.cpu().numpy())
                    dev_target_list.append(filtered_target.cpu().numpy())
            
            dev_pred = np.concatenate(dev_pred_list, axis=0)
            dev_target = np.concatenate(dev_target_list, axis=0)
            dev_auc = roc_auc_score(dev_target, dev_pred)
            dev_pred_binary = (dev_pred >= 0.5).astype(float)
            dev_acc = accuracy_score(dev_target, dev_pred_binary)
            dev_pos_rate = float(np.mean(dev_target))
            
            logger.info(f"[Epoch {epoch}] dev_loss: {dev_loss/len(dev_loader):.6f}, "
                       f"dev_auc: {dev_auc:.6f}, dev_acc: {dev_acc:.6f}")
            logger.info(
                f"[Sanity][Epoch {epoch}] dev label pos_rate={dev_pos_rate:.6f}; "
                f"pred(mean/std/min/max)=({dev_pred.mean():.6f}/{dev_pred.std():.6f}/{dev_pred.min():.6f}/{dev_pred.max():.6f})"
            )
            
            if dev_auc > best_dev_auc:
                best_dev_auc = dev_auc
                dkvmn.save(args.model_path)
                logger.info(f"Model saved at epoch {epoch} with dev_auc: {dev_auc:.6f}")
        
        # 训练结束后，加载 dev 最优模型，在 test 上汇报最终指标
        dkvmn.load(args.model_path)
        model = dkvmn.model.to(device)
        model.eval()
        test_pred_list = []
        test_target_list = []
        with torch.no_grad():
            for q_data, qa_data, semantic_inputs, lengths in tqdm.tqdm(test_loader, "evaluating(test)"):
                q_data = q_data.to(device)
                qa_data = qa_data.to(device)
                semantic_inputs = semantic_inputs.to(device)

                batch_size = q_data.shape[0]
                target = (qa_data.float() - 1) / params['n_question']
                target = torch.floor(target)
                target_to_1d = torch.chunk(target, batch_size, 0)
                target_1d = torch.cat([target_to_1d[i] for i in range(batch_size)], 1)
                target_1d = target_1d.permute(1, 0)

                _, filtered_pred, filtered_target = model.forward(q_data, qa_data, target_1d, semantic_inputs)
                test_pred_list.append(filtered_pred.cpu().numpy())
                test_target_list.append(filtered_target.cpu().numpy())

        test_pred = np.concatenate(test_pred_list, axis=0)
        test_target = np.concatenate(test_target_list, axis=0)
        test_auc = roc_auc_score(test_target, test_pred)
        test_pred_binary = (test_pred >= 0.5).astype(float)
        test_acc = accuracy_score(test_target, test_pred_binary)
        test_pos_rate = float(np.mean(test_target))
        logger.info(f"[Final] test_auc: {test_auc:.6f}, test_acc: {test_acc:.6f}")
        logger.info(
            f"[Sanity][Final] test label pos_rate={test_pos_rate:.6f}; "
            f"pred(mean/std/min/max)=({test_pred.mean():.6f}/{test_pred.std():.6f}/{test_pred.min():.6f}/{test_pred.max():.6f})"
        )
        logger.info("DKVMN model saved with semantic early fusion (selected by dev)!")

def AKT_data_helper(args,logger):
    # 1. read the original file
    with open(args.data_dir, 'r', encoding='utf-8') as f_in:
        json_data = json.loads(f_in.read())
        # for line in f_in.readlines():
        #     dic = json.loads(line)
        #     json_data.append(dic)

    df_nested_list = pd.json_normalize(json_data, record_path =['seq'])

    # 2. define skills  
    raw_question = df_nested_list.problem_id.unique().tolist()
    n_problem = len(raw_question)
    # question id from 0 to (num_skill - 1)
    questions = { p: i for i, p in enumerate(raw_question) }
    logger.info("number of questions: %d" % n_problem)
    args.num_questions = n_problem

    raw_skill = df_nested_list.skill_id.unique().tolist()
    num_skill = len(raw_skill)
    # question id from 0 to (num_skill - 1)
    skills = { p: i for i, p in enumerate(raw_skill) }
    logger.info("number of skills: %d" % num_skill)
    args.num_skills = num_skill

    # 3. s-q-a list 
    # [(skill_seq_0, problem_seq_0, answer_seq_0), ..., (skill_seq_n, problem_seq_n, answer_seq_n)]
    sequences = parse_all_seq(args,df_nested_list.user_id.unique(),df_nested_list,questions,skills)

    # 4. split dataset
    train_sequences, test_sequences = train_test_split(sequences,args.data_split,args.data_shuffle)
    logger.info("data split with ratio {} and shuffle {}".format(args.data_split, args.data_shuffle))

    # 5. save triple line format for other tasks
    sequences2tl(args, train_sequences, args.saved_train_dir)
    sequences2tl(args, test_sequences, args.saved_test_dir)
    logger.info("4 line format trainset saved at {}".format(args.saved_train_dir))
    logger.info("4 line format testset saved at {}".format(args.saved_test_dir))

def AKT_baseline(args, logger):
    batch_size = args.batch_size
    model_type = 'pid'
    n_question = args.num_skills
    n_pid = args.num_questions
    seqlen = 200
    n_blocks = 1
    d_model = 256
    dropout = 0.05
    kq_same = 1
    l2 = 1e-5
    maxgradnorm = -1

    if model_type == 'pid':
        dat = PID_DATA(n_question=n_question, seqlen=seqlen, separate_char=',')
    else:
        dat = DATA(n_question=n_question, seqlen=seqlen, separate_char=',')
    train_data = dat.load_data(args.saved_train_dir)
    test_data = dat.load_data(args.saved_test_dir)


    akt = AKT(n_question, n_pid, n_blocks, d_model, dropout, kq_same, l2, batch_size, maxgradnorm)
    akt.train(train_data, test_data, epoch=args.epoch)
    akt.save(args.model_path)

    akt.load(args.model_path)
    _, auc, accuracy = akt.eval(test_data)
    print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))

if __name__ == '__main__':
    args = config()
    logger = set_logger(args)

    if args.data_process:
        if args.models == 'DKT': 
            DKT_data_helper(args,logger)
        elif args.models == 'DKVMN':
            DKVMN_data_helper(args, logger)
        elif args.models == 'NCDM' or args.models == 'MIRT' or args.models == 'GDIRT':
            CDBD_data_helper(args,logger)
        elif args.models == 'DKT+':
            three_line_format_into_json(args) 
        elif args.models == 'AKT':
            AKT_data_helper(args,logger)

    
    if args.models == 'DKT':
        DKT_Baseline(args,logger)
    elif args.models == 'NCDM' or args.models == 'MIRT' or args.models == 'GDIRT':
        CDBD_Baseline(args,logger)
    elif args.models == 'DKT+':
        DKT_plus_Baseline(args,logger)
    elif args.models == 'DKVMN':
        DKVMN_Baseline(args, logger)
    elif args.models == 'AKT':
        AKT_baseline(args,logger)

        




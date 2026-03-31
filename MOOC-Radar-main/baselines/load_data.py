# Code reused from https://github.com/arghosh/AKT.git
import numpy as np
import math
import pickle
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class DATA(object):
    def __init__(self, n_question, seqlen, separate_char):
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen

    ### data format
    ### 15
    ### 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    ### 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
    def load_data(self, path):
        f_data = open(path , 'r')
        q_data = []
        qa_data = []
        for line_id, line in enumerate(f_data):
            line = line.strip( )
            # line_id starts from 0
            if line_id % 3 == 1:
                Q = line.split(self.separate_char)
                if len( Q[len(Q)-1] ) == 0:
                    Q = Q[:-1]
                #print(len(Q))
            elif line_id % 3 == 2:
                A = line.split(self.separate_char)
                if len( A[len(A)-1] ) == 0:
                    A = A[:-1]
                #print(len(A),A)

                # start split the data
                n_split = 1
                #print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                #print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        end_index  = len(A)
                    else:
                        end_index = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, end_index):
                        if len(Q[i]) > 0 :
                            # int(A[i]) is in {0,1}
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    #print('instance:-->', len(instance),instance)
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        ### convert data into ndarrays for better speed during training
        q_data_array = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_data_array[j, :len(dat)] = dat

        qa_data_array = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_data_array[j, :len(dat)] = dat
        # data_array: [ array([[],[],..])] Shape: (3633, 200)
        return q_data_array, qa_data_array


class PID_DATA(object):
    def __init__(self, n_question, seqlen, separate_char):
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_question = n_question

    # data format
    # length
    # pid1, pid2, ...
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        qa_data = []
        p_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 4 == 2:
                Q = line.split(self.separate_char)
                if len(Q[len(Q) - 1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            if lineID % 4 == 1:
                P = line.split(self.separate_char)
                if len(P[len(P) - 1]) == 0:
                    P = P[:-1]

            elif lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A) - 1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    problem_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k + 1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            Xindex = int(P[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            problem_sequence.append(int(P[i]))
                            answer_sequence.append(Xindex)
                        else:
                            print(Q[i])
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    p_data.append(problem_sequence)

        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat
        return p_dataArray, qa_dataArray
        # return q_dataArray, qa_dataArray, p_dataArray


class SemanticDKTDataset(Dataset):
    """
    DKT Dataset with optional early-fusion semantic embeddings.

    - Input file format (three-line):
        length
        problem_ids separated by `separate_char`
        labels(0/1) separated by `separate_char`

    - Pickle embedding format:
        List[List[np.ndarray]] aligned with the dataset order (index).
        Each inner list is a sequence of vectors, each vector shape=(512,).

    __getitem__ returns:
        x: LongTensor [seq_len] problem_ids
        y: FloatTensor [seq_len] labels (0/1)
        semantic_input: FloatTensor [seq_len, 512]
    """

    def __init__(
        self,
        data_path: str,
        *,
        embedding_path: str = "cognitive_embeddings.pkl",
        separate_char: str = ",",
        semantic_dim: int = 512,
        load_semantic: bool = True,
    ) -> None:
        self.data_path = data_path
        self.separate_char = separate_char
        # semantic_dim 在加载语义 pkl 后会自动推断真实维度（避免硬编码 512）。
        self.semantic_dim = int(semantic_dim)
        self.load_semantic = load_semantic

        self._q_seqs: List[List[int]] = []
        self._a_seqs: List[List[float]] = []

        # Parse 3-line format into per-student sequences
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip() != ""]
        # Every sample is 3 lines: len, q, a
        if len(lines) % 3 != 0:
            raise ValueError(f"Invalid 3-line data file: {data_path}, total non-empty lines={len(lines)}")

        for i in range(0, len(lines), 3):
            # seq_len_line = lines[i]  # not strictly needed, trust content lines
            q_line = lines[i + 1]
            a_line = lines[i + 2]
            q = [int(x) for x in q_line.split(self.separate_char) if x != ""]
            a = [float(x) for x in a_line.split(self.separate_char) if x != ""]
            if len(q) != len(a):
                raise ValueError(f"Length mismatch in {data_path} at sample {i//3}: len(q)={len(q)} len(a)={len(a)}")
            self._q_seqs.append(q)
            self._a_seqs.append(a)

        self._semantic: Optional[Sequence[Sequence[np.ndarray]]] = None
        if self.load_semantic:
            with open(embedding_path, "rb") as f:
                self._semantic = pickle.load(f)
            if len(self._semantic) != len(self._q_seqs):
                raise ValueError(
                    f"Semantic embeddings outer length mismatch: {len(self._semantic)} vs samples {len(self._q_seqs)}. "
                    f"Ensure pickle aligns with the 3-line file order."
                )
            # Auto-infer semantic dim from the first non-empty vector.
            # This keeps training/data code compatible with different embedding backbones.
            inferred_dim: Optional[int] = None
            for sem_seq in self._semantic:
                if not sem_seq:
                    continue
                first_vec = sem_seq[0]
                v = first_vec if isinstance(first_vec, np.ndarray) else np.asarray(first_vec, dtype=np.float32)
                inferred_dim = int(v.shape[-1])
                break
            if inferred_dim is None:
                raise ValueError(f"Failed to infer semantic_dim from empty semantics in {embedding_path}")
            self.semantic_dim = inferred_dim

    def __len__(self) -> int:
        return len(self._q_seqs)

    def __getitem__(self, idx: int):
        q = self._q_seqs[idx]
        a = self._a_seqs[idx]

        x = torch.tensor(q, dtype=torch.long)
        y = torch.tensor(a, dtype=torch.float32)

        if not self.load_semantic:
            semantic_input = torch.zeros((len(q), self.semantic_dim), dtype=torch.float32)
            return x, y, semantic_input

        assert self._semantic is not None
        sem_seq = self._semantic[idx]
        if len(sem_seq) != len(q):
            raise ValueError(f"Semantic seq length mismatch at idx={idx}: len(sem)={len(sem_seq)} vs len(q)={len(q)}")
        sem = np.stack(
            [
                (v.astype(np.float32, copy=False) if isinstance(v, np.ndarray) else np.asarray(v, dtype=np.float32))
                for v in sem_seq
            ],
            axis=0,
        )
        if sem.shape[0] != len(q) or sem.shape[-1] != self.semantic_dim:
            raise ValueError(
                f"Unexpected semantic shape at idx={idx}: {sem.shape}, expected ({len(q)}, {self.semantic_dim})"
            )

        semantic_input = torch.from_numpy(sem)
        return x, y, semantic_input


def semantic_dkt_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    semantic_dim: int = 512,
):
    """
    Pad x/y/semantic_input to max length in batch.
    - x pad value: 0
    - y pad value: 0
    - semantic_input pad value: 0-vector, shape [max_len, semantic_dim]
    Returns:
      x: LongTensor [B, max_len]
      y: FloatTensor [B, max_len]
      semantic_input: FloatTensor [B, max_len, semantic_dim]
      lengths: LongTensor [B]
    """
    xs, ys, sems = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) > 0 else 0

    x_pad = torch.zeros((len(xs), max_len), dtype=torch.long)
    y_pad = torch.zeros((len(xs), max_len), dtype=torch.float32)
    sem_pad = torch.zeros((len(xs), max_len, semantic_dim), dtype=torch.float32)

    for i, (x, y, sem) in enumerate(zip(xs, ys, sems)):
        l = x.shape[0]
        x_pad[i, :l] = x
        y_pad[i, :l] = y
        if sem.numel() > 0:
            sem_pad[i, :l, :] = sem

    return x_pad, y_pad, sem_pad, lengths


class SemanticDKVMNDataset(Dataset):
    """
    DKVMN Dataset with optional early-fusion semantic embeddings.
    
    - Input file format (four-line):
        length
        problem_ids (pid)
        skill_ids (q)
        labels(0/1) (a)
    
    - Pickle embedding format:
        List[List[np.ndarray]] aligned with the dataset order.
        Each inner list is a sequence of vectors, each vector shape=(512,).
    """
    
    def __init__(
        self,
        data_path: str,
        *,
        embedding_path: str = "",
        separate_char: str = ",",
        semantic_dim: int = 512,
        load_semantic: bool = True,
    ) -> None:
        self.data_path = data_path
        self.separate_char = separate_char
        self.semantic_dim = semantic_dim
        self.load_semantic = load_semantic
        
        self._p_seqs: List[List[int]] = []
        self._q_seqs: List[List[int]] = []
        self._a_seqs: List[List[float]] = []
        
        # Parse 4-line format
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip() != ""]
        
        if len(lines) % 4 != 0:
            raise ValueError(f"Invalid 4-line data file: {data_path}, total lines={len(lines)}")
        
        for i in range(0, len(lines), 4):
            p_line = lines[i + 1]
            q_line = lines[i + 2]
            a_line = lines[i + 3]
            p = [int(x) for x in p_line.split(self.separate_char) if x != ""]
            q = [int(x) for x in q_line.split(self.separate_char) if x != ""]
            a = [float(x) for x in a_line.split(self.separate_char) if x != ""]
            
            if len(p) != len(q) or len(q) != len(a):
                raise ValueError(f"Length mismatch at sample {i//4}: len(p)={len(p)}, len(q)={len(q)}, len(a)={len(a)}")
            
            self._p_seqs.append(p)
            self._q_seqs.append(q)
            self._a_seqs.append(a)
        
        self._semantic: Optional[Sequence[Sequence[np.ndarray]]] = None
        if self.load_semantic and embedding_path:
            with open(embedding_path, "rb") as f:
                self._semantic = pickle.load(f)
            if len(self._semantic) != len(self._p_seqs):
                raise ValueError(f"Semantic embeddings length mismatch: {len(self._semantic)} vs {len(self._p_seqs)}")
    
    def __len__(self) -> int:
        return len(self._p_seqs)
    
    def __getitem__(self, idx: int):
        p = self._p_seqs[idx]
        q = self._q_seqs[idx]
        a = self._a_seqs[idx]
        
        p_tensor = torch.tensor(p, dtype=torch.long)
        q_tensor = torch.tensor(q, dtype=torch.long)
        a_tensor = torch.tensor(a, dtype=torch.float32)
        
        if not self.load_semantic or self._semantic is None:
            semantic_input = torch.zeros((len(p), self.semantic_dim), dtype=torch.float32)
            return p_tensor, q_tensor, a_tensor, semantic_input
        
        sem_seq = self._semantic[idx]
        if len(sem_seq) != len(p):
            raise ValueError(f"Semantic seq length mismatch at idx={idx}: len(sem)={len(sem_seq)} vs len(p)={len(p)}")
        
        sem = np.stack([v.astype(np.float32) if isinstance(v, np.ndarray) else np.asarray(v, dtype=np.float32) 
                       for v in sem_seq], axis=0)
        semantic_input = torch.from_numpy(sem)
        
        return p_tensor, q_tensor, a_tensor, semantic_input


def semantic_dkvmn_collate_fn(batch, n_question: int, seqlen: int = 50, semantic_dim: int = 512):
    """
    Collate function for SemanticDKVMNDataset.
    Returns padded sequences and computes qa_data.
    """
    ps, qs, as_, sems = zip(*batch)
    lengths = torch.tensor([len(p) for p in ps], dtype=torch.long)
    max_len = min(int(lengths.max().item()) if len(lengths) > 0 else 0, seqlen)
    
    batch_size = len(ps)
    
    # Pad sequences
    p_pad = np.zeros((batch_size, max_len), dtype=np.int64)
    q_pad = np.zeros((batch_size, max_len), dtype=np.int64)
    a_pad = np.zeros((batch_size, max_len), dtype=np.float32)
    sem_pad = np.zeros((batch_size, max_len, semantic_dim), dtype=np.float32)
    
    for i, (p, q, a, sem) in enumerate(zip(ps, qs, as_, sems)):
        l = min(len(p), max_len)
        p_pad[i, :l] = np.array(p[:l])
        q_pad[i, :l] = np.array(q[:l])
        a_pad[i, :l] = np.array(a[:l])
        if sem.numel() > 0:
            sem_pad[i, :l, :] = sem[:l].numpy()
    
    # Compute qa_data: problem_id + answer * n_question
    qa_pad = p_pad + (a_pad * n_question).astype(np.int64)
    
    return (
        torch.from_numpy(q_pad),  # q_data
        torch.from_numpy(qa_pad),  # qa_data
        torch.from_numpy(sem_pad),  # semantic_inputs
        lengths
    )

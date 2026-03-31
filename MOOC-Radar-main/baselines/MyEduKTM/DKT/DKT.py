# coding: utf-8
# 2021/4/23 @ zengxiaonan

import logging

import numpy as np
import torch
import tqdm
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

from EduKTM import KTM

# 添加设备检测
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Net(nn.Module):
    def __init__(
        self,
        num_questions: int,
        hidden_size: int,
        num_layers: int,
        input_semantic_dim: int = 512,
        emb_dim: int = 64,
        force_early_fusion: bool = False,
    ):
        super(Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.num_questions = num_questions
        self.force_early_fusion = force_early_fusion

        # 用于问题 / 交互 ID 的嵌入
        # 这里默认输入为 2 * num_questions 维的 ID（题目+对错编码）
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_questions * 2, emb_dim)

        # 语义向量投影层，将 512 维降到 emb_dim
        self.semantic_proj = nn.Sequential(
            nn.Linear(input_semantic_dim, emb_dim),
            nn.ReLU(),
        )

        # 早期融合后输入维度为 emb_dim * 2
        self.rnn_fused = nn.RNN(emb_dim * 2, hidden_size, num_layers, batch_first=True)

        # 为了兼容旧流程：直接输入 one-hot 向量 [B, T, 2*num_questions]
        self.rnn_onehot = nn.RNN(num_questions * 2, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, num_questions)

    def forward(self, x, semantic_inputs: torch.Tensor = None):
        """
        两种调用方式：
        1) 兼容旧版 DKT：只传入 x，形状为 [B, T, 2 * num_questions] 的 one-hot 序列；
           此时走 self.rnn_onehot 分支。
        2) 早期融合：传入 problem_ids 序列 x（LongTensor, [B, T]）和
           semantic_inputs（FloatTensor, [B, T, semantic_dim]），
           使用嵌入 + 语义投影后拼接送入 self.rnn_fused。
        """
        if self.force_early_fusion and semantic_inputs is None:
            raise ValueError(
                "Net is in force_early_fusion mode, but semantic_inputs is None. "
                "请确认训练/评估代码调用 model(qa_ids, semantic_inputs)。"
            )

        if semantic_inputs is None:
            # 旧逻辑：x 已经是 one-hot 特征
            if x.dim() != 3:
                raise ValueError(f"Expected one-hot x with dim=3 [B,T,2Q], got shape={tuple(x.shape)}")
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device))
            out, _ = self.rnn_onehot(x, h0)
        else:
            # 新逻辑：x 为 ID，semantic_inputs 为语义向量
            # x: [B, T] (LongTensor)
            # semantic_inputs: [B, T, semantic_dim]
            if x.dim() != 2:
                raise ValueError(f"Expected qa_ids x with dim=2 [B,T], got shape={tuple(x.shape)}")
            if semantic_inputs.dim() != 3:
                raise ValueError(
                    f"Expected semantic_inputs with dim=3 [B,T,D], got shape={tuple(semantic_inputs.shape)}"
                )
            if x.dtype != torch.long:
                x = x.long()
            # 题目/交互 ID 嵌入
            id_emb = self.embedding(x)  # [B, T, emb_dim]
            # 语义向量投影
            sem_emb = self.semantic_proj(semantic_inputs)  # [B, T, emb_dim]
            # 早期融合
            fused_input = torch.cat([id_emb, sem_emb], dim=-1)  # [B, T, 2 * emb_dim]

            h0 = Variable(torch.zeros(self.layer_dim, fused_input.size(0), self.hidden_dim).to(fused_input.device))
            out, _ = self.rnn_fused(fused_input, h0)

        res = torch.sigmoid(self.fc(out))
        return res


def process_raw_pred(raw_question_matrix, raw_pred, num_questions: int) -> tuple:
    questions = torch.nonzero(raw_question_matrix)[1:, 1] % num_questions
    length = questions.shape[0]
    pred = raw_pred[: length]
    pred = pred.gather(1, questions.view(-1, 1)).flatten()
    truth = torch.nonzero(raw_question_matrix)[1:, 1] // num_questions
    return pred, truth


class DKT(KTM):
    def __init__(
        self,
        num_questions: int,
        hidden_size: int,
        num_layers: int,
        input_semantic_dim: int = 512,
        emb_dim: int = 64,
        use_early_fusion: bool = False,
    ):
        super(DKT, self).__init__()
        self.num_questions = num_questions
        self.dkt_model = Net(
            num_questions,
            hidden_size,
            num_layers,
            input_semantic_dim=input_semantic_dim,
            emb_dim=emb_dim,
            force_early_fusion=use_early_fusion,
        )
        # 将模型移到 GPU（如果可用）
        self.dkt_model = self.dkt_model.to(device)
        logging.info(f"DKT model using device: {device}")

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002) -> ...:
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr)

        for e in range(epoch):
            all_pred, all_target = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
            for batch in tqdm.tqdm(train_data, "Epoch %s" % e):
                # 将 batch 移到 GPU
                batch = batch.to(device)
                integrated_pred = self.dkt_model(batch)
                batch_size = batch.shape[0]
                for student in range(batch_size):
                    pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
                    all_pred = torch.cat([all_pred, pred])
                    all_target = torch.cat([all_target, truth.float()])

            loss = loss_function(all_pred, all_target)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("[Epoch %d] LogisticLoss: %.6f" % (e, loss))

            if test_data is not None:
                auc = self.eval(test_data)
                print("[Epoch %d] auc: %.6f" % (e, auc))

    def eval(self, test_data) -> float:
        self.dkt_model.eval()
        y_pred = torch.Tensor([]).to(device)
        y_truth = torch.Tensor([]).to(device)
        for batch in tqdm.tqdm(test_data, "evaluating"):
            # 将 batch 移到 GPU
            batch = batch.to(device)
            integrated_pred = self.dkt_model(batch)
            batch_size = batch.shape[0]
            for student in range(batch_size):
                pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_questions)
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])

        # 将结果移回 CPU 再转 numpy
        return roc_auc_score(y_truth.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

    def save(self, filepath):
        torch.save(self.dkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_model.load_state_dict(torch.load(filepath, map_location=device))
        logging.info("load parameters from %s" % filepath)

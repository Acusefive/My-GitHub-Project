# coding: utf-8
# 2022/3/18 @ ouyangjie


import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn import metrics
from tqdm import tqdm
from EduKTM import KTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Cell(nn.Module):
    def __init__(self, memory_size, memory_state_dim):
        super(Cell, self).__init__()
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim

    def addressing(self, control_input, memory):
        """
        Parameters
        ----------
        control_input: tensor
            embedding vector of input exercise, shape = (batch_size, control_state_dim)
        memory: tensor
            key memory, shape = (memory_size, memory_state_dim)

        Returns
        -------
        correlation_weight: tensor
            correlation weight, shape = (batch_size, memory_size)
        """
        similarity_score = torch.matmul(control_input, torch.t(memory))
        correlation_weight = F.softmax(similarity_score, dim=1)  # Shape: (batch_size, memory_size)
        return correlation_weight

    def read(self, memory, read_weight):
        """
        Parameters
        ----------
        memory: tensor
            value memory, shape = (batch_size, memory_size, memory_state_dim)
        read_weight: tensor
            correlation weight, shape = (batch_size, memory_size)

        Returns
        -------
        read_content: tensor
            read content, shape = (batch_size, memory_size)
        """
        read_weight = read_weight.view(-1, 1)
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)
        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)
        read_content = torch.sum(read_content, dim=1)
        return read_content


class WriteCell(Cell):
    def __init__(self, memory_size, memory_state_dim):
        super(WriteCell, self).__init__(memory_size, memory_state_dim)
        self.erase = torch.nn.Linear(memory_state_dim, memory_state_dim, bias=True)
        self.add = torch.nn.Linear(memory_state_dim, memory_state_dim, bias=True)
        nn.init.kaiming_normal_(self.erase.weight)
        nn.init.kaiming_normal_(self.add.weight)
        nn.init.constant_(self.erase.bias, 0)
        nn.init.constant_(self.add.bias, 0)

    def write(self, control_input, memory, write_weight):
        """
        Parameters
        ----------
        control_input: tensor
            embedding vector of input exercise and students' answer, shape = (batch_size, control_state_dim)
        memory: tensor
            value memory, shape = (batch_size, memory_size, memory_state_dim)
        read_weight: tensor
            correlation weight, shape = (batch_size, memory_size)

        Returns
        -------
        new_memory: tensor
            updated value memory, shape = (batch_size, memory_size, memory_state_dim)
        """
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        erase_mult = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        new_memory = memory * (1 - erase_mult) + add_mul
        return new_memory


class DKVMNCell(nn.Module):
    def __init__(self, memory_size, key_memory_state_dim, value_memory_state_dim, init_key_memory):
        super(DKVMNCell, self).__init__()
        """
        Parameters
        ----------
        memory_size: int
            size of memory
        key_memory_state_dim: int
            dimension of key memory
        value_memory_state_dim:  int
            dimension of value memory
        init_key_memory: tensor
            intial key memory
        """
        self.memory_size = memory_size
        self.key_memory_state_dim = key_memory_state_dim
        self.value_memory_state_dim = value_memory_state_dim

        self.key_head = Cell(memory_size=self.memory_size, memory_state_dim=self.key_memory_state_dim)
        self.value_head = WriteCell(memory_size=self.memory_size, memory_state_dim=self.value_memory_state_dim)

        self.key_memory = init_key_memory
        self.value_memory = None

    def init_value_memory(self, value_memory):
        self.value_memory = value_memory

    def attention(self, control_input):
        correlation_weight = self.key_head.addressing(control_input=control_input, memory=self.key_memory)
        return correlation_weight

    def read(self, read_weight):
        read_content = self.value_head.read(memory=self.value_memory, read_weight=read_weight)
        return read_content

    def write(self, write_weight, control_input):
        value_memory = self.value_head.write(control_input=control_input,
                                             memory=self.value_memory,
                                             write_weight=write_weight)
        self.value_memory = nn.Parameter(value_memory.data)

        return self.value_memory


class Net(nn.Module):
    def __init__(self, n_question, batch_size, key_embedding_dim, value_embedding_dim,
                 memory_size, key_memory_state_dim, value_memory_state_dim, final_fc_dim, 
                 student_num=None, input_semantic_dim: int = 512, use_early_fusion: bool = False):
        super(Net, self).__init__()
        self.n_question = n_question
        self.batch_size = batch_size
        self.key_embedding_dim = key_embedding_dim
        self.value_embedding_dim = value_embedding_dim
        self.memory_size = memory_size
        self.key_memory_state_dim = key_memory_state_dim
        self.value_memory_state_dim = value_memory_state_dim
        self.final_fc_dim = final_fc_dim
        self.student_num = student_num
        self.use_early_fusion = use_early_fusion
        self.input_semantic_dim = input_semantic_dim

        self.input_embed_linear = nn.Linear(self.key_embedding_dim, self.final_fc_dim, bias=True)
        # read_embed_linear 会在下方根据是否使用语义融合方式来设置输入维度
        # 默认（无语义）为: value_read + q_embed -> final_fc_dim
        self.read_embed_linear = None
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)
        self.init_key_memory = nn.Parameter(torch.randn(self.memory_size, self.key_memory_state_dim))
        nn.init.kaiming_normal_(self.init_key_memory)
        self.init_value_memory = nn.Parameter(torch.randn(self.memory_size, self.value_memory_state_dim))
        nn.init.kaiming_normal_(self.init_value_memory)

        self.mem = DKVMNCell(memory_size=self.memory_size, key_memory_state_dim=self.key_memory_state_dim,
                             value_memory_state_dim=self.value_memory_state_dim, init_key_memory=self.init_key_memory)

        value_memory = nn.Parameter(torch.cat([self.init_value_memory.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(value_memory)

        self.q_embed = nn.Embedding(self.n_question + 1, self.key_embedding_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.value_embedding_dim, padding_idx=0)

        # ======== 语义 FiLM 融合组件（调制 read_value，而非直接拼接） ========
        if self.use_early_fusion:
            self.sem_ln = nn.LayerNorm(input_semantic_dim)
            self.sem_proj = nn.Sequential(
                nn.Linear(input_semantic_dim, self.final_fc_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            # FiLM 发生器：根据语义特征生成 gamma/beta，直接作用于 value_memory_state_dim
            self.film_gamma = nn.Linear(self.final_fc_dim, self.value_memory_state_dim)
            self.film_beta = nn.Linear(self.final_fc_dim, self.value_memory_state_dim)

            # 不再做语义拼接，read_embed_linear 输入维度恢复为原版 DKVMN
            self.read_embed_linear = nn.Linear(
                self.value_memory_state_dim + self.key_embedding_dim,
                self.final_fc_dim,
                bias=True,
            )
        else:
            self.read_embed_linear = nn.Linear(
                self.value_memory_state_dim + self.key_embedding_dim,
                self.final_fc_dim,
                bias=True,
            )

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)

    def init_embeddings(self):

        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

    def forward(self, q_data, qa_data, target, semantic_inputs: torch.Tensor = None):
        """
        Args:
            q_data: [B, T] question/skill IDs
            qa_data: [B, T] question-answer IDs (problem_id + answer * n_question)
            target: [B, T] target labels
            semantic_inputs: [B, T, semantic_dim] semantic embeddings (optional)
        """
        if self.use_early_fusion:
            if semantic_inputs is None:
                raise ValueError(
                    "use_early_fusion=True 但未提供 semantic_inputs；为保证语义/FiLM 实验真实生效，已禁止自动回退到原版。"
                )
            if semantic_inputs.dim() != 3:
                raise ValueError(f"semantic_inputs 维度应为 [B,T,D]，但得到 shape={tuple(semantic_inputs.shape)}")
            if semantic_inputs.shape[0] != q_data.shape[0] or semantic_inputs.shape[1] != q_data.shape[1]:
                raise ValueError(
                    f"semantic_inputs 的 [B,T] 必须与 q_data 对齐："
                    f"q_data={tuple(q_data.shape)} semantic_inputs={tuple(semantic_inputs.shape)}"
                )
        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        
        # 1. 保持最纯粹的 Embedding，不在前面阶段注入语义
        q_embed_data = self.q_embed(q_data)          # [B, T, key_embedding_dim]
        qa_embed_data = self.qa_embed(qa_data)       # [B, T, value_embedding_dim]

        value_memory = nn.Parameter(torch.cat([self.init_value_memory.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(value_memory)

        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)

        value_read_content_l = []
        input_embed_l = []
        for i in range(seqlen):
            # Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)

            # Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)
            # Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            self.mem.write(correlation_weight, qa)

        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)

        # 2. FiLM：用语义特征调制 read_value（比简单拼接更平滑）
        if self.use_early_fusion and semantic_inputs is not None:
            sem_norm = self.sem_ln(semantic_inputs)
            sem_feat = self.sem_proj(sem_norm)  # [B, T, final_fc_dim]

            gamma = self.film_gamma(sem_feat)  # [B, T, value_memory_state_dim]
            beta = self.film_beta(sem_feat)    # [B, T, value_memory_state_dim]

            modulated_read_value = all_read_value_content * (1.0 + gamma) + beta
            predict_input = torch.cat([modulated_read_value, input_embed_content], dim=2)
        else:
            predict_input = torch.cat([all_read_value_content, input_embed_content], dim=2)

        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(batch_size * seqlen, -1)))

        pred = self.predict_linear(read_content_embed)
        target_1d = target                   # [batch_size * seq_len, 1]
        mask = target_1d.ge(0)               # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)           # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        loss = F.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target


class DKVMN(KTM):
    def __init__(self, n_question, batch_size, key_embedding_dim, value_embedding_dim,
                 memory_size, key_memory_state_dim, value_memory_state_dim, final_fc_dim, 
                 student_num=None, input_semantic_dim: int = 512, use_early_fusion: bool = False):
        super(DKVMN, self).__init__()
        self.batch_size = batch_size
        self.n_question = n_question
        self.use_early_fusion = use_early_fusion
        self.model = Net(n_question, batch_size, key_embedding_dim, value_embedding_dim,
                         memory_size, key_memory_state_dim, value_memory_state_dim, final_fc_dim, 
                         student_num, input_semantic_dim=input_semantic_dim, use_early_fusion=use_early_fusion)

    def train_epoch(self, epoch, model, params, optimizer, q_data, qa_data, semantic_data=None):
        N = int(math.floor(len(q_data) / params['batch_size']))

        pred_list = []
        target_list = []
        epoch_loss = 0

        model.train()

        for idx in tqdm(range(N), "Epoch %s" % epoch):
            q_one_seq = q_data[idx * params['batch_size']:(idx + 1) * params['batch_size'], :]
            qa_batch_seq = qa_data[idx * params['batch_size']:(idx + 1) * params['batch_size'], :]
            target = qa_data[idx * params['batch_size']:(idx + 1) * params['batch_size'], :]

            target = (target - 1) / params['n_question']
            target = np.floor(target)
            input_q = torch.LongTensor(q_one_seq).to(device)
            input_qa = torch.LongTensor(qa_batch_seq).to(device)
            target = torch.FloatTensor(target).to(device)
            target_to_1d = torch.chunk(target, params['batch_size'], 0)
            target_1d = torch.cat([target_to_1d[i] for i in range(params['batch_size'])], 1)
            target_1d = target_1d.permute(1, 0)

            model.zero_grad()
            semantic_inputs = None
            if semantic_data is not None:
                sem_batch = semantic_data[idx * params['batch_size']:(idx + 1) * params['batch_size'], :, :]
                semantic_inputs = torch.as_tensor(sem_batch, dtype=torch.float32, device=device)
            loss, filtered_pred, filtered_target = model.forward(input_q, input_qa, target_1d, semantic_inputs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params['maxgradnorm'])
            optimizer.step()
            epoch_loss += loss.item()

            right_target = np.asarray(filtered_target.data.tolist())
            right_pred = np.asarray(filtered_pred.data.tolist())
            pred_list.append(right_pred)
            target_list.append(right_target)

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)
        auc = metrics.roc_auc_score(all_target, all_pred)
        all_pred[all_pred >= 0.5] = 1.0
        all_pred[all_pred < 0.5] = 0.0
        accuracy = metrics.accuracy_score(all_target, all_pred)

        return epoch_loss / N, accuracy, auc

    def train(self, params, train_data, test_data=None):
        # train_data can be (q_data, qa_data) or (q_data, qa_data, semantic_data)
        if len(train_data) == 2:
            q_data, qa_data = train_data
            semantic_data = None
        elif len(train_data) == 3:
            q_data, qa_data, semantic_data = train_data
        else:
            raise ValueError(f"Unexpected train_data tuple length: {len(train_data)} (expected 2 or 3)")
        if self.use_early_fusion and semantic_data is None:
            raise ValueError(
                "DKVMN(use_early_fusion=True) 的训练需要 semantic_data，但当前 train_data 未提供；"
                "为保证语义/FiLM 实验真实生效，已禁止自动回退到原版。"
            )

        model = self.model
        model.init_embeddings()
        model.init_params()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'], betas=(0.9, 0.9))

        model.to(device)

        all_valid_loss = {}
        all_valid_accuracy = {}
        all_valid_auc = {}
        best_valid_auc = 0
        best_state_dict = None

        for idx in range(params['max_iter']):
            train_loss, train_accuracy, train_auc = self.train_epoch(idx, model, params, optimizer, q_data, qa_data, semantic_data)
            print('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f' %
                  (idx + 1, params['max_iter'], train_loss, train_auc, train_accuracy))
            if test_data is not None:
                valid_loss, valid_accuracy, valid_auc = self.eval(params, test_data)
                all_valid_loss[idx + 1] = valid_loss
                all_valid_accuracy[idx + 1] = valid_accuracy
                all_valid_auc[idx + 1] = valid_auc
                # output the epoch with the best validation auc
                if valid_auc > best_valid_auc:
                    print('valid auc improve: %3.4f to %3.4f' % (best_valid_auc, valid_auc))
                    best_valid_auc = valid_auc
                    # 保存当前最优模型参数（避免使用 test 集做 epoch 选择时的泄漏）
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # 训练结束后，如果有验证集，则回滚到 best_valid_auc 对应的参数
        if best_state_dict is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        return best_valid_auc

    def eval(self, params, data):
        # data can be (q_data, qa_data) or (q_data, qa_data, semantic_data)
        if len(data) == 2:
            q_data, qa_data = data
            semantic_data = None
        elif len(data) == 3:
            q_data, qa_data, semantic_data = data
        else:
            raise ValueError(f"Unexpected data tuple length: {len(data)} (expected 2 or 3)")
        if self.use_early_fusion and semantic_data is None:
            raise ValueError(
                "DKVMN(use_early_fusion=True) 的评估需要 semantic_data，但当前 data 未提供；"
                "为保证语义/FiLM 实验真实生效，已禁止自动回退到原版。"
            )
        model = self.model
        N = int(math.floor(len(q_data) / params['batch_size']))

        pred_list = []
        target_list = []
        epoch_loss = 0
        model.eval()

        for idx in tqdm(range(N), "Evaluating"):

            q_one_seq = q_data[idx * params['batch_size']:(idx + 1) * params['batch_size'], :]
            qa_batch_seq = qa_data[idx * params['batch_size']:(idx + 1) * params['batch_size'], :]
            target = qa_data[idx * params['batch_size']:(idx + 1) * params['batch_size'], :]

            target = (target - 1) / params['n_question']
            target = np.floor(target)

            input_q = torch.LongTensor(q_one_seq).to(device)
            input_qa = torch.LongTensor(qa_batch_seq).to(device)
            target = torch.FloatTensor(target).to(device)

            target_to_1d = torch.chunk(target, params['batch_size'], 0)
            target_1d = torch.cat([target_to_1d[i] for i in range(params['batch_size'])], 1)
            target_1d = target_1d.permute(1, 0)

            semantic_inputs = None
            if semantic_data is not None:
                sem_batch = semantic_data[idx * params['batch_size']:(idx + 1) * params['batch_size'], :, :]
                semantic_inputs = torch.as_tensor(sem_batch, dtype=torch.float32, device=device)
            loss, filtered_pred, filtered_target = model.forward(input_q, input_qa, target_1d, semantic_inputs)

            right_target = np.asarray(filtered_target.data.tolist())
            right_pred = np.asarray(filtered_pred.data.tolist())
            pred_list.append(right_pred)
            target_list.append(right_target)
            epoch_loss += loss.item()

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)

        auc = metrics.roc_auc_score(all_target, all_pred)
        all_pred[all_pred >= 0.5] = 1.0
        all_pred[all_pred < 0.5] = 0.0
        accuracy = metrics.accuracy_score(all_target, all_pred)
        print('valid auc : %3.5f, valid accuracy : %3.5f' % (auc, accuracy))

        return epoch_loss / N, accuracy, auc

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

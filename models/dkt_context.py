import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot

from models.context_fusion import ContextFusion


class DKTContext(Module):
    def __init__(self, num_q, emb_size, hidden_size, ctx_dim, dropout=0.1, fusion_type="gate"):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.ctx_dim = ctx_dim
        self.fusion_type = fusion_type

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.context_fusion = ContextFusion(self.hidden_size, self.ctx_dim, mode=self.fusion_type)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_q)

    def forward(self, q, r, qry, ctx=None):
        x = q + self.num_q * r
        h, _ = self.lstm_layer(self.interaction_emb(x))

        if ctx is not None:
            h = self.context_fusion(h, ctx)

        y = self.out_layer(self.dropout_layer(h))
        y = torch.sigmoid(y)
        p = (y * one_hot(qry.long(), self.num_q)).sum(-1)
        return p

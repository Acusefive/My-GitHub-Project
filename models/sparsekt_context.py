import torch

from models.pykt_sparsekt_base import SparseKT
from models.simplekt_context import SimpleKTContext


class SparseKTContext(SimpleKTContext):
    def __init__(
        self,
        num_q,
        num_c,
        q_to_concept,
        ctx_dim,
        d_model=100,
        n_blocks=1,
        dropout=0.2,
        d_ff=256,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        num_attn_heads=5,
        separate_qa=False,
        l2=1e-5,
        max_seq_len=512,
        sparse_ratio=0.8,
        k_index=5,
        stride=1,
        **kwargs,
    ):
        super().__init__(
            num_q=num_q,
            num_c=num_c,
            q_to_concept=q_to_concept,
            ctx_dim=ctx_dim,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
            d_ff=d_ff,
            kq_same=kq_same,
            final_fc_dim=final_fc_dim,
            final_fc_dim2=final_fc_dim2,
            num_attn_heads=num_attn_heads,
            separate_qa=separate_qa,
            l2=l2,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        # SparseKT uses the same input/output contract as SimpleKT; only its
        # knowledge-retrieval attention is replaced by causal Top-K attention.
        self.base = SparseKT(
            self.num_c + 1,
            self.num_q,
            d_model=int(d_model),
            n_blocks=int(n_blocks),
            dropout=float(dropout),
            d_ff=int(d_ff),
            kq_same=int(kq_same),
            final_fc_dim=int(final_fc_dim),
            final_fc_dim2=int(final_fc_dim2),
            num_attn_heads=int(num_attn_heads),
            separate_qa=bool(separate_qa),
            l2=float(l2),
            max_seq_len=int(max_seq_len),
            sparse_ratio=float(sparse_ratio),
            k_index=int(k_index),
            stride=int(stride),
        )

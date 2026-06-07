"""
flash_attn.modules.mha shim — drop-in MHA replacement using PyTorch SDPA.

Provides the same constructor and forward() signature as flash_attn's MHA
so that UniRig works without the CUDA flash-attention kernels.

Used interface (from UniRig):
    MHA(embed_dim=D, num_heads=H, cross_attn=True)
    forward(q, x_kv=kv)  -> Tensor same shape as q

The real flash_attn MHA uses a combined Wkv projection (key+value packed
into one Linear). The checkpoint stores weights as "Wkv.weight" and
"Wkv.bias", so this shim must match that parameter naming.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    """Multi-Head Attention compatible with flash_attn's MHA API.

    Falls back to PyTorch's scaled_dot_product_attention (SDPA),
    which uses Flash Attention v2 or memory-efficient attention
    automatically when available on the hardware.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        cross_attn: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,  # absorb extra flash_attn kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.cross_attn = cross_attn
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Q projection (always from the query input)
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Combined K+V projection — matches flash_attn checkpoint format
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, x_kv=None, **kwargs):
        """
        Args:
            x:    (batch, seq_q, embed_dim)  -- query input
            x_kv: (batch, seq_kv, embed_dim) -- key/value input for cross-attn
                  If None, self-attention is performed (x_kv = x).
        Returns:
            Tensor of shape (batch, seq_q, embed_dim)
        """
        if x_kv is None:
            x_kv = x

        B, S_q, _ = x.shape
        S_kv = x_kv.shape[1]

        # Project Q
        q = self.Wq(x).reshape(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)

        # Project K, V from combined Wkv
        kv = self.Wkv(x_kv).reshape(B, S_kv, 2, self.num_heads, self.head_dim)
        k = kv[:, :, 0].transpose(1, 2)  # (B, heads, S_kv, head_dim)
        v = kv[:, :, 1].transpose(1, 2)  # (B, heads, S_kv, head_dim)

        # Use PyTorch's SDPA -- automatically selects Flash Attn v2 or
        # memory-efficient attention based on hardware support
        drop_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)

        # Reshape back: (B, heads, S_q, head_dim) -> (B, S_q, embed_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, S_q, self.embed_dim)

        return self.out_proj(attn_out)

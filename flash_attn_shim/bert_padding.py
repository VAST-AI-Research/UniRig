"""
flash_attn.bert_padding shim — stub implementations for transformers compatibility.

These functions handle padding/unpadding for variable-length attention.
Since we use ATTN_IMPLEMENTATION=eager, these are never called at runtime,
but they must exist so the import in modeling_flash_attention_utils.py succeeds.
"""

import torch


def index_first_axis(hidden_states, indices):
    """Select rows from hidden_states using indices."""
    return hidden_states[indices]


def pad_input(hidden_states, indices, batch, seqlen):
    """Pad hidden_states back to (batch, seqlen, dim)."""
    dim = hidden_states.shape[-1]
    output = torch.zeros(batch * seqlen, dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.reshape(batch, seqlen, dim)


def unpad_input(hidden_states, attention_mask):
    """Remove padding tokens based on attention_mask.

    Returns:
        hidden_states_unpadded: (total_tokens, dim)
        indices: positions of non-pad tokens
        cu_seqlens: cumulative sequence lengths
        max_seqlen: maximum sequence length in batch
    """
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens.max().item()
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return hidden_states[indices], indices, cu_seqlens, max_seqlen

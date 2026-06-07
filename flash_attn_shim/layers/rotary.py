"""
flash_attn.layers.rotary shim — stub for apply_rotary_emb.

Since we use ATTN_IMPLEMENTATION=eager, transformers won't call this,
but the import must succeed when modeling_flash_attention_utils loads.
"""

import torch


def apply_rotary_emb(x, cos, sin, interleaved=False):
    """Apply rotary positional embeddings (stub — not used with eager attention)."""
    raise NotImplementedError(
        "flash_attn shim: apply_rotary_emb should not be called when "
        "ATTN_IMPLEMENTATION=eager. If you see this, check your attention config."
    )

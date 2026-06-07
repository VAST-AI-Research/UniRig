"""
flash_attn shim — provides the MHA interface using standard PyTorch attention.
This allows UniRig to run on Windows where flash_attn CUDA kernels
cannot be compiled due to Long Path limitations.

Submodules provided:
  - flash_attn.modules.mha         (MHA class using PyTorch SDPA)
  - flash_attn.bert_padding        (pad/unpad stubs for transformers)
  - flash_attn.layers.rotary       (rotary embedding stub)
"""
__version__ = "2.8.3.shim"


def flash_attn_func(*args, **kwargs):
    """Stub — not used with ATTN_IMPLEMENTATION=eager."""
    raise NotImplementedError("flash_attn shim: flash_attn_func not available, use eager attention")


def flash_attn_varlen_func(*args, **kwargs):
    """Stub — not used with ATTN_IMPLEMENTATION=eager."""
    raise NotImplementedError("flash_attn shim: flash_attn_varlen_func not available, use eager attention")

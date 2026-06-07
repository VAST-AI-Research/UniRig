# UniRig — Windows + RTX 5090 Installation Guide

This fork adds Windows compatibility for UniRig, tested on:
- **GPU**: NVIDIA RTX 5090 (Blackwell)
- **CUDA**: 12.8 driver + CUDA Toolkit 12.8
- **PyTorch**: 2.11+ (cu128)
- **OS**: Windows 11 Pro

## What's Changed

| Fix | Why |
|-----|-----|
| CUDA DLL path (`run.py`) | spconv/cumm can't find CUDA DLLs on Windows without `os.add_dll_directory()` |
| `weights_only=False` patch (`run.py`) | PyTorch 2.11 defaults to safe unpickling, but UniRig checkpoints contain `box.Box` objects |
| `torch_scatter` CPU fallback (`run.py`) | torch_scatter/torch_cluster compile CPU-only on Windows (no nvcc); ops fall back to CPU transparently |
| `enable_flash: False` (config) | PTv3's flash path calls `flash_attn_varlen_qkvpacked_func` which isn't available; non-flash SDPA path works |
| `eager` attention (config) | OPT LLM uses PyTorch SDPA instead of flash_attention_2 |
| flash_attn SDPA shim | Drop-in replacement using `F.scaled_dot_product_attention` with combined Wkv projection matching checkpoint format |
| Shell script venv activation | Scripts auto-detect and activate `.venv` on Windows or Linux |
| Backslash path fix (shell scripts) | Convert `\` to `/` for Git Bash compatibility |
| NPZ export in user_mode (`ar.py`) | Skeleton NPZ must be written even when `--output` is passed, otherwise skinning step has no input |
| `fps()` CPU fallback (`sal_perceiver.py`) | torch_cluster's farthest-point-sampling falls back to CPU when CUDA kernel unavailable |

## Installation

### Prerequisites
- Python 3.10+
- [CUDA Toolkit 12.4+](https://developer.nvidia.com/cuda-toolkit) installed (CUDA_HOME must be set)
- Git Bash (comes with Git for Windows)

### Steps

```bash
# 1. Clone this fork
git clone https://github.com/HurtzDonutStudios/UniRig.git
cd UniRig

# 2. Create venv
python -m venv .venv
.venv\Scripts\activate  # Windows cmd/powershell
# or: source .venv/Scripts/activate  # Git Bash

# 3. Install PyTorch (CUDA 12.8 example — adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Install UniRig dependencies
pip install -r requirements.txt

# 5. Install spconv — MUST use cu124 (cu120 causes heap corruption on CUDA 12.8)
pip install spconv-cu124==2.3.8

# 6. Install torch_scatter and torch_cluster (CPU-only wheels are fine)
pip install torch_scatter torch_cluster

# 7. Install the flash_attn shim (replaces CUDA flash-attention with PyTorch SDPA)
python install_flash_attn_shim.py

# 8. Set environment variables
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set ATTN_IMPLEMENTATION=eager
```

### Run

```bash
# Full pipeline: mesh → skeleton → skin → merge
bash launch/inference/generate_skeleton.sh \
    --input /path/to/mesh.glb \
    --output /path/to/output_skeleton.fbx \
    --output_dir /path/to/output_dir

bash launch/inference/generate_skin.sh \
    --input /path/to/mesh.glb \
    --output /path/to/output_skin.fbx \
    --output_dir /path/to/output_dir

bash launch/inference/merge.sh \
    --source /path/to/output_skin.fbx \
    --target /path/to/mesh.glb \
    --output /path/to/rigged_mesh.glb
```

## Known Issues

- **bpy segfault on exit (code 139)**: Blender's Python module crashes during interpreter shutdown. Output files are written before the crash — this is harmless.
- **Softmax dimension warning**: `F.softmax()` without `dim=` at `unirig_skin.py:434` — cosmetic warning, doesn't affect results.
- **spconv-cu120 heap corruption**: Using `spconv-cu120` with CUDA 12.8 driver causes `STATUS_HEAP_CORRUPTION`. Always use `spconv-cu124` or newer.

## Performance (RTX 5090, 884K vertex mesh)

| Step | Time |
|------|------|
| Extract | 28.5s |
| Skeleton inference | 4s |
| Skinning inference | 3s |
| Merge | 10s |
| **Total** | **~46s** |

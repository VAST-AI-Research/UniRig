"""
Install the flash_attn shim into the active Python environment.

Usage:
    python install_flash_attn_shim.py

This copies flash_attn_shim/ into site-packages as flash_attn/,
providing a drop-in replacement that uses PyTorch SDPA instead of
the CUDA flash-attention kernels (which don't compile on Windows).
"""

import os
import sys
import shutil
import site


def main():
    # Find site-packages
    sp = site.getsitepackages()
    if not sp:
        sp = [site.getusersitepackages()]
    target_dir = None
    for d in sp:
        if os.path.isdir(d):
            target_dir = d
            break
    if target_dir is None:
        print("ERROR: Could not find site-packages directory", file=sys.stderr)
        sys.exit(1)

    src = os.path.join(os.path.dirname(__file__), "flash_attn_shim")
    dst = os.path.join(target_dir, "flash_attn")

    if os.path.exists(dst):
        # Check if it's the real flash_attn or our shim
        init_file = os.path.join(dst, "__init__.py")
        if os.path.exists(init_file):
            with open(init_file, "r") as f:
                content = f.read()
            if ".shim" in content:
                print(f"flash_attn shim already installed at {dst}")
                return
            else:
                print(f"Real flash_attn found at {dst} — skipping (not needed)")
                return
        # Directory exists but no __init__.py — replace it
        shutil.rmtree(dst)

    print(f"Installing flash_attn shim: {src} -> {dst}")
    shutil.copytree(src, dst)
    print("Done. flash_attn shim installed successfully.")


if __name__ == "__main__":
    main()

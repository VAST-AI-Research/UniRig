# merge texture

# === Auto-activate UniRig venv (Windows/Linux) ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIRIG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "$UNIRIG_ROOT/.venv/Scripts/activate" ]; then
    source "$UNIRIG_ROOT/.venv/Scripts/activate"
elif [ -f "$UNIRIG_ROOT/.venv/bin/activate" ]; then
    source "$UNIRIG_ROOT/.venv/bin/activate"
fi
export PYTHONPATH="$UNIRIG_ROOT:$PYTHONPATH"
export ATTN_IMPLEMENTATION=eager
cd "$UNIRIG_ROOT"
# ================================================

require_suffix="obj,fbx,FBX,dae,glb,gltf,vrm"
source=""
target=""
output=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --require_suffix) require_suffix="$2"; shift ;;
        --source) source="$2"; shift ;;
        --target) target="$2"; shift ;;
        --output) output="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# === Fix Windows backslash paths ===
if [ -n "$source" ]; then source=$(echo "$source" | sed 's|\\|/|g'); fi
if [ -n "$target" ]; then target=$(echo "$target" | sed 's|\\|/|g'); fi
if [ -n "$output" ]; then output=$(echo "$output" | sed 's|\\|/|g'); fi
# ===================================

cmd=" \
    python -m src.inference.merge \
    --require_suffix=$require_suffix \
    --num_runs=1 \
    --id=0 \
    --source=$source \
    --target=$target \
    --output=$output \
"

cmd="$cmd &"
eval $cmd

wait

echo "done"
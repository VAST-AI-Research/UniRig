# extract mesh

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

config="configs/data/quick_inference.yaml"
require_suffix="obj,fbx,FBX,dae,glb,gltf,vrm"
num_runs=1
force_override="false"
faces_target_count=50000

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) config="$2"; shift ;;
        --require_suffix) require_suffix="$2"; shift ;;
        --num_runs) num_runs="$2"; shift ;;
        --force_override) force_override="$2"; shift ;;
        --faces_target_count) faces_target_count="$2"; shift ;;
        --time) time="$2"; shift ;;
        --input) input="$2"; shift ;;
        --input_dir) input_dir="$2"; shift ;;
        --output_dir) output_dir="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# === Fix Windows backslash paths ===
if [ -n "$input" ]; then input=$(echo "$input" | sed 's|\\|/|g'); fi
if [ -n "$input_dir" ]; then input_dir=$(echo "$input_dir" | sed 's|\\|/|g'); fi
if [ -n "$output_dir" ]; then output_dir=$(echo "$output_dir" | sed 's|\\|/|g'); fi
# ===================================

# ensure psutil is installed for memory management
pip install psutil --quiet
if [ $? -ne 0 ]; then
    echo "Warning: Failed to install psutil. Memory management may not work properly."
fi

# set the time for all processes to use
time=$(date "+%Y_%m_%d_%H_%M_%S")

for (( i=0; i<num_runs; i++ ))
do
    cmd=" \
    python -m src.data.extract \
    --config=$config \
    --require_suffix=$require_suffix \
    --force_override=$force_override \
    --num_runs=$num_runs \
    --id=$i \
    --time=$time \
    --faces_target_count=$faces_target_count \
    "
    if [ -n "$input" ]; then
        cmd="$cmd --input=$input"
    fi
    if [ -n "$input_dir" ]; then
        cmd="$cmd --input_dir=$input_dir"
    fi
    if [ -n "$output_dir" ]; then
        cmd="$cmd --output_dir=$output_dir"
    fi
    cmd="$cmd &"
    eval $cmd
done

wait

echo "done"
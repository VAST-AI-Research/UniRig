# generate skin

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
force_override="true"
faces_target_count=50000
skin_task="configs/task/quick_inference_unirig_skin.yaml"
seed=12345
npz_dir="tmp"
data_name="raw_data.npz"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) config="$2"; shift ;;
        --require_suffix) require_suffix="$2"; shift ;;
        --num_runs) num_runs="$2"; shift ;;
        --force_override) force_override="$2"; shift ;;
        --faces_target_count) faces_target_count="$2"; shift ;;
        --skin_task) skin_task="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --input) input="$2"; shift ;;
        --input_dir) input_dir="$2"; shift ;;
        --output_dir) output_dir="$2"; shift ;;
        --output) output="$2"; shift ;;
        --data_name) data_name="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# === Fix Windows backslash paths ===
if [ -n "$input" ]; then input=$(echo "$input" | sed 's|\\|/|g'); fi
if [ -n "$output" ]; then output=$(echo "$output" | sed 's|\\|/|g'); fi
if [ -n "$input_dir" ]; then input_dir=$(echo "$input_dir" | sed 's|\\|/|g'); fi
if [ -n "$output_dir" ]; then output_dir=$(echo "$output_dir" | sed 's|\\|/|g'); fi
# ===================================

# 1. extract mesh
cmd=" \
    bash ./launch/inference/extract.sh \
    --config $config \
    --require_suffix $require_suffix \
    --force_override $force_override \
    --num_runs $num_runs \
    --faces_target_count $faces_target_count \
"
if [ -n "$input" ]; then
    cmd="$cmd --input $input"
fi
if [ -n "$input_dir" ]; then
    cmd="$cmd --input_dir $input_dir"
fi
if [ -n "$npz_dir" ]; then
    cmd="$cmd --output_dir $npz_dir"
fi

cmd="$cmd &"
eval $cmd

wait

# 2. inference skin
cmd="\
    python run.py \
    --task=$skin_task \
    --seed=$seed \
"
if [ -n "$input" ]; then
    cmd="$cmd --input=$input"
fi
if [ -n "$input_dir" ]; then
    cmd="$cmd --input_dir=$input_dir"
fi
if [ -n "$output" ]; then
    cmd="$cmd --output=$output"
fi
if [ -n "$output_dir" ]; then
    cmd="$cmd --output_dir=$output_dir"
fi
if [ -n "$npz_dir" ]; then
    cmd="$cmd --npz_dir=$npz_dir"
fi
if [ -n "$data_name" ]; then
    cmd="$cmd --data_name=$data_name"
fi

eval $cmd

wait

echo "done"
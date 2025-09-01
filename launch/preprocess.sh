#!/bin/bash

config=""
require_suffix="fbx,FBX,dae,glb,gltf,vrm"
num_runs=1
force_override="true"
faces_target_count=50000

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) config="$2"; shift ;;
        --require_suffix) require_suffix="$2"; shift ;;
        --num_runs) num_runs="$2"; shift ;;
        --force_override) force_override="$2"; shift ;;
        --faces_target_count) faces_target_count="$2"; shift ;;
        --time) time="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# validate required arguments
if [ -z "$config" ]; then
  echo "Error: --config is required"
  exit 1
fi

# set the time for all processes to use
time=$(date "+%Y_%m_%d_%H_%M_%S")

for (( i=0; i<num_runs; i++ ))
do
    python3.11 -m src.data.extract "--config=$config" "--require_suffix=$require_suffix" "--force_override=$force_override" "--num_runs=$num_runs" "--id=$i" "--time=$time" "--faces_target_count=$faces_target_count" &
done

wait

echo "All tasks completed."
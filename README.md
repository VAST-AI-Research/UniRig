# UniRig: One Model to Rig Them All

<div align="center">

[![Project Page](https://img.shields.io/badge/🏠-Project%20Page-blue.svg)](https://zjp-shadow.github.io/works/UniRig/)
[![Paper](https://img.shields.io/badge/📑-Paper-green.svg)](https://arxiv.org/abs/2504.12451)
[![Model](https://img.shields.io/badge/🤗-Model-yellow.svg)](https://huggingface.co/VAST-AI/UniRig)

</div>

![teaser](assets/doc/unirig_teaser.png)

This repository contains the official implementation for the **SIGGRAPH'25 (TOG) UniRig** framework, a unified solution for automatic 3D model rigging, developed by Tsinghua University and [Tripo](https://www.tripo3d.ai).

**Paper:** [One Model to Rig Them All: Diverse Skeleton Rigging with UniRig](https://arxiv.org/abs/2504.12451)

## Overview

Rigging 3D models – creating a skeleton and assigning skinning weights – is a crucial but often complex and time-consuming step in 3D animation. UniRig tackles this challenge by introducing a novel, unified framework leveraging large autoregressive models to automate the process for a diverse range of 3D assets.

Combining UniRig with keyframe animation produces these following results:

| ![devil](assets/doc/devil.gif) | ![dragon](assets/doc/dragon.gif) | ![rabbit](assets/doc/rabbit.gif) |
|:-----------------------------:|:-------------------------------:|:-------------------------------:|

The full UniRig system consists of two main stages:
1.  **Skeleton Prediction:** An GPT-like transformer autoregressively predicts a topologically valid skeleton hierarchy using a novel **Skeleton Tree Tokenization** scheme.
2.  **Skinning Weight & Attribute Prediction:** A **Bone-Point Cross Attention** mechanism predicts per-vertex skinning weights and relevant bone attributes (e.g., for physics simulation) based on the predicted skeleton and input mesh geometry.

This repository provides the code implementation for the entire framework vision, with components being released progressively.

## Key Features (Full UniRig Framework)

*   **Unified Model:** Aims to handle diverse model categories (humans, animals, objects) with a single framework.
*   **Automated Skeleton Generation:** Predicts topologically valid skeleton structures. **(✅ Available in current release)**
*   **Automated Skinning Prediction:** Predicts per-vertex skinning weights. **(✅ Available in current release)**
*   **Bone Attribute Prediction:** Predicts attributes like stiffness for physics-based secondary motion. **(⏳ Coming Soon)**
*   **High Accuracy & Robustness:** Achieves state-of-the-art results on challenging datasets (as shown in the paper with Rig-XL/VRoid training).
*   **Efficient Tokenization:** Uses Skeleton Tree Tokenization for compact representation and efficient processing.
*   **Human-in-the-Loop Ready:** Designed to potentially support iterative refinement workflows.

## 🚨 Current Release Status & Roadmap 🚨

We are open-sourcing UniRig progressively. Please note the current status:

**Available Now (Initial Release):**
*   ✅ **Code:** Implementation for skeleton and skinning prediction.
*   ✅ **Model:** Skeleton & Skinning Prediction checkpoint trained on [**Articulation-XL2.0**](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0). Available on [Hugging Face](https://huggingface.co/VAST-AI/UniRig).
*   ✅ **Dataset:** Release of the **Rig-XL** and **VRoid** datasets used in the paper. We also filtered out 31 broken models in the training dataset which do not affect the performance of the final model.

**Planned Future Releases:**
*   ⏳ Full UniRig model checkpoints (Skeleton + Skinning) trained on Rig-XL/VRoid, replicating the paper's main results.
*   ⏳ Training code.

We appreciate your patience as we prepare these components for release. Follow [VAST-AI-Research](https://github.com/orgs/VAST-AI-Research) announcements for updates!

## Installation

1.  **Prerequisites:**
    *   Python 3.11
    *   PyTorch (tested with version >=2.3.1)

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/VAST-AI-Research/UniRig
    cd UniRig
    ```

3.  **Set up a virtual environment (recommended):**
    ```bash
    conda create -n UniRig python=3.11
    conda activate UniRig
    ```

4.  **Install dependencies:**
    ```bash
    python -m pip install torch torchvision
    python -m pip install -r requirements.txt
    python -m pip install spconv-{you-cuda-version}
    python -m pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-{your-torch-version}+{your-cuda-version}.html --no-cache-dir
    python -m pip install numpy==1.26.4
    ```

5.  **Download Model Checkpoint:**
    The currently available skeleton prediction model checkpoint is hosted on Hugging Face and will typically be downloaded automatically by the provided scripts/functions.

6.  **(Optional, for importing/exporting .vrm) Install the blender addon:**
    The blender addon is modifed from [VRM-Addon-for-Blender](https://github.com/saturday06/VRM-Addon-for-Blender).

    Make sure you are in the root directory of the project, then:
    ```bash
    python -c "import bpy, os; bpy.ops.preferences.addon_install(filepath=os.path.abspath('blender/add-on-vrm-v2.20.77_modified.zip'))"
    ```

## RigXL Dataset

[processed data link](https://huggingface.co/VAST-AI/UniRig/tree/main/data/rigxl)

Notice that aside from vroid, all models are selected from [Objaverse](https://huggingface.co/datasets/allenai/objaverse-xl). Just download `mapping.json` if you already have Objaverse dataset (or need to download from web).

The json contains all ids of the models with `type` indicating their category and `url` specifying where to download. `url` is the same with `fileIdentifier` in Objaverse.

Training/validation split is put in `datalist` folder.

📝 **Note**:  
All floating-point values are stored in **`float16`** format for compression.

### Visualize Data

Put the dataset in `dataset_clean`, go back to root, and run the command to export FBX model:

```python
from src.data.raw_data import RawData
raw_data = RawData.load("dataset_clean/rigxl/12345/raw_data.npz")
raw_data.export_fbx("res.fbx")
```

<details>
<summary><strong>📁 Dataset Format</strong> (click to expand)</summary>

### 🔑 Keys of Data

All models are converted into world space.

- **`vertices`**:  
  Position of the vertices of the mesh, shape `(N, 3)`.

- **`vertex_normals`**:  
  Normals of the vertices, processed by `Trimesh`, shape `(N, 3)`.

- **`faces`**:  
  Indices of mesh faces (triangles), starting from 0, shape `(F, 3)`.

- **`face_normals`**:  
  Normals of the faces, shape `(F, 3)`.

- **`joints`**:  
  Positions of the armature joints, shape `(J, 3)`.

- **`skin`**:  
  Skinning weights for each vertex, shape `(N, J)`.

- **`parents`**:  
  Parent index of each joint, where `parents[0]` is always `None` (root), shape `(J)`.

- **`names`**:  
  Name of each joint.

- **`matrix_local`**:  
  The local axis of each bone; aligned to Y-up axis, consistent with Blender.

</details>

## Usage

### Skeleton Prediction

Generate a skeleton for your 3D model using our pre-trained model. The process automatically analyzes the geometry and predicts an appropriate skeletal structure.

```bash
# Process a single file
bash launch/inference/generate_skeleton.sh --input examples/giraffe.glb --output results/giraffe_skeleton.fbx

# Process multiple files in a directory
bash launch/inference/generate_skeleton.sh --input_dir <your_input_directory> --output_dir <your_output_directory>

# Try different skeleton variations by changing the random seed
bash launch/inference/generate_skeleton.sh --input examples/giraffe.glb --output results/giraffe_skeleton.fbx --seed 42
```

Supported input formats: `.obj`, `.fbx`, `.glb`, and `.vrm`

### Skinning Weight Prediction
```bash
# Skin a single file
bash launch/inference/generate_skin.sh --input examples/skeleton/giraffe.fbx --output results/giraffe_skin.fbx

# Process multiple files in a directory
bash launch/inference/generate_skin.sh --input_dir <your_input_directory> --output_dir <your_output_directory>
```

Note that the command above uses an **edited-version** from skeleton phase. The results may degrade significantly if the skeleton is inaccurate — for example, if tail bones or wing bones are missing. Therefore, it is recommended to refine the skeleton before performing skinning in order to achieve better results.

### Merge the Predicted Results

Combine the predicted skeleton with your original 3D model to create a fully rigged asset:

```bash
# Merge skeleton from skeleton prediction
bash launch/inference/merge.sh --source results/giraffe_skeleton.fbx --target examples/giraffe.glb --output results/giraffe_rigged.glb

# Or merge skin from skin prediction
bash launch/inference/merge.sh --source results/giraffe_skin.fbx --target examples/giraffe.glb --output results/giraffe_rigged.glb
```

### Rignet Dataset Validation

Validate the metrics mentioned in the paper. This is for academic usage.

First, Download the processed dataset from [Hugging Face](https://huggingface.co/VAST-AI/UniRig/blob/main/data/rignet/processed.zip) and extract it to the `dataset_clean`.

Then run the following command:

```bash
python run.py --task=configs/task/validate_rignet.yaml
```

To export skeleton & mesh, set `record_res` to `True` in the config file `configs/system/ar_validate_rignet.yaml`.

## Models

Available models are hosted on the: https://huggingface.co/VAST-AI/UniRig

## System Requirements

- CUDA-enabled GPU with at least 8GB VRAM

## Citation

```
@article{zhang2025unirig,
  title={One Model to Rig Them All: Diverse Skeleton Rigging with UniRig},
  author={Zhang, Jia-Peng and Pu, Cheng-Feng and Guo, Meng-Hao and Cao, Yan-Pei and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2504.12451},
  year={2025}
}
```

## Acknowledgements

We would like to thank the following open-source projects and research works:

- [OPT](https://huggingface.co/facebook/opt-350m) for model architecture
- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) for 3D shape representation
- [SAMPart3D](https://github.com/Pointcept/SAMPart3D) and [Michelangelo](https://github.com/NeuralCarver/Michelangelo/) for shape encoder implementation
- [Articulation-XL2.0](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0) for a curated dataset

We are grateful to the broader research community for their open exploration and contributions to the field of 3D generation.

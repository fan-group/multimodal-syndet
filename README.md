# 🚀 Universal VLM: Modular Embedding & Classification Pipeline

**Project Status:** Implementation code for [Project Name]. Currently under review for publication; full release forthcoming.

## ℹ️ Overview
This framework provides an end-to-end pipeline for **cross-domain deepfake analysis**. It leverages state-of-the-art Vision-Language Models (VLMs) to extract high-dimensional embeddings, which are then used to train and evaluate linear classifiers for synthetic media detection.

The core of this project is a **unified inference engine** that abstracts the complexities of multiple VLM backbones into a single command-line interface, allowing for rapid benchmarking across different architectural families.

1. Create and activate an environment:

```bash
   conda create --name universal_vlm python=3.8
   conda activate universal_vlm
```

   Or with venv:

```bash
   python3.8 -m venv universal_vlm
   source universal_vlm/bin/activate
```

2. Install dependencies:

```bash
   pip install -r requirements.txt
```

   The PyTorch wheels are built for **CUDA 12.1**, so the target machine needs an NVIDIA driver supporting CUDA ≥ 12.1 (check with `nvidia-smi`).



## 🏗️ Architecture
1. **Feature Extraction:** Maps raw images to a unified latent space using the specified VLM.
2. **Linear Probing:** Trains a linear classification layer on the training set embeddings.
3. **Evaluation:** Benchmarks the classifier on the test set to measure cross-domain generalization.

## 🛠️ Usage
Execute cross-domain training and inference by specifying your dataset paths and selecting a model backbone:

```
python cross_domain.py \
  --real_train_path images/mmcelebahq_train \
  --real_test_path images/mmcelebahq_test \
  --fake_train_path images/DDIM_train \
  --fake_test_path images/DDIM_test \
  --model imagebind \
  --batch_size 2
```

## Supported Model Arguments:

- clip-vit-base-patch32

- clip-vit-base-patch16

- clip-vit-base-large14

- imagebind

- flamingo9b

## 📚 References & Credits

This framework is built upon and utilizes the following research and repositories:

### Core Architectures & Pipelines
* **[Multimodal-Probes](https://github.com/peterhan91/Multimodal-Probes)** 
* **[LVLM-DFD](https://github.com/botianzhe/LVLM-DFD)** 
* **[UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)** 

### Dataset Sources
* **[DiffFace](https://github.com/Rapisurazurite/DiffFace)** – Source for Diffusion-based synthetic imagery (DDIM).

# 🚀 Universal VLM: Modular Embedding & Classification Pipeline

**Project Status:** Implementation code for [Project Name]. Currently under review for publication; full release forthcoming.

## ℹ️ Overview
This framework provides an end-to-end pipeline for **cross-domain deepfake analysis**. It leverages state-of-the-art Vision-Language Models (VLMs) to extract high-dimensional embeddings, which are then used to train and evaluate linear classifiers for synthetic media detection.
The purpose of this project is to probe the **separability of the vision backbones** of state-of-the-art Vision-Language Models for deepfake detection: by training a simple linear classifier on the extracted embeddings, it measures how well each backbone's representation space alone separates real from synthetic media.

## ⚙️ Installation
This project requires **Python 3.8**.
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

3. **(ImageBind only)** Download the pretrained checkpoint at **[imagebind_checkpoint](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) and place it at `checkpoints/ckpt.pth`:

   This step is only required if you intend to run the `imagebind` backbone. The CLIP and IDEFICS backbones do not need it.

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

- clip-vit-base-patch32 **[CLIP](https://arxiv.org/abs/2103.00020)**

- clip-vit-base-patch16 

- clip-vit-large-patch14 

- imagebind **[ImageBind](https://arxiv.org/abs/2305.05665)** 

- idefics9b **[OBELICS](https://arxiv.org/abs/2306.16527)**

## 📚 References & Credits

This framework is built upon and utilizes the following research and repositories:

### Inspired By 
* **[Multimodal-Probes](https://github.com/peterhan91/Multimodal-Probes)** 
* **[UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)**
* * **[LVLM-DFD](https://github.com/botianzhe/LVLM-DFD)** 

### Dataset Sources
* **[DiffFace](https://github.com/Rapisurazurite/DiffFace)** – Source for Diffusion-based synthetic imagery (DDIM).

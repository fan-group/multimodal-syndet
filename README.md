# 🚀 Multimodal-SynDet: Modular Embedding & Classification Pipeline

Implementation code for **Pretrained Image Encoders of Vision-Language and Multimodal Models: An Evaluation on Synthetic Facial Image Detection**

## ℹ️ Overview
This project leverages image encoders of state-of-the-art Vision-Language (VLM) and Multimodal (MM) Models for accurate, efficient synthetic facial image detection.  A simple linear classifier is trained on sample embeddings extracted from the pretrained image encoder.  


## ⚙️ Installation
This project requires **Python 3.8**.
1. Create and activate an environment:

```bash
   conda create --name multimodal-sydnet python=3.8
   conda activate multimodal-sydnet
```

   Or with venv:

```bash
   python3.8 -m venv multimodal-sydnet
   source multimodal-sydnet/bin/activate
```

2. Install dependencies:

```bash
   pip install -r requirements.txt
```

   The PyTorch wheels are built for **CUDA 12.1**, so the target machine needs an NVIDIA driver supporting CUDA ≥ 12.1 (check with `nvidia-smi`).

3. (`imagebind` only) Download the pretrained checkpoint at [ImageBind Repository](https://github.com/facebookresearch/imagebind) and place it at `checkpoints/ckpt.pth`.   This step is only required if you intend to run the `imagebind` option. 

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
* **[LVLM-DFD](https://github.com/botianzhe/LVLM-DFD)** 

### Dataset Sources
* **[DiffFace](https://github.com/Rapisurazurite/DiffFace)** – Source for Diffusion-based synthetic imagery (DDIM).

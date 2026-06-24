import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, IdeficsImageProcessor
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from transformers import CLIPModel, CLIPImageProcessor
from model.openllama import OpenLLAMAPEFTModel
from huggingface_hub import hf_hub_download
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clip_embeddings(args):
    def get_paths(directory):
        return [str(p.absolute()) for p in Path(directory).rglob("*") if p.suffix.lower() in IMG_EXTS]

    class ImgDataset(Dataset):
        def __init__(self, paths, labels, processor):
            self.paths = paths
            self.labels = labels
            self.processor = processor
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            return pixel_values, self.paths[idx], self.labels[idx]

    # --- 1. Path & Label Setup ---
    train_paths = get_paths(args.real_train_path) + get_paths(args.fake_train_path)
    train_labels = [0] * len(get_paths(args.real_train_path)) + [1] * len(get_paths(args.fake_train_path))

    test_paths = get_paths(args.real_test_path) + get_paths(args.fake_test_path)
    test_labels = [0] * len(get_paths(args.real_test_path)) + [1] * len(get_paths(args.fake_test_path))

    # --- 2. Model Setup ---
    model = CLIPModel.from_pretrained(f"openai/{args.model}").to(device).eval()
    image_proc = CLIPImageProcessor.from_pretrained(f"openai/{args.model}")
    batch_size = int(args.batch_size)

    # --- 3. Feature Extraction Helper ---
    def get_features(paths, labels, desc):
        ds = ImgDataset(paths, labels, image_proc)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        all_feats = []
        pbar = tqdm(total=len(ds), desc=desc, unit="img", dynamic_ncols=True)
        
        with torch.inference_mode():
            for pixel_values, _, _ in loader:
                pixel_values = pixel_values.to(device, non_blocking=True)
                feats = model.get_image_features(pixel_values=pixel_values)
                feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 Normalize
                all_feats.append(feats.cpu())
                pbar.update(pixel_values.size(0))
        pbar.close()
        return torch.cat(all_feats, dim=0)

    # --- 4. Execution ---

    start_time = time.time()

    train_embeds = get_features(train_paths, train_labels, "Processing Train")
    test_embeds = get_features(test_paths, test_labels, "Processing Test")

    return {
        'train_embeds': train_embeds,
        'train_paths': train_paths,
        'train_labels': torch.tensor(train_labels),
        'test_embeds': test_embeds,
        'test_paths': test_paths,
        'test_labels': torch.tensor(test_labels)
    }, start_time



def imagebind_embeddings(args):
    
    
    checkpoint_path = hf_hub_download(
        repo_id="jrobe187/LVLMDFD-checkpoint",
        filename="imagebind_ckpt.pth"
    )



    # 1. Model Initialization (Done once)
    model_args = {
        'model': 'openllama_peft',
        'ckpt_path': 'checkpoint/ckpt.pth',
        'max_tgt_len': 128,
        'lora_r': 32,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
    }
    
    model = OpenLLAMAPEFTModel(**model_args)
    delta_ckpt = torch.load(model_args['ckpt_path'], map_location=torch.device('cpu'))
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.cuda().eval()

    def get_paths_and_labels(real_p, fake_p):
        r_path = Path(real_p)
        f_path = Path(fake_p)
        exts = [".jpg", ".jpeg", ".png", ".webp"]
        
        rp = [p for p in r_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
        fp = [p for p in f_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
        
        paths = rp + fp
        labels = [0] * len(rp) + [1] * len(fp)
        return paths, labels

    def extract_features(image_paths, desc):
        all_embeddings = []
        # Using range with batch_size as you did originally
        for idx in range(0, len(image_paths), args.batch_size):
            batch = image_paths[idx : idx + args.batch_size]
            full_paths = [str(p.absolute()) for p in batch]

            with torch.no_grad():
                # Extract embeddings
                embeds = model.encode_image_for_web_demo(image_paths=full_paths, embeddings=True)
                all_embeddings.append(embeds.detach().cpu())
                
                # Cleanup to prevent OOM
                torch.cuda.empty_cache()
                
        return torch.cat(all_embeddings, dim=0)

    # --- 2. Process Train Set ---
    start_time = time.time()

    train_paths, train_labels = get_paths_and_labels(args.real_train_path, args.fake_train_path)
    print(f"Extracting {len(train_paths)} training embeddings...")
    train_embeds = extract_features(train_paths, "Train")

    # --- 3. Process Test Set ---
    test_paths, test_labels = get_paths_and_labels(args.real_test_path, args.fake_test_path)
    print(f"Extracting {len(test_paths)} test embeddings...")
    test_embeds = extract_features(test_paths, "Test")

    # --- 4. Package Results ---
    return {
        'train_embeds': train_embeds,
        'train_paths': [str(p) for p in train_paths],
        'train_labels': torch.tensor(train_labels),
        'test_embeds': test_embeds,
        'test_paths': [str(p) for p in test_paths],
        'test_labels': torch.tensor(test_labels)
    }, start_time



def flamingo_embeddings(args):
    # --- 1. Model Setup ---
    model_name = "HuggingFaceM4/idefics-9b"
    model = IdeficsForVisionText2Text.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    ).eval()
    #processor = AutoProcessor.from_pretrained(model_name)
    image_processor = IdeficsImageProcessor.from_pretrained(model_name)

    def get_paths_and_labels(real_dir, fake_dir):
        exts = ['.jpg', '.png', '.jpeg', '.webp']
        rp = [p for p in Path(real_dir).iterdir() if p.suffix.lower() in exts]
        fp = [p for p in Path(fake_dir).iterdir() if p.suffix.lower() in exts]
        
        paths = rp + fp
        labels = [0] * len(rp) + [1] * len(fp)
        return paths, labels

    def extract_features(all_paths, desc):
        all_embeds = []
        batch_size = int(args.batch_size)

        for i in tqdm(range(0, len(all_paths), batch_size), desc=desc):
            batch_paths = all_paths[i : i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]

            # 1. Process images - this is returning a raw Tensor per your debug
            inputs = image_processor(images=imgs, return_tensors="pt")
            
            # 2. Since 'inputs' IS the tensor, move it directly to device/dtype
            pixel_values = inputs.to(model.device, dtype=torch.bfloat16)

            with torch.inference_mode():
                if pixel_values.ndim == 5:
                    pixel_values = pixel_values.squeeze(1)

                vision_out = model.model.vision_model(pixel_values, return_dict=True)
                
                image_tokens = model.model.perceiver_resampler(vision_out.last_hidden_state)

                batch_embeds = image_tokens.mean(dim=1).detach().cpu().float()
                all_embeds.append(batch_embeds)
            torch.cuda.empty_cache()

        return torch.cat(all_embeds, dim=0)

    # --- 3. Execution for Train and Test ---
    start_time = time.time()

    train_paths, train_labels = get_paths_and_labels(args.real_train_path, args.fake_train_path)
    train_embeds = extract_features(train_paths, "Flamingo Train")

    test_paths, test_labels = get_paths_and_labels(args.real_test_path, args.fake_test_path)
    test_embeds = extract_features(test_paths, "Flamingo Test")

    # --- 4. Result Aggregation ---
    return {
        'train_embeds': train_embeds,
        'train_paths': [str(p) for p in train_paths],
        'train_labels': torch.tensor(train_labels),
        'test_embeds': test_embeds,
        'test_paths': [str(p) for p in test_paths],
        'test_labels': torch.tensor(test_labels)
    }, start_time

import types
import sys
import torchvision.transforms.functional as _F

_shim = types.ModuleType("torchvision.transforms.functional_tensor")
_shim.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _shim

import argparse
from embedding_generation import *
from sklearn.model_selection import train_test_split
import os
import time
import copy
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False



def set_seed(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, num_classes):
    """Run eval and return metrics dict, can be called at each epoch."""
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0

    # For per-class stats
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    # For acc & AUC
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for emb, lab in loader:
            emb = emb.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)

            logits = model(emb)               # (B, C)
            loss = criterion(logits, lab)

            total_loss += loss.item() * emb.size(0)

            preds = logits.argmax(dim=1)      # (B,)
            correct_batch = (preds == lab).sum().item()
            correct += correct_batch
            total += lab.size(0)

            # per-class stats
            for c in range(num_classes):
                mask = (lab == c)
                class_total[c] += mask.sum().item()
                class_correct[c] += (preds[mask] == lab[mask]).sum().item()

            all_logits.append(logits.cpu())
            all_labels.append(lab.cpu())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    # Concatenate for acc and AUC
    logits_all = torch.cat(all_logits, dim=0)      # (N, C)
    labels_all = torch.cat(all_labels, dim=0)      # (N,)


    # ---- Macro AUC (OVR) ----
    auc_macro_score = float("nan")
    if _HAS_SKLEARN:
        probs_all = torch.softmax(logits_all, dim=1).numpy()
        y_true = labels_all.numpy()

        try:
            if num_classes == 2:
                auc_macro_score = roc_auc_score(y_true, probs_all[:, 1])
            else:
                auc_macro_score = roc_auc_score(
                    y_true,
                    probs_all,
                    multi_class="ovr",
                    average="macro"
                )
        except Exception as e:
            print(f"[warn] AUC computation failed: {e}")
            auc_macro_score = float("nan")
    else:
        print("[warn] sklearn not installed, AUC will be NaN.")

    metrics = {
        "avg_loss": avg_loss,
        "acc": acc,
        "auc_macro": auc_macro_score,
        "class_correct": class_correct,
        "class_total": class_total,
        "total": total,
        "correct": correct,
    }
    return metrics



def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot Deepfake Detection Training Script")
    parser.add_argument('--real_train_path', type=str, 
                        help='Path of the folder containing real train images')
    parser.add_argument('--real_test_path', type=str,
                        help='Path of folder containing real test images')
    parser.add_argument('--fake_train_path', type=str, 
                        help='Name of the folder containing fake train images')
    parser.add_argument('--fake_test_path', type=str,
                        help='Path to folder containing fake test images')
    #parser.add_argument('--train_frac', type=float, default=0.8,
    #                    help='Proportion of train samples')
    #parser.add_argument('--val_frac', type=float, default=0.0, 
    #                    help='Proportion of val samples')
    #parser.add_argument('--test_frac', type=int, default=0.2,
    #                    help='Proportion of test samples')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for training')
    parser.add_argument('--model', type=str, choices=['clip-vit-base-patch32', 'clip-vit-base-patch16', 'clip-vit-large-patch14', 'imagebind', 'idefics9b'],
                        help='Foundation model to use for embedding generation')
    parser.add_argument("--logdir", type=str, default="./logs_linear_head",
                        help="Directory for TensorBoard logs and checkpoints")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

#TODO add early stop metric as argument
    
def main():
    args = parse_args()
    args.logdir = os.path.join(args.logdir, f"{args.model}_seed_{args.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #TODO add seed logic
    

    if args.model in ['clip-vit-base-patch32', 'clip-vit-base-patch16', 'clip-vit-large-patch14']:
        embeds, embed_time = clip_embeddings(args)
    elif args.model == 'imagebind':
        embeds, embed_time = imagebind_embeddings(args)
    elif args.model == 'idefics9b':
        embeds, embed_time = idefics_embeddings(args)
    else:
        print("Input a correct model type")
        sys.exit(1)
    

    X_train = embeds['train_embeds'].float().numpy() 
    y_train = embeds['train_labels'].numpy()


    X_test = embeds['test_embeds'].float().numpy()
    y_test = embeds['test_labels'].numpy()


    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=args.test_frac, random_state=42, stratify=y
    #)


    print(f"Final Sets: Train={len(X_train)}, Test={len(X_test)}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(), 
        torch.from_numpy(y_train).long()
    )

    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

     # Infer dim and num_classes
    dim = train_ds[0][0].shape[-1]
    num_classes = len(set(y_train))

    model = nn.Linear(dim, num_classes).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.1, patience=2, verbose=True
    )



    os.makedirs(args.logdir, exist_ok=True)
    train_writer = SummaryWriter(os.path.join(args.logdir, "train"))

    print(f"Emb dim: {dim} | Train samples: {len(train_ds)} | Num classes: {num_classes}\n")

    global_step = 0
    best_acc = -1.0
    best_auc = -1.0


    print("TRAINING".center(30, "="))
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct_acc = 0
        total = 0

        acc = 0


        # For metrics across the entire epoch
        logits_list = []
        labels_list = []

        # Per-class stats
        class_correct = np.zeros(num_classes, dtype=np.int64)
        class_total = np.zeros(num_classes, dtype=np.int64)

        for emb, lab in train_loader:
            global_step += 1
            emb = emb.to(device, non_blocking=True)   # (B, D)
            lab = lab.to(device, non_blocking=True)   # (B,) long

            logits = model(emb)                       # (B, C)
            loss = crit(logits, lab)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item() * emb.size(0)

            # For metrics
            preds = logits.argmax(dim=1)
            correct_acc += (preds == lab).sum().item()
            total += lab.size(0)

            # Store logits / labels for acc + AUC
            logits_list.append(logits.detach().cpu())
            labels_cpu = lab.detach().cpu()
            labels_list.append(labels_cpu)

            # Per-class stats for this batch
            preds_cpu = preds.detach().cpu()
            for c in range(num_classes):
                mask = (labels_cpu == c)
                if mask.any():
                    class_total[c] += mask.sum().item()
                    class_correct[c] += (preds_cpu[mask] == labels_cpu[mask]).sum().item()

            if global_step % 50 == 0:
                #it_time = (time.time() - start_time) / max(global_step, 1)
         #       print(f"[ep {epoch} | step {global_step}] "
         #             f"loss={loss.item():.4f} | iter_time={it_time:.4f}s")
                train_writer.add_scalar("loss/step", loss.item(), global_step)

        # ---- Epoch-level metrics ----
        epoch_loss = running_loss / len(train_ds)
        acc = correct_acc / total if total > 0 else 0.0

        # Concatenate logits/labels for acc and AUC
        logits_all = torch.cat(logits_list, dim=0)      # (N, C)
        labels_all = torch.cat(labels_list, dim=0)      # (N,)


        auc_macro_score = float("nan")
        if _HAS_SKLEARN:
            probs_all = torch.softmax(logits_all, dim=1).numpy()
            y_true = labels_all.numpy()
            try:
                auc_macro_score = roc_auc_score(y_true, probs_all[:, 1])
            except Exception as e:
                print(f"[warn] AUC computation failed: {e}")
        else:
            print("[warn] sklearn not installed, AUC will be NaN.")


        model.eval()
        val_running_loss = 0.0
        val_total = 0

        val_logits_list = []
        val_labels_list = []

        with torch.no_grad():
            for val_emb, val_lab in train_loader:
                val_emb = val_emb.to(device, non_blocking=True)
                val_lab = val_lab.to(device, non_blocking=True)

                # Forward pass
                val_logits = model(val_emb)
                v_loss = crit(val_logits, val_lab)

                # Accumulate loss and basic stats
                val_running_loss += v_loss.item() * val_emb.size(0)
                val_total += val_lab.size(0)

                # Collect for acc and AUC
                val_logits_list.append(val_logits.cpu())
                val_labels_list.append(val_lab.cpu())

        # --- Final Validation Metric Calculations ---
        val_logits_all = torch.cat(val_logits_list, dim=0)
        val_labels_all = torch.cat(val_labels_list, dim=0)

        # 1. Avg Loss
        val_loss_avg = val_running_loss / val_total if val_total > 0 else 0.0

        # 2. Accuracy
        val_preds = val_logits_all.argmax(dim=1)
        val_acc = (val_preds == val_labels_all).float().mean().item() if val_total > 0 else 0.0

        # 3. Macro AUC
        val_auc_macro = float("nan")
        if _HAS_SKLEARN and val_total > 0:
            val_probs = torch.softmax(val_logits_all, dim=1).numpy()
            val_y_true = val_labels_all.numpy()
            try:
                if num_classes == 2:
                    val_auc_macro = roc_auc_score(val_y_true, val_probs[:, 1])
                else:
                    val_auc_macro = roc_auc_score(val_y_true, val_probs, multi_class="ovr", average="macro")
            except Exception as e:
                print(f"[warn] Val AUC failed: {e}")

        # Print Results
        print(f"\n[EPOCH {epoch}] Train Metrics:\nLoss: {val_loss_avg:.4f} \nAcc1: {val_acc:.4f} \nAUC: {val_auc_macro:.4f}\n")


        # Step LR scheduler on loss
        sched.step(val_loss_avg)

        #sched.step(val_acc)
        if val_acc > best_acc:
            # --- NEW BEST MODEL HEADER ---
            print(f"\n{'#'*15} NEW BEST MODEL FOUND {'#'*15}")
            print(f"Old Best: {best_acc:.4f} | New Best: {val_acc:.4f}")
            print(f"Validation AUC: {val_auc_macro:.4f}")
            
            best_acc = val_acc
            best_auc = val_auc_macro
            best_train_auc = auc_macro_score
            epochs_no_improve = 0
            
            # --- SAVING LOGIC ---
            ckpt_path = os.path.join(args.logdir, "model_best.pth")
            print(f"Saving checkpoint to: {ckpt_path}")
            torch.save({
                "model": model.state_dict(),
                "dim": dim,
                "num_classes": num_classes,
            }, ckpt_path)

            best_model = copy.deepcopy(model)
            print(f"{'#'*50}\n")

        else:
            epochs_no_improve += 1
            # --- PATIENCE/NO IMPROVEMENT LOG ---
            print(f"\n{'-'*10} NO IMPROVEMENT {'-'*10}")
            print(f"Current Patience: {epochs_no_improve}/{args.patience}")
            
            if epochs_no_improve >= args.patience:
                # --- EARLY STOPPING TRIGGER ---
                print(f"\n{'!'*15} EARLY STOPPING {'!'*15}")
                print(f"Patience of {args.patience} reached. Stopping training.")
                #print(f"Final Best Accuracy: {best_acc:.4f}")
                print(f"{'!'*46}\n")
                break
    
    #print(f"Training done. Best acc: {best_acc:.4f}, AUC: {best_auc:4f}")
    end_time = time.time()

    duration = end_time - embed_time
    print(f"Total processing time: {duration:.4f} seconds")
    
    model.eval()
    

    print("TEST RESULTS".center(30, "="))
    metrics = evaluate(best_model, test_loader, device, num_classes)
    print(f"\nAUC: {metrics['auc_macro']}")




if __name__ == "__main__":
    main()

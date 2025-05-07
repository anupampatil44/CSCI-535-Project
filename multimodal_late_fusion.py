#!/usr/bin/env python
# multimodal_late_fusion.py
import os, json, random, numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


# ──────────────────────────────── CONFIG ──────────────────────────────── #
ROOT        = Path(__file__).resolve().parent       # current directory
DATA_DIR    = ROOT                                  # modalities live here
LABELS_FILE = ROOT / "ground_truth.json"
SAVE_PATH   = ROOT / "fusion_model.pt"

BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 3e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42
# ──────────────────────────────────────────────────────────────────────── #


# ------------------------------ DATASET --------------------------------- #
class MultiModalSarcasm(Dataset):
    MODALITIES = ["audio", "visual", "text_context", "text_utterance"]

    def __init__(self, split: str):
        self.split = split
        with open(LABELS_FILE) as f:
            self.labels = json.load(f)

        pattern = DATA_DIR / "audio" / split / "*.npy"
        self.files = [Path(p).stem for p in glob(str(pattern))]
        if not self.files:
            raise RuntimeError(f"No files found under {pattern}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        data = {}
        for mod in self.MODALITIES:
            arr = np.load(DATA_DIR / mod / self.split / f"{name}.npy")
            arr = arr.squeeze()             # <── flattens (2868,1) ➜ (2868,)
            data[mod] = torch.from_numpy(arr).float()
        label = torch.tensor(int(self.labels[name])).long()
        return data, label


# ------------------------------ MODEL ----------------------------------- #
class DenseBlock(nn.Module):
    def __init__(self, in_dim, mid_dim=None, out_dim=256, p=0.1):
        super().__init__()
        layers = [nn.Linear(in_dim, mid_dim or out_dim), nn.ReLU(), nn.Dropout(p)]
        if mid_dim is not None:
            layers += [nn.Linear(mid_dim, out_dim), nn.ReLU(), nn.Dropout(p)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_enc = DenseBlock(2868, 1024)
        self.audio_enc = DenseBlock(1540, 512)
        self.text_enc  = DenseBlock(768)

        self.cls_video = nn.Linear(256, 2)
        self.cls_audio = nn.Linear(256, 2)
        self.cls_text  = nn.Linear(256, 2)

        self.meta = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, batch, modality_dropout=0.):
        h_v = self.video_enc(batch["visual"])
        h_a = self.audio_enc(batch["audio"])
        h_c = self.text_enc(batch["text_context"])
        h_u = self.text_enc(batch["text_utterance"])

        p_v = self.cls_video(h_v)
        p_a = self.cls_audio(h_a)
        p_c = self.cls_text(h_c)
        p_u = self.cls_text(h_u)

        if self.training and modality_dropout:
            stacked = torch.stack([p_v, p_a, p_c, p_u], 1)
            mask = (torch.rand_like(stacked[..., :1]) > modality_dropout)
            meta_in = (stacked * mask).view(stacked.size(0), -1)
        else:
            meta_in = torch.cat([p_v, p_a, p_c, p_u], -1)

        return self.meta(meta_in)


# ---------------------------- TRAIN / EVAL ------------------------------ #
def run_epoch(model, loader, crit, optim=None, bar_desc="train"):
    is_train = optim is not None
    model.train() if is_train else model.eval()
    losses, ys, ŷs = [], [], []

    bar = tqdm(loader, desc=bar_desc, leave=False)
    for data, y in bar:
        data = {k: v.to(DEVICE) for k, v in data.items()}
        y = y.to(DEVICE)

        with torch.set_grad_enabled(is_train):
            logits = model(data, modality_dropout=0.1 if is_train else 0)
            loss   = crit(logits, y)

        if is_train:
            optim.zero_grad(); loss.backward(); optim.step()

        losses.append(loss.item())
        ys.append(y.cpu()); ŷs.append(logits.argmax(1).cpu())
        bar.set_postfix(loss=np.mean(losses))

    ys = torch.cat(ys); ŷs = torch.cat(ŷs)
    return np.mean(losses), accuracy_score(ys, ŷs), f1_score(ys, ŷs)


# ------------------------------ MAIN ------------------------------------ #
if __name__ == "__main__":
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

    loaders = {
        split: DataLoader(MultiModalSarcasm(split), BATCH_SIZE,
                          shuffle=(split=="train"), num_workers=4)
        for split in ["train", "val", "test"]
    }

    model = LateFusionModel().to(DEVICE)
    crit  = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_f1 = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        tr_loss, tr_acc, tr_f1 = run_epoch(model, loaders["train"], crit, optim, "train")
        vl_loss, vl_acc, vl_f1 = run_epoch(model, loaders["val"],   crit, None,  "val  ")

        print(f"  train  L={tr_loss:.3f} A={tr_acc:.3f} F1={tr_f1:.3f}")
        print(f"  val    L={vl_loss:.3f} A={vl_acc:.3f} F1={vl_f1:.3f}")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  🔥 new best model saved → {SAVE_PATH}")

    # final test
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    _, test_acc, test_f1 = run_epoch(model, loaders["test"], crit, None, "test ")
    print(f"\n🧪  Test  accuracy={test_acc:.3f}  F1={test_f1:.3f}")

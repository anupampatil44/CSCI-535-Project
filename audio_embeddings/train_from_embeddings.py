#!/usr/bin/env python3
"""
Train a sarcasm classifier **from pre-computed 1540-D embeddings**.
Embeddings live in  audio_embeddings/{train,val}_embeddings/*.npy
Labels come from    audio_embeddings/sarcasm_data.json
"""

import json, argparse, pathlib, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ───────────────────────────── dataset ───────────────────────────── #

class EmbeddingDataset(Dataset):
    def __init__(self, emb_dir: pathlib.Path, json_path: pathlib.Path):
        self.emb_dir = emb_dir
        with open(json_path) as f:
            meta = json.load(f)                     # id → dict{…,"sarcasm":bool}
        self.files = []
        for fp in emb_dir.glob("*.npy"):
            sid = fp.stem            # "1_60"
            if sid not in meta:
                continue
            label = 1 if meta[sid]["sarcasm"] else 0
            self.files.append((fp, label))

    def __len__(self):  return len(self.files)

    def __getitem__(self, idx):
        fp, lbl = self.files[idx]
        emb = np.load(fp)                      # (1540,)
        return torch.from_numpy(emb).float(), torch.tensor(lbl)

# ──────────────────────────── model ──────────────────────────────── #

class SarcasmMLP(nn.Module):
    def __init__(self, in_dim=1540, hid=512, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hid, n_cls),
        )
    def forward(self, x): return self.net(x)

# ──────────────────────────── helpers ────────────────────────────── #

def accuracy(pred, gold):  # pred: (B,2) logits
    return (pred.argmax(1) == gold).float().mean().item()

@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    tot_acc, tot, tot_loss = 0, 0, 0.
    ce = nn.CrossEntropyLoss()
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        tot_loss += ce(out, y).item() * x.size(0)
        tot_acc  += (out.argmax(1)==y).sum().item()
        tot      += x.size(0)
    return tot_loss/tot, tot_acc/tot

# ───────────────────────────── main ─────────────────────────────── #

def main(args):
    root = pathlib.Path(args.root)

    train_ds = EmbeddingDataset(root/"train_embeddings", root/"sarcasm_data.json")
    val_ds   = EmbeddingDataset(root/"val_embeddings",   root/"sarcasm_data.json")

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SarcasmMLP().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-4)
    ce     = nn.CrossEntropyLoss()

    best_f1, best_path = 0, root/"sarcasm_model_emb.pth"
    for epoch in range(1, args.epochs+1):
        model.train()
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward(); opt.step()

        val_loss, val_acc = eval_loop(model, val_dl, device)
        print(f"Epoch {epoch:02d}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    torch.save(model.state_dict(), best_path)
    print("✔ saved", best_path)

# ────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root",   default="audio_embeddings",
                   help="folder with train|val_embeddings and sarcasm_data.json")
    p.add_argument("--epochs", type=int, default=10)
    main(p.parse_args())

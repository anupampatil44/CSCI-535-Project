#!/usr/bin/env python3
"""
Generate id,prediction,confidence CSV from *.npy embeddings + trained checkpoint.
"""

import csv, argparse, pathlib, json, numpy as np, torch, torch.nn as nn

class SarcasmMLP(nn.Module):
    def __init__(self, in_dim=1540, hid=512, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid, n_cls))
    def forward(self,x): return self.net(x)

@torch.no_grad()
def main(args):
    emb_dir = pathlib.Path(args.emb_dir)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SarcasmMLP().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    rows = []
    for fp in sorted(emb_dir.glob("*.npy")):
        x = torch.from_numpy(np.load(fp)).unsqueeze(0).to(device)  # (1,1540)
        probs = torch.softmax(model(x),1).squeeze()
        pred  = int(probs.argmax())
        conf  = float(probs.max())
        rows.append((f"{fp.stem}_embedding", pred, conf))

    with open(args.out_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["id","prediction","confidence"])
        for r in rows: w.writerow([r[0], r[1], f"{r[2]:.4f}"])
    print("wrote", args.out_csv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",    default="audio_embeddings/sarcasm_model_emb.pth")
    ap.add_argument("--emb_dir", default="audio_embeddings/test_embeddings")
    ap.add_argument("--out_csv", default="predictions.csv")
    main(ap.parse_args())

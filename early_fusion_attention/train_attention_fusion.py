#!/usr/bin/env python3
import os, json, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────────────────────────────────
def load_embeddings(split_folders, split_name):
    txt = os.path.join(split_folders['text'],  split_name)
    aud = os.path.join(split_folders['audio'], split_name)
    vid = os.path.join(split_folders['video'], split_name)
    e = {}
    for fn in os.listdir(txt):
        if not fn.endswith('.npy'): continue
        sid = fn[:-4]
        t = np.load(os.path.join(txt, fn )).astype(np.float32).ravel()
        a = np.load(os.path.join(aud, fn)).astype(np.float32).ravel()
        v = np.load(os.path.join(vid, fn)).astype(np.float32).ravel()
        e[sid] = (t, a, v)
    return e

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return {sid: float(info['sarcasm']) for sid,info in data.items()}

def prepare_dataset(emb_dict, labels):
    Ts, As, Vs, Ys = [], [], [], []
    for sid,(t,a,v) in emb_dict.items():
        if sid not in labels: continue
        Ts.append(t); As.append(a); Vs.append(v); Ys.append(labels[sid])
    T = torch.tensor(np.stack(Ts), dtype=torch.float32)
    A = torch.tensor(np.stack(As), dtype=torch.float32)
    V = torch.tensor(np.stack(Vs), dtype=torch.float32)
    Y = torch.tensor(Ys, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(T,A,V,Y)

class AttentionFusionModel(nn.Module):
    def __init__(self, dims, proj_dim=128, heads=4, layers=2, mlp_h=64, d=0.3):
        super().__init__()
        t_dim,a_dim,v_dim = dims
        self.tp = nn.Linear(t_dim, proj_dim)
        self.ap = nn.Linear(a_dim, proj_dim)
        self.vp = nn.Linear(v_dim, proj_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=heads,
            dim_feedforward=proj_dim*2,
            dropout=d, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim,    mlp_h),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(mlp_h,       1)
        )

    def forward(self, t,a,v):
        pt,pa,pv = self.tp(t), self.ap(a), self.vp(v)
        x = torch.stack([pt,pa,pv], dim=1)   # (B,3,proj_dim)
        x = self.enc(x)                     # (B,3,proj_dim)
        x = x.mean(dim=1)                   # (B,proj_dim)
        return self.mlp(x)                  # (B,1)

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    return (
      accuracy_score(y_true, y_pred),
      precision_score(y_true, y_pred, zero_division=0),
      recall_score(y_true, y_pred, zero_division=0),
      f1_score(y_true, y_pred, zero_division=0)
    )

def evaluate(model, loader, loss_fn, device):
    model.eval()
    tot_loss, Ys, Ps = 0.0, [], []
    with torch.no_grad():
        for T,A,V,Y in loader:
            T,A,V,Y = [x.to(device) for x in (T,A,V,Y)]
            logits = model(T,A,V)
            tot_loss += loss_fn(logits, Y).item() * T.size(0)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs>=0.5).astype(int)
            Ys.extend(Y.cpu().numpy().flatten())
            Ps.extend(preds)
    n = len(loader.dataset)
    loss = tot_loss/n
    acc, prec, rec, f1 = compute_metrics(Ys, Ps)
    return loss, acc, prec, rec, f1

def main():
    base = os.getcwd()
    folders = {
      'text':  os.path.join(base, 'text_embeddings'),
      'audio': os.path.join(base, 'audio_embeddings'),
      'video': os.path.join(base,
                 'video_embeddings/non-framewise/mediapipe_embeddings/context_with_utterence')
    }
    labels = load_labels(os.path.join(base,'data_splits','sarcasm_data.json'))

    tags   = ['train','val','test']
    splits = [f"{t}_embeddings" for t in tags]

    ds = {}
    for tag,split in zip(tags,splits):
        emb = load_embeddings(folders, split)
        ds[tag] = prepare_dataset(emb, labels)

    # infer dims from one example
    t0,a0,v0,_ = ds['train'][0]
    dims = (t0.shape[0], a0.shape[0], v0.shape[0])

    loaders = {
      tag: DataLoader(ds[tag], batch_size=32, shuffle=(tag=='train'))
      for tag in tags
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AttentionFusionModel(dims).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn= nn.BCEWithLogitsLoss()

    best_val_f1=0.0
    for epoch in range(1,31):
        model.train()
        for T,A,V,Y in loaders['train']:
            T,A,V,Y = [x.to(device) for x in (T,A,V,Y)]
            opt.zero_grad()
            l = loss_fn(model(T,A,V), Y)
            l.backward(); opt.step()

        _,tr_acc, tr_p, tr_r, tr_f = evaluate(model, loaders['train'], loss_fn, device)
        _,va_acc, va_p, va_r, va_f = evaluate(model, loaders['val'],   loss_fn, device)
        print(f"Epoch {epoch:2d}  train F1 {tr_f:.3f}  |  val F1 {va_f:.3f}")

        if va_f > best_val_f1:
            best_val_f1=va_f
            torch.save(model.state_dict(), "best_attention_model.pth")

    # test
    model.load_state_dict(torch.load("best_attention_model.pth"))
    _,te_acc, te_p, te_r, te_f = evaluate(model, loaders['test'], loss_fn, device)
    print(f"\nTest →  Acc {te_acc:.3f}  Prec {te_p:.3f}  Rec {te_r:.3f}  F1 {te_f:.3f}")

if __name__=='__main__':
    main()


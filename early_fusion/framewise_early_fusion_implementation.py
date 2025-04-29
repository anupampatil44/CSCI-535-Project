import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Config ---
batch_size = 256
hidden_dim = 128
num_epochs = 45
learning_rate = 5e-4

# --- Load Labels ---
def load_labels(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    labels = {sample_id: float(info["sarcasm"]) for sample_id, info in data.items()}  # Force float
    return labels

# --- Dataset ---
class FramewiseSarcasmDataset(Dataset):
    def __init__(self, audio_dir, video_dir, text_dir, labels_dict):
        self.sample_ids = sorted([f.replace(".npz", "") for f in os.listdir(audio_dir) if f.endswith(".npz")])
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        audio_npz = np.load(os.path.join(self.audio_dir, f"{sample_id}.npz"))
        audio = np.concatenate([audio_npz['context'], audio_npz['utter']], axis=0)
        video = np.load(os.path.join(self.video_dir, f"{sample_id}.npy")).T
        text = np.load(os.path.join(self.text_dir, f"{sample_id}.npy")).flatten()

        # Debug: Check individually
        if np.isnan(audio).any() or np.isinf(audio).any():
            print(f"❗ Corrupt AUDIO in sample {sample_id}")
            audio = np.zeros_like(audio)

        if np.isnan(video).any() or np.isinf(video).any():
            print(f"❗ Corrupt VIDEO in sample {sample_id}")
            video = np.zeros_like(video)

        if np.isnan(text).any() or np.isinf(text).any():
            print(f"❗ Corrupt TEXT in sample {sample_id}")
            text = np.zeros_like(text)

        fused = np.concatenate([audio, video], axis=1)

        # Normalize after fixing
        fused = (fused - fused.mean()) / (fused.std() + 1e-6)
        text = (text - text.mean()) / (text.std() + 1e-6)

        label = float(self.labels_dict.get(sample_id, 0.0))

        return torch.tensor(fused, dtype=torch.float32), torch.tensor(text, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# --- Collate Function ---
def pad_collate(batch):
    seqs, texts, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    max_len = max(lengths)
    feat_dim = seqs[0].size(1)

    padded_seqs = torch.zeros(len(seqs), max_len, feat_dim)
    for i, s in enumerate(seqs):
        padded_seqs[i, :s.size(0)] = s

    texts = torch.stack(texts)
    labels = torch.stack(labels)

    return padded_seqs, texts, labels, lengths

# --- Model ---
class SequenceBasedSarcasmClassifier(nn.Module):
    def __init__(self, input_dim, text_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)  # BiGRU
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim * 3)  # hidden_dim*2 from BiGRU + hidden_dim from text
        self.dropout = nn.Dropout(0.4)
        self.final_fc = nn.Linear(hidden_dim * 3, 1)

    def forward(self, seq, text, lengths):
        packed_seq = nn.utils.rnn.pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden_seq = self.gru(packed_seq)
        hidden_seq = torch.cat([hidden_seq[0], hidden_seq[1]], dim=1)  # [batch_size, hidden_dim*2]

        text_feat = self.text_proj(text)  # [batch_size, hidden_dim]

        combined = torch.cat([hidden_seq, text_feat], dim=1)
        combined = self.norm(combined)
        combined = self.dropout(combined)

        output = self.final_fc(combined)
        return output.squeeze(1)



# --- Training and Evaluation ---
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    for batch in loader:
        seqs, texts, labels, lengths = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(seqs, texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * seqs.size(0)
    return running_loss / len(loader.dataset)

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    acc = np.mean(y_pred == y_true)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for batch in loader:
            seqs, texts, labels, lengths = [b.to(device) for b in batch]
            logits = model(seqs, texts, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * seqs.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            y_true_all.extend(labels.cpu().numpy().flatten())
            y_pred_all.extend(preds.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(y_true_all, y_pred_all)
    return avg_loss, acc, prec, rec, f1


class EarlyStopping:
    def __init__(self, patience=8, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if ((self.mode == "max" and current_score <= self.best_score) or
            (self.mode == "min" and current_score >= self.best_score)):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0  # reset counter if improvement
        return self.early_stop


# --- Main ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    labels = load_labels("data_splits/sarcasm_data.json")

    train_ds = FramewiseSarcasmDataset("audio_frame_embeddings/train", "video_embeddings/framewise/with_context_fixed/train", "text_embeddings/train_embeddings", labels)
    val_ds = FramewiseSarcasmDataset("audio_frame_embeddings/val", "video_embeddings/framewise/with_context_fixed/val", "text_embeddings/val_embeddings", labels)
    test_ds = FramewiseSarcasmDataset("audio_frame_embeddings/test", "video_embeddings/framewise/with_context_fixed/test", "text_embeddings/test_embeddings", labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    sample_seq, sample_text, _ = train_ds[0]
    model = SequenceBasedSarcasmClassifier(input_dim=sample_seq.size(1), text_dim=sample_text.size(0), hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)


    best_f1 = 0.0

    early_stopper = EarlyStopping(patience=8, mode="max")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_loader, criterion, device)
        train_loss, train_acc, train_prec, train_rec, train_f1 = evaluate_model(model, train_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, F1 {train_f1:.4f} | Val Loss {val_loss:.4f}, F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_framewise_model.pt")

        scheduler.step(val_f1)

        # ✅ Early Stopping check
        if early_stopper(val_f1):
            print(f"Early stopping triggered at epoch {epoch}. Best Val F1 was {best_f1:.4f}")
            break

    print("Training complete.")

    model.load_state_dict(torch.load("best_framewise_model.pt"))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Performance: Loss {test_loss:.4f} | Accuracy {test_acc:.4f} | Precision {test_prec:.4f} | Recall {test_rec:.4f} | F1 {test_f1:.4f}")



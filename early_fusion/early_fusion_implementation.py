import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------
# 1. Helper to load embeddings
# ----------------------------------------
def load_embeddings(split_folder, split_name):
    """
    Loads embeddings for a given split (train, val, test) across text, audio, video modalities.

    Args:
        split_folder (dict): Dictionary with keys {"text", "audio", "video"} and path values.
        split_name (str): One of "train_embeddings", "val_embeddings", or "test_embeddings".

    Returns:
        dict: sample_id -> (text, audio, video embeddings)
    """
    text_path = os.path.join(split_folder['text'], split_name)
    audio_path = os.path.join(split_folder['audio'], split_name)
    video_path = os.path.join(split_folder['video'], split_name)

    embeddings = {}
    for filename in os.listdir(text_path):
        if filename.endswith(".npy"):
            sample_id = filename.replace(".npy", "")
            if "_embedding" in sample_id:
                sample_id = filename.replace("_embedding", "")
            text_embed = np.load(os.path.join(text_path, filename))
            audio_embed = np.load(os.path.join(audio_path, filename))
            video_embed = np.load(os.path.join(video_path, filename))

            text_embed = text_embed.flatten()
            audio_embed = audio_embed.flatten()
            video_embed = video_embed.flatten()

            fused = np.concatenate([text_embed, audio_embed, video_embed])
            embeddings[sample_id] = fused

    return embeddings

# ----------------------------------------
# 2. Load labels
# ----------------------------------------
def load_labels(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    labels = {}
    for sample_id, info in data.items():
        labels[sample_id] = 1 if info["sarcasm"] else 0
    return labels

# ----------------------------------------
# 3. Dataset Preparation
# ----------------------------------------
def prepare_dataset(embeddings_dict, labels_dict):
    X = []
    y = []
    for sample_id, features in embeddings_dict.items():
        if sample_id in labels_dict:
            X.append(features)
            y.append(labels_dict[sample_id])
    X = np.vstack(X)
    y = np.array(y)
    return X, y

# ----------------------------------------
# 4. Define Model
# ----------------------------------------
class SarcasmDetectionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, hidden_dim2=64, dropout_rate=0.5):
        super(SarcasmDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc_out = nn.Linear(hidden_dim2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        out = self.fc_out(x)
        return out

# ----------------------------------------
# 5. Metrics
# ----------------------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    acc  = np.mean(y_pred == y_true)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            y_true_all.extend(y_batch.cpu().numpy().flatten())
            y_pred_all.extend(preds.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(y_true_all, y_pred_all)
    return avg_loss, acc, prec, rec, f1

# ----------------------------------------
# 6. Main Training Script
# ----------------------------------------
def main():
    # Paths
    project_dir = os.getcwd()  # Get the script's directory: os.path.abspath(__file__)
    split_folders = {
        "text": os.path.join(project_dir, "text_embeddings"),
        "audio": os.path.join(project_dir, "audio_embeddings"),
        "video": os.path.join(project_dir, "video_embeddings/libreface_embeddings"),
    }
    labels_json = os.path.join(project_dir, "data_splits", "sarcasm_data.json")

    batch_size = 32
    num_epochs = 40

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    print("Loading embeddings and labels...")
    labels_dict = load_labels(labels_json)

    train_embeds = load_embeddings(split_folders, "train_embeddings")
    val_embeds = load_embeddings(split_folders, "val_embeddings")
    test_embeds = load_embeddings(split_folders, "test_embeddings")

    X_train, y_train = prepare_dataset(train_embeds, labels_dict)
    X_val, y_val = prepare_dataset(val_embeds, labels_dict)
    X_test, y_test = prepare_dataset(test_embeds, labels_dict)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

    # Model
    input_dim = X_train.shape[1]
    model = SarcasmDetectionModel(input_dim)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_f1 = -1.0
    best_model_path = os.path.join(project_dir, "best_sarcasm_model.pth")

    print("Starting training...")
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_loss, train_acc, train_prec, train_rec, train_f1 = evaluate_model(model, train_loader, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train F1 {train_f1:.4f} | Val Loss {val_loss:.4f}, Val F1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            best_epoch = epoch

    print(f"✅ Training complete. Best model at epoch {best_epoch} with Validation F1 {best_val_f1:.4f}")

    # Evaluate on Test Set
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_loader, criterion, device)
    print(f"\nTest Set Performance: Loss {test_loss:.4f}, Accuracy {test_acc:.4f}, Precision {test_prec:.4f}, Recall {test_rec:.4f}, F1 {test_f1:.4f}")

# ----------------------------------------
# Run
# ----------------------------------------
if __name__ == "__main__":
    main()

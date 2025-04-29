import os
import numpy as np
import json
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

base = os.getcwd()
# ---------------- CONFIGURATION ----------------
train_embeddings_folder = base + "/video_embeddings/non-framewise/mediapipe_embeddings/context_with_utterence/train_embeddings"
val_embeddings_folder = base + "/video_embeddings/non-framewise/mediapipe_embeddings/context_with_utterence/val_embeddings"
test_embeddings_folder = base + "/video_embeddings/non-framewise/mediapipe_embeddings/context_with_utterence/test_embeddings"
label_json_path = base + "/data_splits/sarcasm_data.json"
# ------------------------------------------------


def load_embeddings_and_labels(folder_path, label_json_path):
    """
    Loads embeddings and sarcasm labels for a given split.
    """
    with open(label_json_path, 'r') as f:
        label_data = json.load(f)

    X_list = []
    y_list = []
    sample_ids = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            sample_id = file_name.replace(".npy", "")
            embed_path = os.path.join(folder_path, file_name)

            if sample_id in label_data:
                embedding = np.load(embed_path)
                if embedding.shape[-1] == 1:
                    embedding = embedding.flatten()  # (features, 1) -> (features,)
                X_list.append(embedding)
                label = 1 if label_data[sample_id]["sarcasm"] else 0
                y_list.append(label)
                sample_ids.append(sample_id)
            else:
                print(f"⚠️ Warning: {sample_id} not found in label JSON.")

    X = np.vstack(X_list)
    y = np.array(y_list)

    return X, y


def train_model(X_train, y_train):
    """
    Train an XGBoost model on provided training data.
    """
    print("Training XGB Model...")
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    optimization_dict = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
    }

    model = GridSearchCV(xgb_model, optimization_dict, scoring='accuracy', verbose=1)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_eval, y_eval, split_name="Validation"):
    """
    Evaluate the model and print metrics.
    """
    y_pred = model.predict(X_eval)

    acc = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)

    print(f"\n📈 {split_name} Evaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"\nClassification Report ({split_name}):")
    print(classification_report(y_eval, y_pred))



print("Loading Data...")

X_train, y_train = load_embeddings_and_labels(train_embeddings_folder, label_json_path)
X_val, y_val = load_embeddings_and_labels(val_embeddings_folder, label_json_path)
X_test, y_test = load_embeddings_and_labels(test_embeddings_folder, label_json_path)

print(f"✅ Loaded: {X_train.shape[0]} train samples, {X_val.shape[0]} val samples, {X_test.shape[0]} test samples.")

# Train
model = train_model(X_train, y_train)

# Evaluate on validation
evaluate_model(model, X_val, y_val, split_name="Validation")

# Evaluate on test
evaluate_model(model, X_test, y_test, split_name="Test")

# Save model if needed
model.best_estimator_.save_model("xgb_sarcasm_model_context.json")
print("\n✅ Model saved to xgb_sarcasm_model.json")
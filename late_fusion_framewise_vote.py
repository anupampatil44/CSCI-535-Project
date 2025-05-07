import csv
import json
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load visual predictions
def load_visual_predictions(path):
    visual_preds = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            visual_preds[row['id']] = {
                "prediction": int(row['prediction']),
                "confidence": float(row['confidence'])
            }
    return visual_preds

# Load audio predictions and labels
def load_audio_predictions(path):
    audio_preds = {}
    true_labels = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row['sample_id']
            audio_preds[sample_id] = {
                "prediction": int(row['pred_label']),
                "confidence": float(row['probability'])
            }
            true_labels[sample_id] = int(row['true_label'])
    return audio_preds, true_labels

# Load text predictions
def load_text_predictions(path):
    text_preds = {}
    with open(path, 'r') as f:
        data = json.load(f)
        for entry in data:
            text_preds[entry['filename']] = {
                "prediction": int(entry['prediction']),
                "confidence": 1.0  # full confidence if not given
            }
    return text_preds

# Perform majority vote
def late_fusion_vote(visual_preds, audio_preds, text_preds, true_labels):
    fused = []
    y_true = []
    y_pred = []

    all_keys = set(visual_preds) | set(audio_preds) | set(text_preds)
    for key in sorted(all_keys):
        votes = []
        confidences = defaultdict(list)

        for modality, source in [("visual", visual_preds), ("audio", audio_preds), ("text", text_preds)]:
            if key in source:
                pred = source[key]["prediction"]
                conf = source[key]["confidence"]
                votes.append(pred)
                confidences[pred].append(conf)

        # Majority vote
        vote_counts = Counter(votes)
        most_common = vote_counts.most_common()
        top_pred, top_count = most_common[0]

        # Tie-break by avg confidence
        tied_preds = [pred for pred, count in most_common if count == top_count]
        if len(tied_preds) > 1:
            avg_conf = {pred: sum(confidences[pred]) / len(confidences[pred]) for pred in tied_preds}
            top_pred = max(avg_conf, key=avg_conf.get)

        if key in true_labels:
            y_true.append(true_labels[key])
            y_pred.append(top_pred)

        fused.append({
            "id": key,
            "final_prediction": top_pred,
            "votes": votes,
            "modalities_used": [m for m, s in [("visual", visual_preds), ("audio", audio_preds), ("text", text_preds)] if key in s],
            "true_label": true_labels.get(key)
        })

    return fused, y_true, y_pred

# Save results
def save_fused_predictions(fused, path):
    with open(path, 'w') as f:
        json.dump(fused, f, indent=2)

# Main
if __name__ == "__main__":
    visual_path = "individual_modality_training_inference/video/cnn_lstm_framewise_predictions_context.csv"
    audio_path = "crossattn_framewise_predictions.csv"
    text_path = "custom_test_results.json"

    visual_preds = load_visual_predictions(visual_path)
    audio_preds, true_labels = load_audio_predictions(audio_path)
    text_preds = load_text_predictions(text_path)

    fused_results, y_true, y_pred = late_fusion_vote(visual_preds, audio_preds, text_preds, true_labels)
    save_fused_predictions(fused_results, "fused_predictions.json")

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

    print(f"[SUCCESS] Saved {len(fused_results)} fused predictions to fused_predictions.json")
    print(f"[METRICS] Accuracy:  {acc:.3f}")
    print(f"[METRICS] Precision: {precision:.3f}")
    print(f"[METRICS] Recall:    {recall:.3f}")
    print(f"[METRICS] F1 Score:  {f1:.3f}")

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

def load_and_pool_embedding(embedding_path, pooling="mean"):
    """
    Load a framewise embedding and pool across frames to get a single feature vector.
    """
    embedding = np.load(embedding_path)

    if len(embedding.shape) == 2:
        if pooling == "mean":
            embedding_pooled = np.mean(embedding, axis=1)
        elif pooling == "max":
            embedding_pooled = np.max(embedding, axis=1)
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
    else:
        raise ValueError(f"Expected (features, frames) shape. Got {embedding.shape}")

    return embedding_pooled.reshape(1, -1)  # (1, features)

def predict_batch_xgb_framewise(model_path, embeddings_folder, output_csv_path, pooling="mean"):
    """
    Predict sarcasm using a framewise-pooled XGBoost model over all embeddings in a folder.
    
    Args:
        model_path (str): Path to the trained XGB model (.json).
        embeddings_folder (str): Folder containing .npy embeddings.
        output_csv_path (str): Path to save the output CSV.
        pooling (str): "mean" or "max" pooling over frames.
    """
    # Load the model
    model = XGBClassifier()
    model.load_model(model_path)
    print(f"XGBoost model loaded from {model_path}")

    results = []

    for filename in sorted(os.listdir(embeddings_folder)):
        if filename.endswith(".npy"):
            file_path = os.path.join(embeddings_folder, filename)
            try:
                embedding = load_and_pool_embedding(file_path, pooling=pooling)

                prob = model.predict_proba(embedding)[0][1]  # Probability for class 1 (sarcastic)
                prediction = "1" if prob >= 0.5 else "0"

                id_value = os.path.splitext(filename)[0]

                results.append({
                    "id": id_value,
                    "prediction": prediction,
                    "confidence": round(float(prob), 4)
                })

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

# Example usage
if __name__ == "__main__":
    MODEL_PATH = "framewise_xgb_sarcasm_model_context.json"
    EMBEDDINGS_FOLDER = "framewise_test_embeddings_with_context"
    OUTPUT_CSV = "xgb_framewise_test_predictions.csv"

    predict_batch_xgb_framewise(MODEL_PATH, EMBEDDINGS_FOLDER, OUTPUT_CSV, pooling="mean")

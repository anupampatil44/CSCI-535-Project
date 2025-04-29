import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_embedding(file_path):
    """
    Load a single .npy embedding file.
    """
    embedding = np.load(file_path)

    # Flatten if shape is (features, 1)
    if len(embedding.shape) == 2 and embedding.shape[1] == 1:
        embedding = embedding.flatten()

    # Reshape to (batch_size=1, timesteps=1, feature_dim)
    embedding = embedding.reshape((1, 1, embedding.shape[0]))
    return embedding

def predict_batch(model_path, embeddings_folder, output_csv_path):
    """
    Predict sarcasm for all embeddings in a folder and save to CSV.

    Args:
        model_path (str): Path to the saved model (.h5).
        embeddings_folder (str): Path to folder containing .npy files.
        output_csv_path (str): Path to save output CSV.
    """
    # Load the model
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Collect results
    results = []

    # Process each file
    for filename in sorted(os.listdir(embeddings_folder)):
        if filename.endswith(".npy"):
            file_path = os.path.join(embeddings_folder, filename)
            try:
                embedding = load_embedding(file_path)
                prob = model.predict(embedding)[0][0]
                prediction = "1" if prob >= 0.5 else "0"

                # Extract ID (remove .npy)
                id_value = os.path.splitext(filename)[0]
                id_value.replace("_embedding", "")

                results.append({
                    "id": id_value,
                    "prediction": prediction,
                    "confidence": round(float(prob), 4)
                })

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)

# Example Usage
# if __name__ == "__main__":
#     MODEL_PATH = "/content/cnn_lstm_sarcasm_model_context.h5"
#     EMBEDDINGS_FOLDER = "/content/test_embeddings_with_context"
#     OUTPUT_CSV = "cnn_lstm_non-framewise_video_predictions_context.csv"

#     predict_batch(MODEL_PATH, EMBEDDINGS_FOLDER, OUTPUT_CSV)
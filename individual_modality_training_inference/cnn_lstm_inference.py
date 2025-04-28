import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_embedding(file_path):
    """
    Load a single .npy embedding file.

    Args:
        file_path (str): Path to the .npy file.

    Returns:
        np.ndarray: Feature embedding reshaped for model input.
    """
    embedding = np.load(file_path)

    if len(embedding.shape) == 2 and embedding.shape[1] == 1:
        embedding = embedding.flatten()

    # Reshape to (batch_size=1, timesteps=1, feature_dim)
    embedding = embedding.reshape((1, 1, embedding.shape[0]))
    return embedding

def predict_sarcasm(model_path, embedding_path):
    """
    Predict sarcasm for a single embedding using a saved model.

    Args:
        model_path (str): Path to the saved CNN-LSTM model (.h5).
        embedding_path (str): Path to the embedding file (.npy).

    Returns:
        str: Prediction result ("Sarcastic" or "Non-Sarcastic").
    """

    model = load_model(model_path)

    # Load and prepare the embedding
    embedding = load_embedding(embedding_path)
    print(f"Loaded embedding of shape {embedding.shape}")

    # Predict
    prob = model.predict(embedding)[0][0]
    prediction = "Sarcastic" if prob >= 0.5 else "Non-Sarcastic"

    print(f"Prediction: {prediction} (Confidence: {prob:.4f})")
    return prediction

# Example Usage
# Provide paths here
# MODEL_PATH = "cnn_lstm_sarcasm_model.h5"
# EMBEDDING_PATH = "/content/test_embeddings/2_278.npy"  # e.g., "test_embeddings/1_60_embedding.npy"

# predict_sarcasm(MODEL_PATH, EMBEDDING_PATH)
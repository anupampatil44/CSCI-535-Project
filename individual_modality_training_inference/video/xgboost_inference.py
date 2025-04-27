import numpy as np
from xgboost import XGBClassifier

# REMEMBER TO INSTALL XGBOOST AND CHANGE MODEL FILENAME
def predict_xgb(embedding_path, trained_model_path="xgb_sarcasm_model_libreface_embeddings.json"):
    """
    Predict sarcasm (0: non-sarcastic, 1: sarcastic) for a given new embedding.
    """
    model = XGBClassifier()
    model.load_model(trained_model_path)

    embedding = np.load(embedding_path)
    if embedding.shape[-1] == 1:
        embedding = embedding.flatten()

    embedding = embedding.reshape(1, -1)
    prediction = model.predict(embedding)

    label = "Sarcastic" if prediction[0] == 1 else "Not Sarcastic"
    return label

# Example usage
# embedding_path = "/content/test_embeddings_libreface/1_410_embedding.npy"
# print(f"Prediction: {predict_xgb(embedding_path)}")

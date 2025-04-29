import os
import numpy as np
import libreface
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch
import numpy as np
from sklearn.decomposition import PCA
import torch


def process_video_libreface(video_path, output_path=None):
    """
    Process a video using LibreFace to extract Landmarks + AU intensities,
    and pool features via PCA.

    Args:
        video_path (str): Path to input video (.mp4).
        output_path (str, optional): Path to save the final pooled embedding (.npy).

    Returns:
        numpy.ndarray: Pooled feature embedding of shape (features, 1).
    """
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        results = libreface.get_facial_attributes(f"{video_path}.mp4", device=device)
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None

    embeddings = []

    if results is not None:
      records = results.to_dict(orient="records")  # List of frame dicts
    else:
      print("Libreface returned None")

    for frame_result in records:
        if isinstance(frame_result, dict):
            # Extract landmarks
            landmark_keys = [k for k in frame_result.keys() if k.startswith('lm_')]
            landmarks = np.array([frame_result[k] for k in landmark_keys])

            # Extract AU intensities
            au_keys = [k for k in frame_result.keys() if k.startswith('au_') and k.endswith('intensity')]
            aus = np.array([frame_result[k] for k in au_keys])

            # Concatenate landmarks and AUs
            feature_vector = np.concatenate([landmarks, aus])
            embeddings.append(feature_vector)
        else:
            # No face detected
            print(f"Warning: Empty frame detected, filling with zeros.")
            dummy_vector = np.zeros(478*3 + 12)  # adjust based on actual number of landmarks and AUs
            embeddings.append(dummy_vector)

    if len(embeddings) == 0:
        print(f"Warning: No embeddings extracted for video: {video_path}")
        return None

    concatenated = np.array(embeddings)  # Shape: (frames, features)

    # PCA Pooling
    if np.allclose(concatenated.var(axis=0), 0):
        print(f"Warning: No variance detected in video {video_path}. Saving zero embedding.")
        pooled_embedding = np.zeros((concatenated.shape[1], 1))
    else:
        pca = PCA(n_components=1)
        pca.fit(concatenated)
        pooled_embedding = pca.components_[0].reshape(-1, 1)

    if output_path:
        np.save(output_path, pooled_embedding)

    return pooled_embedding



def batch_process_videos_libreface(txt_file_path, videos_folder, output_folder):
    """
    Batch processes videos listed in a .txt file using LibreFace.

    Args:
        txt_file_path (str): Path to the .txt file listing video filenames.
        videos_folder (str): Path to the folder containing videos.
        output_folder (str): Path to save embeddings.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Read list of videos
    with open(txt_file_path, 'r') as f:
        video_files = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Found {len(video_files)} videos to process.")

    for video_file in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(videos_folder, video_file)
        output_name = os.path.splitext(video_file)[0] + "_embedding.npy"
        output_path = os.path.join(output_folder, output_name)

        try:
            process_video_libreface(video_path, output_path=output_path)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

    print(f"\n✅ All embeddings saved in {output_folder}")

# Train
batch_process_videos_libreface(txt_file_path="train.txt", videos_folder="utterances_final", output_folder="train_embeddings_libreface")

# Val
batch_process_videos_libreface(txt_file_path="val.txt", videos_folder="utterances_final", output_folder="val_embeddings_libreface")

# Test
batch_process_videos_libreface(txt_file_path="test.txt", videos_folder="utterances_final", output_folder="test_embeddings_libreface")
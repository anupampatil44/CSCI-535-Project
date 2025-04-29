import os
import numpy as np

def pad_embeddings_to_match(embedding_A, embedding_B):
    """
    Pad the shorter embedding along the frame dimension to match the longer one.
    """
    frames_A = embedding_A.shape[1]
    frames_B = embedding_B.shape[1]

    if frames_A == frames_B:
        return embedding_A, embedding_B

    max_frames = max(frames_A, frames_B)

    def pad(embedding, target_frames):
        feature_dim = embedding.shape[0]
        padding_needed = target_frames - embedding.shape[1]
        if padding_needed > 0:
            padding = np.zeros((feature_dim, padding_needed), dtype=embedding.dtype)
            embedding_padded = np.concatenate([embedding, padding], axis=1)
            return embedding_padded
        else:
            return embedding

    embedding_A_padded = pad(embedding_A, max_frames)
    embedding_B_padded = pad(embedding_B, max_frames)

    return embedding_A_padded, embedding_B_padded

def batch_pad_folders(folder_A, folder_B, output_folder_A, output_folder_B):
    """
    Batch loads embeddings from two folders, aligns them by padding,
    and saves the corrected embeddings to new folders.

    Args:
        folder_A (str): Path to first set of embeddings (folder).
        folder_B (str): Path to second set of embeddings (folder).
        output_folder_A (str): Output path for padded A embeddings.
        output_folder_B (str): Output path for padded B embeddings.
    """
    os.makedirs(output_folder_A, exist_ok=True)
    os.makedirs(output_folder_B, exist_ok=True)

    files_A = sorted([f for f in os.listdir(folder_A) if f.endswith(".npy")])
    files_B = sorted([f for f in os.listdir(folder_B) if f.endswith(".npy")])

    common_files = set(files_A).intersection(set(files_B))
    print(f"✅ Found {len(common_files)} common files to process.")

    for file_name in common_files:
        path_A = os.path.join(folder_A, file_name)
        path_B = os.path.join(folder_B, file_name)

        try:
            embedding_A = np.load(path_A)
            embedding_B = np.load(path_B)

            embedding_A_padded, embedding_B_padded = pad_embeddings_to_match(embedding_A, embedding_B)

            np.save(os.path.join(output_folder_A, file_name), embedding_A_padded)
            np.save(os.path.join(output_folder_B, file_name), embedding_B_padded)

        except Exception as e:
            print(f"⚠️ Error processing {file_name}: {e}")

    print("\n🎯 Finished saving all padded embeddings.")


# folder_A = "framewise_test_embeddings_video"
# folder_B = "framewise_test_embeddings_audio"

# output_folder_A = "framewise_test_embeddings_libreface_context_10fps_padded"
# output_folder_B = "framewise_test_embeddings_colleague_padded"

# batch_pad_folders(folder_A, folder_B, output_folder_A, output_folder_B)

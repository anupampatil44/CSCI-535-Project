import os
import numpy as np
import pandas as pd

# Directories for audio and video embeddings
audio_dir = "audio_frame_embeddings/train"
video_dir = "video_embeddings/framewise/with_context/train"

# Get sorted list of common files (without extensions)
audio_files = {f.replace(".npz", "") for f in os.listdir(audio_dir) if f.endswith(".npz")}
video_files = {f.replace(".npy", "") for f in os.listdir(video_dir) if f.endswith(".npy")}
common_ids = sorted(audio_files & video_files)

# Prepare comparison data
comparison = []
audio_feat_dims = set()
video_feat_dims = set()

for sample_id in common_ids:
    audio_path = os.path.join(audio_dir, f"{sample_id}.npz")
    video_path = os.path.join(video_dir, f"{sample_id}.npy")

    audio_npz = np.load(audio_path)
    audio_combined = np.concatenate([audio_npz["context"], audio_npz["utter"]], axis=0)  # shape [frames, features]
    video_array = np.load(video_path).T  # shape [frames, features] after transpose

    # Record feature dimensions
    audio_feat_dims.add(audio_combined.shape[1])
    video_feat_dims.add(video_array.shape[1])

    comparison.append({
        "sample_id": sample_id,
        "audio_shape": audio_combined.shape,
        "video_shape": video_array.shape,
        "frames_audio": audio_combined.shape[0],
        "frames_video": video_array.shape[0],
        "frame_diff": abs(audio_combined.shape[0] - video_array.shape[0])
    })

    print({
        "sample_id": sample_id,
        "audio_shape": audio_combined.shape,
        "video_shape": video_array.shape,
        "frames_audio": audio_combined.shape[0],
        "frames_video": video_array.shape[0],
        "frame_diff": abs(audio_combined.shape[0] - video_array.shape[0])
    })

# Create DataFrame for display
df_comparison = pd.DataFrame(comparison)

# Check if feature dimension is consistent
print("\n✅ Audio Feature Dimensions Found:", audio_feat_dims)
print("✅ Video Feature Dimensions Found:", video_feat_dims)

if len(audio_feat_dims) == 1:
    print(f"✅ All audio embeddings have consistent feature dimension: {list(audio_feat_dims)[0]}")
else:
    print(f"⚠️ Audio embeddings have inconsistent feature dimensions: {audio_feat_dims}")

if len(video_feat_dims) == 1:
    print(f"✅ All video embeddings have consistent feature dimension: {list(video_feat_dims)[0]}")
else:
    print(f"⚠️ Video embeddings have inconsistent feature dimensions: {video_feat_dims}")

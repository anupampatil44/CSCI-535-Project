import os
import numpy as np
from tqdm import tqdm

def fix_video_embeddings(video_folder, audio_folder, output_folder, expected_video_feature_dim=1434):
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(video_folder) if f.endswith(".npy")]

    for file in tqdm(video_files, desc="Fixing video embeddings"):
        video_path = os.path.join(video_folder, file)
        audio_path = os.path.join(audio_folder, file.replace(".npy", ".npz"))
        output_path = os.path.join(output_folder, file)

        if not os.path.exists(audio_path):
            print(f"⚠️ Audio file not found for {file}, skipping.")
            continue

        # Load
        video = np.load(video_path)  # Shape could be (features, frames) or (frames, features)
        audio_npz = np.load(audio_path)
        audio = np.concatenate([audio_npz['context'], audio_npz['utter']], axis=0)  # Shape: (frames, audio_features)

        # Check if video needs transpose
        if video.shape[0] == expected_video_feature_dim:
            video = video.T  # Transpose to (frames, features)

        video_frames, video_features = video.shape
        audio_frames = audio.shape[0]

        # Check feature dimension
        if video_features != expected_video_feature_dim:
            raise ValueError(f"⚠️ Video feature dimension mismatch in {file}: found {video_features}, expected {expected_video_feature_dim}")

        # Pad or crop **only frames** (first dimension)
        if video_frames < audio_frames:
            pad_amount = audio_frames - video_frames
            video = np.pad(video, ((0, pad_amount), (0, 0)), mode='constant')
        elif video_frames > audio_frames:
            video = video[:audio_frames, :]

        assert video.shape[1] == expected_video_feature_dim, f"Feature dimension changed for {file}!"

        print(f"✅ Fixed {file}: video shape {video.shape}, audio shape {audio.shape}")

        np.save(output_path, video.T)  # Save back as (features, frames)

    print(f"\n✅ All video embeddings fixed and saved to {output_folder}")

if __name__ == "__main__":
    fix_video_embeddings(
        video_folder="video_embeddings/framewise/with_context/train",
        audio_folder="audio_frame_embeddings/train",
        output_folder="video_embeddings/framewise/with_context_fixed/train"
    )

    fix_video_embeddings(
        video_folder="video_embeddings/framewise/with_context/val",
        audio_folder="audio_frame_embeddings/val",
        output_folder="video_embeddings/framewise/with_context_fixed/val"
    )

    fix_video_embeddings(
        video_folder="video_embeddings/framewise/with_context/test",
        audio_folder="audio_frame_embeddings/test",
        output_folder="video_embeddings/framewise/with_context_fixed/test"
    )

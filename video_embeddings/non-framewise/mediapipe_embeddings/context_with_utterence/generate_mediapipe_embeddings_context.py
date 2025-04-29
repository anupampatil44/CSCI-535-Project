import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.decomposition import PCA
import os
import asyncio
import nest_asyncio
nest_asyncio.apply()

def process_video(video_path):
    """
    Process a single video into a PCA-pooled landmark embedding.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None

    embeddings = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]).flatten()
            embeddings.append(landmark_array)
        else:
            embeddings.append(np.zeros(478 * 3))  # 1434-dim null frame

    cap.release()
    face_mesh.close()

    concatenated = np.array(embeddings)

    if np.allclose(concatenated.var(axis=0), 0):
        pooled_embedding = np.zeros((1434, 1))
    else:
        pca = PCA(n_components=1)
        pca.fit(concatenated)
        pooled_embedding = pca.components_[0].reshape(-1, 1)

    return pooled_embedding

async def batch_process_videos_with_context_async(txt_file_path, utterances_folder, context_folder, output_folder):
    """
    Async version of batch processing for utterance + context videos.
    """
    os.makedirs(output_folder, exist_ok=True)

    with open(txt_file_path, 'r') as f:
        video_files = [line.strip() for line in f.readlines() if line.strip()]

    for video_file in tqdm(video_files, desc=f"Processing {output_folder}"):
        utterance_path = os.path.join(utterances_folder, f"{video_file}.mp4")
        context_path = os.path.join(context_folder, f"{video_file}_c.mp4")
        output_path = os.path.join(output_folder, f"{video_file}.npy")

        try:
            # Important: run CPU-heavy process in a thread to not block event loop
            utterance_embedding = await asyncio.to_thread(process_video, utterance_path)
            context_embedding = await asyncio.to_thread(process_video, context_path)

            if utterance_embedding is None or context_embedding is None:
                print(f"Skipping {video_file} due to missing embedding.")
                continue

            fused_embedding = np.vstack([utterance_embedding, context_embedding])
            np.save(output_path, fused_embedding)

        except Exception as e:
            print(f"Error processing {video_file}: {e}")

    print(f"\nFinished {output_folder}")



async def main():
    tasks = [
        batch_process_videos_with_context_async(
            txt_file_path="train.txt",
            utterances_folder="utterances_final",
            context_folder="context_final",
            output_folder="train_embeddings_with_context"
        ),
        batch_process_videos_with_context_async(
            txt_file_path="val.txt",
            utterances_folder="utterances_final",
            context_folder="context_final",
            output_folder="val_embeddings_with_context"
        ),
        batch_process_videos_with_context_async(
            txt_file_path="test.txt",
            utterances_folder="utterances_final",
            context_folder="context_final",
            output_folder="test_embeddings_with_context"
        )
    ]

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.decomposition import PCA
import os

def process_video(video_path, output_path=None):
    """
    Process video using MediaPipe FaceMesh for face detection and landmark extraction

    Args:
        video_path: Path to input video
        output_path: Optional path to save embeddings

    Returns:
        numpy array: Array of face landmark coordinates per frame
    """
    # Initialize MediaPipe FaceMesh
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

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    embeddings = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with MediaPipe
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            # Extract and normalize landmarks
            face_landmarks = results.multi_face_landmarks[0]
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]).flatten()
            embeddings.append(landmark_array)
        else:
            # Create null embedding (478 landmarks * 3 coordinates)
            embeddings.append(np.zeros(478 * 3))

    cap.release()
    face_mesh.close()

    concatenated = np.array(embeddings)

    if np.allclose(concatenated.var(axis=0), 0):
      # No variance, fallback
      pooled_embedding = np.zeros((1434, 1))

    else:
        pca = PCA(n_components=1)
        pca.fit(concatenated)
        pooled_embedding = pca.components_[0]
        pooled_embedding = pooled_embedding.reshape(-1, 1)

    if output_path:
        np.save(output_path, pooled_embedding)

    return pooled_embedding



def batch_process_videos(txt_file_path, videos_folder, output_folder):
    """
    Processes a batch of videos listed in a txt file and saves PCA-pooled embeddings.

    Args:
        txt_file_path: Path to the .txt file containing video file names (one per line)
        videos_folder: Folder where input videos are located
        output_folder: Folder to save the embeddings
    """
    # Make sure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Read video filenames
    with open(txt_file_path, 'r') as f:
        video_files = [line.strip() for line in f.readlines() if line.strip()]  # remove empty lines

    # Iterate through videos
    for video_file in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(videos_folder, f"{video_file}.mp4")
        
        # You can customize output name here
        video_name = os.path.splitext(video_file)[0]
        output_path = os.path.join(output_folder, f"{video_name}.npy")

        # Process and save
        try:
            process_video(video_path, output_path=output_path)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

    print(f"\nDone! Saved all embeddings to {output_folder}")

# Train
batch_process_videos(txt_file_path="train.txt", videos_folder="utterances_final", output_folder="train_embeddings")

# Val
batch_process_videos(txt_file_path="val.txt", videos_folder="utterances_final", output_folder="val_embeddings")

# Test
batch_process_videos(txt_file_path="test.txt", videos_folder="utterances_final", output_folder="test_embeddings")

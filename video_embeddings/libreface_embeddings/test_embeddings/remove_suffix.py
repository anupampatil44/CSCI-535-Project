import os

def remove_embedding_suffix(folder_path):
    """
    Removes '_embedding' from all filenames in the specified folder.

    Args:
        folder_path (str): Path to the folder containing the files.
    """
    for filename in os.listdir(folder_path):
        if "_embedding" in filename:
            old_path = os.path.join(folder_path, filename)
            new_filename = filename.replace("_embedding", "")
            new_path = os.path.join(folder_path, new_filename)

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} --> {new_filename}")

folder_path = os.getcwd() # 🔵 Change this
remove_embedding_suffix(folder_path)
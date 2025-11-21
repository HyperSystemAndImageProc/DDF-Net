# Create folders
# import os
#
# base_path = r'D:\code\DiffusionData1\diffusion\data\BCI-2a'
#
# for subject in range(1, 10):  # s1 to s9
#     subject_dir = os.path.join(base_path, f's{subject}')
#
#     for condition in ['A01T', 'A01E']:  # create A and T folders
#         condition_dir = os.path.join(subject_dir, condition)
#
#         for session in range(1, 10):  # create session folders 1-9
#             session_dir = os.path.join(condition_dir, str(session))
#             os.makedirs(session_dir, exist_ok=True)

import os
import shutil
import send2trash  # requires installation: pip install Send2Trash


def safe_delete_files(root_dir, suffix="_restored.mat", use_trash=True):
    """
    Safely delete files with a specific suffix
    :param root_dir: root directory path
    :param suffix: target file suffix
    :param use_trash: send to recycle bin if True (False will permanently delete)
    """
    deleted_count = 0
    error_files = []

    # Recursively traverse directories
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(root, file)

                try:
                    if use_trash:
                        # send to recycle bin (safer)
                        send2trash.send2trash(file_path)
                    else:
                        # permanently delete
                        os.remove(file_path)

                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    error_files.append((file_path, str(e)))

    # Print summary report
    print("\n" + "=" * 50)
    print(f"Operation completed! Deleted {deleted_count} files")

    if error_files:
        print("\nThe following files failed to delete:")
        for path, err in error_files:
            print(f"- {path}\n  Error: {err}")


if __name__ == "__main__":
    # Configure parameters (edit here)
    target_directory = r"D:\code\DiffusionData1\diffusion\data\BCI-2a_mat"
    file_suffix = "_restored.mat"  # file suffix to delete


    # Execute safe deletion (send to recycle bin by default)
    print("======= Start safe deletion =======")
    safe_delete_files(target_directory, suffix=file_suffix)

    # If permanent deletion is needed (use with caution!)
    # safe_delete_files(target_directory, suffix=file_suffix, use_trash=False)

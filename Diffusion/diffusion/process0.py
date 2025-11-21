# Extract single-channel txt files
import os
import numpy as np
import scipy.io as sio

source_root = r'D:\datasets\BCI-2a'
target_root = r'D:\code\DiffusionData1\diffusion\data\BCI-2a'

for subj in range(1, 10):
    subj_dir = f's{subj}'
    source_subj_path = os.path.join(source_root, subj_dir)

    base_name = f"A{subj:02d}"
    mat_files = [f"{base_name}E.mat", f"{base_name}T.mat"]

    for mat_file in mat_files:
        source_file = os.path.join(source_subj_path, mat_file)

        if not os.path.exists(source_file):
            print(f"Skipping missing file: {source_file}")
            continue

        target_folder = os.path.join(target_root, subj_dir, mat_file.replace('.mat', ''))

        try:
            # Load MAT data
            mat_data = sio.loadmat(source_file)
            a_data = mat_data['data']

            print(f"\nProcessing file: {os.path.basename(source_file)}")
            print(f"Total structures: {a_data.size}")

            # Iterate each structure
            for struct_idx in range(a_data.size):
                # Correct unpacking logic
                struct_item = a_data[0, struct_idx]  # get structure container [1×1 cell]
                struct_cell = struct_item[0, 0]  # unpack first cell [1×1 cell]
                X_data = struct_cell[0]  # final data matrix [N×25 array]

                # validate dimensions
                if X_data.ndim != 2:
                    print(f"Structure {struct_idx + 1} abnormal shape: {X_data.shape}, reshaping")
                    X_data = X_data.reshape(-1, 25)

                rows, cols = X_data.shape
                print(f"Structure {struct_idx + 1}: {rows} rows x {cols} cols")

                # Create output directory
                struct_dir = os.path.join(target_folder, str(struct_idx + 1))
                os.makedirs(struct_dir, exist_ok=True)

                # save all 25 channels
                for col in range(25):
                    if col >= cols:
                        print(f"! Structure {struct_idx + 1} has <25 columns, padding channel {col + 1} with zeros")
                        col_data = np.zeros(rows)
                    else:
                        col_data = X_data[:, col].flatten()

                    txt_path = os.path.join(struct_dir, f"{col + 1}.txt")
                    np.savetxt(txt_path, col_data, fmt='%.6f')

                print(f"Saved 25 channels → {struct_dir}")

        except Exception as e:
            print(f"\nFailed: {source_file}")
            print(f"Error: {str(e)}")
            continue

print("\nAll processing completed!")







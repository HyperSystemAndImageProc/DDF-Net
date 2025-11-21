# Restore to 1D series
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat

# Load processed data (demo)
mat_data = loadmat('/diffusion/data/BCI-2a_mat/s1/A01E/1/1_t.mat')
interp_matrix = mat_data['Trento']
mat_data1 = loadmat('/diffusion/data/BCI-2a_mat/s1/A01E/1/1.mat')
x_old = mat_data1['x_old'].flatten()
x_new = mat_data1['x_new'].flatten()

# Flatten and inverse interpolate
processed_data = interp_matrix.flatten()
interp_func_reverse = interp1d(x_new, processed_data, kind='linear')
restored_data = interp_func_reverse(x_old).reshape(-1, 1)

# Save restored result
savemat(
    '/diffusion/data/huanyuan/x_restored.mat',
    {'restored_data': restored_data}
)

print("Restoration complete and saved to huanyuan folder")


from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat
import traceback
import os
import scipy.io as sio
import numpy as np


def restore_single_file(processed_path):
    """Restore a single file from _t.mat to _restored.mat"""
    try:
        # derive original path (remove _t suffix)
        original_path = processed_path.replace("_t.mat", ".mat")
        restore_path = processed_path.replace("_t.mat", "_restored.mat")

        # skip if already exists
        if os.path.exists(restore_path):
            print(f"Skip existing: {restore_path}")
            return

        # ensure original exists
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Original file {original_path} not found")

        # load data
        processed_data = loadmat(processed_path)['diffusion_result']
        original_data = loadmat(original_path)

        # coordinates
        x_old = original_data['x_old'].flatten()
        x_new = original_data['x_new'].flatten()

        # inverse interpolation
        processed_flat = processed_data.flatten()
        interp_func = interp1d(x_new, processed_flat, kind='linear', fill_value="extrapolate")
        restored_data = interp_func(x_old).reshape(-1, 1)

        # save result
        savemat(restore_path, {'restored_data': restored_data})
        print(f"Restored: {os.path.basename(processed_path)} -> {os.path.basename(restore_path)}")

    except Exception as e:
        error_msg = f"Failed: {processed_path}\nError: {type(e).__name__}\nDetail: {str(e)}"
        print(error_msg)
        print("-" * 50)
        traceback.print_exc()
        with open("restore_errors.log", "a") as f:
            f.write(error_msg + "\n")


def batch_restore(root_dir):
    """Batch restore all _t.mat files under root_dir"""
    # Iterate all _t.mat files
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("_t.mat"):
                processed_path = os.path.join(root, file)
                print("\n" + "=" * 50)
                print(f"Processing: {processed_path}")
                restore_single_file(processed_path)
def merge_restored_mat(source_root, target_root):
    """
    Merge 25 _restored.mat files into a 25-column array per leaf folder,
    preserving the same directory structure in target_root
    """
    for root, dirs, files in os.walk(source_root):
        # only process leaf numeric folders (e.g. 1,2..)
        if not os.path.basename(root).isdigit():
            continue

        # collect all _restored.mat files
        restored_files = [f for f in files if f.endswith('_restored.mat')]
        if len(restored_files) != 25:
            print(f"Skip {root}: found {len(restored_files)} files, need 25")
            continue

        # sort by numeric filename (1_restored.mat, 2_restored.mat,...)
        restored_files.sort(key=lambda x: int(x.split('_')[0]))

        # load and merge
        merged_data = []
        try:
            for f in restored_files:
                data = sio.loadmat(os.path.join(root, f))['diffusion_result']  # confirm field name
                if data.shape[1] != 1:
                    raise ValueError(f"File {f} column count != 1")
                merged_data.append(data)

            # check row consistency
            rows = merged_data[0].shape[0]
            if any(d.shape[0] != rows for d in merged_data):
                raise ValueError("Row count mismatch")

            merged_array = np.hstack(merged_data)  # horizontal concatenation

            # build target path
            rel_path = os.path.relpath(root, source_root)
            target_dir = os.path.join(target_root, rel_path)
            os.makedirs(target_dir, exist_ok=True)

            # save (use folder name)
            folder_name = os.path.basename(root)
            sio.savemat(
                os.path.join(target_dir, f"{folder_name}.mat"),
                {"merged_data": merged_array}
            )
            print(f"Merged: {rel_path} -> {folder_name}.mat")

        except Exception as e:
            print(f"Failed: {root}")
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Batch restore
    # root dir
    root_directory = r"D:\code\DiffusionData1\diffusion\data\BCI-2a_mat"
    # clear error log
    if os.path.exists("restore_errors.log"):
        os.remove("restore_errors.log")
    # start
    print("======= Start batch restore =======")
    batch_restore(root_directory)
    print("======= Restore done =======")



    # Batch merge

    # paths
    source_root = r"D:\code\DiffusionData1\diffusion\data\BCI-2a_mat"
    target_root = r"D:\code\DiffusionData1\diffusion\data\BCI-2a_diff"

    # execute merge
    merge_restored_mat(source_root, target_root)
    print("Merge complete! Saved to BCI-2a_diff")





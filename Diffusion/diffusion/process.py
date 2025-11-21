import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import savemat

# Paths
source_root = r'D:\code\DiffusionData1\diffusion\data\BCI-2a'
target_root = r'D:\code\DiffusionData1\diffusion\data\BCI-2a_mat'
target_shape = (166, 600)  # target matrix shape


def process_txt_with_metadata(source_path):
    """Process a single txt and return interpolation metadata"""
    try:
        # read raw series
        raw_data = np.loadtxt(source_path).flatten()
        original_length = len(raw_data)

        # generate coordinate system
        x_old = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, np.prod(target_shape))

        # interpolate
        interp_func = interp1d(x_old, raw_data, kind='linear', fill_value="extrapolate")
        interp_matrix = interp_func(x_new).reshape(target_shape)

        return {
            'interp_matrix': interp_matrix,
            'x_old': x_old,
            'x_new': x_new,
            'original_length': original_length,
            'source_path': source_path
        }
    except Exception as e:
        print(f"Failed: {source_path}")
        print(f"Error: {str(e)}")
        return None


def generate_target_path(source_path):
    """Generate mirrored path and create directories"""
    # Preserve path structure
    relative_path = os.path.relpath(source_path, source_root)
    mat_path = relative_path.replace('.txt', '.mat')
    target_path = os.path.join(target_root, mat_path)

    # Create target directory
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    return target_path


def process_all_files():
    # Walk through entire source directory
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.endswith('.txt'):
                source_path = os.path.join(root, file)
                target_path = generate_target_path(source_path)

                # skip existing
                if os.path.exists(target_path):
                    print(f"Exists, skip: {target_path}")
                    continue

                # process and save
                result = process_txt_with_metadata(source_path)
                if result is not None:
                    savemat(target_path, {
                        'interp_matrix': result['interp_matrix'],
                        'x_old': result['x_old'],
                        'x_new': result['x_new'],
                        'original_info': {
                            'length': result['original_length'],
                            'source_path': result['source_path']
                        }
                    })
                    print(f"Converted: {source_path} → {target_path}")


if __name__ == "__main__":
    process_all_files()
    print("Batch processing done!")

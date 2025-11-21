# Stage 1 (Diffusion) Workflow

This document focuses on Stage 1 (DDPM-based synthetic data generation). It clarifies each script’s responsibility and the execution order: split raw EEG captures → interpolate into 2D → train and sample via diffusion → inverse interpolate and merge channels.

## File Responsibilities

- `process0.py`
  - Split each capture in original BCI-2a `.mat` into 25 single-channel `.txt` files.
  - Preserve directory layout: `s?/A01E|A01T/<session>/<1..25>.txt`.

- `process.py`
  - Interpolate single-channel `.txt` into 2D `.mat` (default shape `166×600`).
  - Save interpolation coordinates `x_old` and `x_new` to support inverse interpolation back to the original 1D length.
  - Output to a mirrored directory tree: `BCI-2a_mat`.

- `DiffLidar.py`
  - Train and sample the diffusion model. Read the `interp_matrix` field from a single-channel `.mat`, reshape to 4D, train UNet noise predictor, and reconstruct via DDPM.
  - Key dependencies:
    - `diffusion.py` core (`forward_diffusion_sample`, `get_loss`, `reconstruct`).
    - `unetIDDPM.py` model `UNetLidar`.
    - `utils.py` with `AvgrageMeter`.
  - Important: edit dataset path inside `DiffLidar.py:138` to point to your own file.

- `process1.py`
  - Inverse interpolation and merging: take reconstructed 2D matrices (e.g., `_t.mat`), use `x_old/x_new` to recover the original 1D series, and merge 25 channels horizontally per capture, preserving directory structure to a new location.
  - Provides batch restore and batch merge functions (configure paths as needed).

- `fen ge.py` (optional)
  - Safely delete `_restored.mat` files (recycle bin) for cleanup before reruns.

## Execution Order

1. Split original `.mat` into per-channel `.txt`
   - Command: `python process0.py`
   - Input root example: `D:\\datasets\\BCI-2a`
   - Output example: `D:\\code\\DiffusionData1\\diffusion\\data\\BCI-2a\\s?/A01E|A01T/<session>/<1..25>.txt`

2. Interpolate `.txt` to 2D `.mat` and save interpolation coordinates
   - Command: `python process.py`
   - Input root: `D:\\code\\DiffusionData1\\diffusion\\data\\BCI-2a`
   - Output mirror: `D:\\code\\DiffusionData1\\diffusion\\data\\BCI-2a_mat`
   - `.mat` fields: `interp_matrix` (2D), `x_old`, `x_new`, `original_info`

3. Diffusion model training and sampling (Stage 1 synthetic generation)
   - Command: `python DiffLidar.py --dataset Trento` (or your dataset)
   - Read example: `data\\BCI-2a_mat\\s1\\A01E\\1\\1.mat` field `interp_matrix`
   - After training, save reconstructed samples next to source files (recommended `_t.mat`, field `diffusion_result`).

4. Inverse interpolation and per-capture channel merging
   - Run batch functions in `process1.py`:
     - Batch restore: iterate `_t.mat` and generate `_restored.mat` (field `restored_data`).
     - Batch merge: merge 25 `_restored.mat` files horizontally per capture, save to a new location (e.g., `BCI-2a_diff`).
   - Command example:
     - `python process1.py` (configure `source_root` and `target_root` inside the script).

## Parameters and Paths

- Field names:
  - Use `diffusion_result` for reconstructed data saved by diffusion; matches `process1.py`.
  - Interpolation coordinates: `x_old`, `x_new`.

- Windows paths:
  - Use drive-letter style paths like `D:\\code\\DiffusionData1\\...`; avoid Unix-style `'/diffusion/...'`.

- Directory existence:
  - Ensure target folders exist before saving `.txt` or `.mat` (scripts call `os.makedirs(..., exist_ok=True)`).

## Relation to Three-Stage Training

- Stage 1 (Synthetic Data Generation): Train DDPM via `DiffLidar.py` + `diffusion.py` + `unetIDDPM.py` on real training data; generate high-quality synthetic samples (`_t.mat` or tensors).
 - Stage 2 (Network Pre-training): Pre-train downstream model (e.g., DDFNet) on synthetic samples (no labels) to obtain robust weights.
 - Stage 3 (Formal Training & Evaluation): Load Stage 2 weights and train/evaluate on labeled real data.

The execution order above corresponds to Stage 1 end-to-end: channelization → interpolation → diffusion training/sampling → inverse restore and merging.

## Notes

- In `DiffLidar.py`, consider fixing the `labels` reshape to `labels = labels.reshape(1, 1, *labels.shape)` if needed.
- If the sample count is not exactly 100, avoid hardcoding `i == 99`; save the final or desired step.
- Ensure `_t.mat` field names and paths match `process1.py` expectations for smooth batch restore/merge.
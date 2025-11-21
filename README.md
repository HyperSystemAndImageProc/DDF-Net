# DDFNet

This repository contains the official implementation of DDFNet (Dual-Domain Fusion Network) for EEG-based motor imagery classification. DDFNet integrates frequency–space and time-domain attention with parallel convolutional branches and temporal convolutional networks to learn discriminative representations from MI-EEG.

## Highlights
- Dual-branch architecture combining a standard convolutional block and a dynamic spatio-temporal residual block.
- Sliding-window encoding of temporal context with per-window attention and TCN processing.
- Pluggable attention modules: `mha`, `mhla`, `cbam`, `se`, and a `custom` fusion of frequency–space and time-domain attention.
- Subject-specific training and evaluation (BCI IV-2a, BCI IV-2b), with optional LOSO subject-independent evaluation.

## Model Overview
DDFNet builds two parallel feature streams over the input EEG:
- Branch 1: `Conv_block_` → attention (`attention1`) → `TCN_block_` → window aggregation.
- Branch 2: `DSTR_block` → attention (`attention2`, supports `custom`) → `TCN_block_` → window aggregation.

Outputs from both branches are fused by either averaging window logits or concatenating window features followed by classification. The `custom` attention stacks a Frequency–Space Attention module with a Time-Domain Attention module.

## Datasets
- BCI Competition IV-2a (`./BCI-2a`): 22 channels, 4 classes.
- BCI Competition IV-2b (`./BCI-2b`): 3 channels, 2 classes.

Place the datasets in the folders above or update `data_path` in the corresponding script.

### Official Download Links
- BCI Competition IV-2a
  - Official description (BBCI): https://www.bbci.de/competition/iv/#dataset2a
  - Description PDF: https://www.bbci.de/competition/iv/desc_2a.pdf
  - Official download (BNCI Horizon 2020): https://bnci-horizon-2020.eu/database/data-sets (find “BCI Competition IV – 2a” four-class motor imagery dataset)
- BCI Competition IV-2b
  - Official description (BBCI): https://www.bbci.de/competition/iv/#dataset2b
  - Description PDF: https://www.bbci.de/competition/iv/desc_2b.pdf
  - Official download (BNCI Horizon 2020): https://bnci-horizon-2020.eu/database/data-sets (find “BCI Competition IV – 2b” two-class motor imagery dataset)

### Direct .mat URLs (examples)
- 2a (subjects 1–9)
  - A01T.mat: https://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat
  - A01E.mat: https://bnci-horizon-2020.eu/database/data-sets/001-2014/A01E.mat
  - A02T.mat: https://bnci-horizon-2020.eu/database/data-sets/001-2014/A02T.mat
  - A02E.mat: https://bnci-horizon-2020.eu/database/data-sets/001-2014/A02E.mat
  - A03T.mat: https://bnci-horizon-2020.eu/database/data-sets/001-2014/A03T.mat
  - A03E.mat: https://bnci-horizon-2020.eu/database/data-sets/001-2014/A03E.mat
  - Replace `01` with `04` … `09` for other subjects.
- 2b (subjects 1–9)
  - B01T.mat: https://bnci-horizon-2020.eu/database/data-sets/004-2014/B01T.mat
  - B01E.mat: https://bnci-horizon-2020.eu/database/data-sets/004-2014/B01E.mat
  - B02T.mat: https://bnci-horizon-2020.eu/database/data-sets/004-2014/B02T.mat
  - B02E.mat: https://bnci-horizon-2020.eu/database/data-sets/004-2014/B02E.mat
  - B03T.mat: https://bnci-horizon-2020.eu/database/data-sets/004-2014/B03T.mat
  - B03E.mat: https://bnci-horizon-2020.eu/database/data-sets/004-2014/B03E.mat
  - Replace `01` with `04` … `09` for other subjects.

## Usage
- BCI-2a pipeline: run `python "main_TrainTest 2a.py"`. Training configuration and dataset setup are defined in `run()`; evaluation loads the best run weights from `results`.
- BCI-2b pipeline: run `python main_2b.py`. Results are saved under `2b results`.

Adjust the hyperparameters and attention types in the `getModel(...)` call and the `train_conf` dictionary inside each script.

## Training Workflow
The architecture of our proposed DDFNet and its unique training paradigm are illustrated in Fig. 1. The workflow can be decomposed into three sequential, independent stages to achieve effective data augmentation and parameter initialization.

- Stage 1: Synthetic Data Generation. A Denoising Diffusion Probabilistic Model (DDPM) is independently trained on the real EEG training dataset. Upon convergence, this trained diffusion model is used to generate a large-scale, high-quality set of synthetic EEG samples that mirror the distribution of the real data.
- Stage 2: Network Pre-training. The DDFNet model, with its dual-branch architecture, is initialized with random weights. It is then pre-trained on the synthetic samples generated in Stage 1. This process does not use any real data labels and aims to learn generalizable features, resulting in a set of robust pre-trained weights.
- Stage 3: Formal Training and Evaluation. The pre-trained weights from Stage 2 are loaded into DDFNet as an advanced initialization. The model is then formally trained on the labeled real EEG dataset, following the unified protocol described in Section 3.2.1. The performance of the final model is evaluated on the held-out test set.



This decoupled training strategy ensures that the knowledge transferred from the synthetic data effectively bootstraps the learning process on the real data, leading to enhanced generalization performance.

### Checkpoints and Reproducibility
- Checkpoints are saved per subject and per run and selected by `val_accuracy` via `ModelCheckpoint`.
- 2a results
  - Best weights: `results/saved models/run-<N>/subject-<M>.h5`
  - Best-run index per subject: `results/best models.txt`
  - Aggregated metrics across runs: `results/perf_allRuns.npz`
- 2b results
  - Best weights: `2b results/saved models/run-<N>/subject-<M>.h5`
  - Best-run index per subject: `2b results/best models.txt`
  - Aggregated metrics across runs: `2b results/perf_allRuns.npz`

To reproduce:
- Training: in each script’s `run()` function, enable the training line `train(dataset_conf, train_conf, results_path)` to generate new checkpoints.
- Evaluation: run the script as-is; it will load the best weights per subject based on the entries in `best models.txt` and report metrics. Ensure datasets are placed under `./BCI-2a` and `./BCI-2b` or update `data_path` accordingly.

## Requirements
- Python 3.7+
- TensorFlow 2.7+
- NumPy, SciPy, matplotlib, scikit-learn

Install packages with your preferred environment manager (e.g., Anaconda) on Windows or Linux with CUDA-enabled GPUs for training.

## License
This project is licensed under the Apache-2.0 License. See `LICENSE` for details.

## Citation
If you find DDFNet useful in your research, please cite this repository. A formal paper describing DDFNet is in preparation.



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

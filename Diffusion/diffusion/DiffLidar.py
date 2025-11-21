import os
import scipy.io as sio
import scipy.io

import argparse
import numpy as np
import torch

from matplotlib import pyplot as plt
from sklearn import preprocessing

from torch.optim import Adam

from torch.backends import cudnn

from diffusion import Diffusion  # diffusion model
from uvit import UViT  # UViT model (optional)
from unetIDDPM import UNetLidar  # UNetLidar model
from utils import AvgrageMeter, show_img  # utility functions

# CLI arguments
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Trento', 'Houston', 'Berlin', 'Muufl', 'Augsburg'], default='Trento',
                    help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='test', help='testing mark')
parser.add_argument('--mode', choices=['HCT', 'CAF'], default='HCT', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=3047, help='number of seed')
parser.add_argument('--BATCH_SIZE', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--Patch_size', type=int, default=11, help='number of patches')
parser.add_argument('--Pca_Components', type=int, default=32, help='number of related band')
parser.add_argument('--Epoch', type=int, default=10, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--model_path', type=str, default='./save_model/model/Muufl50000.pkl',
                    help='path to the trained model for testing')
args = parser.parse_args()

# Set random seed for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

# Hyperparameters
batch_size = 64
patch_size = 11
select_spectral = []
spe = 200
channel = 1  # 3D channel

# Training schedule
epochs = 1
lr = 1e-4
T = 1

# RGB channels
rgb = [30, 50, 90]
path_prefix = "./save_model/model"  # model save path




# Sampling at t = T-1 to reconstruct
def sample_by_t1(diffusion, model, X):
    X = X.cpu().numpy()
    # convert to torch tensor
    x0 = torch.from_numpy(X[:, :, :, :]).float()

    t = torch.full((1,), diffusion.T - 1, device=device, dtype=torch.long)
    print(t)
    xt, tmp_noise = diffusion.forward_diffusion_sample(x0, t, device)  # diffusion sampling

    ##########################################################
    # reconstruct from noise
    _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num=5)  # reconstruction

    return xt, recon_from_xt

# Save trained model
def save_model(model, path):
    torch.save(model.state_dict(), path)  # save model state dict
    print("save model done. path=%s" % path)

# Diffusion training function
# Diffusion training function
def DiffusionTrain(lables, Hsi_train):
    global recon_from_xt, xt
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = 100  # number of diffusion steps
    diffusion = Diffusion(T=T)  # instantiate diffusion with T steps
    model = UNetLidar(
        T=T, in_ch=1, out_ch=1, dropout=0.1, tdim=512)  # UNetLidar training model
    model.to(device)  # move model to device

    optimizer = Adam(model.parameters(), lr=lr)  # Adam optimizer
    loss_metric = AvgrageMeter()  # average loss tracker

    os.makedirs(path_prefix, exist_ok=True)  # ensure model dir exists

    # start training
    for epoch in range(5000):
        loss_metric.reset()  # reset loss meter
        batch = lables.to(device)
        optimizer.zero_grad()

        t = torch.full((1,), diffusion.T - 1, device=device, dtype=torch.long)  # current timestep

        # compute loss via diffusion (xt, noise, predicted noise)
        loss, temp_xt, temp_noise, temp_noise_pred = diffusion.get_loss(model, batch, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Epoch-step] {epoch} |  Loss: {loss.item()} ")

        # save model and outputs periodically
        if (epoch + 1) % 500 == 0:
            xt, recon_from_xt = sample_by_t1(diffusion, model, Hsi_train)
            path = "%s/Muufl%s.pkl" % (path_prefix, epoch + 1)
            save_model(model, path)

    return xt, recon_from_xt  # return generated results


# Data loading
def loadData():
    if args.dataset == 'Trento':

        # data_lidar = sio.loadmat('data/Trento/Lidar1_Trento.mat')['lidar1_trento']
        # data_lidar = sio.loadmat('data/shiyan/2.mat')['interp_matrix']
        labels = sio.loadmat('data/Trento/GT_Trento.mat')['gt_trento']
        data_lidar = sio.loadmat('data\BCI-2a_mat\s1\A01E/1/1.mat')['interp_matrix']  # IMPORTANT: edit dataset path here to your own data location
    elif args.dataset == 'Muufl':
        data_lidar = sio.loadmat('data\Muufl\Muufl_lidar.mat')['lidar']
        labels = sio.loadmat('data\Muufl\Muufl_gt0.mat')['gt']
    elif args.dataset == 'Houston':
        data_lidar = sio.loadmat('data/Houston2013/Houston_Lidar.mat')['lidar']
        labels = sio.loadmat('data/Houston2013/Houston_gt.mat')['gt']
    elif args.dataset == 'Berlin':
        data_lidar = sio.loadmat('data/Berlin/Ber/Berlin4_sar.mat')['sar']
        labels = sio.loadmat('data/Berlin/Ber/Berlin_gt.mat')['gt']
    elif args.dataset == 'Augsburg':
        data_lidar = sio.loadmat('data/Augsburg/Au/Augsburg_DSM.mat')['Augsburg']
        labels = sio.loadmat('data/Augsburg/Au/Augsburg_gt.mat')['gt']
    else:
        print("NO dataset")
    return labels, data_lidar





# Main
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load data
    labels, data_lidar = loadData()
    data_lidar = data_lidar.astype(np.float32)  # ensure float32
    data_lidar = data_lidar.reshape(1, 1, *data_lidar.shape)  # reshape to 4D
    labels = data_lidar.reshape(1, 1, *labels.shape)
    data_lidar = torch.tensor(data_lidar)  # numpy to tensor
    labels = torch.tensor(labels)

    # train
    xt, recon_from_xt = DiffusionTrain(labels, data_lidar)

    # save results
    xt_dict = {}
    for i in range(len(recon_from_xt)):
        if i == 99:  # only save the last sample (assuming 100 samples)
            recon_from_xt[i] = torch.squeeze(recon_from_xt[i])
            xt_i = recon_from_xt[i].cpu().numpy()
            var_name = f'xt{i}'
            xt_dict[var_name] = xt_i
            file_path = 'DiffusionData/' + args.dataset + '/' + args.dataset + f'{i}.mat'
            scipy.io.savemat(file_path, {args.dataset: xt_dict[var_name]})
            print(f"Saved only the last sample (i=99) to {file_path}")


# import os
# import scipy.io as sio
# import numpy as np
# import torch
# from torch.optim import Adam
# from diffusion import Diffusion
# from unetIDDPM import UNetLidar
#
# # configure paths
# source_root = r'D:\code\DiffusionData1\diffusion\data\BCI-2a_mat'
# fixed_label_path = 'data/Trento/GT_Trento.mat'  # fixed label path
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# def process_single_file(mat_path):
#     """Process a single mat file (fixed save logic)"""
#     try:
#         # generate output path
#         dir_path = os.path.dirname(mat_path)
#         base_name = os.path.basename(mat_path).split('.')[0]
#         output_path = os.path.join(dir_path, f"{base_name}_t.mat")  # strictly keep original directory
#
#         # skip already processed file
#         if os.path.exists(output_path):
#             print(f"Skip existing file: {output_path}")
#             return
#
#         # data loading and processing (preserve original logic)
#         data_lidar = sio.loadmat(mat_path)['interp_matrix'].astype(np.float32)
#         labels = sio.loadmat(fixed_label_path)['gt_trento'].astype(np.float32)
#
#         # convert to tensor (preserve original dimensions)
#         data_tensor = torch.tensor(data_lidar).unsqueeze(0).unsqueeze(0).to(device)
#         label_tensor = torch.tensor(labels).reshape(1, 1, *labels.shape).to(device)
#
#         # model training (preserve original workflow)
#         diffusion = Diffusion(T=100)
#         model = UNetLidar(T=100, in_ch=1, out_ch=1, dropout=0.1, tdim=512).to(device)
#         optimizer = Adam(model.parameters(), lr=1e-4)
#
#         # training loop
#         for epoch in range(100):
#             optimizer.zero_grad()
#             t = torch.full((1,), diffusion.T - 1, device=device, dtype=torch.long)
#             loss, _, _, _ = diffusion.get_loss(model, label_tensor, t)
#             loss.backward()
#             optimizer.step()
#             print(f"[{base_name}] Epoch {epoch} | Loss: {loss.item():.4f}")
#
#         # final sampling (preserve original sampling logic)
#         with torch.no_grad():
#             t = torch.full((1,), diffusion.T - 1, device=device, dtype=torch.long)
#             _, recon_from_xt = diffusion.reconstruct(model, xt=data_tensor, tempT=t, num=5)
#
#         # fixed save logic --------------------------------------------------
#         xt_dict = {}
#         for i in range(len(recon_from_xt)):
#             if i == len(recon_from_xt) - 1:  # only save the last sample (original i=99 logic)
#                 # handle dimensions (consistent with original code)
#                 result = torch.squeeze(recon_from_xt[i]).cpu().numpy()
#
#                 # validate shape (166,600)
#                 if result.shape != (166, 600):
#                     raise ValueError(f"Incorrect result shape: {result.shape}, should be (166, 600)")
#
#                 # save with original filename + _t
#                 sio.savemat(output_path, {'diffusion_result': result})
#                 print(f"Saved: {os.path.basename(mat_path)} → {os.path.basename(output_path)}")
#
#     except Exception as e:
#         print(f"Processing failed: {os.path.basename(mat_path)}")
#         print(f"Error type: {type(e).__name__}, detail: {str(e)}")
#         if 'output_path' in locals():
#             print(f"Target path: {output_path}")
#
#
# # keep other code unchanged
#
#
# def batch_process():
#     """Process all mat files in strict numeric order"""
#     for root, _, files in os.walk(source_root):
#         # sort files numerically (1.mat, 2.mat, ... 25.mat)
#         sorted_files = sorted(
#             [f for f in files if f.endswith('.mat') and not f.endswith('_t.mat')],
#             key=lambda x: int(x.split('.')[0])
#         )
#
#         for file in sorted_files:
#             mat_path = os.path.join(root, file)
#             process_single_file(mat_path)
#
#
# if __name__ == '__main__':
#     # keep original random seed settings
#     np.random.seed(3047)
#     torch.manual_seed(3047)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(3047)
#
#     batch_process()
#     print("Batch processing completed! Original training logic preserved")

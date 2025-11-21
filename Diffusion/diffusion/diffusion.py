import torch
import torchvision
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
# from data import HSIDataLoader
import numpy as np

# from plot import show_tensor_image

device = "cuda" if torch.cuda.is_available() else "cpu"


class Diffusion(object):
    def __init__(self, T=1000) -> None:
        """
        Initialize diffusion process parameters.
        :param T: number of diffusion timesteps, default 1000
        """
        self.T = T
        # generate linear beta schedule controlling noise magnitude
        self.betas = self._linear_beta_schedule(timesteps=self.T)

        # precompute parameters used in diffusion
        self.alphas = 1. - self.betas  # alpha = 1 - beta
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)  # cumulative product of alphas
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)  # pad with initial 1
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # sqrt reciprocal alphas
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # sqrt cumulative alphas
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)  # sqrt(1 - cumulative alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)  # posterior variance

    def _linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        """
        Generate linearly increasing beta values.
        :param timesteps: total diffusion steps
        :param start: start beta
        :param end: end beta
        :return: linearly spaced beta values between start and end
        """
        return torch.linspace(start, end, timesteps)

    def _get_index_from_list(self, vals, t, x_shape):
        """
        Gather value for timestep `t` from a list, respecting batch dimension.
        :param vals: parameter list (tensor)
        :param t: current timestep tensor
        :param x_shape: input tensor shape
        :return: value for timestep t reshaped to match batch and broadcast over x shape
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())  # take values at indices t
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)  # reshape for broadcasting

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """
        Add noise to `x_0` at timestep `t` and return noisy image and noise.
        :param x_0: clean input image/tensor
        :param t: current timestep
        :param device: target device
        :return: (noisy image, noise)
        """
        noise = torch.randn_like(x_0)  # random noise same shape as x_0
        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)  # sqrt(alpha_bar_t)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)  # sqrt(1 - alpha_bar_t)
        return (sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
               + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device))

    def get_loss(self, model, x_0, t):
        """
        Compute loss between predicted noise and true noise at timestep `t`.
        :return: L1 loss, noisy image, true noise, predicted noise
        """
        # get noisy image and noise
        x_noisy, noise = self.forward_diffusion_sample(x_0, t, device)

        # predict noise
        noise_pred = model(x_noisy, t)

        # L1 loss
        return F.l1_loss(noise, noise_pred), x_noisy, noise, noise_pred

    @torch.no_grad()
    def sample_timestep(self, x, t, model):
        """
        Sample the previous timestep given current `x` and `t` using the model.
        Computes model mean and, if not the last step, adds noise scaled by posterior variance.
        """
        # get beta, alpha and related parameters at t
        betas_t = self._get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # model mean using predicted noise
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self._get_index_from_list(self.posterior_variance, t, x.shape)

        # last step returns mean; otherwise add noise
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def reconstruct(self, model, xt=None, tempT=None, num=10, from_noise=False, shape=None):
        """
        Reconstruct images stepwise from noise or given `xt`.
        :return: indices list and images list for each step
        """
        stepsize = int(tempT.cpu().numpy()[0] / num)  # step size per reconstruction chunk
        index = []
        res = []

        # initialize from noise or given xt
        if from_noise:
            img = torch.randn(shape, device=device)
        else:
            img = xt

        if tempT is None:
            tempT = self.T

        for i in reversed(range(tempT)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = self.sample_timestep(img, t, model)
            index.append(i)
            res.append(img.detach().cpu())

        index.append(i)
        res.append(img.detach().cpu())
        return index, res

    @torch.no_grad()
    def reconstruct_v2(self, model, xt=None, tempT=None, use_index=[], from_noise=False, shape=None):
        """
        Reconstruct while saving only specified timesteps in `use_index`.
        :return: indices list and images list at specified steps
        """
        index = []
        res = []

        # initialize from noise or given xt
        if from_noise:
            img = torch.randn(shape, device=device)
        else:
            img = xt

        if tempT is None:
            tempT = self.T

        # iterate timesteps high to low
        for i in range(0, tempT)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = self.sample_timestep(img, t, model)
            # save if current t in use_index
            if i in use_index:
                index.append(i)
                res.append(img.detach().cpu())

        index.append(i)
        res.append(img.detach().cpu())
        return index, res

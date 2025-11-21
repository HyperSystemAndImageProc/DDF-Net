import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class Swish(nn.Module):
    '''Swish activation'''

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)  # [d_model/2]
        pos = torch.arange(T).float()  # [t]
        # emb = pos[:, None] * emb[None, :]
        # emb = pos[T,1]*emb[1,d_model/2] = [T,d_model/2]
        emb = pos.unsqueeze(1) * emb.unsqueeze(0)
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)  # stack creates new dim; cat only concatenates
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),  # effectively nn.Embedding(T, d_model) with pretrained weights
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        # If input is [B], output is [B, dim]
        emb = self.timembedding(t - 1)
        return emb


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_ch),  # batch norm over channels
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            # project time embedding from tdim to out_ch
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = Attention(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]  # add via broadcasting
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 1024 // 8 = 128
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)  # linear to get qkv, e.g., 768 -> 2304
        self.proj = nn.Linear(dim, dim)  # projection

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.reshape(B, C, -1)  # [B,C,N]
        x = torch.einsum('xyz->xzy', x)  # [B,N,C]
        qkv = self.qkv(x)  # [B,N,3*C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # [B,N,3,8,128]
        qkv = torch.einsum('abcde->cadbe', qkv)  # [3,B,8,N,128]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,8,N,128]
        attn = (q @ torch.einsum('abcd->abdc', k)) * self.scale
        # softmax by row
        attn = attn.softmax(dim=-1)
        x = (attn @ v)  # [B,8,N,128]
        x = torch.einsum('abcd->acbd', x)  # [B,N,8,128]
        x = x.reshape(B, N, C)  # [B,N,1024]
        x = self.proj(x)
        x = x.reshape(B, H, W, C)  # [B,H,W,C]
        x = torch.einsum('abcd->adbc', x)
        return x


def ConcatAlign(out, x4):
    if out.shape == x4.shape:
        out = torch.cat([out, x4], dim=1)
    else:
        if out.shape[2] < x4.shape[2]:
            pad_size = (0, 0, 0, 1)
            out = torch.nn.functional.pad(out, pad_size, mode='constant', value=1)
        elif out.shape[2] > x4.shape[2]:
            pad_size = (0, 0, 0, 1)
            x4 = torch.nn.functional.pad(x4, pad_size, mode='constant', value=1)

        if out.shape[3] < x4.shape[3]:
            pad_size = (0, 1, 0, 0)
            out = torch.nn.functional.pad(out, pad_size, mode='constant', value=1)
        elif out.shape[3] > x4.shape[3]:
            pad_size = (0, 1, 0, 0)
            x4 = torch.nn.functional.pad(x4, pad_size, mode='constant', value=1)

        out = torch.cat([out, x4], dim=1)

    return out


class UNetLidar(nn.Module):
    def __init__(self, T=1000, in_ch=1, out_ch=1, dropout=0.1, tdim=512):
        # tdim: time embedding dimension
        super(UNetLidar, self).__init__()
        time_emb_dim = 512
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # time embedding
        # self.time_embedding = TimeEmbedding(T, 128, tdim)
        # encoder
        self.res1 = ResBlock(1, 64, tdim, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = ResBlock(64, 128, tdim, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = ResBlock(128, 256, tdim, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.res4 = ResBlock(256, 512, tdim, dropout)
        self.pool4 = nn.MaxPool2d(2)
        # middle with self-attention
        self.midblock = ResBlock(512, 1024, tdim, dropout, attn=True)
        # predict linear weight from middle outputs
        self.var_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        # decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.res5 = ResBlock(1024, 512, tdim, dropout)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.res6 = ResBlock(512, 256, tdim, dropout)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.res7 = ResBlock(256, 128, tdim, dropout)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.res8 = ResBlock(128, 64, tdim, dropout)
        # output layer
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x, t):
        # temb = self.time_embedding(t)
        temb = self.time_mlp(t)
        # encoder
        x1 = self.res1(x, temb)
        x2 = self.res2(self.pool1(x1), temb)
        x3 = self.res3(self.pool2(x2), temb)
        x4 = self.res4(self.pool3(x3), temb)
        # middle with self-attention
        out = self.midblock(self.pool4(x4), temb)
        var_weight = self.var_weight(out)  # predicted linear weight
        # decoder
        out = self.up1(out)
        out = ConcatAlign(out, x4)
        # out = torch.cat([out, x4], dim=1)
        out = self.res5(out, temb)

        out = self.up2(out)
        out = ConcatAlign(out, x3)
        # out = torch.cat([out, x3], dim=1)
        out = self.res6(out, temb)

        out = self.up3(out)
        out = ConcatAlign(out, x2)
        # out = torch.cat([out, x2], dim=1)
        out = self.res7(out, temb)

        out = self.up4(out)
        out = ConcatAlign(out, x1)
        # out = torch.cat([out, x1], dim=1)
        out = self.res8(out, temb)
        out = self.out(out)
        return out  # return predicted noise and linear weight


if __name__ == '__main__':
    batch_size = 1
    model = UNetLidar(
        T=1000, in_ch=1, out_ch=1, dropout=0.1, tdim=512)
    x = torch.randn(batch_size, 1, 32, 32)
    # t = torch.randint(1000, (batch_size,))
    t = torch.full((1,), 0, dtype=torch.long)  # create tensor of shape (1,), filled with 0, dtype long
    y = model(x, t)
    print(y.shape)

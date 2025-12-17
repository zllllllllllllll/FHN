import numpy as np
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net1_ = nn.Linear(dim, hidden_dim)
        self.net2_ = nn.Linear(hidden_dim, dim)
        self.net = nn.Sequential(
            self.net1_,
            nn.GELU(),
            self.net2_
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, attn_drop=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.dim = dim
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        h = self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        v_B, v_head, v_N, v_C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class Embedding(nn.Module):
    def __init__(self, num_patches, min_dim, dim):
        super(Embedding, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.num = num_patches
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, num_patches, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(min_dim * min_dim, dim)

    def forward(self, x):
        b, n, c, c = x.shape
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        # print('x', x.shape)
        x = x.view(b, self.num, -1)
        # print('x', x.shape)
        x = self.fc(x)
        return x


class ViT(nn.Module):
    def __init__(self, rule_center, var, code_length, n_clusters, cls, image_size, patch_size, channels, dim, depth, heads, mlp_dim):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.code_length = code_length
        self.patch_size = patch_size
        self.cls = cls
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.end_to_hash = nn.Linear(n_clusters*code_length, code_length)
        self.end_to_cls = nn.Linear(n_clusters*cls, cls)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.num_patches = num_patches
        self.to_cls_token = nn.Identity()
        self.center = torch.nn.Parameter(torch.FloatTensor(rule_center))
        self.var = torch.nn.Parameter(torch.FloatTensor(var))
        self.mlp_head1_ = nn.Linear(dim, mlp_dim)
        self.mlp_head2_ = nn.Linear(mlp_dim, code_length)
        self.mlp_head = nn.Sequential(
            self.mlp_head1_,
            # nn.GELU(),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.mlp_head2_,
            nn.Tanh()
        )

        self.classifier1_ = nn.Linear(dim, 2 * dim)
        self.classifier2_ = nn.Linear(2 * dim, dim)
        self.classifier3_ = nn.Linear(dim, cls)
        self.classifier = nn.Sequential(
            self.classifier1_,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.classifier2_,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.classifier3_,
        )

    def forward(self, img):
        x_h = np.zeros((img.shape[0], self.code_length), dtype=np.float32)
        x_h = torch.tensor(x_h).cuda()
        x_cls = np.zeros((img.shape[0], self.cls), dtype=np.float32)
        x_cls = torch.tensor(x_cls).cuda()
        for i in range(img.shape[2]):
            x = img[:, :, i]
            x = x.to(torch.float32)
            b, fea = x.shape
            x = x.reshape(b, self.num_patches, -1)
            x += self.pos_embedding
            x = self.transformer(x)
            x = self.to_cls_token(x[:, 0])
            x_h_per = self.mlp_head(x)
            # print('x_h_per', x_h_per)
            x_cls_per = self.classifier(x)
            # print('x_cls_per', x_cls_per)
            x_h += x_h_per.cuda()  # [b, code_length]
            x_cls += x_cls_per.cuda()

        x_h = F.softsign(x_h)
        x_cls = F.softsign(x_cls)
        return x_h, x_cls
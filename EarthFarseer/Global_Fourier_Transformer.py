import torch
from torch import nn
from modules import *
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
from functools import partial
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint_sequential


class PatchEmbed(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_c=13, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # h, w
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection= nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Error..."
        '''
        [32, 3, 224, 224] -> [32, 768, 14, 14] -> [32, 768, 196] -> [32, 196, 768]
        Conv2D: [32, 3, 224, 224] -> [32, 768, 14, 14]
        Flatten: [B, C, H, W] -> [B, C, HW]
        Transpose: [B, C, HW] -> [B, HW, C]
        '''
        x = self.projection(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc3 = nn.AdaptiveAvgPool1d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, F_dim: int, H_dim: int, D: int, gamma: float):

        super().__init__()
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):

        B, N, M = x.shape
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        Y = self.mlp(F)
        PEx = Y.reshape((B, N, self.D))
        return PEx

class AdativeFourierNeuralOperator(nn.Module):
    def __init__(self, dim, h=14, w=14, is_fno_bias=True):
        super(AdativeFourierNeuralOperator, self).__init__()
        self.hidden_size = dim
        self.h = h
        self.w = w
        self.num_blocks = 2
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()
        self.is_fno_bias = is_fno_bias

        if self.is_fno_bias:
            self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        else:
            self.bias = None

        self.softshrink = 0.00

    def multiply(self, input, weights):
        return torch.einsum('...bd, bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape

        if self.bias:
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros(x.shape, device=x.device)

        x = x.reshape(B, self.h, self.w, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0], inplace=True)
        x_imag = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1], inplace=True)
        x_real = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1]

        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1,2), norm='ortho')
        x = x.reshape(B, N, C)

        return x+bias

class FourierNetBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=2.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 h=14,
                 w=14):
        super(FourierNetBlock, self).__init__()
        self.normlayer1 = norm_layer(dim)
        self.filter = AdativeFourierNeuralOperator(dim, h=h, w=w)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.normlayer2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.double_skip = True

    def forward(self, x):
        x = x + self.drop_path(self.filter(self.normlayer1(x)))
        x = x + self.drop_path(self.mlp(self.normlayer2(x)))
        return x

class GF_Block(nn.Module):
    def __init__(self,
                 img_size=128,
                 patch_size=16,
                 in_channels=20,
                 out_channels=20,
                 input_frames=20,
                 embed_dim=768,
                 depth=12,
                 mlp_ratio=4.,
                 uniform_drop=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 dropcls=0.):
        super(GF_Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = input_frames
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_c=in_channels,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # [1, 196, 768]
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = self.patch_embed.grid_size[0]
        self.w = self.patch_embed.grid_size[1]
        '''
        stochastic depth decay rule
        '''
        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([FourierNetBlock(
            dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            drop_path=dpr[i],
            act_layer=nn.GELU,
            norm_layer=norm_layer,
            h=self.h,
            w=self.w)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.linearprojection = nn.Sequential(OrderedDict([
            ('transposeconv1', nn.ConvTranspose2d(embed_dim, out_channels * 16, kernel_size=(2, 2), stride=(2, 2))),
            ('act1', nn.Tanh()),
            ('transposeconv2', nn.ConvTranspose2d(out_channels * 16, out_channels * 4, kernel_size=(2, 2), stride=(2, 2))),
            ('act2', nn.Tanh()),
            ('transposeconv3', nn.ConvTranspose2d(out_channels * 4, out_channels, kernel_size=(4, 4), stride=(4, 4)))
        ]))

        # Modify
        self.reshapemlp =  nn.Linear(64 * 64, img_size * img_size)

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        '''
        patch_embed:
        [B, T, C, H, W] -> [B*T, num_patches, embed_dim] L D
        '''
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.patch_embed(x)
        #enc = LearnableFourierPositionalEncoding(768, 768, 64, 768, 10)
       # fourierpos_embed = enc(x)
        x = self.pos_drop(x + self.pos_embed)
        #x = self.pos_drop(x + fourierpos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).transpose(1, 2)
        x = torch.reshape(x, [-1, self.embed_dim, self.h, self.w])
        return x

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.linearprojection(x)
        x = self.reshapemlp(x.flatten(-2,-1))
        x = x.reshape(B, T, C, H, W)
        return x



class Fourier_Model(nn.Module):
    def __init__(self, shape_in, hid_S=512, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(Fourier_Model, self).__init__()
        self.fourier = GF_Block(img_size=128,
                                patch_size=16,
                                in_channels=2,
                                out_channels=2,
                                input_frames=12,
                                embed_dim=768,
                                depth=12,
                                mlp_ratio=4.,
                                uniform_drop=False,
                                drop_rate=0.,
                                drop_path_rate=0.,
                                norm_layer=None,
                                dropcls=0.)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        pde_features = self.fourier(x_raw)
        return pde_features


if __name__ == '__main__':
    x = torch.randn((1, 12, 2, 128, 128))
    y = torch.randn((1, 12, 2, 128, 128))
    model1 = Fourier_Model(shape_in=(12,2,128,128))
    output = model1(x)
    print("input shape:", x.shape)
    print("output shape:", output.shape)

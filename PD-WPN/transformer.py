import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from scipy.optimize import fsolve
import torch.nn.init as init


class SpaceTimeTransformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        d_model = configs.d_model
        self.device = configs.device
        self.input_dim = configs.input_dim
        self.kernal = configs.patch_size[0] * configs.patch_size[1]
        self.src_emb = input_embedding(self.input_dim, self.kernal, d_model, configs.emb_spatial_size,
                                       configs.input_length, self.device)
        self.tgt_emb = input_embedding(self.input_dim, self.kernal, d_model, configs.emb_spatial_size,
                                       configs.output_length, self.device)

        encoder_layer = EncoderLayer(d_model, configs.nheads, configs.dim_feedforward, configs.dropout)
        decoder_layer = DecoderLayer(d_model, configs.nheads, configs.dim_feedforward, configs.dropout,configs)
        self.encoder = Encoder(encoder_layer, num_layers=configs.num_encoder_layers)
        self.decoder = Decoder(decoder_layer, num_layers=configs.num_decoder_layers)
        self.linear_output = nn.Linear(d_model, self.kernal)

    def forward(self, src, tgt, src_mask=None, memory_mask=None,
                train=True, ssr_ratio=0, heights = [10, 100]):
        """
        Args:
            src: (N, T_src, C, H, W)
            tgt: (N, T_tgt, C, H, W), T_tgt is 1 during test
            src_mask: (T_src, T_src)
            tgt_mask: (T_tgt, T_tgt)
            memory_mask: (T_tgt, T_src)
        Returns:
            sst_pred: (N, T_tgt, C, H, W)
            nino_pred: (N, 24)
        """
        truth = tgt
        memory = self.encode(src, src_mask)
        if train:
            with torch.no_grad():
                # tgt = torch.cat([src[:, -1:], tgt[:, :-1]], dim=1)  # (N, T_tgt, C, H, W)
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
                sst_pred = self.decode(torch.cat([src[:, -1:], tgt[:, :-1]], dim=1),
                                       memory, tgt_mask, memory_mask)  # (N, T_tgt, C, H, W)
            if ssr_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(ssr_ratio *
                        torch.ones(tgt.size(0), tgt.size(1) - 1, 1, 1, 1)).to(self.device)
            else:
                teacher_forcing_mask = 0
            # print(tgt.shape,tgt[:, :-1].shape,sst_pred.shape,sst_pred[:, :-1].shape)
            tgt = teacher_forcing_mask * tgt[:, :-1] + (1 - teacher_forcing_mask) * sst_pred[:, :-1]
            tgt = torch.cat([src[:, -1:], tgt], dim=1)
            sst_pred = self.decode(tgt, memory, tgt_mask, memory_mask)
        else:
            if tgt is None:
                tgt = src[:, -1:]  # use last src as the input during test
            else:
                assert tgt.size(1) == 1
            for t in range(self.configs.output_length):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
                sst_pred = self.decode(tgt, memory, tgt_mask, memory_mask)
                tgt = torch.cat([tgt, sst_pred[:, -1:]], dim=1)

            for h, height in enumerate(np.array(heights)):
                if height==10:
                    result = sst_pred[:,:,:2]
                elif height==100:
                    result = sst_pred[:,:,2:]
                else:
                    result = calculate_wind_fields(sst_pred, height, kappa=0.4)

                if h==0:
                    r = result
                else:
                    r = torch.cat([r,result],2)
            sst_pred = r
        return sst_pred

    def encode(self, src, src_mask):
        """
        Args:
            src: (N, T_src, C, H, W)
            src_mask: (T_src, T_src)
        Returns:
            memory: (N, S, C, T_src, D)
        """
        T = src.size(1)
        src = unfold_StackOverChannel(src, self.configs.patch_size)  # (N, T_src, C, H_k*N_k, H_output, W_output)
        src = src.reshape(src.size(0), T, self.input_dim, self.kernal, -1).permute(0, 4, 2, 1,3)  # (N, S, C, T_src, C_)
        src = self.src_emb(src)  # (N, S, C ,T_src, D)
        memory = self.encoder(src, src_mask)  # (N, S, C, T_src, D)
        return memory

    def decode(self, tgt, memory, tgt_mask, memory_mask):
        """
        Args:
            tgt: (N, T_tgt, C, H, W)
            memory: (N, S, C, T_src, D)
            tgt_mask: (T_tgt, T_tgt)
            memory_mask: (T_tgt, T_src)
        Returns:
            (N, T_tgt, C, H, W)
        """
        H, W = tgt.size()[-2:]
        T = tgt.size(1)
        tgt = unfold_StackOverChannel(tgt, self.configs.patch_size)  # (N, T_tgt, C, C_, H_, W_) / (N, T_src, C, H_k*N_k, H_output, W_output)
        tgt = tgt.reshape(tgt.size(0), T, self.input_dim, self.kernal, -1).permute(0, 4, 2, 1,3)  # (N, S, C, T_tgt, C_)
        tgt = self.tgt_emb(tgt)  # (N, S, C, T_tgt, D)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        output = self.linear_output(output).permute(0, 3, 2, 4, 1)  # (N, T_tgt, C, C_, S)

        # (N, T_tgt, C, C_, H_, W_)
        output = output.reshape(tgt.size(0), T, self.input_dim, self.kernal,
                                H // self.configs.patch_size[0], W // self.configs.patch_size[1])
        # (N, T_tgt, C, H, W)
        output = fold_tensor(output, output_size=(H, W), kernel_size=self.configs.patch_size)



        return output

    def generate_square_subsequent_mask(self, sz: int):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf')
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 0).T
        return mask.to(self.configs.device)


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, x, memory, tgt_mask, memory_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout):
        super().__init__()
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.time_attn = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout)
        self.channel_attn = MultiHeadedAttention(d_model, nheads, ChannelAttention, dropout)
        self.space_attn = MultiHeadedAttention(d_model, nheads, SpaceAttention, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )

    def divided_space_time_attn(self, query, key, value, mask):
        """
        Apply space and time attention sequentially
        Args:
            query (N, S, C, T, D)
            key (N, S, C, T, D)
            value (N, S, C, T, D)
        Returns:
            (N, S, C, T, D)
        """
        m = self.time_attn(query, key, value, mask)
        n = self.channel_attn(m, m, m, mask)+m
        return self.space_attn(n, n, n, mask)+n

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.divided_space_time_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout,configs):
        super().__init__()
        self.sublayer = clones(SublayerConnection(d_model, dropout), 4)
        self.encoder_attn = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout)
        self.time_attn = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout)
        self.channel_attn = MultiHeadedAttention(d_model, nheads, ChannelAttention, dropout)
        self.space_attn = MultiHeadedAttention(d_model, nheads, SpaceAttention, dropout)
        self.time_attn_2 = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout)
        self.channel_attn_2 = MultiHeadedAttention(d_model, nheads, ChannelAttention, dropout)
        self.space_attn_2 = MultiHeadedAttention(d_model, nheads, SpaceAttention, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )
        self.phycell = PhyCell(input_shape=(100, 256), input_dim=4, F_hidden_dims=[49], n_layers=1, kernel_size=(7, 7),device=configs.device)
        self.param1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.param2 = nn.Parameter(torch.zeros(1, requires_grad=True))

    def divided_space_time_attn(self, query, key, value, mask=None):
        """
               Apply space and time attention sequentially
               Args:
                   query (N, S, C, T, D)
                   key (N, S, C, T, D)
                   value (N, S, C, T, D)
               Returns:
                   (N, S, C, T, D)
        """
        m = self.time_attn(query, key, value, mask)
        n = self.channel_attn(m, m, m, mask)
        return self.space_attn(n, n, n, mask)

    def divided_space_time_attn_2(self, query, key, value, mask=None):
        """
               Apply space and time attention sequentially
               Args:
                   query (N, S, C, T, D)
                   key (N, S, C, T, D)
                   value (N, S, C, T, D)
               Returns:
                   (N, S, C, T, D)
        """
        m = self.time_attn_2(query, key, value, mask)
        n = self.channel_attn_2(m, m, m, mask)+m
        return self.space_attn_2(n, n, n, mask)+n

    def phy(self, x):
        """
               Apply space and time attention sequentially
               Args:
                   x (N, S, C, T, D)
               Returns:
                   (N, S, C, T, D)
        """

        x = x.permute(0,3,2,1,4) # (N, T, C, S, D)
        for i in range(x.shape[1]):
            if i==0:
                temp = self.phycell(x[:, i], first_timestep=True)[-1][:, None]
            else:
                temp = self.phycell(x[:, i], first_timestep=False)[-1][:, None]
            r = temp if i == 0 else torch.cat([r, temp], 1)
        r = r.permute(0,3,2,1,4)
        return r

    def forward(self, x, memory, tgt_mask, memory_mask):
        x0 = self.sublayer[0](x, lambda x: self.divided_space_time_attn(x, x, x, tgt_mask))
        x0 = self.sublayer[1](x0, lambda x0: self.divided_space_time_attn_2(x0, memory, memory, memory_mask))
        x1 = self.sublayer[2](x, lambda x: self.phy(x))
        return self.sublayer[3](self.param1*x0+self.param2*x1, self.feed_forward)

def unfold_StackOverChannel(img, kernel_size):
    """
    divide the original image to patches, then stack the grids in each patch along the channels
    Args:
        img (N, *, C, H, W): the last two dimensions must be the spatial dimension
        kernel_size: tuple of length 2
    Returns:
        output (N, T_src, C, H_k*N_k, H_output, W_output)
    """
    n_dim = len(img.size())
    assert n_dim == 4 or n_dim == 5

    pt = img.unfold(-2, size=kernel_size[0], step=kernel_size[0])
    pt = pt.unfold(-2, size=kernel_size[1], step=kernel_size[1]).flatten(-2)  # (N, *, C, n0, n1, k0*k1)
    # if n_dim == 4:  # (N, C, H, W)
    #     pt = pt.permute(0, 1, 4, 2, 3).flatten(1, 2)
    # elif n_dim == 5:  # (N, T, C, H, W)
    #     pt = pt.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    if n_dim == 4:  # (N, C, H, W)
        pt = pt.permute(0, 1, 4, 2, 3)
    elif n_dim == 5:  # (N, T, C, H, W)
        pt = pt.permute(0, 1, 2, 5, 3, 4) # (N, T_src, C, H_k*N_k, H_output, W_output)
    assert pt.size(-5) == img.size(-4)
    assert pt.size(-3) == kernel_size[0] * kernel_size[1]
    return pt


def fold_tensor(tensor, output_size, kernel_size):
    """
    reconstruct the image from its non-overlapping patches
    Args:
        input tensor of size (N, *, C, k_h*k_w, n_h, n_w)
        output_size of size(H, W), the size of the original image to be reconstructed
        kernel_size: (k_h, k_w)
        stride is usually equal to kernel_size for non-overlapping sliding window
    Returns:
        (N, *, C, H=n_h*k_h, W=n_w*k_w)
    """
    tensor = tensor.float()
    n_dim = len(tensor.size())
    assert n_dim == 4 or n_dim == 6
    f = tensor.flatten(0, 2) if n_dim == 6 else tensor
    folded = F.fold(f.flatten(-2), output_size=output_size, kernel_size=kernel_size, stride=kernel_size)
    if n_dim == 6:
        folded = folded.reshape(tensor.size(0), tensor.size(1), tensor.size(2), *folded.size()[1:]).squeeze(-3)
    return folded


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class input_embedding(nn.Module):
    def __init__(self, input_dim, kernal, d_model, emb_spatial_size, max_len, device):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe_time = pe[None, None, None] .to(device)  # (1, 1, 1, T, D)

        self.spatial_pos = torch.arange(emb_spatial_size)[None, :, None, None].to(device)
        self.emb_space = nn.Embedding(emb_spatial_size, d_model)

        self.channel_pos = torch.arange(input_dim)[None, None, :, None].to(device)
        self.emb_channel = nn.Embedding(input_dim, d_model)

        self.linear = nn.Linear(kernal, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Add temporal encoding and learnable spatial embedding to the input (after patch)
        Args:
            input x of size (N, S, C, T_src, C_)
        Returns:
            embedded input (N, S, C, T, D)
        """
        assert len(x.size()) == 5
        embedded_space = self.emb_space(self.spatial_pos) # (1, S, 1, 1, D)
        embedded_channel = self.emb_channel(self.channel_pos)  # (1, S, 1, 1, D)
        x = self.linear(x) + self.pe_time[:, :, :, :x.size(3)] + embedded_channel + embedded_space  # (N, S, C, T, D)
        # x[50, 192, 3, 12, 256] = [50, 192, 3, 12, 256] + [1, 1, 1, 12, 256] + [1, 1, 3, 1, 256] + [1, 192, 1, 1, 256]
        return self.norm(x)


def TimeAttention(query, key, value, mask=None, dropout=None):
    """
    attention over the time axis
    Args:
        query, key, value: linearly-transformed query, key, value (N, h, S, C, T, D)
        mask: of size (T (query), T (key)) specifying locations (which key) the query can and cannot attend to
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, S, C, T, T)
    if mask is not None:
        assert mask.dtype == torch.bool
        assert len(mask.size()) == 2
        scores = scores.masked_fill(mask[None, None, None, None], float("-inf"))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)  # (N, h, S, C, T, D)

def ChannelAttention(query, key, value, mask=None, dropout=None):
    """
    attention over the two Channel axes
    Args:
        query, key, value: linearly-transformed query, key, value (N, h, S, C, T, D)
        mask: None (space attention does not need mask), this argument is intentionally set for consistency
    """
    d_k = query.size(-1)
    query = query.transpose(3, 4)  # (N, h, S, T, C, D)
    key = key.transpose(3, 4)  # (N, h, S, T, C, D)
    value = value.transpose(3, 4)  # (N, h, S, T, C, D)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, S, T, C, C)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).transpose(3, 4)  # (N, h, S, C, T, D)

def SpaceAttention(query, key, value, mask=None, dropout=None):
    """
    attention over the two space axes
    Args:
        query, key, value: linearly-transformed query, key, value (N, h, S, C, T, d_k)
        mask: None (Channel attention does not need mask), this argument is intentionally set for consistency
    """
    d_k = query.size(-1)
    query = query.transpose(2, 4)  # (N, h, T, C, S, D)
    key = key.transpose(2, 4)  # (N, h, T, C, S, D)
    value = value.transpose(2, 4)  # (N, h, T, C, S, D)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, T, C, S, S)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).transpose(2, 4)  # (N, h, S, C, T_q, D)

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, nheads, attn, dropout):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.attn = attn

    def forward(self, query, key, value, mask=None):
        """
        Transform the query, key, value into different heads, then apply the attention in parallel
        Args:
            query, key, value: size (N, S, C, T, D)
        Returns:
            (N, S, C, T, D)
        """
        nbatches = query.size(0)
        nspace = query.size(1)
        nchannel = query.size(2)
        ntime = query.size(3)
        # (N, h, S, C, T, d_k)
        query, key, value = \
            [l(x).view(x.size(0), x.size(1), x.size(2), x.size(3), self.nheads, self.d_k).permute(0, 4, 1, 2, 3, 5)
             for l, x in zip(self.linears, (query, key, value))]

        # (N, h, S, C, T, d_k)
        x = self.attn(query, key, value, mask=mask, dropout=self.dropout)

        # (N, S, C, T, D)
        x = x.permute(0, 2, 3, 4, 1, 5).contiguous() \
             .view(nbatches, nspace, nchannel, ntime, self.nheads * self.d_k)
        return self.linears[-1](x)

class GradientConvLayer(nn.Module):
    def __init__(self):
        super(GradientConvLayer, self).__init__()
        # 定义水平梯度卷积核 (G_x)，尺寸为 (C, C, 1, 2)，即每个通道的卷积核
        self.G_x = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), padding=(0, 1), bias=False)
        # 定义垂直梯度卷积核 (G_y)，尺寸为 (C, C, 2, 1)
        self.G_y = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 1), padding=(1, 0), bias=False)

        # 使用 Xavier 初始化来避免权重过大导致梯度爆炸
        init.xavier_normal_(self.G_x.weight)
        init.xavier_normal_(self.G_y.weight)

        # 采用批归一化来稳定训练
        self.batch_norm = nn.BatchNorm2d(4)

    def forward(self, x):
        # 水平梯度计算 (∇x h_t)
        nabla_x_h_t = self.G_x(x)  # 水平卷积，padding=(0,1) 处理边缘
        # 垂直梯度计算 (∇y h_t)
        nabla_y_h_t = self.G_y(x)  # 垂直卷积，padding=(1,0) 处理边缘
        # 计算整体空间梯度 Ω(∇h_t)，使用L2范数
        Omega_nabla_h_t = torch.sqrt(nabla_x_h_t**2 + nabla_y_h_t**2 + 1e-12)
        Omega_nabla_h_t = self.batch_norm(Omega_nabla_h_t)

        return Omega_nabla_h_t

class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.F = nn.Sequential()
        self.F.add_module('conv1',
                          nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size,
                                    stride=(1, 1), padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(7, F_hidden_dim))
        self.F.add_module('conv2',
                          nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                  out_channels=self.input_dim,
                                  kernel_size=(3, 3),
                                  padding=(1, 1), bias=self.bias)

        self.gradient_layer = GradientConvLayer()

    def forward(self, x, hidden):  # x [batch_size, hidden_dim, height, width]
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden) + self.gradient_layer(hidden)  # prediction
        # hidden_tilde = hidden + self.F(hidden)
        next_hidden = hidden_tilde + K * (x - hidden_tilde)  # correction , Haddamard product
        return next_hidden


class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []
        self.device = device

        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):  # input_ [batch_size, 1, channels, width, height]

        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)  # init Hidden at each forward start

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j - 1], self.H[j])

        return self.H

    def initHidden(self, batch_size):
        self.H = []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, H):
        self.H = H



def solve_z0_closed(U10, U100, z10=10, z100=100, default_z0=0.1):
    ratio = U100 / (U10 + 1e-6)
    near_one = np.abs(ratio - 1.0) < 0.05  # 更合理地放宽判定 (5%范围)

    L10 = np.log(z10)
    L100 = np.log(z100)
    safe_ratio = np.where(near_one, 1.0 + 1e-3, ratio)  # 避免除0
    delta = safe_ratio - 1.0
    exponent = (L100 - safe_ratio * L10) / np.where(np.abs(delta) < 1e-3, 1e-3, delta)
    exponent = np.clip(exponent, -50, 50)

    z0 = np.exp(exponent)
    z0 = np.where(near_one, default_z0, z0)
    return np.clip(z0, 1e-3, z10 - 1e-3)

# === 幂律剖面 fallback ===
def power_law_interpolation(U10, U100, z=50, z10=10, z100=100, default_alpha=0.2):
    ratio = U100 / (U10 + 1e-6)
    U50 = np.full_like(U10, fill_value=0.0)
    valid = (ratio > 0) & np.isfinite(ratio)

    alpha = np.full_like(U10, fill_value=default_alpha)
    safe_ratio = np.where(valid, ratio, 1.0)
    log_safe_ratio = np.log(safe_ratio)
    log_z = np.log(z100 / z10 + 1e-6)
    alpha[valid] = log_safe_ratio[valid] / log_z

    U50[valid] = U10[valid] * (z / z10) ** alpha[valid]
    U50[~valid] = U10[~valid] + (U100[~valid] - U10[~valid]) * (z - z10) / (z100 - z10)

    return U50

# === 对数风速公式 ===
def calculate_wind_speed(U0, z, z0, kappa=0.4):
    z0 = np.clip(z0, 1e-6, z - 1e-3)
    return (U0 / kappa) * np.log(z / z0)

# === 主要计算函数（合理 fallback，10%容忍度）===
def calculate_wind_fields(UV_data, height, kappa=0.4):
    assert UV_data.shape[2] == 4, "Expected 4 channels: U10, V10, U100, V100"
    B, T, _, H, W = UV_data.shape
    wind_speeds = np.zeros((B, T, 2, H, W))

    fallback_count = 0
    total_count = 0
    threshold = 0.1  # 允许10%的浮动容忍

    for b in range(B):
        for t in range(T):
            U10 = UV_data[b, t, 0]
            V10 = UV_data[b, t, 1]
            U100 = UV_data[b, t, 2]
            V100 = UV_data[b, t, 3]

            z0_U = solve_z0_closed(U10, U100)
            z0_V = solve_z0_closed(V10, V100)

            U_target = calculate_wind_speed(U10, height, z0_U, kappa)
            V_target = calculate_wind_speed(V10, height, z0_V, kappa)

            U_target_fallback = power_law_interpolation(U10, U100, z=height)
            V_target_fallback = power_law_interpolation(V10, V100, z=height)

            # 按高度分段处理 + 10% 容忍度
            if height < 10:
                U_invalid = (U_target > (1 + threshold) * U10)
                V_invalid = (V_target > (1 + threshold) * V10)
            elif height <= 100:
                U_invalid = (U_target > (1 + threshold) * np.maximum(U10, U100)) | (U_target < (1 - threshold) * np.minimum(U10, U100))
                V_invalid = (V_target > (1 + threshold) * np.maximum(V10, V100)) | (V_target < (1 - threshold) * np.minimum(V10, V100))
            else:  # height > 100
                U_invalid = (U_target < (1 - threshold) * U100)
                V_invalid = (V_target < (1 - threshold) * V100)

            fallback_count += np.sum(U_invalid) + np.sum(V_invalid)
            total_count += U_target.size + V_target.size

            U_target[U_invalid] = U_target_fallback[U_invalid]
            V_target[V_invalid] = V_target_fallback[V_invalid]

            wind_speeds[b, t, 0] = U_target
            wind_speeds[b, t, 1] = V_target

    if total_count > 0:
        fallback_rate = fallback_count / total_count * 100
        print(f"Fallback used at {fallback_count}/{total_count} points ({fallback_rate:.2f}%)")

    return wind_speeds
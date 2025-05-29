import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import MAUCell
from config import configs


class MAU_Model(nn.Module):
    r"""MAU Model

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, configs):
        super(MAU_Model, self).__init__()
        self.configs = configs
        self.device = configs.device
        C, H, W = configs.in_shape
        num_layers, num_hidden = configs.num_layers, configs.num_hidden

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs.tau
        self.cell_mode = configs.cell_mode
        self.states = ['recall', 'normal']
        if not self.configs.model_mode in self.states:
            raise AssertionError
        cell_list = []

        width = W // configs.patch_size // configs.sr_size
        height = H // configs.patch_size // configs.sr_size

        self.h, self.w = H // configs.patch_size, W // configs.patch_size

        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(
                MAUCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                        configs.stride, self.tau, self.cell_mode)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(configs.sr_size))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []

        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.merge = nn.Conv2d(
            self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        self.conv_last_sr = nn.Conv2d(
            self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, input, target, teacher_forcing=False, scheduled_sampling_ratio=0, train=True):

        frames_tensor = torch.cat([input, target], 1).to(self.configs.device)
        device = frames_tensor.device

        B, T, C, H, W = frames_tensor.shape
        frames_tensor = F.unfold(frames_tensor.view(B * T, C, H, W), kernel_size=self.configs.patch_size,
                                 stride=self.configs.unfoldstride)
        frames_tensor = frames_tensor.view(B, T, C * self.configs.patch_size * self.configs.patch_size, self.h, self.w)

        frames = frames_tensor.contiguous()

        batch_size = frames.shape[0]
        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        frame_channels = frames.shape[2]
        next_frames = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None
        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros(
                    [batch_size, in_channel, height, width]).to(device))
                tmp_s.append(torch.zeros(
                    [batch_size, in_channel, height, width]).to(device))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)

        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                mask_true = torch.bernoulli(scheduled_sampling_ratio *torch.ones(frames.size(0), self.configs.aft_seq_length - 1, 1, 1, 1)).to(self.device)
            else:
                teacher_forcing = False
        else:
            teacher_forcing = False

        for t in range(self.configs.total_length - 1):

            if t < self.configs.pre_seq_length:
                net = frames[:, t]
            elif not teacher_forcing:
                net = x_gen
            else:
                time_diff = t - self.configs.pre_seq_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen

            frames_feature = net
            frames_feature_encoded = []
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros(
                        [batch_size, self.num_hidden[i], height, width]).to(device)
                    T_t.append(zeros)
            S_t = frames_feature
            for i in range(self.num_layers):
                t_att = T_pre[i][-self.tau:]
                t_att = torch.stack(t_att, dim=0)
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0)
                S_pre[i].append(S_t)
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
                T_pre[i].append(T_t[i])
            out = S_t

            for i in range(len(self.decoders)):
                out = self.decoders[i](out)
                if self.configs.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]

            x_gen = self.srcnn(out)
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        next_frames = F.fold(next_frames.view(B * (T - 1), C * self.configs.patch_size * self.configs.patch_size, self.h * self.w),output_size=(H, W), kernel_size=self.configs.patch_size, stride=configs.unfoldstride)
        next_frames = next_frames.view(B, T - 1, C, H, W)

        if train:
            next_frames = next_frames
        else:
            next_frames = next_frames[:,-self.configs.aft_seq_length:]

        return next_frames

if __name__ == '__main__':
    x = torch.randn((1, 24, 4, 40, 40)).to(configs.device)
    y = torch.randn((1, 24, 4, 40, 40)).to(configs.device)
    model1 = MAU_Model(configs).to(configs.device)
    output = model1(x,y,teacher_forcing=True, scheduled_sampling_ratio=0.5, train=False)
    print("input shape:", x.shape)
    print("output shape:", output.shape)
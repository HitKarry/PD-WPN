import torch
import torch.nn as nn
from modules import Eidetic3DLSTMCell
from config import configs
import torch.nn.functional as F


class E3DLSTM_Model(nn.Module):
    r"""E3D-LSTM Model

    Implementation of `EEidetic 3D LSTM: A Model for Video Prediction and Beyond
    <https://openreview.net/forum?id=B1lKS2AqtX>`_.

    """

    def __init__(self, configs):
        super(E3DLSTM_Model, self).__init__()
        C, H, W = configs.in_shape
        num_layers, num_hidden = configs.num_layers, configs.num_hidden
        self.configs = configs
        self.device = configs.device
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        self.window_length = 2
        self.window_stride = 1

        height = H // configs.patch_size
        width = W // configs.patch_size
        self.h, self.w = height, width
        self.MSE_criterion = nn.MSELoss()
        self.L1_criterion = nn.L1Loss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                Eidetic3DLSTMCell(in_channel, num_hidden[i],
                                  self.window_length, height, width, (2, 5, 5),
                                  configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv3d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=(self.window_length, 1, 1),
                                   stride=(self.window_length, 1, 1), padding=0, bias=False)

    def forward(self, input, target, teacher_forcing=False, scheduled_sampling_ratio=0, train=True):

        frames_tensor = torch.cat([input, target],1).to(self.configs.device)
        device = frames_tensor.device

        B, T, C, H, W = frames_tensor.shape
        frames_tensor = F.unfold(frames_tensor.view(B * T, C, H, W), kernel_size=self.configs.patch_size, stride=self.configs.stride)
        frames_tensor = frames_tensor.view(B, T, C * self.configs.patch_size * self.configs.patch_size, self.h, self.w)

        frames = frames_tensor.contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        c_history = []
        input_list = []

        for t in range(self.window_length - 1):
            input_list.append(
                torch.zeros_like(frames[:, 0]))

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], self.window_length, height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)
            c_history.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], self.window_length, height, width], device=device)

        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                mask_true = torch.bernoulli(scheduled_sampling_ratio *
                    torch.ones(frames.size(0), self.configs.aft_seq_length - 1, 1, 1, 1)).to(self.device)
            else:
                teacher_forcing = False
        else:
            teacher_forcing = False

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):

            if t < self.configs.pre_seq_length:
                net = frames[:, t]
            elif not teacher_forcing:
                net = x_gen
            else:
                net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            input_list.append(net)

            if t % (self.window_length - self.window_stride) == 0:
                net = torch.stack(input_list[t:], dim=0)
                net = net.permute(1, 2, 0, 3, 4).contiguous()

            for i in range(self.num_layers):
                if t == 0:
                    c_history[i] = c_t[i]
                else:
                    c_history[i] = torch.cat((c_history[i], c_t[i]), 1)
                
                input = net if i == 0 else h_t[i-1]
                h_t[i], c_t[i], memory = self.cell_list[i](input, h_t[i], c_t[i], memory, c_history[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1]).squeeze(2)
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        next_frames = F.fold(next_frames.view(B * (T - 1), C * self.configs.patch_size * self.configs.patch_size, self.h*self.w),output_size=(H, W), kernel_size=self.configs.patch_size, stride=self.configs.stride)
        next_frames = next_frames.view(B,T-1,C,H,W)

        if train:
            next_frames = next_frames
        else:
            next_frames = next_frames[:, -self.configs.aft_seq_length:]

        return next_frames

if __name__ == '__main__':
    x = torch.randn((1, 24, 4, 40, 40)).to(configs.device)
    y = torch.randn((1, 24, 4, 40, 40)).to(configs.device)
    model1 = E3DLSTM_Model(configs).to(configs.device)
    output = model1(x,y,teacher_forcing=True, scheduled_sampling_ratio=0.5, train=True)
    print("input shape:", x.shape)
    print("output shape:", output.shape)
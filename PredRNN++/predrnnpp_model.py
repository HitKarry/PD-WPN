import torch
import torch.nn as nn
from config import configs
import torch.nn.functional as F
from modules import CausalLSTMCell, GHU


class PredRNNpp_Model(nn.Module):
    r"""PredRNN++ Model

    Implementation of `PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma
    in Spatiotemporal Predictive Learning <https://arxiv.org/abs/1804.06300>`_.

    """

    def __init__(self, configs):
        super(PredRNNpp_Model, self).__init__()

        C, H, W = configs.in_shape
        num_layers, num_hidden = configs.num_layers, configs.num_hidden

        self.configs = configs
        self.device = configs.device
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs.patch_size
        width = W // configs.patch_size

        self.h, self.w = height, width
        
        self.gradient_highway = GHU(num_hidden[0], num_hidden[0], height, width,
                                    configs.filter_size, configs.stride, configs.layer_norm)

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                CausalLSTMCell(in_channel, num_hidden[i], height, width,
                               configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input, target, teacher_forcing=False, scheduled_sampling_ratio=0, train=True):

        frames_tensor = torch.cat([input, target], 1).to(self.configs.device)
        device = frames_tensor.device

        B, T, C, H, W = frames_tensor.shape
        frames_tensor = F.unfold(frames_tensor.view(B * T, C, H, W), kernel_size=self.configs.patch_size,
                                 stride=self.configs.unfoldstride)
        frames_tensor = frames_tensor.view(B, T, C * self.configs.patch_size * self.configs.patch_size, self.h, self.w)

        frames = frames_tensor.contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width], device=device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], height, width], device=device)
        z_t = None

        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                mask_true = torch.bernoulli(scheduled_sampling_ratio *torch.ones(frames.size(0), self.configs.aft_seq_length - 1, 1, 1, 1)).to(self.device)
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

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            z_t = self.gradient_highway(h_t[0], z_t)
            h_t[1], c_t[1], memory = self.cell_list[1](z_t, h_t[1], c_t[1], memory)

            for i in range(2, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        next_frames = F.fold(
            next_frames.view(B * (T - 1), C * self.configs.patch_size * self.configs.patch_size, self.h * self.w),
            output_size=(H, W), kernel_size=self.configs.patch_size, stride=self.configs.unfoldstride)
        next_frames = next_frames.view(B, T - 1, C, H, W)

        if train:
            next_frames = next_frames
        else:
            next_frames = next_frames[:, -self.configs.aft_seq_length:]

        return next_frames

if __name__ == '__main__':
    x = torch.randn((1, 24, 4, 40, 40)).to(configs.device)
    y = torch.randn((1, 24, 4, 40, 40)).to(configs.device)
    model1 = PredRNNpp_Model(configs).to(configs.device)
    output = model1(x,y,teacher_forcing=True, scheduled_sampling_ratio=0.5, train=True)
    print("input shape:", x.shape)
    print("output shape:", output.shape)
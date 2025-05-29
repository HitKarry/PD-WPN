import torch
import torch.nn as nn
from config import configs

# class SimVP(nn.Module):
#     def __init__(self, configs):
#         super(SimVP, self).__init__()
#         self.configs = configs
#         self.model = nn.Sequential(
#             nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, padding=1),  # (B, 24, 8, 40, 40)
#             nn.ReLU(),
#             nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, padding=1),  # 回到原通道数
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         x = x.to(self.configs.device)
#         # x shape: (B, 24, 4, 40, 40)
#         # 交换时间和通道维度以适配 Conv3d
#         x = x.permute(0, 2, 1, 3, 4)  # (B, 4, 24, 40, 40)
#         x = self.model(x)  # (B, 4, 24, 40, 40)
#         x = x.permute(0, 2, 1, 3, 4)  # (B, 24, 4, 40, 40)
#         return x

class SimVP(nn.Module):
    def __init__(self, configs):
        super(SimVP, self).__init__()
        self.configs = configs
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, padding=1),  # 回到 4 通道
        )

    def forward(self, x):
        # 输入 x:
        B, T, C, H, W = x.shape
        y = x.view(B * T, C, H, W).to(self.configs.device)  # 合并 batch 和时间维度 -> (B*T, 4, 40, 40)
        y = self.conv_block(y)  # 卷积处理
        y = y.view(B, T, 4, H, W)  # 恢复原状
        y = y.mean(dim=1, keepdim=True)

        return y
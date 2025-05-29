import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:1')
configs.batch_size_test = 4
configs.batch_size = 4
configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 520
configs.num_epochs = 100
configs.early_stopping = True
configs.patience = 20
configs.gradient_clipping = True
configs.clipping_threshold = 1.

# data related
configs.input_dim = 4
configs.input_length = 24
configs.input_h = 40
configs.input_w = 40
configs.input_gap = 1
configs.pred_shift = 24

# model related
configs.kernel_size = (3, 3)
configs.bias = True
configs.hidden_dim = (32, 48, 48, 32)
configs.d_attn = 32
configs.ssr_decay_rate = 0.8e-4

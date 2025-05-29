import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 1
configs.device = torch.device('cuda:0')
configs.batch_size_test = 8
configs.batch_size = 8
configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 260
configs.num_epochs = 100
configs.early_stopping = True
configs.patience = 20
configs.gradient_clipping = True
configs.clipping_threshold = 1.


# data related
configs.input_dim = 4
configs.output_dim = 4

configs.input_length = 24
configs.output_length = 24

configs.input_gap = 1
configs.pred_shift = 24

# model related
configs.kernel_size = (3, 3)
configs.bias = True
configs.steps = 3
assert configs.steps < configs.input_length

configs.order = 2
assert configs.order <= configs.steps

configs.ranks = 8

configs.layers_per_block = (3, 3, 3, 3)
configs.hidden_channels = (32, 48, 48, 32)
assert len(configs.layers_per_block) == len(configs.hidden_channels)

configs.skip_stride = 2
configs.ssr_decay_rate = 2.e-4

import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:1')
configs.batch_size = 8
configs.batch_size_test = 8
configs.lr = 0.0005
configs.weight_decay = 0
configs.display_interval = 260
configs.num_epochs = 1000
configs.early_stopping = True
configs.patience = 20
configs.gradient_clipping = False
configs.clipping_threshold = 1.


# data related
configs.pre_seq_length = 24
configs.aft_seq_length = 24
configs.total_length = configs.pre_seq_length + configs.aft_seq_length

# model
configs.ssr_decay_rate = 1.e-4
configs.in_shape = (4,40,40)
configs.num_layers = 4
configs.num_hidden = [128,128,128,128]
configs.sr_size = 2
configs.tau = 5
configs.cell_mode = 'normal'
configs.model_mode = 'normal'

configs.filter_size = 5
configs.patch_size = 4
configs.unfoldstride = configs.patch_size
configs.stride = 1
configs.layer_norm = 0
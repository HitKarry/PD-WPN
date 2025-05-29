import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:1')
configs.batch_size_test = 16
configs.batch_size = 16
configs.lr = 0.005
configs.weight_decay = 0
configs.display_interval = 130
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
configs.shape_in = (24,4,40,40)
configs.hid_S = 64
configs.hid_T = 256
configs.N_S = 4
configs.N_T = 8
configs.incep_ker = [3,5,7,11]
configs.groups = 4

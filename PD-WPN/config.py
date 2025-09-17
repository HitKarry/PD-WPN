import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:1')
configs.batch_size = 4
configs.batch_size_test = 4
configs.lr = 1.e-5
configs.weight_decay = 0
configs.display_interval = 520
configs.num_epochs = 300
configs.patience = 30
configs.reg_epoch = 50
configs.early_stopping = True
configs.gradient_clipping = False
configs.clipping_threshold = 1.

# lr warmup
configs.warmup = 3000

# data related
configs.input_dim = 4
configs.output_dim = 4

configs.input_length = int(1 * 24 / 1)
configs.output_length = int(1 * 24 / 1)

configs.input_gap = 1
configs.pred_shift = 24

# model
configs.d_model = 256
configs.patch_size = (4, 4)
configs.emb_spatial_size = 10*10
configs.nheads = 4
configs.dim_feedforward = 512
configs.dropout = 0.2
configs.num_encoder_layers = 3
configs.num_decoder_layers = 3

configs.ssr_decay_rate = 5.e-5

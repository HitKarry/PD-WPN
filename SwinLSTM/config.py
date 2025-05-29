import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:0')
configs.batch_size = 4
configs.batch_size_test = 4
configs.lr = 0.0003
configs.weight_decay = 0
configs.display_interval = 520
configs.num_epochs = 1000
configs.early_stopping = True
configs.patience = 1000
configs.gradient_clipping = False
configs.clipping_threshold = 1.


configs.input_dim = 4
configs.output_length = int(1 * 24 / 1)
configs.num_layers = [1,2]

configs.img_size = (64,64)
configs.patch_size = 2
configs.in_chans = 4
configs.embed_dim = 128
configs.depths_downsample = [2, 6]
configs.depths_upsample = [6, 2]
configs.num_heads = [4, 8]
configs.window_size = 2


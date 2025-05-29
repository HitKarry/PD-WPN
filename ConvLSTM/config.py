import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:0')
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



configs.input_dim = 4
configs.output_dim = 4
configs.num_layers = 2
configs.hidden_dim = [128,configs.output_dim]
configs.kernel_size = (3,3)
configs.input_length = int(1 * 24 / 1)
configs.output_length = int(1 * 24 / 1)



import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:0')
configs.batch_size = 8
configs.batch_size_test = 8
configs.lr = 0.001
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


# model
configs.ssr_decay_rate = 5.e-5
# # reverse scheduled sampling
# r_sampling_step_1 = 25000
# r_sampling_step_2 = 50000
# r_exp_alpha = 5000
# # scheduled sampling
# scheduled_sampling = 1
# sampling_stop_iter = 50000
# sampling_start_value = 1.0
# sampling_changing_rate = 0.00002
# # model
configs.in_shape = (4,40,40)
configs.num_layers = 4
configs.num_hidden = [128,128,128,128]
# filter_size = 5
configs.patch_size = 4
configs.stride = configs.patch_size
configs.layer_norm = 0
# # training
# lr = 5e-3
# batch_size = 16
# sched = 'cosine'
# warmup_epoch = 5

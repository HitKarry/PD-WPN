import numpy as np
from torch.utils.data import Dataset
from convttlstm_net import ConvTTLSTMNet
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
import math
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau



if __name__ == '__main__':
    print('Configs:\n', configs.__dict__)

    data = np.load('../wind_data/1_wind_data_45n_35n_90e_100e_slide.npz')
    data_x, data_y, mean, std = data['data_x'],data['data_y'],data['mean'],data['std']
    
    test_x = np.concatenate([data_x[8406:],data_y[8406:]],1)
    test_y = data_y[8406:]
    
    dataset_test = dataset_package(train_x=test_x, train_y=test_y)
    print('Dataset_test Shape:\n', dataset_test.GetDataShape())


    device = configs.device
    model = ConvTTLSTMNet(configs.order, configs.steps, configs.ranks, configs.kernel_size,
                                     configs.bias, configs.hidden_channels, configs.layers_per_block,
                                     configs.skip_stride, configs.input_dim, configs.output_dim).to(configs.device)
    net = torch.load('checkpoint_convttlstm.chk')
    model.load_state_dict(net['net'])
    model.eval()
    dataloader_eval = DataLoader(dataset_test, batch_size=8, shuffle=False)
    with torch.no_grad():
        start_time = datetime.datetime.now()
        for j, (input_sst, nino_true) in enumerate(dataloader_eval):
            nino_pred = model(input_sst.float(), train=False)
            if j == 0:
                pred = nino_pred
                label = nino_true
            else:
                pred = torch.cat((pred, nino_pred), 0)
                label = torch.cat((label, nino_true), 0)
        end_time = datetime.datetime.now()
        print(end_time - start_time)
    np.savez('result.npz', pred=pred.cpu(), label=label.cpu())

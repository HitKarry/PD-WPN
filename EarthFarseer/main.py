import numpy as np
from torch.utils.data import Dataset
from model import Earthfarseer_model
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
    
    test_x = data_x[8406:]
    test_y = data_y[8406:]
    
    dataset_test = dataset_package(train_x=test_x, train_y=test_y)
    print('Dataset_test Shape:\n', dataset_test.GetDataShape())
    

    device = configs.device
    model = Earthfarseer_model(shape_in=(configs.input_length, configs.input_dim, configs.input_h, configs.input_w)).to(configs.device)
    net = torch.load('checkpoint_earthfarseer.chk')
    model.load_state_dict(net['net'])
    model.eval()
    dataloader_eval = DataLoader(dataset_test, batch_size=8, shuffle=False)
    with torch.no_grad():
        start_time = datetime.datetime.now()
        for j, (input_sst, nino_true) in enumerate(dataloader_eval):
            nino_pred = model(input_sst.float())
            if j == 0:
                pred = nino_pred
                label = nino_true
            else:
                pred = torch.cat((pred, nino_pred), 0)
                label = torch.cat((label, nino_true), 0)
        end_time = datetime.datetime.now()
        print(end_time - start_time)
    np.savez('result.npz', pred=pred.cpu(), label=label.cpu())


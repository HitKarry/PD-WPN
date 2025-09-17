import numpy as np
from torch.utils.data import Dataset
from src.score import *
from transformer import SpaceTimeTransformer
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
from regularizer import *
from constrain_moments import K2M
import pickle
import math
import datetime
import torch.nn.functional as F



if __name__ == '__main__':
    print('Configs:\n', configs.__dict__)

    data = np.load('../wind_data/1_wind_data_45n_35n_90e_100e_slide.npz')
    data_x, data_y, mean, std = data['data_x'],data['data_y'],data['mean'],data['std']


    test_x = data_x[8406:]
    test_y = data_y[8406:]


    dataset_test = dataset_package(train_x=test_x, train_y=test_y)
    print('Dataset_test Shape:\n', dataset_test.GetDataShape())

    device = configs.device
    model = SpaceTimeTransformer(configs).to(device)
    net = torch.load('checkpoint_pdwpn.chk')
    model.load_state_dict(net['net'])
    model.eval()
    
    data = DataLoader(dataset_test, batch_size=3, shuffle=False)
    
    with torch.no_grad():
        starttime = datetime.datetime.now()
        for i, (input, target) in enumerate(data):
            pred_temp = model(src=input.float().to(device), tgt=None, train=False)
            if i == 0:
                pred = pred_temp
                label = target
            else:
                pred = torch.cat((pred, pred_temp), 0)
                label = torch.cat((label, target), 0)
        endtime=datetime.datetime.now()
        print('SPEND TIME:',(endtime-starttime))
    
    np.savez('result.npz', pred=pred.cpu(), target=label.cpu())
    



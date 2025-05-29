import numpy as np
from torch.utils.data import Dataset
from model import CNN
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
    test_y = data_y[8406:,0:1]

    dataset_test = dataset_package(train_x=test_x, train_y=test_y)
    print('Dataset_test Shape:\n', dataset_test.GetDataShape())

    device = configs.device
    model = SimVP(configs).to(configs.device)
    net = torch.load('checkpoint_cnn.chk')
    model.load_state_dict(net['net'])
    model.eval()
    dataloader_eval = DataLoader(dataset_test, batch_size=8, shuffle=False)
    with torch.no_grad():
        start_time = datetime.datetime.now()
        for j, (input_sst, nino_true) in enumerate(dataloader_eval):
            for t in range(24):
                nino_pred = model(input_sst.float())
                input_sst = torch.cat([input_sst[:,1:].to(configs.device),nino_pred.to(configs.device)],1)
                if t==0:
                    r = nino_pred
                    s = nino_true
                else:
                    r = torch.cat([r,nino_pred],1)
                    s = torch.cat([s, nino_true], 1)
            if j == 0:
                pred = r
                label = s
            else:
                pred = torch.cat((pred, r), 0)
                label = torch.cat((label, s), 0)
        end_time = datetime.datetime.now()
        print(end_time - start_time)
    np.savez('result.npz', pred=pred.cpu(), label=label.cpu())

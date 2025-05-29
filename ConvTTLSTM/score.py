from convttlstm_net import ConvTTLSTMNet
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from utils import *
import math
import os

device = 'cuda:0'

def coreff(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    c1 = sum((x - x_mean) * (y - y_mean))
    c2 = sum((x - x_mean) ** 2) * sum((y - y_mean) ** 2)
    return c1 / np.sqrt(c2)

def rmse(preds, y):
    rmse = np.sqrt(sum((preds - y) ** 2) / preds.shape[0])
    return rmse

def eval_score(preds, label):
    acskill = 0
    RMSE = 0
    a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
    for i in range(24):
        RMSE += rmse(label[:, i], preds[:, i])
        cor = coreff(label[:, i], preds[:, i])
        acskill += a[i] * np.log(i + 1) * cor
    score = 2 / 3 * acskill - RMSE
    return score

test_path = 'C:/PythonProject/enso/enso_round1_test_20210201/'
files = os.listdir(test_path)
sst_sum = []
test_feas_dict = {}
for file in files:
    test_feas_dict[file] = np.load(test_path + file)
test_predicts_dict = {}
for file_name, val in test_feas_dict.items():
    sst = val[:, :, :, 0]
    sst = torch.tensor(sst).float().reshape(-1, 12, 24, 72) # [1, 12, 24, 72]
    sst_sum.append(sst.numpy().tolist())
input_sst = torch.tensor(np.array(sst_sum)[:,:,:,:,19:67]).permute(0, 2, 1, 3, 4).squeeze() # (104, 12, 24, 48)
del test_path,files ,sst_sum,test_feas_dict,test_predicts_dict

label_path = 'C:/PythonProject/enso/enso_round1_test_20210201_labels/'
files = os.listdir(label_path)
label_feas_dict = {}
for file in files:
    label_feas_dict[file] = np.load(label_path + file)
label = []
for file_name, val in label_feas_dict.items():
     label.append(val.tolist())
label = torch.tensor(label)


# test
print('loading test dataloader')
dataloader_test = DataLoader(input_sst, batch_size=150, shuffle=False)
model = ConvTTLSTMNet(configs.order, configs.steps, configs.ranks, configs.kernel_size,
                                     configs.bias, configs.hidden_channels, configs.layers_per_block,
                                     configs.skip_stride, configs.input_dim, configs.output_dim).to(configs.device)
chk = torch.load('checkpoint.chk')
model.load_state_dict(chk['net'])
model.eval()
print('testing...')
with torch.no_grad():
    for i, input_sst in enumerate(dataloader_test):
        sst_pred, nino_pred = model(input_sst.float().to(device),train=False)

print('Score:',eval_score(nino_pred.cpu().numpy(),label.cpu().numpy()))
# Score: 29.956186676951273
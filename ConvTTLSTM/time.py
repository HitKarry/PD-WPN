from convttlstm_net import ConvTTLSTMNet
from config import configs
from torch.utils.data import DataLoader
from utils import *
import os
import datetime

device = 'cuda:0'


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

model = ConvTTLSTMNet(configs.order, configs.steps, configs.ranks, configs.kernel_size,
                                     configs.bias, configs.hidden_channels, configs.layers_per_block,
                                     configs.skip_stride, configs.input_dim, configs.output_dim).to(configs.device)
chk = torch.load('checkpoint.chk')
model.load_state_dict(chk['net'])
model.eval()

dataloader_test = DataLoader(input_sst, batch_size=10, shuffle=False)
starttime = datetime.datetime.now()
with torch.no_grad():
    for i, sst in enumerate(dataloader_test):
        sst_pred, nino_pred = model(sst.float().to(device),train=False)
endtime = datetime.datetime.now()
print(input_sst.shape[0])
print('SPEND TIME:',(endtime - starttime) / input_sst.shape[0])
# SPEND TIME: 0:00:00.058530
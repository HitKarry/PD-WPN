import numpy as np
from torch.utils.data import Dataset
from predrnn_model import PredRNN_Model
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
import math
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(5)
        self.network = PredRNN_Model(configs).to(configs.device)
        self.opt = torch.optim.Adam(self.network.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(self.opt, mode='max', factor=0.3, patience=0, verbose=True,
                                              min_lr=0.0001)
        self.z500, self.t850, self.t2m, self.new = 'z500', 't850', 't2m', 'new'
        self.loss = nn.MSELoss()

    def eva_loss(self, y_pred, y_true,idx):
        if idx == 'z500':
            idx = 0
        if idx == 't850':
            idx = 1
        if idx == 't2m':
            idx = 2
        if idx == 'new':
            idx = 3
        rmse = torch.mean((y_pred[:, :, idx] - y_true[:, :, idx])**2, dim=[2, 3])
        rmse = torch.sum(rmse.sqrt().mean(dim=0))
        return rmse

    def train_once(self, input_sst, sst_true, ssr_ratio):
        sst_pred = self.network(input_sst.float().to(self.device), sst_true.float().to(self.device), teacher_forcing=True,
                                           scheduled_sampling_ratio=ssr_ratio, train=True)
        self.opt.zero_grad()
        loss = self.loss(sst_pred, torch.cat([input_sst[:,1:].float(),sst_true.float()],1).to(self.device))
        loss.backward()
        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
        self.opt.step()
        return loss

    def test(self, dataloader_test):
        sst_pred = []
        with torch.no_grad():
            for input_sst, target_sst in dataloader_test:
                sst = self.network(input_sst.float().to(self.device),target_sst.float().to(self.device), train=False)
                sst_pred.append(sst)

        return torch.cat(sst_pred, dim=0)

    def infer(self, dataset, dataloader):
        self.network.eval()
        with torch.no_grad():
            sst_pred = self.test(dataloader)
            sst_true = torch.from_numpy(dataset.target).float().to(self.device)
            loss_z500 = self.eva_loss(sst_pred, sst_true, self.z500).item()
            loss_t850 = self.eva_loss(sst_pred, sst_true, self.t850).item()
            loss_t2m = self.eva_loss(sst_pred, sst_true, self.t2m).item()
            loss_new = self.eva_loss(sst_pred, sst_true, self.new).item()

        return loss_z500, loss_t850, loss_t2m, loss_new

    def train(self, dataset_train, dataset_eval, chk_path):
        torch.manual_seed(0)
        print('loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        print('loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)

        count = 0
        best = math.inf
        ssr_ratio = 1
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i+1))
            # train
            self.network.train()
            for j, (input_sst, sst_true) in enumerate(dataloader_train):
                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.configs.ssr_decay_rate, 0)
                loss  = self.train_once(input_sst.to(), sst_true, ssr_ratio)  # y_pred for one batch

                if (j+1) % self.configs.display_interval == 0:
                    print('batch training loss: {:.2f}, ssr: {:.5f}'.format(loss, ssr_ratio))

                # increase the number of evaluations in order not to miss the optimal point
                # which is feasible because of the less training time of ticsformer
                if (i+1 >= 6) and (j+1)%(self.configs.display_interval * 2) == 0:
                    loss_z500_eval_0, loss_t850_eval_0, loss_t2m_eval_0, loss_new_eval_0 = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
                    loss_eval_0 = loss_z500_eval_0 + loss_t850_eval_0 + loss_t2m_eval_0 + loss_new_eval_0
                    print('batch eval loss: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(loss_z500_eval_0, loss_t850_eval_0, loss_t2m_eval_0, loss_new_eval_0, loss_eval_0))
                    # self.save_model(chk_path + '_' + str(i) + '_' + str(loss_eval_0) + '.chk')
                    if loss_eval_0 < best:
                        self.save_model(chk_path + '_' + str(loss_eval_0) + '.chk')
                        best = loss_eval_0
                        count = 0

            # evaluation
            loss_z500_eval, loss_t850_eval, loss_t2m_eval, loss_new_eval = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
            loss_eval = loss_z500_eval + loss_t850_eval + loss_t2m_eval + loss_new_eval
            print('epoch eval loss: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(loss_z500_eval, loss_t850_eval, loss_t2m_eval, loss_new_eval, loss_eval))
            self.lr_scheduler.step(loss_eval)
            if loss_eval >= best:
                count += 1
                print('eval loss is not reduced for {} epoch'.format(count))
            else:
                count = 0
                print('eval loss is reduced from {:.5f} to {:.5f}, saving model'.format(best, loss_eval))
                self.save_model(chk_path + '_' + str(loss_eval) + '.chk')
                best = loss_eval

            if count == self.configs.patience:
                print('early stopping reached, best score is {:5f}'.format(best))
                break

    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.opt.state_dict()}, path)


def dataset_generator(dataset,history_size,target_size, rolling_step=1, sampling_step=1, single_step=False):
    data = []
    labels = []
    start_index = history_size
    end_index = len(dataset) - target_size + 1
    for i in range(start_index, end_index, rolling_step):
        indices = range(i-history_size, i, sampling_step)
        data.append(dataset[indices])
        indices1=range(i, i+target_size, sampling_step)
        if single_step:
            labels.append(dataset[i+target_size])
        else:
            labels.append(dataset[indices1])
    return np.array(data), np.array(labels)

class dataset_package(Dataset):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.input = train_x
        self.target = train_y

    def GetDataShape(self):
        return {'input': self.input.shape,
                'target': self.target.shape}

    def __len__(self, ):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

if __name__ == '__main__':
    print('Configs:\n', configs.__dict__)

    data = np.load('../wind_data/1_wind_data_45n_35n_90e_100e_slide.npz')
    data_x, data_y, mean, std = data['data_x'],data['data_y'],data['mean'],data['std']

    train_x = data_x[:8406]
    train_y = data_y[:8406]
    test_x = data_x[8406:]
    test_y = data_y[8406:]

    dataset_train = dataset_package(train_x=train_x, train_y=train_y)
    dataset_test = dataset_package(train_x=test_x, train_y=test_y)
    del train_x, train_y, test_x, test_y
    print('Dataset_train Shape:\n', dataset_train.GetDataShape())
    print('Dataset_test Shape:\n', dataset_test.GetDataShape())

    trainer = Trainer(configs)
    # trainer.save_configs('config.pkl')
    # trainer.train(dataset_train, dataset_test, 'checkpoint')

    #########################################################################################################################################

    device = configs.device
    model = PredRNN_Model(configs).to(configs.device)
    net = torch.load('checkpoint_49.135154724121094.chk')
    model.load_state_dict(net['net'])
    model.eval()
    dataloader_eval = DataLoader(dataset_test, batch_size=8, shuffle=False)
    with torch.no_grad():
        start_time = datetime.datetime.now()
        for j, (input_sst, nino_true) in enumerate(dataloader_eval):
            nino_pred = model(input_sst.float(),nino_true.float(), train=False)
            if j == 0:
                pred = nino_pred
                label = nino_true
            else:
                pred = torch.cat((pred, nino_pred), 0)
                label = torch.cat((label, nino_true), 0)
        end_time = datetime.datetime.now()
        print(end_time - start_time)
    np.savez('result49.135154724121094.npz', pred=pred.cpu(), label=label.cpu())
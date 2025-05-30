import os
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from scipy import interpolate
import matplotlib.pyplot as plt


def data_standardization(data):
    mean = data.mean([0, 2, 3])[None, :, None, None]
    std = data.std([0, 2, 3])[None, :, None, None]
    data = (data - mean) / std

    return data, mean, std


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


history_size,target_size,rolling_step,sampling_step,single_step = 24,24,5,1,False

data_npz = np.load('1_wind_data_45n_35n_90e_100e.npz')
data,mean,std = data_npz['data'],data_npz['mean'],data_npz['std']
data,labels = dataset_generator(data,history_size,target_size, rolling_step, sampling_step, single_step)
np.savez('1_wind_data_45n_35n_90e_100e_slide.npz',data_x=np.array(data),data_y=np.array(labels),mean=np.array(mean),std=np.array(std))

data_npz = np.load('2_wind_data_47n_37n_105e_115e.npz')
data,mean,std = data_npz['data'],data_npz['mean'],data_npz['std']
data,labels = dataset_generator(data,history_size,target_size, rolling_step, sampling_step, single_step)
np.savez('2_wind_data_47n_37n_105e_115e_slide.npz',data_x=np.array(data),data_y=np.array(labels),mean=np.array(mean),std=np.array(std))

data_npz = np.load('3_wind_data_33n_23n_117e_127e.npz')
data,mean,std = data_npz['data'],data_npz['mean'],data_npz['std']
data,labels = dataset_generator(data,history_size,target_size, rolling_step, sampling_step, single_step)
np.savez('3_wind_data_33n_23n_117e_127e_slide.npz',data_x=np.array(data),data_y=np.array(labels),mean=np.array(mean),std=np.array(std))

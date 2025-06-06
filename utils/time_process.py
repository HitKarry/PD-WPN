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

data_npz = np.load('wind_data_time.npz')
data = data_npz['data']
data,labels = dataset_generator(data,history_size,target_size, rolling_step, sampling_step, single_step)
np.savez('wind_data_slide_time.npz',data_x=np.array(data),data_y=np.array(labels))


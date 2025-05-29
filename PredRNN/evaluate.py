import numpy as np
import torch

# lat: [45.   44.75 44.5  ...  35.5  35.25 35.  ]
# lat = np.linspace(45, 35, num=40, dtype=float, endpoint=True)

def compute_composite_wind_speed(U, V):
    """
    计算合成风速，假设 U 和 V 的形状为 (B, T, C, H, W)

    参数:
    U : ndarray
        风速U分量，形状为 (B, T, C, H, W)
    V : ndarray
        风速V分量，形状为 (B, T, C, H, W)

    返回:
    composite_wind_speed : ndarray
        合成风速，形状为 (B, T, C, H, W)
    """
    # 计算合成风速
    composite_wind_speed = torch.sqrt(U ** 2 + V ** 2)
    return composite_wind_speed

def weighted_rmse(y_pred,y_true):
    """
            Args:
                y_pred: (time, channel, lat, lon)
                y_true: (time, channel, lat, lon)
            Returns:
                RMSE: (channel, )
            """
    lat = np.linspace(45, 35, num=40, dtype=float, endpoint=True)
    RMSE = np.empty([y_pred.size(1)])
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    for i in range(y_pred.size(1)):
        RMSE[i] = np.sqrt(((y_pred[:,i,:,:] - y_true[:,i,:,:]).permute(0,2,1)**2*weights_lat).mean([-2,-1])).mean(axis=0)
    return RMSE

def weighted_mae(y_pred,y_true):
    """
            Args:
                y_pred: (time, channel, lat, lon)
                y_true: (time, channel, lat, lon)
            Returns:
                MAE: (channel, )
            """
    lat = np.linspace(45, 35, num=40, dtype=float, endpoint=True)
    MAE = np.empty([y_pred.size(1)])
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    for i in range(y_pred.size(1)):
        MAE[i] = (abs(y_pred[:, i, :, :] - y_true[:, i, :, :]).permute(0, 2, 1) * weights_lat).mean([0, -2, -1])
    return MAE

def weighted_acc(y_pred,y_true):
    """
            Args:
                y_pred: (time, channel, lat, lon)
                y_true: (time, channel, lat, lon)
            Returns:
                ACC: (channel, )
            """
    lat = np.linspace(45, 35, num=40, dtype=float, endpoint=True)
    ACC = np.empty([y_pred.size(1)])
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    w = torch.tensor(weights_lat)
    for i in range(y_pred.size(1)):
        clim = y_true[:,i,:,:].mean(0)
        a = y_true[:,i,:,:] - clim
        a_prime = (a - a.mean()).permute(0,2,1)
        fa = y_pred[:,i,:,:] - clim
        fa_prime = (fa - fa.mean()).permute(0,2,1)
        ACC[i] = (
                torch.sum(w * fa_prime * a_prime) /
                torch.sqrt(
                    torch.sum(w * fa_prime ** 2) * torch.sum(w * a_prime ** 2)
                )
        )
    return ACC

def calculate_wdfa(pred, true, alpha):
    """
    Calculate the Wind Direction Forecast Accuracy (WDFA) metric.

    Parameters:
        pred (np.ndarray): Predicted wind components (shape: B1, T1, C1, H1, W1).
        true (np.ndarray): True wind components (shape: B2, T2, C2, H2, W2).
        alpha (float): Angle threshold in degrees.

    Returns:
        float: WDFA_alpha value.
    """
    # Ensure the input arrays have matching spatial dimensions
    assert pred.shape[-3:] == true.shape[-3:], "Spatial dimensions of pred and true must match."

    # Calculate wind direction in degrees for both predicted and true values
    pred_u, pred_v = pred[..., 0, :, :], pred[..., 1, :, :]
    true_u, true_v = true[..., 0, :, :], true[..., 1, :, :]

    pred_dir = np.arctan2(pred_v, pred_u) * (180 / np.pi)  # Convert from radians to degrees
    true_dir = np.arctan2(true_v, true_u) * (180 / np.pi)  # Convert from radians to degrees

    # Normalize angles to [0, 360)
    pred_dir = np.mod(pred_dir, 360)
    true_dir = np.mod(true_dir, 360)

    # Calculate angular difference
    diff = np.abs(pred_dir - true_dir)
    diff = np.minimum(diff, 360 - diff)  # Handle circular nature of angles

    # Count grid points where diff < alpha
    count = torch.sum(diff < alpha)

    # Calculate WDFA_alpha
    B, T, H, W = pred.shape[0], pred.shape[1], pred.shape[-2], pred.shape[-1]  # Spatial dimensions
    wdfa_alpha = (count / (B * T * H * W)) * 100

    return wdfa_alpha.item()

def evaluate(data_path,time,std,mean):
    i=time
    data = np.load(data_path)
    y_pred, y_true = torch.tensor(data['pred']), torch.tensor(data['label']) # (719, 20, 2, 32, 64) (719, 20, 2, 32, 64)
    y_pred, y_true = y_pred*std[None]+mean[None], y_true*std[None]+mean[None]
    y_pred_WDFA, y_true_WDFA = y_pred[:,i,:,:,:], y_true[:,i,:,:,:]
    y_pred = torch.cat([compute_composite_wind_speed(y_pred[:,:,0:1], y_pred[:,:,1:2]),compute_composite_wind_speed(y_pred[:,:,2:3], y_pred[:,:,3:])],2)
    y_true = torch.cat([compute_composite_wind_speed(y_true[:,:,0:1], y_true[:,:,1:2]),compute_composite_wind_speed(y_true[:,:,2:3], y_true[:,:,3:])],2)
    y_pred, y_true = y_pred[:,i,:,:,:], y_true[:,i,:,:,:]
    print('RMSE:', weighted_rmse(y_pred,y_true))
    print('MAE: ', weighted_mae(y_pred,y_true))
    print('ACC: ', weighted_acc(y_pred,y_true))
    print('WDFA90: [', calculate_wdfa(y_pred_WDFA[:,None,:2], y_true_WDFA[:,None,:2],90),',',calculate_wdfa(y_pred_WDFA[:,None,2:], y_true_WDFA[:,None,2:],90),']')
    print('WDFA45: [', calculate_wdfa(y_pred_WDFA[:, None,:2], y_true_WDFA[:, None,:2], 45),',',calculate_wdfa(y_pred_WDFA[:,None,2:], y_true_WDFA[:,None,2:],45),']')
    print('WDFA22.5: [', calculate_wdfa(y_pred_WDFA[:, None,:2], y_true_WDFA[:, None,:2], 22.5),',',calculate_wdfa(y_pred_WDFA[:,None,2:], y_true_WDFA[:,None,2:],22.5),']')




data = np.load('../wind_data/1_wind_data_45n_35n_90e_100e_slide.npz')
mean, std = data['mean'],data['std']


result_npz = 'result49.135154724121094.npz'

for leadtime in range(24):
    print('Lead Time:',leadtime+1,'h')
    evaluate(result_npz,leadtime,std, mean)

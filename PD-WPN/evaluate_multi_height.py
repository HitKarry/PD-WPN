import numpy as np
import torch

def compute_composite_wind_speed(U, V):
    composite_wind_speed = torch.sqrt(U ** 2 + V ** 2)
    return composite_wind_speed

def weighted_rmse(y_pred, y_true):
    lat = np.linspace(45, 35, num=40, dtype=float, endpoint=True)
    RMSE = np.empty([y_pred.size(1)])
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    for i in range(y_pred.size(1)):
        RMSE[i] = np.sqrt(((y_pred[:, i, :, :] - y_true[:, i, :, :]).permute(0, 2, 1) ** 2 * weights_lat).mean([-2, -1])).mean(axis=0)
    return RMSE

def weighted_mae(y_pred, y_true):
    lat = np.linspace(45, 35, num=40, dtype=float, endpoint=True)
    MAE = np.empty([y_pred.size(1)])
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    for i in range(y_pred.size(1)):
        MAE[i] = (abs(y_pred[:, i, :, :] - y_true[:, i, :, :]).permute(0, 2, 1) * weights_lat).mean([0, -2, -1])
    return MAE

def weighted_acc(y_pred, y_true):
    lat = np.linspace(45, 35, num=40, dtype=float, endpoint=True)
    ACC = np.empty([y_pred.size(1)])
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    w = torch.tensor(weights_lat)
    for i in range(y_pred.size(1)):
        clim = y_true[:, i, :, :].mean(0)
        a = y_true[:, i, :, :] - clim
        a_prime = (a - a.mean()).permute(0, 2, 1)
        fa = y_pred[:, i, :, :] - clim
        fa_prime = (fa - fa.mean()).permute(0, 2, 1)
        ACC[i] = (
            torch.sum(w * fa_prime * a_prime) /
            torch.sqrt(
                torch.sum(w * fa_prime ** 2) * torch.sum(w * a_prime ** 2)
            )
        )
    return ACC

def calculate_wdfa(pred, true, alpha):
    assert pred.shape[-3:] == true.shape[-3:], "Spatial dimensions of pred and true must match."
    pred_u, pred_v = pred[..., 0, :, :], pred[..., 1, :, :]
    true_u, true_v = true[..., 0, :, :], true[..., 1, :, :]

    pred_dir = np.arctan2(pred_v, pred_u) * (180 / np.pi)
    true_dir = np.arctan2(true_v, true_u) * (180 / np.pi)

    pred_dir = np.mod(pred_dir, 360)
    true_dir = np.mod(true_dir, 360)

    diff = np.abs(pred_dir - true_dir)
    diff = np.minimum(diff, 360 - diff)

    count = torch.sum(diff < alpha)

    B, T, H, W = pred.shape[0], pred.shape[1], pred.shape[-2], pred.shape[-1]
    wdfa_alpha = (count / (B * T * H * W)) * 100

    return wdfa_alpha.item()


def evaluate(data_path, time):
    i = time
    data = np.load(data_path)
    y_pred, y_true = torch.tensor(data['data_pred']), torch.tensor(data['data_true'])  # (B, T, 10, H, W)

    y_pred_WDFA, y_true_WDFA = y_pred[:, i, :, :, :], y_true[:, i, :, :, :]

    composite_pred = []
    composite_true = []
    levels = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]  # (U,V)对：10m, 100m, 30m, 50m, 75m
    for u_idx, v_idx in levels:
        composite_pred.append(
            compute_composite_wind_speed(y_pred[:, :, u_idx:u_idx + 1], y_pred[:, :, v_idx:v_idx + 1]))
        composite_true.append(
            compute_composite_wind_speed(y_true[:, :, u_idx:u_idx + 1], y_true[:, :, v_idx:v_idx + 1]))

    y_pred = torch.cat(composite_pred, dim=2)
    y_true = torch.cat(composite_true, dim=2)

    y_pred = y_pred[:, i, :, :, :]
    y_true = y_true[:, i, :, :, :]

    print('RMSE:', weighted_rmse(y_pred, y_true))
    print('MAE :', weighted_mae(y_pred, y_true))
    print('ACC :', weighted_acc(y_pred, y_true))

    print('WDFA90:',
          [calculate_wdfa(y_pred_WDFA[:, None, u_idx:u_idx + 2], y_true_WDFA[:, None, u_idx:u_idx + 2], 90) for u_idx in
           range(0, 10, 2)])
    print('WDFA45:',
          [calculate_wdfa(y_pred_WDFA[:, None, u_idx:u_idx + 2], y_true_WDFA[:, None, u_idx:u_idx + 2], 45) for u_idx in
           range(0, 10, 2)])
    print('WDFA22.5:',
          [calculate_wdfa(y_pred_WDFA[:, None, u_idx:u_idx + 2], y_true_WDFA[:, None, u_idx:u_idx + 2], 22.5) for u_idx
           in range(0, 10, 2)])



result_npz = 'height_extend_data.npz'

for leadtime in range(24):
    print('Lead Time:', leadtime + 1, 'h')
    evaluate(result_npz, leadtime)

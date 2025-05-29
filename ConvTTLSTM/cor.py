
import numpy as np



def coreff(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    c1 = sum((x - x_mean) * (y - y_mean))
    c2 = sum((x - x_mean) ** 2) * sum((y - y_mean) ** 2)
    return c1 / np.sqrt(c2)


def score(preds, label):
    acskill = 0
    a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
    for i in range(24):
        cor = coreff(label[:, i], preds[:, i])
        acskill += a[i] * np.log(i + 1) * cor
    score = 2 / 3 * acskill
    return score

i = 24
data = np.load('result.npz')
pred,label = data['pred'],data['label'],
preds = np.concatenate([pred[:,:i],label[:,i:]],1)
print(score(preds, label))
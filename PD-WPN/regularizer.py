import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ 遍历物理分支 conv1（你已有） ------
def _iter_phycell_conv1_weights(model):
    for name, m in model.named_modules():
        if hasattr(m, 'F') and isinstance(m.F, nn.Sequential) and hasattr(m.F, 'conv1'):
            conv1 = m.F.conv1
            if isinstance(conv1, nn.Conv2d):
                yield name, conv1.weight   # [F_hidden, C_in, kh, kw]

# ------ 遍历梯度分支 Gx / Gy（新增） ------
def _iter_gradient_convs(model):
    for name, m in model.named_modules():
        # 针对你的 GradientConvLayer 命名
        if hasattr(m, 'G_x') and isinstance(m.G_x, nn.Conv2d):
            yield f"{name}.G_x", m.G_x.weight, 'x'  # [C_out, C_in, 1, 3]
        if hasattr(m, 'G_y') and isinstance(m.G_y, nn.Conv2d):
            yield f"{name}.G_y", m.G_y.weight, 'y'  # [C_out, C_in, 3, 1]

# ------ 你已有：通用 PDE 项的矩正则（可保留） ------
def _ij_terms(q):
    return [(i, j) for i in range(q+1) for j in range(q+1-i)]

def moment_regularizer(model, K2M_class, q=2, lamb=1e-3, target_scale=1.0):
    reg = 0.0
    n = 0
    terms = _ij_terms(q)
    for name, W in _iter_phycell_conv1_weights(model):
        _, Cin, kh, kw = W.shape
        k2m = K2M_class([kh, kw]).to(W.device)
        K2d = W.mean(dim=1)  # [F_hidden, kh, kw]
        for oc in range(K2d.shape[0]):
            i, j = terms[oc % len(terms)]
            if i + j > q:
                continue
            m = k2m(K2d[oc])  # [kh, kw]，行= y 阶，列= x 阶
            mask = torch.zeros_like(m)
            tgt  = torch.zeros_like(m)
            for p in range(q+1):
                for qy in range(q+1-p):
                    mask[p, qy] = 1.0
                    tgt[p, qy]  = 0.0
            mask[i, j] = 1.0
            tgt[i, j]  = target_scale
            reg = reg + F.mse_loss(m * mask, tgt, reduction='mean')
            n += 1
    if n == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return lamb * (reg / n)

# ------ 新增：专门把 Gx / Gy 约束为“一阶偏导” ------
def gradient_first_order_regularizer(model, K2M_class, lamb=1e-3, target_scale=1.0, suppress_second=True):
    """
    把 GradientConvLayer 的 1x3 / 3x1 卷积核约束为一阶偏导：
      - 对 G_x(1x3)：  m[0,1] = target_scale, m[0,0] = 0, (可选) m[0,2] = 0
      - 对 G_y(3x1)：  m[1,0] = target_scale, m[0,0] = 0, (可选) m[2,0] = 0
    说明：K2M 约定 moment 矩阵 m 的下标 [p, q] = y^p * x^q 的矩。
    """
    reg = 0.0
    n   = 0

    for name, W, axis in _iter_gradient_convs(model):
        # W: [C_out, C_in, kh, kw]，你的实现里 kh,kw 分别是 (1,3) 或 (3,1)
        _, Cin, kh, kw = W.shape
        k2m = K2M_class([kh, kw]).to(W.device)

        # 与 conv1 一样，先把 in_channel 聚合（平均或求和）
        K2d = W.mean(dim=1)  # [C_out, kh, kw]

        for oc in range(K2d.shape[0]):
            m = k2m(K2d[oc])  # [kh, kw]，行=y 阶(0..kh-1)，列=x 阶(0..kw-1)
            mask = torch.zeros_like(m)
            tgt  = torch.zeros_like(m)

            # 零阶矩为 0（去 DC）
            mask[0, 0] = 1.0
            tgt[0, 0]  = 0.0

            if axis == 'x':     # 1x3 核，约束 ∂x
                # 一阶 x 矩为常数
                mask[0, 1] = 1.0
                tgt[0, 1]  = target_scale
                if suppress_second and kw >= 3:
                    # 抑制二阶 x 矩（可选，提升纯度）
                    mask[0, 2] = 1.0
                    tgt[0, 2]  = 0.0

            elif axis == 'y':   # 3x1 核，约束 ∂y
                mask[1, 0] = 1.0
                tgt[1, 0]  = target_scale
                if suppress_second and kh >= 3:
                    mask[2, 0] = 1.0
                    tgt[2, 0]  = 0.0

            reg = reg + F.mse_loss(m * mask, tgt, reduction='mean')
            n  += 1

    if n == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return lamb * (reg / n)

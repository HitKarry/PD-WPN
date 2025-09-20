import torch
import torch.nn as nn
import torch.nn.functional as F

def _iter_phycell_conv1_weights(model):
    for name, m in model.named_modules():
        if hasattr(m, 'F') and isinstance(m.F, nn.Sequential) and hasattr(m.F, 'conv1'):
            conv1 = m.F.conv1
            if isinstance(conv1, nn.Conv2d):
                yield name, conv1.weight 

def _iter_gradient_convs(model):
    for name, m in model.named_modules():
        if hasattr(m, 'G_x') and isinstance(m.G_x, nn.Conv2d):
            yield f"{name}.G_x", m.G_x.weight, 'x'  
        if hasattr(m, 'G_y') and isinstance(m.G_y, nn.Conv2d):
            yield f"{name}.G_y", m.G_y.weight, 'y' 

def _ij_terms(q):
    return [(i, j) for i in range(q+1) for j in range(q+1-i)]

def moment_regularizer(model, K2M_class, q=2, lamb=1e-3, target_scale=1.0):
    reg = 0.0
    n = 0
    terms = _ij_terms(q)
    for name, W in _iter_phycell_conv1_weights(model):
        _, Cin, kh, kw = W.shape
        k2m = K2M_class([kh, kw]).to(W.device)
        K2d = W.mean(dim=1) 
        for oc in range(K2d.shape[0]):
            i, j = terms[oc % len(terms)]
            if i + j > q:
                continue
            m = k2m(K2d[oc]) 
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

def gradient_first_order_regularizer(model, K2M_class, lamb=1e-3, target_scale=1.0, suppress_second=True):
    reg = 0.0
    n   = 0

    for name, W, axis in _iter_gradient_convs(model):
        _, Cin, kh, kw = W.shape
        k2m = K2M_class([kh, kw]).to(W.device)

        K2d = W.mean(dim=1) 

        for oc in range(K2d.shape[0]):
            m = k2m(K2d[oc]) 
            mask = torch.zeros_like(m)
            tgt  = torch.zeros_like(m)

            mask[0, 0] = 1.0
            tgt[0, 0]  = 0.0

            if axis == 'x':   
                mask[0, 1] = 1.0
                tgt[0, 1]  = target_scale
                if suppress_second and kw >= 3:
                    mask[0, 2] = 1.0
                    tgt[0, 2]  = 0.0

            elif axis == 'y': 
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

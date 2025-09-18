import torch
import torch.nn.functional as F


def _iter_phycell_conv1_weights(model):
    for name, m in model.named_modules():
        if hasattr(m, 'F') and isinstance(m.F, torch.nn.Sequential) and hasattr(m.F, 'conv1'):
            conv1 = m.F.conv1
            if isinstance(conv1, torch.nn.Conv2d):
                yield name, conv1.weight  


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

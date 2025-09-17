import torch
import torch.nn.functional as F

# === 1. 找到所有 PhyCell_Cell.conv1 的权重 ===
def _iter_phycell_conv1_weights(model):
    for name, m in model.named_modules():
        # 精准匹配到你的结构：PhyCell_Cell里有 self.F.conv1
        if hasattr(m, 'F') and isinstance(m.F, torch.nn.Sequential) and hasattr(m.F, 'conv1'):
            conv1 = m.F.conv1
            if isinstance(conv1, torch.nn.Conv2d):
                yield name, conv1.weight  # shape: [F_hidden_dim, input_dim, kh, kw]

# === 2. 给定 q，生成 (i,j) 列表：i+j<=q ===
def _ij_terms(q):
    return [(i, j) for i in range(q+1) for j in range(q+1-i)]

# === 3. 计算“矩约束”正则（soft constraint）===
def moment_regularizer(model, K2M_class, q=2, lamb=1e-3, target_scale=1.0):
    """
    只依赖模型权重，不改任何模块。
    K2M_class: 你的 K2M 实现类（例如 from k2m import K2M）
    q: 最高阶
    lamb: 正则权重
    target_scale: 目标矩的常数（可按网格步长缩放）
    """
    reg = 0.0
    n = 0
    terms = _ij_terms(q)

    for name, W in _iter_phycell_conv1_weights(model):
        # W: [F_hidden_dim, input_dim, kh, kw]
        _, Cin, kh, kw = W.shape
        k2m = K2M_class([kh, kw]).to(W.device)

        # 把每个 out_channel 的核在 in_channel 上做平均 -> 2D 核；你也可以用“求和”
        K2d = W.mean(dim=1)  # [F_hidden_dim, kh, kw]

        # 每个 out_channel 分配一个 (i,j)，循环使用
        for oc in range(K2d.shape[0]):
            i, j = terms[oc % len(terms)]
            # 只约束 i+j<=q
            if i + j > q: 
                continue

            m = k2m(K2d[oc])  # [kh, kw] 矩矩阵

            # 构造 mask / target：只约束到阶数<=q，低阶为0，(i,j)为常数
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

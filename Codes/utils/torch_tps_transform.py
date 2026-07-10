import torch
import numpy as np
import torch.nn.functional as F  # 新增导入

# transforming an image (U) from target (control points) to source (control points)
# all the points should be normalized from -1 ~1
#rigid->mesh
def transformer(U, source, target, out_size):

    def _meshgrid(height, width, source):

        x_t = torch.matmul(torch.ones([height, 1]), torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1), torch.ones([1, width]))
        if torch.cuda.is_available():
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        x_t_flat = x_t.reshape([1, 1, -1])
        y_t_flat = y_t.reshape([1, 1, -1])

        num_batch = source.size()[0]
        px = torch.unsqueeze(source[:,:,0], 2)  # [bn, pn, 1]
        py = torch.unsqueeze(source[:,:,1], 2)  # [bn, pn, 1]
        if torch.cuda.is_available():
            px = px.cuda()
            py = py.cuda()
        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)
        r = d2 * torch.log(d2 + 1e-6) # [bn, pn, h*w]
        x_t_flat_g = x_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        y_t_flat_g = y_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        ones = torch.ones_like(x_t_flat_g) # [bn, 1, h*w]
        if torch.cuda.is_available():
            ones = ones.cuda()

        grid = torch.cat((ones, x_t_flat_g, y_t_flat_g, r), 1) # [bn, 3+pn, h*w]
        return grid

    def _transform(T, source, input_dim, out_size):
        num_batch, num_channels, height, width = input_dim.size()
        out_height, out_width = out_size[0], out_size[1]
        
        grid = _meshgrid(out_height, out_width, source) # [bn, 3+pn, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = torch.matmul(T, grid)
        
        # ==========================================
        # 核心修改：使用 PyTorch 官方 grid_sample 替代手工 gather
        # ==========================================
        # 1. 提取 x 和 y，并 reshape 为 grid_sample 需要的形状 [batch, H, W]
        x_s = T_g[:, 0, :].reshape(num_batch, out_height, out_width)
        y_s = T_g[:, 1, :].reshape(num_batch, out_height, out_width)
        
        # 2. 堆叠成 [batch, H, W, 2] 的采样网格
        sample_grid = torch.stack([x_s, y_s], dim=-1)
        
        # 3. 使用 bicubic 模式进行双三次插值
        # padding_mode='border' 可以防止网格轻微越界时出现黑边
        output = F.grid_sample(
            input_dim, 
            sample_grid, 
            mode='bicubic', 
            padding_mode='border', 
            align_corners=True
        )
        return output

    def _solve_system(source, target):
        num_batch  = source.size()[0]
        num_point  = source.size()[1]

        np.set_printoptions(precision=8)

        ones = torch.ones(num_batch, num_point, 1).float()

        if torch.cuda.is_available():
            ones = ones.cuda()
        
        p = torch.cat([ones, source], 2) # [bn, pn, 3]

        p_1 = p.reshape([num_batch, -1, 1, 3]) # [bn, pn, 1, 3]
        p_2 = p.reshape([num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = torch.sum(torch.square(p_1-p_2), 3) # p1 - p2: [bn, pn, pn, 3]   final output: [bn, pn, pn]

        r = d2 * torch.log(d2 + 1e-6) # [bn, pn, pn]

        zeros = torch.zeros(num_batch, 3, 3).float()
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        W_0 = torch.cat((p, r), 2) # [bn, pn, 3+pn]
        W_1 = torch.cat((zeros, p.permute(0,2,1)), 2) # [bn, 3, pn+3]
        W = torch.cat((W_0, W_1), 1) # [bn, pn+3, pn+3]
        
        # =========================================================
        # 核心修改：加入 Tikhonov 正则化 (防止 Singular Matrix 报错)
        # =========================================================
        W_64 = W.type(torch.float64) # 提前转为 float64 以保证加扰动时的最高精度
        dim = W_64.size(-1)          # 获取方阵的维度，即 pn + 3
        eps = 1e-4                   # 极小的扰动值 (1e-4 到 1e-5 都可以)
        
        # 生成对角单位矩阵，形状为 [1, dim, dim]，利用广播机制自动匹配 batch_size
        eye = torch.eye(dim, dtype=W_64.dtype, device=W_64.device).unsqueeze(0)
        
        # 给矩阵对角线加上微小扰动，强制其满秩可逆
        W_stable = W_64 + eps * eye
        
        # 对稳定后的矩阵求逆
        W_inv = torch.inverse(W_stable)
        # =========================================================

        zeros2 = torch.zeros(num_batch, 3, 2)
        if torch.cuda.is_available():
            zeros2 = zeros2.cuda()
        tp = torch.cat((target, zeros2), 1) # [bn, pn+3, 2]

        T = torch.matmul(W_inv, tp.type(torch.float64)) # [bn, pn+3, 2]
        T = T.permute(0, 2, 1) # [bn, 2, pn+3]

        return T.type(torch.float32)

    T = _solve_system(source , target)
    output = _transform(T, source , U, out_size)
  
    return output
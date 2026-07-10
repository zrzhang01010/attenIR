import torch
import torch.nn.functional as F

import predefine
import utils.torch_tps_transform as torch_tps_transform

"""
Pure pixel-level saliency-aligned shape loss version

核心区别：
1) 保留 cell 级几何能量定义；
2) 将 E_shape 上采样为像素级 D_shape；
3) 使用 warp 后显著图 S_warp 做像素级加权；
4) salient_shape_loss 直接返回纯像素级版本，不做 grid / pixel 融合。
"""

# 定义全局网格尺寸
grid_w = predefine.GRID_W
grid_h = predefine.GRID_H


def get_rigid_mesh(batch_size, height, width, device=None, dtype=None):
    """构造规则控制点网格: [B, gh+1, gw+1, 2]"""
    base_dtype = torch.float32 if dtype is None else dtype
    ww = torch.matmul(
        torch.ones([grid_h + 1, 1], device=device, dtype=base_dtype),
        torch.unsqueeze(
            torch.linspace(0.0, float(width), grid_w + 1, device=device, dtype=base_dtype),
            0,
        ),
    )
    hh = torch.matmul(
        torch.unsqueeze(
            torch.linspace(0.0, float(height), grid_h + 1, device=device, dtype=base_dtype),
            1,
        ),
        torch.ones([1, grid_w + 1], device=device, dtype=base_dtype),
    )

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)), 2)  # [gh+1, gw+1, 2]
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, gh+1, gw+1, 2]
    return ori_pt


def get_norm_mesh(mesh, height, width, eps=1e-6):
    """将控制点坐标归一化到 [-1, 1]"""
    width = float(max(width, eps))
    height = float(max(height, eps))
    mesh_w = mesh[..., 0] * 2.0 / width - 1.0
    mesh_h = mesh[..., 1] * 2.0 / height - 1.0
    norm_mesh = torch.stack([mesh_w, mesh_h], -1)  # [B, gh+1, gw+1, 2]
    return norm_mesh


def get_stack_mesh(mesh):
    """将网格展平为 TPS 需要的 [B, N, 2]"""
    batch_size = mesh.size(0)
    mesh_w = mesh[..., 0]
    mesh_h = mesh[..., 1]
    norm_mesh = torch.stack([mesh_w, mesh_h], 3)  # [B, gh+1, gw+1, 2]
    return norm_mesh.reshape([batch_size, -1, 2])  # [B, (gh+1)*(gw+1), 2]


def get_salient_omega(iim, grid_size, tau=8.0):
    """
    使用 Softmax Pooling 提取显著性权重。
    tau 越大，越能突出图像中最核心的显著目标（抑制背景的微弱显著性）。
    """
    B, C, H, W = iim.shape
    gh, gw = grid_size
    kh, kw = H // gh, W // gw

    # LogSumExp 聚合，防止下采样时小目标丢失
    iim_exp = torch.exp(tau * iim)
    avg_exp = F.avg_pool2d(iim_exp, kernel_size=(kh, kw), stride=(kh, kw))
    omega = torch.log(avg_exp + 1e-6) / tau

    return omega[:, 0]  # 返回形状: [B, gh, gw]


def _compute_cell_shape_energy(input_tensor, warp_image, ori_mesh, mesh, iim=None, eps=1e-6):
    """
    计算 cell 级形变几何能量 E_shape。
    返回:
        E_shape: [B, gh, gw]
    """
    del input_tensor, warp_image, iim  # 预留参数，保持接口统一

    # 原始网格 cell 宽高
    x0, y0 = ori_mesh[..., 0], ori_mesh[..., 1]  # [B, gh+1, gw+1]
    w0 = 0.5 * ((x0[:, :-1, 1:] - x0[:, :-1, :-1]) + (x0[:, 1:, 1:] - x0[:, 1:, :-1]))  # [B, gh, gw]
    h0 = 0.5 * ((y0[:, 1:, :-1] - y0[:, :-1, :-1]) + (y0[:, 1:, 1:] - y0[:, :-1, 1:]))  # [B, gh, gw]

    # 变形后网格 cell 宽高
    x, y = mesh[..., 0], mesh[..., 1]  # [B, gh+1, gw+1]
    w = 0.5 * ((x[:, :-1, 1:] - x[:, :-1, :-1]) + (x[:, 1:, 1:] - x[:, 1:, :-1]))  # [B, gh, gw]
    h = 0.5 * ((y[:, 1:, :-1] - y[:, :-1, :-1]) + (y[:, 1:, 1:] - y[:, :-1, 1:]))  # [B, gh, gw]

    # 数值稳定：防止 log 或除法出现 NaN
    w0 = w0.clamp_min(eps)
    h0 = h0.clamp_min(eps)
    w = w.clamp_min(eps)
    h = h.clamp_min(eps)

    # 局部缩放率
    sx = w / w0  # [B, gh, gw]
    sy = h / h0  # [B, gh, gw]

    # 保持当前版本几何定义
    E_iso = (torch.log(sx) - torch.log(sy)).pow(2)
    s_target = 1.0
    E_scale = (sx - s_target).pow(2) + (sy - s_target).pow(2)

    lambda_iso = 0.0
    lambda_scale = 1.0
    E_shape = lambda_iso * E_iso + lambda_scale * E_scale
    return E_shape


def _cell_energy_to_vertex_energy(E_cell, eps=1e-6):
    """
    将 cell-level energy 转换为 vertex-level energy。

    输入:
        E_cell: [B, gh, gw]
    输出:
        V_energy: [B, gh+1, gw+1]

    说明：
    每个顶点的 energy 由其相邻 cell energy 平均得到。
    这样可以把 piecewise-constant cell penalty 转换成连续场的顶点采样值，
    后续再在 deformed cell 内做局部 bilinear interpolation。
    """
    # 2. 创建全1张量：形状/设备/数据类型 和 E_cell 完全一致
    ones = torch.ones_like(E_cell)

    # cell -> four adjacent vertices
    v_sum = (
        F.pad(E_cell, (0, 1, 0, 1)) +  # top-left vertex
        F.pad(E_cell, (1, 0, 0, 1)) +  # top-right vertex
        F.pad(E_cell, (0, 1, 1, 0)) +  # bottom-left vertex
        F.pad(E_cell, (1, 0, 1, 0))    # bottom-right vertex
    )
    v_cnt = (
        F.pad(ones, (0, 1, 0, 1)) +
        F.pad(ones, (1, 0, 0, 1)) +
        F.pad(ones, (0, 1, 1, 0)) +
        F.pad(ones, (1, 0, 1, 0))
    )
    return v_sum / (v_cnt + eps)


def _soft_rasterize_quad_masks(vertices, out_h, out_w, softness=1.5, eps=1e-6):
    """
    对 deformed quad cell 生成 soft support mask。

    输入:
        vertices: [B, N, 4, 2] 批次B, N个四边形, 每个4个顶点, 每个顶点(x,y)
            顶点顺序为 p00, p01, p11, p10。
        out_h, out_w: 输出图像尺寸。
        softness: 边界软化宽度，单位近似为 pixel。（数值越大，边界越模糊）
    输出:
        masks: [B, N, out_h, out_w]
    """
    # ===================== 1. 基础张量信息提取 =====================
    B, N, _, _ = vertices.shape
    device = vertices.device
    dtype = vertices.dtype

    # ===================== 2. 生成图像所有像素的坐标 =====================
    # 生成【垂直方向y坐标】：形状 [1,1,out_h,1]，+0.5是取像素**中心**坐标
    # 像素是整数网格，真实中心在 (x+0.5, y+0.5)，这是图像处理标准做法
    py = torch.arange(out_h, device=device, dtype=dtype).view(1, 1, out_h, 1) + 0.5
    px = torch.arange(out_w, device=device, dtype=dtype).view(1, 1, 1, out_w) + 0.5

    # ===================== 3. 拆分顶点坐标，准备遍历四边形边 =====================
    xv = vertices[..., 0] # 提取所有顶点的 x 坐标：[B, N, 4]
    yv = vertices[..., 1] # 提取所有顶点的 y 坐标：[B, N, 4]

    # torch.roll：循环移位，把顶点顺序向后挪1位 → 快速获取「当前边的下一个顶点」
    # 例：顶点[0,1,2,3] → roll后[1,2,3,0]，完美匹配四边形的4条边：0→1,1→2,2→3,3→0
    xv_next = torch.roll(xv, shifts=-1, dims=2)
    yv_next = torch.roll(yv, shifts=-1, dims=2)

    # ===================== 4. 计算四边形面积 + 确定环绕方向（关键） =====================
    # 叉乘求和：计算 2倍的四边形面积（带符号），符号表示顶点是顺时针/逆时针排列
    area2 = (xv * yv_next - yv * xv_next).sum(dim=2)  # [B,N]

    # orient：方向系数，面积正→1，负→-1，用于统一「内部为正、外部为负」
    # .detach()：固定方向，不参与梯度反向传播（纯几何判断，无需学习）
    orient = torch.where(area2 >= 0, torch.ones_like(area2), -torch.ones_like(area2)).detach()
    orient = orient.unsqueeze(-1).unsqueeze(-1)  # [B,N,1,1]

    d_min = None
    for k in range(4):
        a = vertices[:, :, k, :]
        b = vertices[:, :, (k + 1) % 4, :]

        ax = a[..., 0].unsqueeze(-1).unsqueeze(-1)
        ay = a[..., 1].unsqueeze(-1).unsqueeze(-1)
        bx = b[..., 0].unsqueeze(-1).unsqueeze(-1)
        by = b[..., 1].unsqueeze(-1).unsqueeze(-1)

        ex = bx - ax
        ey = by - ay
        cross = ex * (py - ay) - ey * (px - ax)
        edge_len = torch.sqrt(ex.pow(2) + ey.pow(2) + eps)
        signed_dist = orient * cross / edge_len

        d_min = signed_dist if d_min is None else torch.minimum(d_min, signed_dist)

    return torch.sigmoid(d_min / softness)


def _bilinear_energy_inside_quad(vertices, vertex_values, out_h, out_w, clamp_uv=True, eps=1e-6):
    """
    在每个 deformed quad 内构造连续 bilinear energy field。

    输入:
        vertices: [B, N, 4, 2]
            p00, p01, p11, p10。
        vertex_values: [B, N, 4]
            与四个顶点对应的 energy 值。
    输出:
        E_pixel: [B, N, out_h, out_w]
    """
    device = vertices.device
    dtype = vertices.dtype

    py = torch.arange(out_h, device=device, dtype=dtype).view(1, 1, out_h, 1) + 0.5
    px = torch.arange(out_w, device=device, dtype=dtype).view(1, 1, 1, out_w) + 0.5

    p00 = vertices[:, :, 0, :]
    p01 = vertices[:, :, 1, :]
    p11 = vertices[:, :, 2, :]
    p10 = vertices[:, :, 3, :]

    # 用平均横向/纵向边近似局部参数坐标轴。
    # 对常见的轻度非仿射 quad，该近似比 piecewise-constant projection 更能保持连续能量变化。
    eu = 0.5 * ((p01 - p00) + (p11 - p10))  # [B,N,2]
    ev = 0.5 * ((p10 - p00) + (p11 - p01))  # [B,N,2]

    ax = p00[..., 0].unsqueeze(-1).unsqueeze(-1)
    ay = p00[..., 1].unsqueeze(-1).unsqueeze(-1)
    eux = eu[..., 0].unsqueeze(-1).unsqueeze(-1)
    euy = eu[..., 1].unsqueeze(-1).unsqueeze(-1)
    evx = ev[..., 0].unsqueeze(-1).unsqueeze(-1)
    evy = ev[..., 1].unsqueeze(-1).unsqueeze(-1)

    dx = px - ax
    dy = py - ay

    det = eux * evy - euy * evx
    det_sign = torch.where(det >= 0, torch.ones_like(det), -torch.ones_like(det))
    det_safe = det_sign * det.abs().clamp_min(eps)

    u = (dx * evy - dy * evx) / det_safe
    v = (eux * dy - euy * dx) / det_safe

    if clamp_uv:
        u = u.clamp(0.0, 1.0)
        v = v.clamp(0.0, 1.0)

    q00 = vertex_values[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    q01 = vertex_values[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    q11 = vertex_values[:, :, 2].unsqueeze(-1).unsqueeze(-1)
    q10 = vertex_values[:, :, 3].unsqueeze(-1).unsqueeze(-1)

    E_pixel = (
        (1.0 - u) * (1.0 - v) * q00 +
        u * (1.0 - v) * q01 +
        u * v * q11 +
        (1.0 - u) * v * q10
    )
    return E_pixel


def _build_deformation_aware_bilinear_energy(
    E_cell,
    mesh,
    out_h,
    out_w,
    softness=None,
    detach_projection_geometry=True,
    chunk_size=32,
    eps=1e-6,
):
    """
    Deformation-aware bilinear energy field.

    与直接 soft projection 的区别：
    - direct projection: 一个 cell 内部所有像素共享同一个 E_ij，容易退化为 grid-level 加权；
    - 本函数: 先将 cell energy 平均到 mesh vertices，
      再在每个 deformed cell 内使用四个顶点 energy 做 bilinear interpolation，
      因此 cell 内部具有连续变化的 pixel-level penalty。

    输入:
        E_cell: [B, gh, gw]
        mesh:   [B, gh+1, gw+1, 2]
    输出:
        D_pixel: [B, 1, out_h, out_w]
    """
    B, gh, gw = E_cell.shape
    device = E_cell.device
    dtype = E_cell.dtype
    N = gh * gw

    if softness is None:
        cell_size = min(float(out_h) / float(max(gh, 1)), float(out_w) / float(max(gw, 1)))
        softness = max(1.0, 0.05 * cell_size)

    mesh_geo = mesh.detach() if detach_projection_geometry else mesh

    # 1) cell energy -> vertex energy: [B, gh+1, gw+1]
    V_energy = _cell_energy_to_vertex_energy(E_cell, eps=eps)

    # 2) 每个 cell 的四个变形后顶点坐标: p00, p01, p11, p10
    p00 = mesh_geo[:, :-1, :-1, :]
    p01 = mesh_geo[:, :-1, 1:, :]
    p11 = mesh_geo[:, 1:, 1:, :]
    p10 = mesh_geo[:, 1:, :-1, :]
    vertices = torch.stack([p00, p01, p11, p10], dim=3).reshape(B, N, 4, 2)

    # 3) 每个 cell 四个顶点对应的 energy
    q00 = V_energy[:, :-1, :-1]
    q01 = V_energy[:, :-1, 1:]
    q11 = V_energy[:, 1:, 1:]
    q10 = V_energy[:, 1:, :-1]
    vertex_values = torch.stack([q00, q01, q11, q10], dim=3).reshape(B, N, 4)

    numerator = torch.zeros((B, out_h, out_w), device=device, dtype=dtype)
    denominator = torch.zeros((B, out_h, out_w), device=device, dtype=dtype)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        v_chunk = vertices[:, start:end, :, :]
        val_chunk = vertex_values[:, start:end, :]

        masks = _soft_rasterize_quad_masks(
            v_chunk,
            out_h=out_h,
            out_w=out_w,
            softness=softness,
            eps=eps,
        )
        E_local = _bilinear_energy_inside_quad(
            v_chunk,
            val_chunk,
            out_h=out_h,
            out_w=out_w,
            clamp_uv=True,
            eps=eps,
        )

        numerator = numerator + (masks * E_local).sum(dim=1)
        denominator = denominator + masks.sum(dim=1)

    return (numerator / (denominator + eps)).unsqueeze(1)


def _build_pixel_shape_energy(E_shape, mesh, out_h, out_w):
    """
    将 cell-level shape penalty 构造为 deformation-aware continuous pixel-level energy field。
    """
    return _build_deformation_aware_bilinear_energy(
        E_cell=E_shape,
        mesh=mesh,
        out_h=out_h,
        out_w=out_w,
        softness=None,
        detach_projection_geometry=True,
        chunk_size=32,
    )


def _build_pixel_rect_energy(E_rect, mesh, out_h, out_w):
    """
    将 cell-level rect penalty 构造为 deformation-aware continuous pixel-level energy field。
    """
    return _build_deformation_aware_bilinear_energy(
        E_cell=E_rect,
        mesh=mesh,
        out_h=out_h,
        out_w=out_w,
        softness=None,
        detach_projection_geometry=True,
        chunk_size=32,
    )


def _warp_saliency_with_mesh(iim, mesh, out_h, out_w, eps=1e-6):
    """
    将输入显著图按当前 mesh 进行 TPS warp。
    输入:
        iim:  [B, 1, H, W]
        mesh: [B, gh+1, gw+1, 2]
    输出:
        S_warp: [B, 1, out_h, out_w]
    """
    b, _, in_h, in_w = iim.shape
    ratio_w = float(out_w) / float(max(in_w, 1))

    rigid_mesh = get_rigid_mesh(b, out_h, out_w, device=iim.device, dtype=iim.dtype)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, out_h, out_w, eps=eps)
    norm_mesh = get_norm_mesh(mesh, out_h, out_w, eps=eps)

    stack_rigid_mesh = get_stack_mesh(norm_rigid_mesh)
    stack_mesh = get_stack_mesh(norm_mesh)

    # 与 test_output_omega.py 保持一致：x 向压缩场景走该分支
    if ratio_w <= 1.0:
        s_warp = torch_tps_transform.transformer(
            iim, stack_mesh, stack_rigid_mesh, (out_h, out_w)
        )
    else:
        s_warp = torch_tps_transform.transformer(
            iim, stack_rigid_mesh, stack_mesh, (out_h, out_w)
        )

    return s_warp.clamp(0.0, 1.0)


def salient_shape_loss(
    input_tensor,
    warp_image,
    ori_mesh,
    mesh,
    iim=None,
    enable=False,
    return_items=False,
):
    """
    纯像素级显著 shape loss：
    L_shape = sum(S_warp * D_shape) / (sum(S_warp) + eps)
    """
    if not enable:
        zero = torch.zeros((), device=mesh.device, dtype=mesh.dtype)
        if return_items:
            return zero, {}
        return zero

    eps = 1e-6
    b = mesh.size(0)
    img_h, img_w = input_tensor.shape[2:]
    out_h, out_w = warp_image.shape[2:]

    # 1) cell 级几何能量: [B, gh, gw]
    E_shape = _compute_cell_shape_energy(
        input_tensor=input_tensor,
        warp_image=warp_image,
        ori_mesh=ori_mesh,
        mesh=mesh,
        iim=iim,
        eps=eps,
    )

    # 2) 上采样到像素级能量图: [B, 1, out_h, out_w]
    D_shape = _build_pixel_shape_energy(E_shape, mesh, out_h, out_w)

    # 3) 准备显著图
    if iim is None:
        iim_use = torch.ones((b, 1, img_h, img_w), device=mesh.device, dtype=mesh.dtype)
    else:
        iim_use = iim.to(device=mesh.device, dtype=mesh.dtype).clamp(0.0, 1.0)

    # 4) warp 显著图到输出空间: [B, 1, out_h, out_w]
    S_warp = _warp_saliency_with_mesh(iim_use, mesh, out_h, out_w, eps=eps)

    # 5) 逐样本像素加权平均
    numerator = (S_warp * D_shape).sum(dim=(1, 2, 3))
    denominator = S_warp.sum(dim=(1, 2, 3)) + eps
    L_per_sample = numerator / denominator

    total_loss = L_per_sample.mean()

    if return_items:
        items = {
            "L_shape_pixel": total_loss.detach(),
            "E_shape_mean": E_shape.mean().detach(),
            "E_shape_max": E_shape.max().detach(),
            "S_warp_mean": S_warp.mean().detach(),
            "D_shape_mean": D_shape.mean().detach(),
        }
        return total_loss, items

    return total_loss

def mesh_orthogonal_reg_loss(input_tensor, warp_image, ori_mesh, mesh, enable=True):
    """
    网格正交正则化损失（纯网格级，无像素插值/无显著图）
    约束网格单元保持水平竖直正交，水平与垂直边 1:1 等权重约束
    用于抑制网格畸变、保持几何规整性
    """
    del ori_mesh

    if not enable:
        return torch.zeros((), device=mesh.device, dtype=mesh.dtype)

    eps = 1e-6
    x, y = mesh[..., 0], mesh[..., 1]

    # 计算网格单元四条边的差分
    dx_top = x[:, :-1, 1:] - x[:, :-1, :-1]
    dy_top = y[:, :-1, 1:] - y[:, :-1, :-1]

    dx_bottom = x[:, 1:, 1:] - x[:, 1:, :-1]
    dy_bottom = y[:, 1:, 1:] - y[:, 1:, :-1]

    dx_left = x[:, 1:, :-1] - x[:, :-1, :-1]
    dy_left = y[:, 1:, :-1] - y[:, :-1, :-1]

    dx_right = x[:, 1:, 1:] - x[:, :-1, 1:]
    dy_right = y[:, 1:, 1:] - y[:, :-1, 1:]

    # 水平/垂直误差计算
    horizontal_error = dy_top.pow(2) + dy_bottom.pow(2)
    vertical_error = dx_left.pow(2) + dx_right.pow(2)

    # 1:1 等权重约束
    cell_ortho_error = horizontal_error + vertical_error

    # 平均得到最终正则损失
    orthogonal_reg_loss = cell_ortho_error.mean()

    return orthogonal_reg_loss

def salient_rect_loss(input_tensor, warp_image, ori_mesh, mesh, iim=None, enable=False):
    """
    [视觉软约束] 显著区域像素级横平竖直损失
    作用：防止显著区域的网格发生严重的剪切(Shear)畸变。
    """
    del ori_mesh

    if not enable:
        return torch.zeros((), device=mesh.device, dtype=mesh.dtype)

    eps = 1e-6
    B = mesh.size(0)
    img_h, img_w = input_tensor.shape[2:]
    out_h, out_w = warp_image.shape[2:]

    if iim is None:
        iim_use = torch.ones((B, 1, img_h, img_w), device=mesh.device, dtype=mesh.dtype)
    else:
        iim_use = iim.to(device=mesh.device, dtype=mesh.dtype).clamp(0.0, 1.0)

    x, y = mesh[..., 0], mesh[..., 1]

    dx_top = x[:, :-1, 1:] - x[:, :-1, :-1]
    dy_top = y[:, :-1, 1:] - y[:, :-1, :-1]
    dx_bottom = x[:, 1:, 1:] - x[:, 1:, :-1]
    dy_bottom = y[:, 1:, 1:] - y[:, 1:, :-1]

    dx_left = x[:, 1:, :-1] - x[:, :-1, :-1]
    dy_left = y[:, 1:, :-1] - y[:, :-1, :-1]
    dx_right = x[:, 1:, 1:] - x[:, :-1, 1:]
    dy_right = y[:, 1:, 1:] - y[:, :-1, 1:]

    # 横向边要求 dy 趋于 0，纵向边要求 dx 趋于 0
    err_top_h = dy_top.pow(2) / (dx_top.pow(2) + dy_top.pow(2) + eps)
    err_bottom_h = dy_bottom.pow(2) / (dx_bottom.pow(2) + dy_bottom.pow(2) + eps)
    err_left_v = dx_left.pow(2) / (dx_left.pow(2) + dy_left.pow(2) + eps)
    err_right_v = dx_right.pow(2) / (dx_right.pow(2) + dy_right.pow(2) + eps)

    E_rect = 0.25 * (err_top_h + err_bottom_h + err_left_v + err_right_v)

    D_rect = _build_pixel_rect_energy(E_rect, mesh, out_h, out_w)
    S_warp = _warp_saliency_with_mesh(iim_use, mesh, out_h, out_w, eps=eps).detach()

    numerator = (S_warp * D_rect).sum(dim=(1, 2, 3))
    denominator = S_warp.sum(dim=(1, 2, 3)) + eps
    L_rect = (numerator / denominator).mean()

    return L_rect


def global_rect_loss(input_tensor, warp_image, ori_mesh, mesh, iim=None, enable=True):
    """
    [背景弱约束] 背景区域像素级横平竖直损失

    作用：
    - 只约束非显著区域（background）
    - 用 warp 后显著图的反图 (1 - S_warp) 作为像素级背景权重
    - 显著区域交给 salient_rect_loss
    - 背景区域负责维持直线结构和整体规整性
    """
    del ori_mesh

    if not enable:
        return torch.zeros((), device=mesh.device, dtype=mesh.dtype)

    eps = 1e-6
    B = mesh.size(0)
    img_h, img_w = input_tensor.shape[2:]
    out_h, out_w = warp_image.shape[2:]

    if iim is None:
        iim_use = torch.zeros((B, 1, img_h, img_w), device=mesh.device, dtype=mesh.dtype)
    else:
        iim_use = iim.to(device=mesh.device, dtype=mesh.dtype).clamp(0.0, 1.0)

    # 1) 计算局部 rect 误差（与 salient_rect_loss 一致）
    x, y = mesh[..., 0], mesh[..., 1]

    dx_top = x[:, :-1, 1:] - x[:, :-1, :-1]
    dy_top = y[:, :-1, 1:] - y[:, :-1, :-1]
    dx_bottom = x[:, 1:, 1:] - x[:, 1:, :-1]
    dy_bottom = y[:, 1:, 1:] - y[:, 1:, :-1]

    dx_left = x[:, 1:, :-1] - x[:, :-1, :-1]
    dy_left = y[:, 1:, :-1] - y[:, :-1, :-1]
    dx_right = x[:, 1:, 1:] - x[:, :-1, 1:]
    dy_right = y[:, 1:, 1:] - y[:, :-1, 1:]

    err_top_h = dy_top.pow(2) / (dx_top.pow(2) + dy_top.pow(2) + eps)
    err_bottom_h = dy_bottom.pow(2) / (dx_bottom.pow(2) + dy_bottom.pow(2) + eps)
    err_left_v = dx_left.pow(2) / (dx_left.pow(2) + dy_left.pow(2) + eps)
    err_right_v = dx_right.pow(2) / (dx_right.pow(2) + dy_right.pow(2) + eps)

    E_rect = 0.25 * (err_top_h + err_bottom_h + err_left_v + err_right_v)  # [B, gh, gw]

    # 2) 将 cell 级 rect 能量和显著图都放到输出像素空间
    D_rect = _build_pixel_rect_energy(E_rect, mesh, out_h, out_w)
    S_warp = _warp_saliency_with_mesh(iim_use, mesh, out_h, out_w, eps=eps).detach()
    B_warp = (1.0 - S_warp).clamp(0.0, 1.0)

    # 3) 只对背景像素区域做加权平均
    numerator = (B_warp * D_rect).sum(dim=(1, 2, 3))
    denominator = B_warp.sum(dim=(1, 2, 3)) + eps
    L_rect_bg = (numerator / denominator).mean()

    return L_rect_bg


def folding_loss(
    mesh,
    out_w,
    out_h,
    grid_w,
    grid_h,
    eta=0.05,
    area_eta=0.02,
    lambda_max=0.1,
    lambda_area=0.5,
    return_items=False,
):
    """
    Anti-fold loss with printable sub-items:
    1) edge_mean
    2) edge_max
    3) area_mean
    4) area_max
    """

    x = mesh[..., 0]
    y = mesh[..., 1]

    # 1) 边间距约束
    dx = x[:, :, 1:] - x[:, :, :-1]  # [B, gh+1, gw]
    dy = y[:, 1:, :] - y[:, :-1, :]  # [B, gh, gw+1]

    gx = out_w / grid_w
    gy = out_h / grid_h
    min_dx = eta * gx
    min_dy = eta * gy

    bad_x = F.relu((min_dx - dx) / gx).pow(2)
    bad_y = F.relu((min_dy - dy) / gy).pow(2)

    edge_mean = bad_x.mean() + bad_y.mean()
    edge_max = bad_x.amax() + bad_y.amax()

    L_edge = edge_mean + lambda_max * edge_max

    # 2) 面积 barrier
    p00 = mesh[:, :-1, :-1, :]
    p01 = mesh[:, :-1, 1:, :]
    p10 = mesh[:, 1:, :-1, :]
    p11 = mesh[:, 1:, 1:, :]

    def cross2(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    area1 = cross2(p01 - p00, p10 - p00)
    area2 = cross2(p11 - p01, p10 - p01)

    min_area = area_eta * gx * gy
    base_area = gx * gy
    bad_a1 = F.relu((min_area - area1) / base_area).pow(2)
    bad_a2 = F.relu((min_area - area2) / base_area).pow(2)

    area_mean = bad_a1.mean() + bad_a2.mean()
    area_max = bad_a1.amax() + bad_a2.amax()

    L_area = area_mean + lambda_max * area_max

    # 3) 总 fold loss
    total_loss = L_edge + lambda_area * L_area

    # 4) 返回子项
    if return_items:
        items = {
            "edge_mean": edge_mean.detach(),
            "edge_max": edge_max.detach(),
            "area_mean": area_mean.detach(),
            "area_max": area_max.detach(),
            "L_edge": L_edge.detach(),
            "L_area": L_area.detach(),
            "total_fold": total_loss.detach(),
        }
        return total_loss, items

    return total_loss


def dynamic_folding_loss(mesh, out_w, out_h, grid_w, grid_h, iim=None, enable=True):
    """
    [物理硬约束] 自适应防翻折损失
    """
    if not enable:
        return torch.zeros((), device=mesh.device, dtype=mesh.dtype)

    x, y = mesh[..., 0], mesh[..., 1]
    dx = x[:, :, 1:] - x[:, :, :-1]  # [B, gh+1, gw]
    dy = y[:, 1:, :] - y[:, :-1, :]  # [B, gh, gw+1]

    gx = out_w / grid_w
    gy = out_h / grid_h

    # 计算动态底线 eta_map
    if iim is not None:
        iim = iim.to(device=mesh.device, dtype=mesh.dtype).clamp(0.0, 1.0)
        omega = F.adaptive_avg_pool2d(iim, (grid_h, grid_w))[:, 0]
    else:
        omega = torch.zeros((mesh.size(0), grid_h, grid_w), device=mesh.device, dtype=mesh.dtype)

    eta_min = 0.1
    alpha = 0.15
    eta_map = eta_min + alpha * omega  # [B, gh, gw]

    # 将中心保护底线广播到网格边上（取相邻更严苛底线）
    eta_map_x = torch.max(F.pad(eta_map, (0, 0, 1, 0)), F.pad(eta_map, (0, 0, 0, 1)))  # [B, gh+1, gw]
    eta_map_y = torch.max(F.pad(eta_map, (1, 0, 0, 0)), F.pad(eta_map, (0, 1, 0, 0)))  # [B, gh, gw+1]

    min_dx = eta_map_x * gx
    min_dy = eta_map_y * gy

    Lx = F.relu(min_dx - dx).mean()
    Ly = F.relu(min_dy - dy).mean()

    return Lx + Ly


def global_smoothness_loss(mesh, ori_mesh):
    """
    一阶 L2 位移平滑损失。
    """
    # 1) 位移场: [B, gh+1, gw+1, 2]
    motion = mesh - ori_mesh

    # 2) 分离 x/y 位移
    u = motion[..., 0]
    v = motion[..., 1]

    # 3) 水平方向一阶差分
    du_dx = u[:, :, 1:] - u[:, :, :-1]  # [B, gh+1, gw]
    dv_dx = v[:, :, 1:] - v[:, :, :-1]  # [B, gh+1, gw]

    # 4) 垂直方向一阶差分
    du_dy = u[:, 1:, :] - u[:, :-1, :]  # [B, gh, gw+1]
    dv_dy = v[:, 1:, :] - v[:, :-1, :]  # [B, gh, gw+1]

    # 5) L2 平均
    loss_x = du_dx.pow(2).mean() + dv_dx.pow(2).mean()
    loss_y = du_dy.pow(2).mean() + dv_dy.pow(2).mean()

    return loss_x + loss_y

def global_smoothness_angle_loss(mesh, ori_mesh=None, eps=1e-6):
    """
    Inter-grid angle smoothness loss, converted from the TensorFlow version.

    说明：
    - 保留 ori_mesh 参数，是为了兼容当前训练脚本中的调用方式：
        smooth_loss = global_smoothness_loss(mesh, ori_mesh)
    - 该 loss 不再约束 motion 的一阶差分幅度，而是约束相邻网格边的方向一致性。
    - 当相邻边方向越一致时，cos 越接近 1，1 - cos 越接近 0。
    """
    del ori_mesh  # angle-based inter-grid loss 不使用 ori_mesh

    # -------------------------
    # 1) 水平方向边向量
    # -------------------------
    # 对应 TF:
    # w_edges = train_mesh[:,:,0:grid_w,:] - train_mesh[:,:,1:grid_w+1,:]
    # shape: [B, grid_h+1, grid_w, 2]
    w_edges = mesh[:, :, :-1, :] - mesh[:, :, 1:, :]

    # 相邻水平边之间的 cosine
    # 对应 TF:
    # cos_w = cos(w_edges[:,:,0:grid_w-1,:], w_edges[:,:,1:grid_w,:])
    # shape: [B, grid_h+1, grid_w-1]
    cos_w = F.cosine_similarity(
        w_edges[:, :, :-1, :],
        w_edges[:, :, 1:, :],
        dim=-1,
        eps=eps,
    )

    delta_w_angle = 1.0 - cos_w

    # -------------------------
    # 2) 垂直方向边向量
    # -------------------------
    # 对应 TF:
    # h_edges = train_mesh[:,0:grid_h,:,:] - train_mesh[:,1:grid_h+1,:,:]
    # shape: [B, grid_h, grid_w+1, 2]
    h_edges = mesh[:, :-1, :, :] - mesh[:, 1:, :, :]

    # 相邻垂直边之间的 cosine
    # 对应 TF:
    # cos_h = cos(h_edges[:,0:grid_h-1,:,:], h_edges[:,1:grid_h,:,:])
    # shape: [B, grid_h-1, grid_w+1]
    cos_h = F.cosine_similarity(
        h_edges[:, :-1, :, :],
        h_edges[:, 1:, :, :],
        dim=-1,
        eps=eps,
    )

    delta_h_angle = 1.0 - cos_h

    return delta_w_angle.mean() + delta_h_angle.mean()

def _get_cell_centers_from_mesh(mesh):
    """
    根据 mesh 顶点坐标，计算每个 cell 的中心。
    输入:
        mesh: [B, gh+1, gw+1, 2]
    输出:
        cx, cy: [B, gh, gw]
    """
    x = mesh[..., 0]
    y = mesh[..., 1]

    cx = 0.25 * (x[:, :-1, :-1] + x[:, :-1, 1:] + x[:, 1:, :-1] + x[:, 1:, 1:])
    cy = 0.25 * (y[:, :-1, :-1] + y[:, :-1, 1:] + y[:, 1:, :-1] + y[:, 1:, 1:])
    return cx, cy


def _core_weight_from_omega(omega, core_ratio=0.6, core_temp=0.08, eps=1e-6):
    """
    从 omega 中提取高置信核心权重。
    输入:
        omega: [B, gh, gw]
    输出:
        omega_core: [B, gh, gw]
    """
    omega_max = omega.amax(dim=(1, 2), keepdim=True)  # [B,1,1]
    thr = core_ratio * omega_max

    gate = torch.sigmoid((omega - thr) / core_temp)
    omega_core = omega * gate
    omega_core = omega_core / (omega_core.sum(dim=(1, 2), keepdim=True) + eps)
    return omega_core


def _weighted_centroid(cx, cy, weight, norm_w, norm_h, eps=1e-6):
    """
    根据 cell center + weight 计算归一化重心。
    输入:
        cx, cy: [B, gh, gw]
        weight: [B, gh, gw]
    输出:
        G: [B, 2], 坐标归一化到 [0,1]
    """
    del eps
    gx = (weight * (cx / float(norm_w))).sum(dim=(1, 2))
    gy = (weight * (cy / float(norm_h))).sum(dim=(1, 2))
    return torch.stack([gx, gy], dim=1)  # [B,2]


def salient_reposition_loss(
    input_tensor,
    warp_image,
    ori_mesh,
    mesh,
    iim=None,
    enable=False,
    use_softmax_pool=True,
    omega_tau=8.0,
    core_ratio=0.6,
    core_temp=0.08,
    center_band=0.15,
    return_items=False,
):
    if not enable:
        zero = torch.zeros((), device=mesh.device, dtype=mesh.dtype)
        if return_items:
            return zero, {}
        return zero

    eps = 1e-6
    B = mesh.size(0)
    img_h, img_w = input_tensor.shape[2:]
    out_h, out_w = warp_image.shape[2:]

    # 1) 提取核心置信区
    if iim is None:
        iim = torch.ones((B, 1, img_h, img_w), device=mesh.device, dtype=mesh.dtype)
    else:
        iim = iim.to(device=mesh.device, dtype=mesh.dtype).clamp(0.0, 1.0)

    if use_softmax_pool:
        omega = get_salient_omega(iim, (grid_h, grid_w), tau=omega_tau)
    else:
        omega = F.adaptive_avg_pool2d(iim, (grid_h, grid_w))[:, 0]

    omega_core = _core_weight_from_omega(omega, core_ratio=core_ratio, core_temp=core_temp, eps=eps)

    # 2) 当前重心 G 和 原重心 G0
    cx0, cy0 = _get_cell_centers_from_mesh(ori_mesh)
    cx, cy = _get_cell_centers_from_mesh(mesh)

    G0 = _weighted_centroid(cx0, cy0, omega_core, img_w, img_h, eps=eps)  # [B,2]
    G = _weighted_centroid(cx, cy, omega_core, out_w, out_h, eps=eps)  # [B,2]

    # 3) 定义锚点
    anchors = torch.tensor(
        [
            [0.5, 0.5],
            [1.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 2.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0],
        ],
        device=mesh.device,
        dtype=mesh.dtype,
    )

    # 4) 到各锚点的距离平方（用于分配）
    dist2 = ((G.unsqueeze(1) - anchors.unsqueeze(0)) ** 2).sum(dim=2)  # [B,5]

    # 5) 象限掩码限制
    gx0, gy0 = G0[:, 0], G0[:, 1]
    allow = torch.zeros((B, 5), device=mesh.device, dtype=mesh.dtype)
    allow[:, 0] = 1.0

    near_center = ((gx0 - 0.5).abs() < center_band) & ((gy0 - 0.5).abs() < center_band)
    left, right, top, bot = gx0 < 0.5, gx0 >= 0.5, gy0 < 0.5, gy0 >= 0.5

    allow[left & top, 1] = 1.0
    allow[right & top, 2] = 1.0
    allow[left & bot, 3] = 1.0
    allow[right & bot, 4] = 1.0
    allow[near_center, 1:] = 1.0

    # 6) 硬分配 + L1 推力
    big_num = 1e4
    masked_dist2 = dist2 + (1.0 - allow) * big_num

    best_anchor_idx = torch.argmin(masked_dist2, dim=1).detach()
    target_anchors = anchors[best_anchor_idx]  # [B, 2]

    L_move = F.l1_loss(G, target_anchors, reduction="mean")

    if return_items:
        items = {
            "L_move": L_move.detach(),
            "L_keep": torch.tensor(0.0, device=mesh.device, dtype=mesh.dtype),
            "shift_norm": torch.norm(G - G0, dim=1).mean().detach(),
            "target_idx": best_anchor_idx.float().mean().detach(),
        }
        return L_move, items

    return L_move


def salient_aesthetic_loss(
    input_tensor,
    warp_image,
    ori_mesh,
    mesh,
    iim=None,
    enable=False,
    tau_anchor=0.02,
    lambda_disp=0.1,
    use_softmax_pool=True,
    omega_tau=5.0,
):
    """
    [美感软约束] 显著主体重心构图损失。
    """
    if not enable:
        return torch.zeros((), device=mesh.device, dtype=mesh.dtype)

    eps = 1e-6
    B = mesh.size(0)

    img_h, img_w = input_tensor.shape[2:]
    out_h, out_w = warp_image.shape[2:]

    # 1) 显著图 -> 网格权重 omega
    if iim is None:
        iim = torch.ones((B, 1, img_h, img_w), device=mesh.device, dtype=mesh.dtype)
    else:
        iim = iim.to(device=mesh.device, dtype=mesh.dtype).clamp(0.0, 1.0)

    if use_softmax_pool:
        omega = get_salient_omega(iim, (grid_h, grid_w), tau=omega_tau)  # [B, gh, gw]
    else:
        omega = F.adaptive_avg_pool2d(iim, (grid_h, grid_w))[:, 0]  # [B, gh, gw]

    omega_sum = omega.sum(dim=(1, 2), keepdim=True).clamp_min(eps)  # [B,1,1]

    # 2) 计算输出图每个 cell 中心
    x = mesh[..., 0]  # [B, gh+1, gw+1]
    y = mesh[..., 1]

    cx = 0.25 * (x[:, :-1, :-1] + x[:, :-1, 1:] + x[:, 1:, :-1] + x[:, 1:, 1:])  # [B, gh, gw]
    cy = 0.25 * (y[:, :-1, :-1] + y[:, :-1, 1:] + y[:, 1:, :-1] + y[:, 1:, 1:])  # [B, gh, gw]

    cx_norm = cx / float(out_w)
    cy_norm = cy / float(out_h)

    gx = (omega * cx_norm).sum(dim=(1, 2), keepdim=True) / omega_sum  # [B,1,1]
    gy = (omega * cy_norm).sum(dim=(1, 2), keepdim=True) / omega_sum  # [B,1,1]
    G = torch.cat([gx, gy], dim=2).squeeze(1)  # [B,2]

    # 3) 计算原图重心 G0（位移约束）
    x0 = ori_mesh[..., 0]
    y0 = ori_mesh[..., 1]

    cx0 = 0.25 * (x0[:, :-1, :-1] + x0[:, :-1, 1:] + x0[:, 1:, :-1] + x0[:, 1:, 1:])
    cy0 = 0.25 * (y0[:, :-1, :-1] + y0[:, :-1, 1:] + y0[:, 1:, :-1] + y0[:, 1:, 1:])

    cx0_norm = cx0 / float(img_w)
    cy0_norm = cy0 / float(img_h)

    gx0 = (omega * cx0_norm).sum(dim=(1, 2), keepdim=True) / omega_sum
    gy0 = (omega * cy0_norm).sum(dim=(1, 2), keepdim=True) / omega_sum
    G0 = torch.cat([gx0, gy0], dim=2).squeeze(1)  # [B,2]

    # 4) 美学锚点：中心 + 四个三分交点
    anchors = torch.tensor(
        [
            [0.5, 0.5],
            [1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 2.0 / 3.0],
            [2.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0],
        ],
        device=mesh.device,
        dtype=mesh.dtype,
    )

    # 5) soft-min 近似最近锚点
    dist2 = ((G.unsqueeze(1) - anchors.unsqueeze(0)) ** 2).sum(dim=2)  # [B,5]
    L_anchor = (-tau_anchor * torch.logsumexp(-dist2 / tau_anchor, dim=1)).mean()

    # 6) 限制不要偏离原布局太远
    L_disp = ((G - G0) ** 2).sum(dim=1).mean()

    # 7) 总损失
    L_beauty = L_anchor + lambda_disp * L_disp
    return L_beauty


def b_loss(warp_image, mesh, motion):
    """
    Boundary tangential loss。
    仅约束边界切向位移不要过大：
    - 上下边界：限制 x 向位移
    - 左右边界：限制 y 向位移
    """
    del mesh
    out_h, out_w = warp_image.shape[2:]

    du = out_w / (4.0 * grid_w)  # 水平边界上允许的切向 x 位移阈值
    dv = out_h / (4.0 * grid_h)  # 垂直边界上允许的切向 y 位移阈值

    # 去掉四角，避免重复统计
    motion_x_top = motion[:, 0, 1:-1, 0].abs()
    motion_x_bottom = motion[:, -1, 1:-1, 0].abs()
    loss_tb_x = torch.relu(motion_x_top - du).mean() + torch.relu(motion_x_bottom - du).mean()

    motion_y_left = motion[:, 1:-1, 0, 1].abs()
    motion_y_right = motion[:, 1:-1, -1, 1].abs()
    loss_lr_y = torch.relu(motion_y_left - dv).mean() + torch.relu(motion_y_right - dv).mean()

    loss = loss_tb_x + loss_lr_y
    return loss

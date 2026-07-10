import argparse
import os
import glob
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from network import build_model, Network
from dataset import TrainDataset

# ===== 导入损失函数 =====
from loss4 import (
    salient_shape_loss,
    salient_rect_loss,
    b_loss,
    folding_loss,
    global_smoothness_loss,
    salient_aesthetic_loss,
    grid_w,
    grid_h,
)

last_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def safe_item(x, default=0.0):
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    return float(x.item())



def train(args):
    batch_size = args.batch_size
    USE_SHAPE = args.enable_w_l  # 复用原参数名，控制是否开启形状保护
    USE_RECT = args.enable_g_l   # 复用原参数名，控制是否开启横平竖直

    set_seed(args.seed)

    run_dir = os.path.join(last_path, args.log_root, args.exp_name)
    ckpt_dir = os.path.join(last_path, args.ckpt_root, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_dir)

    train_data = TrainDataset(
        data_path=args.train_path,
        saliency_dirname=args.saliency_root,
        use_saliency=True,
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    print("len(train_data) =", len(train_data))
    print("len(train_loader) =", len(train_loader), "batch_size =", batch_size)

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=3e-5, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97, last_epoch=-1)

    ckpt_list = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    ckpt_list.sort()
    if args.resume and len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.97, last_epoch=start_epoch - 1
        )
        print('resume model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from scratch!')

    # ==========================================
    # Hyper-parameters
    # ==========================================
    lam_fold = 15.0
    lam_shape = 40.0
    lam_smooth = 0.02
    lam_b = 0.0
    lam_rect = 150.0
    # lam_beauty = 2.0
    # ==========================================

    print("################## start training #######################")
    print(f"s_target_mode = {args.s_target_mode}")
    score_print_fre = 50
    start_time = time.time()

    for epoch in range(start_epoch, args.max_epoch):
        net.train()

        # 当前 epoch 的累计 loss
        loss_sigma = 0.0
        shape_sigma = 0.0
        rect_sigma = 0.0
        fold_sigma = 0.0
        smooth_sigma = 0.0
        boundary_sigma = 0.0
        # beauty_sigma = 0.0

        # s_target 调试信息累计
        s_target_sigma = 0.0
        s_actual_sigma = 0.0
        target_gap_sigma = 0.0
        iso_sigma = 0.0

        # fold 四个子项累计
        fold_edge_mean_sigma = 0.0
        fold_edge_max_sigma = 0.0
        fold_area_mean_sigma = 0.0
        fold_area_max_sigma = 0.0

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader):
            inpu_tensor = batch_value[0].float()
            if len(batch_value) == 3:
                iim_tensor = batch_value[2].float()
            else:
                iim_tensor = None

            if torch.cuda.is_available():
                inpu_tensor = inpu_tensor.cuda()
                if iim_tensor is not None:
                    iim_tensor = iim_tensor.cuda()

            optimizer.zero_grad()

            # 1. 前向传播
            batch_out = build_model(net, inpu_tensor, is_training=True)
            warp_image = batch_out['warp_primary']
            mesh = batch_out['mesh_pri']
            ori_mesh = batch_out['ori_mesh']
            motion = batch_out['motion_primary']
            _, _, out_h, out_w = warp_image.shape

            # 2. 计算损失
            shape_loss, shape_debug = salient_shape_loss(
                inpu_tensor,
                warp_image,
                ori_mesh,
                mesh,
                iim=iim_tensor,
                enable=USE_SHAPE,
                s_target_mode=args.s_target_mode,
                return_debug=True,
            )

            rect_loss = salient_rect_loss(
                inpu_tensor,
                warp_image,
                ori_mesh,
                mesh,
                iim=iim_tensor,
                enable=USE_RECT,
            )

            fold_loss, fold_items = folding_loss(
                mesh,
                out_w,
                out_h,
                grid_w,
                grid_h,
                eta=0.15,
                area_eta=0.15,
                lambda_max=2.0,
                lambda_area=0.8,
                return_items=True,
            )

            smooth_loss = global_smoothness_loss(mesh, ori_mesh)
            boundary_loss = b_loss(warp_image, mesh, motion)

            # 3. 权重加和
            shape_scaled = lam_shape * shape_loss
            rect_scaled = lam_rect * rect_loss
            fold_scaled = lam_fold * fold_loss
            smooth_scaled = lam_smooth * smooth_loss
            boundary_scaled = lam_b * boundary_loss
            total_loss = shape_scaled + fold_scaled + smooth_scaled + boundary_scaled + rect_scaled

            # 4. 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            # 5. 累加日志
            loss_sigma += safe_item(total_loss)
            shape_sigma += safe_item(shape_scaled)
            rect_sigma += safe_item(rect_scaled)
            fold_sigma += safe_item(fold_scaled)
            smooth_sigma += safe_item(smooth_scaled)
            boundary_sigma += safe_item(boundary_scaled)

            s_target_sigma += safe_item(shape_debug.get("s_target_mean", 0.0))
            s_actual_sigma += safe_item(shape_debug.get("s_actual_mean", 0.0))
            target_gap_sigma += safe_item(shape_debug.get("target_gap_mean", 0.0))
            iso_sigma += safe_item(shape_debug.get("iso_mean", 0.0))

            fold_edge_mean_sigma += safe_item(fold_items.get("edge_mean", 0.0))
            fold_edge_max_sigma += safe_item(fold_items.get("edge_max", 0.0))
            fold_area_mean_sigma += safe_item(fold_items.get("area_mean", 0.0))
            fold_area_max_sigma += safe_item(fold_items.get("area_max", 0.0))

            # 6. 打印与 TensorBoard
            if i % score_print_fre == 0 and i != 0:
                avg_loss = loss_sigma / score_print_fre
                avg_shape = shape_sigma / score_print_fre
                avg_rect = rect_sigma / score_print_fre
                avg_fold = fold_sigma / score_print_fre
                avg_smooth = smooth_sigma / score_print_fre
                avg_boundary = boundary_sigma / score_print_fre

                avg_s_target = s_target_sigma / score_print_fre
                avg_s_actual = s_actual_sigma / score_print_fre
                avg_target_gap = target_gap_sigma / score_print_fre
                avg_iso = iso_sigma / score_print_fre

                avg_fold_edge_mean = fold_edge_mean_sigma / score_print_fre
                avg_fold_edge_max = fold_edge_max_sigma / score_print_fre
                avg_fold_area_mean = fold_area_mean_sigma / score_print_fre
                avg_fold_area_max = fold_area_max_sigma / score_print_fre

                print(
                    "Epoch[{:0>3}/{:0>3}] Iter[{:<4d}/{:<4d}] "
                    "Total: {:.4f} | Shape: {:.4f} | Fold: {:.4f} | Rect: {:.4f} | Smooth: {:.4f} "
                    "| s_target: {:.4f} | s_actual: {:.4f} | gap: {:.4f} | iso: {:.4f} "
                    "| edge_mean: {:.4f} | edge_max: {:.4f} | area_mean: {:.4f} | area_max: {:.4f} "
                    "| lr: {:.6f}".format(
                        epoch + 1,
                        args.max_epoch,
                        i + 1,
                        len(train_loader),
                        avg_loss,
                        avg_shape,
                        avg_fold,
                        avg_rect,
                        avg_smooth,
                        avg_s_target,
                        avg_s_actual,
                        avg_target_gap,
                        avg_iso,
                        avg_fold_edge_mean,
                        avg_fold_edge_max,
                        avg_fold_area_mean,
                        avg_fold_area_max,
                        optimizer.state_dict()['param_groups'][0]['lr'],
                    )
                )

                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('Loss/total', avg_loss, glob_iter)
                writer.add_scalar('Loss/shape', avg_shape, glob_iter)
                writer.add_scalar('Loss/rect', avg_rect, glob_iter)
                writer.add_scalar('Loss/fold', avg_fold, glob_iter)
                writer.add_scalar('Loss/smooth', avg_smooth, glob_iter)
                writer.add_scalar('Loss/boundary', avg_boundary, glob_iter)

                writer.add_scalar('ShapeDebug/s_target_mean', avg_s_target, glob_iter)
                writer.add_scalar('ShapeDebug/s_actual_mean', avg_s_actual, glob_iter)
                writer.add_scalar('ShapeDebug/target_gap_mean', avg_target_gap, glob_iter)
                writer.add_scalar('ShapeDebug/iso_mean', avg_iso, glob_iter)

                writer.add_scalar('Loss_fold_items/edge_mean', avg_fold_edge_mean, glob_iter)
                writer.add_scalar('Loss_fold_items/edge_max', avg_fold_edge_max, glob_iter)
                writer.add_scalar('Loss_fold_items/area_mean', avg_fold_area_mean, glob_iter)
                writer.add_scalar('Loss_fold_items/area_max', avg_fold_area_max, glob_iter)

                # 清零
                loss_sigma = 0.0
                shape_sigma = 0.0
                rect_sigma = 0.0
                fold_sigma = 0.0
                smooth_sigma = 0.0
                boundary_sigma = 0.0

                s_target_sigma = 0.0
                s_actual_sigma = 0.0
                target_gap_sigma = 0.0
                iso_sigma = 0.0

                fold_edge_mean_sigma = 0.0
                fold_edge_max_sigma = 0.0
                fold_area_mean_sigma = 0.0
                fold_area_max_sigma = 0.0

            glob_iter += 1

        scheduler.step()

        # 7. 保存模型
        if ((epoch + 1) % 5 == 0) or ((epoch + 1) == args.max_epoch):
            filename = 'epoch' + str(epoch + 1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(ckpt_dir, filename)
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'glob_iter': glob_iter,
            }
            torch.save(state, model_save_path)

    end_time = time.time()
    print("################## end training #######################", end_time - start_time)
    writer.close()


if __name__ == "__main__":
    print('<==================== setting arguments ===================>\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--train_path', type=str, default='/paddle/Zzr/Object-IR-saliency/Data/train')

    parser.add_argument('--exp_name', type=str, default='v4_shape_fold', help='experiment name')
    parser.add_argument('--log_root', type=str, default='runs', help='root dir for tensorboard logs')
    parser.add_argument('--ckpt_root', type=str, default='checkpoints', help='root dir for model checkpoints')
    parser.add_argument('--resume', action='store_true', help='resume training')

    # 兼容原有的入参习惯
    parser.add_argument('--enable_w_l', action='store_true', help='enable shape loss')
    parser.add_argument('--enable_g_l', action='store_true', help='enable rect loss')
    parser.add_argument('--saliency_root', type=str, default='/paddle/Zzr/MDSAM-master/out/input')

    # 新增：正式版只保留 s_target 模式切换，不做 fast_dev_steps 截断
    parser.add_argument('--s_target_mode', type=str, default='adaptive', choices=['global', 'adaptive'])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(args)
    train(args)

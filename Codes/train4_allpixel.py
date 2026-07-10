import os
import sys

def _configure_cuda_from_argv(default_gpu=""):
    """
    在 import torch 之前，根据命令行中的 --gpu 提前设置 CUDA 环境变量
    """
    gpu = default_gpu
    argv = sys.argv[1:]

    for i, arg in enumerate(argv):
        if arg == "--gpu" and i + 1 < len(argv):
            gpu = argv[i + 1]
            break
        if arg.startswith("--gpu="):
            gpu = arg.split("=", 1)[1]
            break

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return gpu

_BOOTSTRAP_GPU = _configure_cuda_from_argv()
import argparse

import glob
import time
import random
import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from network2 import build_model, Network
from dataset import TrainDataset

from loss4_allpixel_dabef import (
    salient_shape_loss,
    salient_rect_loss,
    global_rect_loss,
    b_loss,
    folding_loss,
    # global_smoothness_loss,
    global_smoothness_angle_loss,
    salient_reposition_loss,
    grid_w,
    grid_h,
)

# from loss4_gridmean_shape import (
#     salient_shape_loss,
#     salient_rect_loss,
#     global_rect_loss,
#     b_loss,
#     folding_loss,
#     global_smoothness_loss,
#     salient_reposition_loss,
#     grid_w,
#     grid_h,
# )

last_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了对照实验更可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    set_seed(args.seed)

    batch_size = args.batch_size
    USE_SHAPE = args.enable_w_l
    USE_RECT = args.enable_g_l

    run_dir = os.path.join(last_path, args.log_root, args.exp_name)
    ckpt_dir = os.path.join(last_path, args.ckpt_root, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_dir)

    train_data = TrainDataset(
        data_path=args.train_path,
        saliency_dirname=args.saliency_root,
        use_saliency=True
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True
    )

    print("len(train_data) =", len(train_data))
    print("len(train_loader) =", len(train_loader), "batch_size =", batch_size)

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.Adam(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=args.lr_gamma,
        last_epoch=-1
    )

    ckpt_list = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    ckpt_list.sort()

    if args.resume and len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        glob_iter = checkpoint["glob_iter"]
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.lr_gamma,
            last_epoch=start_epoch - 1
        )
        print(f"resume model from {model_path}!")
    else:
        start_epoch = 0
        glob_iter = 0
        print("training from scratch!")

    print("################## start training #######################")
    score_print_fre = 50
    start_time = time.time()

    # # ==============================
    # # 固定静态权重（除 smooth 外）
    # # ==============================
    # lam_move = args.lam_move
    # lam_shape = args.lam_shape
    # lam_rect = args.lam_rect
    # lam_global_rect = args.lam_global_rect
    # lam_fold = args.lam_fold
    # lam_b = args.lam_b

    # enable_global_rect = (lam_global_rect > 0.0)

    # print(
    #     f"Base Weights -> "
    #     f"Move:{lam_move:.1f} | Shape:{lam_shape:.1f} | Rect:{lam_rect:.1f} | "
    #     f"GRect:{lam_global_rect:.1f} | "
    #     f"Fold:{lam_fold:.1f} | Boundary:{lam_b:.1f}"
    # )

    # for epoch in range(start_epoch, args.max_epoch):
    #     net.train()

    #     # smooth 权重调度：
    #     # 前 30 轮保持 0.05
    #     # 第 30~40 轮 cosine 平滑回升到 0.07
    #     # 40 轮后保持 0.07
    #     if epoch < 30:
    #         lam_smooth = 0.05
    #     elif epoch < 40:
    #         t = (epoch - 30) / 10.0
    #         lam_smooth = 0.05 + 0.5 * (0.07 - 0.05) * (1 - math.cos(math.pi * t))
    #     else:
    #         lam_smooth = 0.07

    #     print(
    #         f"Epoch {epoch:03d} | SmoothSchedule | "
    #         f"lr={optimizer.state_dict()['param_groups'][0]['lr']:.6f} | "
    #         f"lam_smooth={lam_smooth:.4f}"
    #     )

    #     loss_sigma = 0.0
    #     shape_sigma = 0.0
    #     rect_sigma = 0.0
    #     global_rect_sigma = 0.0
    #     fold_sigma = 0.0
    #     smooth_sigma = 0.0
    #     boundary_sigma = 0.0
    #     move_sigma = 0.0
    #     move_l1_sigma = 0.0
    #     move_shift_sigma = 0.0
    # # ==============================
    # # 固定静态权重
    # # ==============================
    lam_move = args.lam_move
    lam_shape = args.lam_shape
    lam_rect = args.lam_rect
    lam_smooth = args.lam_smooth
    lam_global_rect = args.lam_global_rect
    lam_fold = args.lam_fold
    lam_b = args.lam_b

    enable_global_rect = (lam_global_rect > 0.0)

    print(
        f"Static Weights -> "
        f"Move:{lam_move:.1f} | Shape:{lam_shape:.1f} | Rect:{lam_rect:.1f} | "
        f"GRect:{lam_global_rect:.1f} | Smooth:{lam_smooth:.3f} | "
        f"Fold:{lam_fold:.1f} | Boundary:{lam_b:.1f}"
    )

    for epoch in range(start_epoch, args.max_epoch):
        net.train()

        # 当前 epoch 的累计 loss
        loss_sigma = 0.0
        shape_sigma = 0.0
        rect_sigma = 0.0
        global_rect_sigma = 0.0
        fold_sigma = 0.0
        smooth_sigma = 0.0
        boundary_sigma = 0.0
        move_sigma = 0.0
        move_l1_sigma = 0.0
        move_shift_sigma = 0.0

        # fold 子项监控
        edge_mean_sigma = 0.0
        edge_max_sigma = 0.0
        area_mean_sigma = 0.0
        area_max_sigma = 0.0

        print(
            f"Epoch {epoch:03d} | AllPixelRect Training | "
            f"lr={optimizer.state_dict()['param_groups'][0]['lr']:.6f}"
        )

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
            warp_image = batch_out["warp_primary"]
            mesh = batch_out["mesh_pri"]
            ori_mesh = batch_out["ori_mesh"]
            motion = batch_out["motion_primary"]
            _, _, out_h, out_w = warp_image.shape

            # 2. 计算各项损失
            shape_loss = salient_shape_loss(
                inpu_tensor, warp_image, ori_mesh, mesh,
                iim=iim_tensor, enable=USE_SHAPE
            )

            rect_loss = salient_rect_loss(
                inpu_tensor, warp_image, ori_mesh, mesh,
                iim=iim_tensor, enable=USE_RECT
            )

            global_rect = global_rect_loss(
                inpu_tensor, warp_image, ori_mesh, mesh, iim=iim_tensor,
                enable=enable_global_rect
            )

            fold_loss, fold_items = folding_loss(
                mesh, out_w, out_h, grid_w, grid_h,
                eta=0.20,
                area_eta=0.20,
                lambda_max=2.0,
                lambda_area=0.8,
                return_items=True
            )

            # smooth_loss = global_smoothness_loss(mesh, ori_mesh)
            smooth_loss = global_smoothness_angle_loss(mesh, ori_mesh)

            move_loss, move_items = salient_reposition_loss(
                inpu_tensor,
                warp_image,
                ori_mesh,
                mesh,
                iim=iim_tensor,
                enable=True,
                use_softmax_pool=True,
                omega_tau=8.0,
                core_ratio=0.6,
                core_temp=0.08,
                center_band=0.15,
                return_items=True,
            )

            boundary_loss = b_loss(warp_image, mesh, motion)

            # 3. 固定权重加和
            shape_scaled = lam_shape * shape_loss
            rect_scaled = lam_rect * rect_loss
            global_rect_scaled = lam_global_rect * global_rect
            fold_scaled = lam_fold * fold_loss
            smooth_scaled = lam_smooth * smooth_loss
            boundary_scaled = lam_b * boundary_loss
            move_scaled = lam_move * move_loss

            total_loss = (
                shape_scaled
                + rect_scaled
                + global_rect_scaled
                + fold_scaled
                + smooth_scaled
                + boundary_scaled
                + move_scaled
            )

            # 4. 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            # 5. 累计
            loss_sigma += float(total_loss.item())
            shape_sigma += float(shape_scaled.item())
            rect_sigma += float(rect_scaled.item())
            global_rect_sigma += float(global_rect_scaled.item())
            fold_sigma += float(fold_scaled.item())
            smooth_sigma += float(smooth_scaled.item())
            boundary_sigma += float(boundary_scaled.item())
            move_sigma += float(move_scaled.item())
            move_l1_sigma += float(move_items["L_move"].item())
            move_shift_sigma += float(move_items["shift_norm"].item())

            edge_mean_sigma += float(fold_items["edge_mean"].item())
            edge_max_sigma += float(fold_items["edge_max"].item())
            area_mean_sigma += float(fold_items["area_mean"].item())
            area_max_sigma += float(fold_items["area_max"].item())

            # 6. 日志
            if i % score_print_fre == 0 and i != 0:
                avg_loss = loss_sigma / score_print_fre
                avg_shape = shape_sigma / score_print_fre
                avg_rect = rect_sigma / score_print_fre
                avg_global_rect = global_rect_sigma / score_print_fre
                avg_fold = fold_sigma / score_print_fre
                avg_smooth = smooth_sigma / score_print_fre
                avg_boundary = boundary_sigma / score_print_fre
                avg_move = move_sigma / score_print_fre
                avg_move_l1 = move_l1_sigma / score_print_fre
                avg_move_shift = move_shift_sigma / score_print_fre

                avg_edge_mean = edge_mean_sigma / score_print_fre
                avg_edge_max = edge_max_sigma / score_print_fre
                avg_area_mean = area_mean_sigma / score_print_fre
                avg_area_max = area_max_sigma / score_print_fre

                print(
                    "Epoch[{:0>3}/{:0>3}] Iter[{:<4d}/{:<4d}] "
                    "Total:{:.4f} | Shape:{:.4f} | Rect:{:.4f} | GRect:{:.4f} "
                    "| Smooth:{:.4f} | Fold:{:.4f} | Move:{:.4f} "
                    "| L1_Dist:{:.4f} | Shift:{:.4f} "
                    "| edge_mean:{:.4f} | edge_max:{:.4f} | area_mean:{:.4f} | area_max:{:.4f}".format(
                        epoch + 1, args.max_epoch, i + 1, len(train_loader),
                        avg_loss, avg_shape, avg_rect, avg_global_rect,
                        avg_smooth, avg_fold, avg_move,
                        avg_move_l1, avg_move_shift,
                        avg_edge_mean, avg_edge_max, avg_area_mean, avg_area_max
                    )
                )

                writer.add_scalar("lr", optimizer.state_dict()["param_groups"][0]["lr"], glob_iter)
                writer.add_scalar("Loss/total", avg_loss, glob_iter)
                writer.add_scalar("Loss/shape", avg_shape, glob_iter)
                writer.add_scalar("Loss/rect", avg_rect, glob_iter)
                writer.add_scalar("Loss/global_rect", avg_global_rect, glob_iter)
                writer.add_scalar("Loss/fold", avg_fold, glob_iter)
                writer.add_scalar("Loss/smooth", avg_smooth, glob_iter)
                writer.add_scalar("Loss/boundary", avg_boundary, glob_iter)
                writer.add_scalar("Loss/move", avg_move, glob_iter)

                writer.add_scalar("Fold/edge_mean", avg_edge_mean, glob_iter)
                writer.add_scalar("Fold/edge_max", avg_edge_max, glob_iter)
                writer.add_scalar("Fold/area_mean", avg_area_mean, glob_iter)
                writer.add_scalar("Fold/area_max", avg_area_max, glob_iter)

                writer.add_scalar("Move/L1_Dist", avg_move_l1, glob_iter)
                writer.add_scalar("Move/Shift", avg_move_shift, glob_iter)

                # 清零
                loss_sigma = 0.0
                shape_sigma = 0.0
                rect_sigma = 0.0
                global_rect_sigma = 0.0
                fold_sigma = 0.0
                smooth_sigma = 0.0
                boundary_sigma = 0.0
                move_sigma = 0.0
                move_l1_sigma = 0.0
                move_shift_sigma = 0.0

                edge_mean_sigma = 0.0
                edge_max_sigma = 0.0
                area_mean_sigma = 0.0
                area_max_sigma = 0.0

            glob_iter += 1

        scheduler.step()

        # 7. 保存模型
        if ((epoch + 1) % 5 == 0) or ((epoch + 1) == args.max_epoch):
            filename = "epoch" + str(epoch + 1).zfill(3) + "_model.pth"
            model_save_path = os.path.join(ckpt_dir, filename)
            state = {
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "glob_iter": glob_iter
            }
            torch.save(state, model_save_path)

    end_time = time.time()
    print("################## end training #######################", end_time - start_time)
    writer.close()


if __name__ == "__main__":
    print("<==================== setting arguments ===================>\n")
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lr_gamma", type=float, default=0.97)

    parser.add_argument("--train_path", type=str, default="/paddle/Zzr/Object-IR-saliency/Data/train")
    parser.add_argument("--saliency_root", type=str, default="/paddle/Zzr/MDSAM-master/out/input")

    parser.add_argument("--exp_name", type=str, default="v5_allpixel_rect", help="experiment name")
    parser.add_argument("--log_root", type=str, default="run5", help="root dir for tensorboard logs")
    parser.add_argument("--ckpt_root", type=str, default="checkpoint5", help="root dir for model checkpoints")
    parser.add_argument("--resume", action="store_true", help="resume training")

    # 注意：这两个默认仍是 False，训练命令里请显式加上
    parser.add_argument("--enable_w_l", action="store_true", help="enable shape loss")
    parser.add_argument("--enable_g_l", action="store_true", help="enable rect loss")

    # 固定静态权重
    parser.add_argument("--lam_move", type=float, default=0.0)
    parser.add_argument("--lam_shape", type=float, default=0.0)
    parser.add_argument("--lam_rect", type=float, default=950.0)#950
    # parser.add_argument("--lam_rect", type=float, default=450.0)#950
    parser.add_argument("--lam_smooth", type=float, default=0.1)
    parser.add_argument("--lam_global_rect", type=float, default=450.0)#450
    # parser.add_argument("--lam_global_rect", type=float, default=450.0)
    parser.add_argument("--lam_fold", type=float, default=25.0)
    
    # parser.add_argument("--lam_shape", type=float, default=300.0)
    # parser.add_argument("--lam_rect", type=float, default=950.0)
    # parser.add_argument("--lam_smooth", type=float, default=0.0)
    # parser.add_argument("--lam_global_rect", type=float, default=450.0)
    # parser.add_argument("--lam_fold", type=float, default=25.0)
    parser.add_argument("--lam_b", type=float, default=0.0)

    args = parser.parse_args()

    print(args)
    print(f"[CUDA bootstrap] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    train(args)

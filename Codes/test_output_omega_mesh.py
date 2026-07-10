# coding: utf-8
import os
import sys


def _configure_cuda_from_argv(default_gpu="0"):

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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import predefine
import utils.torch_tps_transform as torch_tps_transform
from dataset import OutDataset
from Codes.network import Network, output_model, get_rigid_mesh, get_norm_mesh, get_stack_mesh


grid_h = predefine.GRID_H
grid_w = predefine.GRID_W


def get_salient_omega(iim_tensor, grid_size, tau=0.001):
    gh, gw = grid_size
    B, C, H, W = iim_tensor.shape
    kh, kw = H // gh, W // gw
    

    local_max = F.max_pool2d(iim_tensor, kernel_size=(kh, kw), stride=(kh, kw))

    iim_stable = iim_tensor - F.interpolate(local_max, size=(H, W), mode='nearest')
    
    iim_exp = torch.exp(tau * iim_stable)
    avg_exp = F.avg_pool2d(iim_exp, kernel_size=(kh, kw), stride=(kh, kw))

    omega = (torch.log(avg_exp + 1e-6) / tau) + local_max
    
    return omega[0, 0].clamp(0.0, 1.0) 

def draw_mesh_on_warp(warp, f_local):

    warp = np.ascontiguousarray(warp)
    point_color = (0, 0, 255)  # BGR red
    thickness = 2
    lineType = 8
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i + 1, j, 0]), int(f_local[i + 1, j, 1])),
                    point_color,
                    thickness,
                    lineType,
                )
            elif i == grid_h:
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i, j + 1, 0]), int(f_local[i, j + 1, 1])),
                    point_color,
                    thickness,
                    lineType,
                )
            else:
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i + 1, j, 0]), int(f_local[i + 1, j, 1])),
                    point_color,
                    thickness,
                    lineType,
                )
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i, j + 1, 0]), int(f_local[i, j + 1, 1])),
                    point_color,
                    thickness,
                    lineType,
                )
    return warp


def load_checkpoint(net, ckpt_path, device):
    if not ckpt_path or (not os.path.isfile(ckpt_path)):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and ("model" in ckpt):
        state = ckpt["model"]
    else:
        state = ckpt
    net.load_state_dict(state, strict=True)
    print(f"[OK] Loaded checkpoint: {ckpt_path}")


def resolve_ckpt_path(ckpt_path, model_dir):
    if ckpt_path:
        return ckpt_path
    ckpt_list = glob.glob(os.path.join(model_dir, "*.pth"))
    ckpt_list.sort()
    if len(ckpt_list) == 0:
        return ""
    return ckpt_list[-1]


def find_saliency_path(saliency_dir, base_name, suffixes, exts):
    if not saliency_dir:
        return ""
    for suf in suffixes:
        for e in exts:
            p = os.path.join(saliency_dir, base_name + suf + e)
            if os.path.isfile(p):
                return p
    return ""


def load_saliency(path, target_hw, warned_flag):
    if not path or (not os.path.isfile(path)):
        if (path and (not warned_flag[0])):
            print(f"[WARN] Missing saliency map, using ones. Example: {path}")
            warned_flag[0] = True
        iim = np.ones(target_hw, dtype=np.float32)
        return iim
    iim = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if iim is None:
        if not warned_flag[0]:
            print(f"[WARN] Unreadable saliency map, using ones. Example: {path}")
            warned_flag[0] = True
        iim = np.ones(target_hw, dtype=np.float32)
        return iim
    iim = cv2.resize(iim, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    iim = iim.astype(np.float32) / 255.0
    return iim


def omega_to_gray(omega, out_hw):
   
    om = omega.astype(np.float32)
    om_min = float(om.min())
    om_max = float(om.max())
    if (om_max - om_min) < 1e-6:
        om_norm = np.zeros_like(om, dtype=np.float32)
    else:
        om_norm = (om - om_min) / (om_max - om_min)
    om_u8 = (om_norm * 255.0).clip(0, 255).astype(np.uint8)
    om_up = cv2.resize(om_u8, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
    return om_up


def warp_saliency(iim_tensor, mesh, ratio_h, ratio_w):

    b, _, img_h, img_w = iim_tensor.shape
    warp_h = int(img_h * ratio_h)
    warp_w = int(img_w * ratio_w)

    rigid_mesh = get_rigid_mesh(b, warp_h, warp_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, warp_h, warp_w)
    norm_mesh = get_norm_mesh(mesh, warp_h, warp_w)

    stack_rigid_mesh = get_stack_mesh(norm_rigid_mesh)
    stack_mesh = get_stack_mesh(norm_mesh)

    if ratio_w <= 1:
        warp = torch_tps_transform.transformer(
            iim_tensor, stack_mesh, stack_rigid_mesh, (warp_h, warp_w)
        )
    else:
        warp = torch_tps_transform.transformer(
            iim_tensor, stack_rigid_mesh, stack_mesh, (warp_h, warp_w)
        )
    return warp


def test(args):
    # os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_data = OutDataset(data_path=args.test_path)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_to_load = resolve_ckpt_path(args.ckpt_path, args.model_dir)
    if not ckpt_to_load:
        raise RuntimeError(
            f"No checkpoint found in dir: {args.model_dir}. "
            f"Or specify --ckpt_path explicitly."
        )
    load_checkpoint(net, ckpt_to_load, device)

    path_warp = args.warp_dir or os.path.join(args.out_dir, "warp")
    path_omega_gray = args.omega_gray_dir or os.path.join(args.out_dir, "omega_gray")
    path_sal_warp = args.saliency_warp_dir or os.path.join(args.out_dir, "saliency_warp")
    path_sal_warp_grid = args.saliency_warp_grid_dir or os.path.join(args.out_dir, "saliency_warp_grid")
    path_omega_warp_grid = args.omega_warp_grid_dir or os.path.join(args.out_dir, "omega_warp_grid")
    path_grid = os.path.join(args.out_dir, "grid")
    path_deformed_mesh = args.deformed_mesh_dir or os.path.join(args.out_dir, "deformed_mesh")
    os.makedirs(path_warp, exist_ok=True)
    os.makedirs(path_omega_gray, exist_ok=True)
    os.makedirs(path_sal_warp, exist_ok=True)
    os.makedirs(path_sal_warp_grid, exist_ok=True)
    os.makedirs(path_omega_warp_grid, exist_ok=True)
    os.makedirs(path_grid, exist_ok=True)
    os.makedirs(path_deformed_mesh, exist_ok=True)

    print("##################start testing#######################")
    num_images = len(test_loader)
    total_time = 0.0
    net.eval()
    warned_missing_sal = [False]

    for i, batch_value in enumerate(test_loader):
        inpu_tensor = batch_value[0].float()
        input_path = batch_value[1]
        if isinstance(input_path, (list, tuple)):
            input_path = input_path[0]
        if torch.cuda.is_available():
            inpu_tensor = inpu_tensor.cuda()

        with torch.no_grad():
            start_time = time.time()
            batch_out = output_model(net, inpu_tensor, args.ratio_h, args.ratio_w)
            end_time = time.time()
            inference_time = end_time - start_time

        warp_image = batch_out["warp_primary"]
        mesh = batch_out["mesh_primary"]

        b, c, out_h, out_w = warp_image.shape
        _, _, img_h, img_w = inpu_tensor.shape

        # ----- save warped image -----
        warp = ((warp_image[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
        base = os.path.splitext(os.path.basename(input_path))[0]
        path1 = os.path.join(path_warp, base + ".jpg")
        cv2.imwrite(path1, warp)

        # ----- save warp grid (mesh on warped image) -----
        warp_draw = warp_image[0].cpu().detach().numpy().transpose(1, 2, 0)
        warp_draw = (warp_draw + 1.0) / 2.0
        mesh_np = mesh[0].cpu().detach().numpy()
        warp_grid = draw_mesh_on_warp(warp_draw, mesh_np)
        warp_grid_u8 = (np.array(warp_grid) * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(path_grid, base + ".jpg"), warp_grid_u8)

        # ----- load saliency -----
        sal_path = ""
        if args.saliency_dir:
            sal_path = find_saliency_path(
                args.saliency_dir,
                base,
                args.saliency_suffixes,
                args.saliency_exts,
            )
        iim = load_saliency(sal_path, (img_h, img_w), warned_missing_sal)
        iim_tensor = torch.tensor(iim, dtype=inpu_tensor.dtype, device=inpu_tensor.device)
        iim_tensor = iim_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        # ----- omega visualization -----
        # omega = F.adaptive_avg_pool2d(iim_tensor, (grid_h, grid_w))[0, 0].cpu().numpy()


        with torch.no_grad():
            omega_tensor = get_salient_omega(iim_tensor, (grid_h, grid_w), tau=0.01)
            omega = omega_tensor.cpu().numpy()

        om_gray = omega_to_gray(omega, (img_h, img_w))
        cv2.imwrite(os.path.join(path_omega_gray, base + ".png"), om_gray)

        # ----- warp saliency with mesh -----
        sal_warp = warp_saliency(iim_tensor, mesh, args.ratio_h, args.ratio_w)
        sal_warp_np = sal_warp[0, 0].cpu().detach().numpy()
        sal_warp_u8 = (sal_warp_np * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(path_sal_warp, base + ".png"), sal_warp_u8)

        # ----- save warped saliency with deformed mesh -----
        # Output: out_dir/saliency_warp_grid/{image_name}.png
        sal_warp_bgr = cv2.cvtColor(sal_warp_u8, cv2.COLOR_GRAY2BGR)
        sal_warp_grid = draw_mesh_on_warp(sal_warp_bgr, mesh_np)
        cv2.imwrite(os.path.join(path_sal_warp_grid, base + ".png"), sal_warp_grid)

        # ----- save deformed mesh only -----
        # Output: out_dir/deformed_mesh/{image_name}.png
        mesh_canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
        mesh_only = draw_mesh_on_warp(mesh_canvas, mesh_np)
        cv2.imwrite(os.path.join(path_deformed_mesh, base + ".png"), mesh_only)

        # ----- warp omega_gray with mesh -----
        om_tensor = torch.tensor(
            om_gray.astype(np.float32) / 255.0,
            dtype=inpu_tensor.dtype,
            device=inpu_tensor.device,
        ).unsqueeze(0).unsqueeze(0)
        om_warp = warp_saliency(om_tensor, mesh, args.ratio_h, args.ratio_w)
        om_warp_np = om_warp[0, 0].cpu().detach().numpy()
        om_warp_u8 = (om_warp_np * 255.0).clip(0, 255).astype(np.uint8)
        # draw grid on warped omega (convert gray -> BGR first)
        om_warp_bgr = cv2.cvtColor(om_warp_u8, cv2.COLOR_GRAY2BGR)
        om_warp_grid = draw_mesh_on_warp(om_warp_bgr, mesh_np)
        cv2.imwrite(os.path.join(path_omega_warp_grid, base + ".png"), om_warp_grid)

        print(f"Image {i+1} processed in {inference_time:.4f} seconds")
        total_time += inference_time
        torch.cuda.empty_cache()

    ave_time = total_time / max(1, num_images)
    print("##################end testing#######################", f"average time is {ave_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--ratio_h", type=float, default=1.0)
    parser.add_argument("--ratio_w", type=float, default=0.5)

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to a specific .pth checkpoint. If set, will load this file only.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Directory containing .pth checkpoints (used when ckpt_path is empty).",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory for warp, omega, saliency_warp, saliency_warp_grid, and mesh visualization results.",
    )
    parser.add_argument("--warp_dir", type=str, default="", help="Override warp output dir")
    parser.add_argument("--omega_gray_dir", type=str, default="", help="Override omega gray output dir")
    parser.add_argument("--saliency_warp_dir", type=str, default="", help="Override warped saliency output dir")
    parser.add_argument(
        "--saliency_warp_grid_dir",
        type=str,
        default="",
        help="Override warped saliency with deformed mesh output dir",
    )
    parser.add_argument("--omega_warp_grid_dir", type=str, default="", help="Override warped omega grid output dir")
    parser.add_argument(
        "--deformed_mesh_dir",
        type=str,
        default="",
        help="Override pure deformed mesh visualization output dir",
    )

    parser.add_argument(
        "--saliency_dir",
        type=str,
        default="",
        help="Directory containing saliency maps (grayscale).",
    )
    parser.add_argument(
        "--saliency_suffixes",
        type=str,
        nargs="+",
        default=["", "_vis"],
        help="Suffixes to try when matching saliency filenames.",
    )
    parser.add_argument(
        "--saliency_exts",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
        help="Extensions to try when matching saliency filenames.",
    )

    args = parser.parse_args()

    print(args)
    print(f"[CUDA bootstrap] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    test(args)

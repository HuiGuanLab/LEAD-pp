#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDT-based cropping on an ImageFolder dataset (NO extra expansion).
Goal:
  1) 只按 DDT 的“第一主成分响应最集中的区域”裁剪（不再外扩）。
  2) 输出裁剪图像为【严格正方形】（以 bbox 最大边为 side，中心不变，边界 clamp，side 不变）。
  3) 对少量异常的超过阈值响应（离群点）做鲁棒处理：多 cut_rate 搜索 + 面积约束，选更“集中”的框。

Usage:
CUDA_VISIBLE_DEVICES=0 python get_crop_2_square.py \
  --root /media/data1/zzh/EAD-FFAB/EAD/bird \
  --trans-vec /media/data1/zzh/LEAD++/DDT/DDT_result/bird1/vec_res50_bird.npy \
  --descriptors-mean-tensor /media/data1/zzh/LEAD++/DDT/DDT_result/bird1/mean_tensor_bird.pth \
  --output /media/data1/zzh/LEAD++/DDT/crop_dataset/bird_crop_square

Notes:
- 兼容 DDT_load.DDT.co_locate() 返回 (x,y,w,h) 或 (x1,y1,x2,y2)：
  自动判断并转成 xyxy 再用。
"""

import argparse
import os
import random
import numpy as np
import torch
from torchvision import datasets as tvdatasets
from DDT_load import DDT


# ---------------------- Args ----------------------
def get_args():
    p = argparse.ArgumentParser("DDT crop & save on ImageFolder (square output, robust)")
    p.add_argument('--seed', default=123, type=int, help='random seed')

    p.add_argument('--root', type=str, required=True, help='ImageFolder root (expects <root>/<class>/*)')
    p.add_argument('--trans-vec', type=str, required=True, help='path to trans_vec .npy')
    p.add_argument('--descriptors-mean-tensor', type=str, required=True, help='path to descriptors_mean_tensor .pth')

    p.add_argument('--pretrain-model', type=str, default='resnet50',
                   help='backbone name used by DDT (e.g., resnet50)')

    p.add_argument('--output', type=str, required=True,
                   help='output root for cropped images (mirror subfolders & filenames)')

    # p.add_argument('--over-num', type=float, default=0.1)
    # p.add_argument('--cut-rates', type=float, nargs='+', default=[1.25, 1.15, 1.05, 0.95, 0.85])

    return p.parse_args()


# ---------------------- Utils ----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageFolderReturnPath(tvdatasets.ImageFolder):
    """ImageFolder, but returns (PIL Image, absolute path)."""
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = self.loader(path)  # PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, path


def _clamp_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(x1 + 1, min(int(x2), W))
    y2 = max(y1 + 1, min(int(y2), H))
    return x1, y1, x2, y2


def _as_xyxy_from_ddt_return(b, W, H):
    """
    兼容 DDT_load.DDT.co_locate() 的两种常见返回：
      - (x, y, w, h)  (cv2.boundingRect 风格)
      - (x1, y1, x2, y2)
    返回统一的 (x1, y1, x2, y2)（clamp 后）
    """
    if b is None or len(b) != 4:
        return None

    a0, a1, a2, a3 = map(float, b)

    # 经验判断：如果 a2,a3 明显小于图像尺寸且 (a0+a2)<=W, (a1+a3)<=H，则更像 (x,y,w,h)
    # 否则按 (x1,y1,x2,y2) 处理
    if (a2 > 0 and a3 > 0) and (a0 + a2 <= W + 1) and (a1 + a3 <= H + 1) and (a2 <= W) and (a3 <= H):
        # treat as xywh
        x1, y1 = a0, a1
        x2, y2 = a0 + a2, a1 + a3
    else:
        # treat as xyxy
        x1, y1, x2, y2 = a0, a1, a2, a3

    return _clamp_xyxy(x1, y1, x2, y2, W, H)


def fit_bbox_to_square_by_max_side(x1, y1, x2, y2, W, H):
    """
    用 bbox 的最大边作为正方形边长 side，中心保持 bbox 中心。
    关键：clamp 时保持 side 不变，只移动左上角，保证输出严格正方形。
    """
    import math

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    side = int(math.ceil(max(bw, bh)))
    side = min(side, W, H)  # 必须能放进图像短边

    x1n = int(math.floor(cx - side / 2))
    y1n = int(math.floor(cy - side / 2))

    # clamp（保持 side 不变）
    x1n = max(0, min(x1n, W - side))
    y1n = max(0, min(y1n, H - side))

    x2n = x1n + side
    y2n = y1n + side

    return _clamp_xyxy(x1n, y1n, x2n, y2n, W, H)


def bbox_quality_score(x1, y1, x2, y2, W, H):
    """
    评价 bbox 是否“集中且合理”：
    - 太大：可能背景连通（阈值偏低）
    - 太小：可能噪点/碎片（阈值过高）
    趋向选择面积更小但仍在合理范围内的框。
    """
    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh
    img_area = W * H
    ar = area / max(img_area, 1e-6)

    MIN_AR = 0.01
    MAX_AR = 0.80
    if ar < MIN_AR or ar > MAX_AR:
        return None

    score = -ar

    # 轻微惩罚贴边
    border_touch = 0
    if x1 <= 1: border_touch += 1
    if y1 <= 1: border_touch += 1
    if x2 >= W - 1: border_touch += 1
    if y2 >= H - 1: border_touch += 1
    score -= 0.02 * border_touch

    return score


def robust_co_locate_bbox(ddt, img, trans_vectors, descriptor_means):
    """
    多 cut_rate 搜索，挑选更集中/更合理的 bbox。
    注意：这里不引入额外外扩参数；cut_rate 是 DDT 内部用来确定 crop side 的系数（不是额外 margin）。
    """
    W, H = img.size

    # 从更严格到更宽松（你原来的顺序保留）
    cut_rates = [1.25, 1.15, 1.05, 0.95, 0.85]

    # 固定 over_num；如果你发现框经常过大，可改为 0.1/0.2（不想加参数就写死）
    over_num = 0.0

    best = None  # (score, (x1,y1,x2,y2), used_cut_rate)
    for cr in cut_rates:
        b = ddt.co_locate(img, trans_vectors, descriptor_means, cut_rate=cr, over_num=over_num)
        b_xyxy = _as_xyxy_from_ddt_return(b, W, H)
        if b_xyxy is None:
            continue

        x1, y1, x2, y2 = b_xyxy
        s = bbox_quality_score(x1, y1, x2, y2, W, H)
        if s is None:
            continue

        if best is None or s > best[0]:
            best = (s, (x1, y1, x2, y2), cr)

    # 全部不合格：回退中庸阈值
    if best is None:
        cr = 1.05
        b = ddt.co_locate(img, trans_vectors, descriptor_means, cut_rate=cr, over_num=over_num)
        b_xyxy = _as_xyxy_from_ddt_return(b, W, H)
        if b_xyxy is None:
            return (0, 0, W, H), cr
        return b_xyxy, cr

    return best[1], best[2]


# ---------------------- Main ----------------------
def main():
    args = get_args()
    set_seed(args.seed)

    ddt = DDT(use_cuda=True, pretrain_model=args.pretrain_model)
    trans_vectors = np.load(args.trans_vec)
    descriptor_means = torch.load(args.descriptors_mean_tensor)

    eval_data = ImageFolderReturnPath(args.root)
    images_root = args.root
    out_root = args.output
    os.makedirs(out_root, exist_ok=True)

    saved, failed = 0, 0
    for img, file_path in eval_data:
        try:
            W, H = img.size

            # 1) 鲁棒选“PC1 最集中”的 bbox（先兼容 co_locate 返回格式）
            (x1, y1, x2, y2), used_cr = robust_co_locate_bbox(ddt, img, trans_vectors, descriptor_means)

            # 2) 输出严格正方形：以 bbox 最大边为 side，中心不变，clamp 不改 side
            x1, y1, x2, y2 = fit_bbox_to_square_by_max_side(x1, y1, x2, y2, W, H)

            # 3) 裁剪并保存
            crop = img.crop((x1, y1, x2, y2))

            # Mirror class subfolder & filename
            rel = os.path.relpath(file_path, start=images_root)
            dst_path = os.path.join(out_root, rel)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            save_kwargs = {}
            ext = os.path.splitext(dst_path)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                save_kwargs.update(dict(quality=95, optimize=True))
            crop.save(dst_path, **save_kwargs)

            if saved % 50 == 0:
                side = (x2 - x1)
                print(f"[OK] {saved:06d} side={side:4d} cut_rate={used_cr:.2f}  {file_path}")

            saved += 1

        except Exception as e:
            failed += 1
            print(f"[WARN] fail on {file_path}: {e}")

    print(f"Done. saved={saved}, failed={failed}")


if __name__ == "__main__":
    main()

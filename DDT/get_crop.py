#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import numpy as np
import torch
from torchvision import datasets as tvdatasets
from DDT_load import DDT


# ---------------------- Args ----------------------
def get_args():
    p = argparse.ArgumentParser("DDT crop & save on ImageFolder (task-wise policy)")
    p.add_argument('--seed', default=123, type=int, help='random seed')

    # Any string is allowed: non bird/car/aircraft defaults to bird-style square crop
    p.add_argument('--task', type=str, default='bird',
                   help='dataset name; bird/car/aircraft special; others default to bird-style square')

    p.add_argument('--root', type=str, required=True, help='ImageFolder root (expects <root>/<class>/*)')
    p.add_argument('--trans-vec', type=str, required=True, help='path to trans_vec .npy')
    p.add_argument('--descriptors-mean-tensor', type=str, required=True, help='path to descriptors_mean_tensor .pth')

    p.add_argument('--pretrain-model', type=str, default='resnet50',
                   help='backbone name used by DDT (e.g., resnet50)')

    p.add_argument('--output', type=str, required=True,
                   help='output root for cropped images (mirror subfolders & filenames)')

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
    Compatible with two common outputs of DDT_load.DDT.co_locate():
      - (x, y, w, h)  (cv2.boundingRect style)
      - (x1, y1, x2, y2)
    Always returns (x1, y1, x2, y2) after clamping.
    """
    if b is None or len(b) != 4:
        return None

    a0, a1, a2, a3 = map(float, b)

    # Heuristic: if a2, a3 look like width/height
    if (a2 > 0 and a3 > 0) and (a0 + a2 <= W + 1) and (a1 + a3 <= H + 1) and (a2 <= W) and (a3 <= H):
        # xywh -> xyxy
        x1, y1 = a0, a1
        x2, y2 = a0 + a2, a1 + a3
    else:
        # already xyxy
        x1, y1, x2, y2 = a0, a1, a2, a3

    return _clamp_xyxy(x1, y1, x2, y2, W, H)


def fit_bbox_to_square_by_max_side(x1, y1, x2, y2, W, H):
    """
    Use the maximum side of the bbox as the square side length.
    Keep the bbox center fixed; clamp without changing side length
    to guarantee a strict square output.
    """
    import math
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    side = int(math.ceil(max(bw, bh)))
    side = min(side, W, H)

    x1n = int(math.floor(cx - side / 2))
    y1n = int(math.floor(cy - side / 2))

    # Clamp while keeping side unchanged
    x1n = max(0, min(x1n, W - side))
    y1n = max(0, min(y1n, H - side))

    x2n = x1n + side
    y2n = y1n + side
    return _clamp_xyxy(x1n, y1n, x2n, y2n, W, H)


def expand_bbox_anisotropic(x1, y1, x2, y2, W, H, pad_w_ratio: float, pad_h_ratio: float):
    """
    Anisotropic expansion: expand width and height with different ratios
    around the bbox center, keeping a rectangular shape.
      new_w = w * (1 + pad_w_ratio)
      new_h = h * (1 + pad_h_ratio)
    Clamp while keeping new_w/new_h unchanged to avoid shrinking.
    """
    import math
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    new_w = int(math.ceil(bw * (1.0 + pad_w_ratio)))
    new_h = int(math.ceil(bh * (1.0 + pad_h_ratio)))

    new_w = min(max(new_w, 2), W)
    new_h = min(max(new_h, 2), H)

    x1n = int(math.floor(cx - new_w / 2))
    y1n = int(math.floor(cy - new_h / 2))

    x1n = max(0, min(x1n, W - new_w))
    y1n = max(0, min(y1n, H - new_h))

    x2n = x1n + new_w
    y2n = y1n + new_h
    return _clamp_xyxy(x1n, y1n, x2n, y2n, W, H)


def bbox_quality_score(x1, y1, x2, y2, W, H):
    """
    Evaluate whether a bbox is concentrated and reasonable:
    - Too large: likely background leakage (threshold too low)
    - Too small: likely noise/fragments (threshold too high)
    Prefer smaller bboxes within a reasonable area range.
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

    # Mild penalty for touching image borders
    border_touch = 0
    if x1 <= 1: border_touch += 1
    if y1 <= 1: border_touch += 1
    if x2 >= W - 1: border_touch += 1
    if y2 >= H - 1: border_touch += 1
    score -= 0.02 * border_touch
    return score


def robust_co_locate_bbox(ddt, img, trans_vectors, descriptor_means):
    """
    Search over multiple cut_rate values to select a more concentrated
    and reasonable bbox.
    Note: no extra expansion is introduced here; cut_rate is an internal
    DDT parameter (not an additional margin).
    """
    W, H = img.size
    cut_rates = [1.25, 1.15, 1.05, 0.95, 0.85]
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

    # Fallback to a moderate threshold if all candidates are invalid
    if best is None:
        cr = 1.05
        b = ddt.co_locate(img, trans_vectors, descriptor_means, cut_rate=cr, over_num=over_num)
        b_xyxy = _as_xyxy_from_ddt_return(b, W, H)
        if b_xyxy is None:
            return (0, 0, W, H), cr
        return b_xyxy, cr

    return best[1], best[2]


# ---------------------- Task Policy ----------------------
# Only car/aircraft use anisotropic rectangular expansion;
# all other datasets follow bird-style square cropping.
TASK_PAD = {
    'car':      dict(pad_w=0.10, pad_h=0.30),   # slight width expansion, larger height expansion
    'aircraft': dict(pad_w=0.2, pad_h=0.8),   # usually flatter; expand height a bit more
}

def apply_taskwise_crop_policy(task, x1, y1, x2, y2, W, H):
    task = (task or "").lower()

    if task in ("car", "aircraft"):
        pad = TASK_PAD[task]
        return expand_bbox_anisotropic(
            x1, y1, x2, y2, W, H,
            pad_w_ratio=pad["pad_w"],
            pad_h_ratio=pad["pad_h"],
        )

    # bird + any other dataset: default bird-style strict square crop
    return fit_bbox_to_square_by_max_side(x1, y1, x2, y2, W, H)


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

            # 1) Robustly select bbox
            (x1, y1, x2, y2), used_cr = robust_co_locate_bbox(ddt, img, trans_vectors, descriptor_means)

            # 2) Apply task-wise crop policy
            x1, y1, x2, y2 = apply_taskwise_crop_policy(args.task, x1, y1, x2, y2, W, H)

            # 3) Crop and save
            crop = img.crop((x1, y1, x2, y2))

            rel = os.path.relpath(file_path, start=images_root)
            dst_path = os.path.join(out_root, rel)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            save_kwargs = {}
            ext = os.path.splitext(dst_path)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                save_kwargs.update(dict(quality=95, optimize=True))
            crop.save(dst_path, **save_kwargs)

            if saved % 50 == 0:
                bw, bh = (x2 - x1), (y2 - y1)
                print(f"[OK] {saved:06d} task={args.task} bw={bw:4d} bh={bh:4d} cut_rate={used_cr:.2f}  {file_path}")

            saved += 1

        except Exception as e:
            failed += 1
            print(f"[WARN] fail on {file_path}: {e}")

    print(f"Done. saved={saved}, failed={failed}")


if __name__ == "__main__":
    main()

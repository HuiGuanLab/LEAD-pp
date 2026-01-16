import torch
import torch.nn as nn
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from numpy import ndarray
from skimage import measure
import cv2
import random


class DDT(object):
    def __init__(self, use_cuda=False, pretrain_model=None, input_size=448):
        """
        统一版 DDT：
        - use_cuda: 是否用 GPU
        - pretrain_model: 'resnet50' / 'resnet18' 等 torchvision.models 名称；None 时走 VGG19 分支
        - input_size: DDT 特征提取时统一 Resize 到的尺寸（必须和 DDT_fit 时一致，例如 256 或 448）
        """
        self.input_size = int(input_size)

        # 是否用 GPU
        self.use_cuda = use_cuda and torch.cuda.is_available()
        print(f"use_cuda = {self.use_cuda}")

        # ---- 构建 backbone ----
        if pretrain_model is not None:
            # 例如 'resnet50'
            backbone = getattr(models, pretrain_model)(pretrained=True)
            # 去掉 avgpool & fc，保留 conv 特征图
            if hasattr(backbone, "avgpool"):
                backbone.avgpool = nn.Identity()
            if hasattr(backbone, "fc"):
                backbone.fc = nn.Identity()
            self.pretrained_feature_model = backbone
            self.dim = 2048  # 对 resnet50 是 2048，其他模型你可以按需修改
        else:
            # vgg19 分支，保持你原来的逻辑
            vgg = models.vgg19(pretrained=True)
            self.pretrained_feature_model = vgg.features
            self.dim = 512

        if self.use_cuda:
            self.pretrained_feature_model = self.pretrained_feature_model.cuda()

        # ---- 统一的预处理流程 ----
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.totensor = transforms.ToTensor()
        self.preprocess = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            self.totensor,
            self.normalize,
        ])

    @torch.no_grad()
    def co_locate(self, pil_image, trans_vector, descriptor_mean_tensor,
                  cut_rate=1.1, over_num=0.3):
        """
        给定一张 PIL.Image、PCA 主方向向量和均值向量，返回 [x1, y1, x2, y2] 的 bbox（原图坐标系）。

        - pil_image: PIL.Image (原图)
        - trans_vector: numpy.ndarray, shape (C,)，DDT_fit 得到的主方向
        - descriptor_mean_tensor: torch.Tensor, shape (C,) or (1,C,1,1)，DDT_fit 保存的 mean
        - cut_rate: 同原始 DDT
        - over_num: 阈值，用于 max_conn_mask
        """
        device = torch.device("cuda" if self.use_cuda else "cpu")

        # ---- 0. 原图尺寸
        orig_W, orig_H = pil_image.size  # (W,H)

        # ---- 1. 预处理到统一尺寸 self.input_size
        img_resized = self.preprocess(pil_image).unsqueeze(0).to(device)  # [1,3,S,S]

        # ---- 2. 前向 backbone，取最后一层 conv 特征图
        x = img_resized
        # 通用 forward：按 named_children 走到 avgpool 前一层（如果有）
        for name, module in self.pretrained_feature_model.named_children():
            if name in ["avgpool", "fc"]:
                break
            x = module(x)
        # x: [1, C, H_f, W_f]
        featmap = x[0, :]  # [C, H_f, W_f]

        C, H_f, W_f = featmap.shape
        assert C == self.dim, f"feat channels {C} != self.dim {self.dim}"

        # ---- 3. flatten & 去均值
        feat_flat = featmap.view(self.dim, -1).t().contiguous()  # [H_f*W_f, C]

        dm = descriptor_mean_tensor
        # 兼容 (1,C,1,1) 或 (C,)
        if isinstance(dm, torch.Tensor):
            dm = dm.view(-1).to(device=device, dtype=feat_flat.dtype)
        else:
            dm = torch.tensor(dm, device=device, dtype=feat_flat.dtype).view(-1)

        feat_flat = feat_flat - dm.unsqueeze(0)  # [N,C] - [1,C]

        # ---- 4. PCA 主方向打分（trans_vector shape = (C,)）
        tv = torch.tensor(trans_vector,
                          device=device,
                          dtype=feat_flat.dtype).view(1, -1)   # [1,C]
        P = torch.matmul(tv, feat_flat.t())  # [1, N]
        P = P.view(H_f, W_f).detach().cpu().numpy()  # (H_f, W_f)

        # ---- 5. 生成 binary mask（在 self.input_size × self.input_size 空间）
        mask = self.max_conn_mask(P, self.input_size, self.input_size, over_num)

        # ---- 6. 得到 bbox（在 self.input_size 空间里的 x,y,w,h）
        bboxes = self.get_bboxes(mask)
        if len(bboxes) == 0:
            # fallback：整图
            x, y, w_box, h_box = 0, 0, self.input_size, self.input_size
        else:
            x, y, w_box, h_box = bboxes[0]

        # ---- 7. random_cut 在 self.input_size 空间中做扩展
        bbox_resized = self.random_cut(mask, w_box, h_box, cut_rate, if_train=False)
        # bbox_resized 是 [x,y,w,h] in (self.input_size, self.input_size) 坐标

        # ---- 8. 映射回原图坐标
        bbox_orig = self._map_bbox_back_to_origin(bbox_resized, orig_W, orig_H)

        return bbox_orig

    def max_conn_mask(self, P, origin_height, origin_width, over_num=0):
        h, w = P.shape[0], P.shape[1]
        # 只用正值范围归一化
        if (P >= 0).sum() == 0:
            highlight = np.ones_like(P)
        else:
            P_pos = P[P >= 0]
            minv, maxv = P_pos.min(), P_pos.max()
            if maxv > minv:
                P_norm = (P - minv) / (maxv - minv)
            else:
                P_norm = np.zeros_like(P)
            highlight = (P_norm > over_num).astype(np.uint8)

        # 寻找最大的连通分量
        labels = measure.label(highlight, connectivity=1, background=0)
        props = measure.regionprops(labels)

        if props:
            max_index = 0
            for i in range(len(props)):
                if props[i].area > props[max_index].area:
                    max_index = i
            max_prop = props[max_index]
            highlights_conn = np.zeros_like(highlight, dtype=np.uint8)
            for each in max_prop.coords:
                highlights_conn[each[0], each[1]] = 1
        else:
            highlights_conn = np.ones_like(highlight, dtype=np.uint8)
            print("---------------this picture is all 0------------------------")

        # 最近邻插值到 (origin_width, origin_height)
        highlight_big = cv2.resize(
            highlights_conn,
            (origin_width, origin_height),
            interpolation=cv2.INTER_NEAREST
        )
        highlight_big = np.array(highlight_big, dtype=np.uint16).reshape(1, origin_height, origin_width)

        return highlight_big

    def bete_cut(self, highlight_big, x, y, w, h, i):
        _, ori_h, ori_w = highlight_big.shape
        choose_x = int(x + Beta(0.5, 0.5).sample() * w)
        choose_y = int(y + Beta(0.5, 0.5).sample() * h)

        if choose_x < (w / 2 + 1):
            choose_x = 0
        elif choose_x > (ori_w - w / 2) - 1:
            choose_x = int(ori_w - w)
        else:
            choose_x = int(choose_x - w / 2)

        if choose_y < (h / 2 + 1):
            choose_y = 0
        elif choose_y > (ori_h - h / 2) - 1:
            choose_y = int(ori_h - h)
        else:
            choose_y = int(choose_y - h / 2)

        bbox = [choose_x, choose_y, int(w), int(h)]
        return bbox

    def random_cut(self, highlight_big, w, h, cut_rate, if_train=True):
        """
        在 mask 高亮区域附近做随机/平均裁剪（依然在 self.input_size 空间）
        """
        _, ori_h, ori_w = highlight_big.shape
        x_index = np.argwhere(highlight_big > 0)

        if x_index.shape[0] == 0:
            # 整图 fallback
            choose_y, choose_x = ori_h // 2, ori_w // 2
        else:
            if if_train:
                random_i = random.randint(0, x_index.shape[0] - 1)
                _, choose_y, choose_x = x_index[random_i]
            else:
                _, choose_y, choose_x = np.average(x_index, axis=0)

        cut_w = min(cut_rate * w, ori_w)
        cut_h = min(cut_rate * h, ori_h)

        if choose_x < (cut_w / 2 + 1):
            choose_x = 0
        elif choose_x > (ori_w - cut_w / 2) - 1:
            choose_x = int(ori_w - cut_w)
        else:
            choose_x = int(choose_x - cut_w / 2)

        if choose_y < (cut_h / 2 + 1):
            choose_y = 0
        elif choose_y > (ori_h - cut_h / 2) - 1:
            choose_y = int(ori_h - cut_h)
        else:
            choose_y = int(choose_y - cut_h / 2)

        bbox = [choose_x, choose_y, int(cut_w), int(cut_h)]
        return bbox

    def get_bboxes(self, bin_img):
        img = np.squeeze(bin_img.copy().astype(np.uint8), axis=0)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for c in contours:
            rect = cv2.boundingRect(c)  # (x,y,w,h)
            bboxes.append(rect)
        return bboxes

    def _map_bbox_back_to_origin(self, bbox_resized, orig_W, orig_H):
        """
        将 self.input_size 空间里的 [x, y, w, h] 映射回原图坐标 [x1, y1, x2, y2]
        """
        x, y, w, h = bbox_resized

        scale_x = orig_W / float(self.input_size)
        scale_y = orig_H / float(self.input_size)

        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + w) * scale_x))
        y2 = int(round((y + h) * scale_y))

        x1 = max(0, min(orig_W - 1, x1))
        y1 = max(0, min(orig_H - 1, y1))
        x2 = max(0, min(orig_W,     x2))
        y2 = max(0, min(orig_H,     y2))

        return [x1, y1, x2, y2]

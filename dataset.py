# dataset.py
import os
from PIL import Image
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataSet(Dataset):
    def __init__(self, img_root, transform=None):
        self.root = os.path.abspath(img_root)
        classes = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        classes.sort()
        class_to_idx = {c: i for i, c in enumerate(classes)}

        file_paths, labels = [], []
        for cls in classes:
            cls_dir = os.path.join(self.root, cls)
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if os.path.isfile(fpath):
                    file_paths.append(os.path.abspath(fpath))
                    labels.append(class_to_idx[cls])

        self.imgs = file_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = default_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)

class MyPair_entropy_DDT_four(MyDataSet):
    def __init__(self, img_root, transform=None, crop_root=None, strict_crop=True):
        """
        crop_root: 裁剪图根目录（与原始 img_root 的相对路径结构一致）
        strict_crop: True 时缺失裁剪图直接报错
        """
        super().__init__(img_root=img_root, transform=transform)
        if not crop_root:
            raise ValueError("必须提供 crop_root（裁剪图根目录，目录结构需与 img_root 一致）")
        self.crop_root = os.path.abspath(crop_root)
        self.strict_crop = strict_crop

    def _to_crop_path(self, img_path: str) -> str:
        # 用相对于 img_root 的路径去映射到 crop_root
        rel = os.path.relpath(os.path.abspath(img_path), start=self.root)
        if rel.startswith(".."):
            rel = os.path.basename(img_path)  # 极端兜底；正常不应触发
        return os.path.join(self.crop_root, rel)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = default_loader(img_path)

        # 原图两路视图
        im_1 = self.transform(img) if self.transform is not None else img
        im_2 = self.transform(img) if self.transform is not None else img

        # 裁剪图两路视图（只从 crop_root 读取）
        crop_path = self._to_crop_path(img_path)
        if not os.path.exists(crop_path):
            if self.strict_crop:
                raise FileNotFoundError(
                    f"未找到裁剪图：{crop_path}\n"
                    f"原图：{img_path}\n"
                    f"请确保裁剪图在 crop_root 下的相对路径与原图一致。"
                )
            img_new = img  # 如果你以后想宽松处理，可以把 strict_crop 设为 False
        else:
            img_new = default_loader(crop_path)

        if self.transform is not None:
            im_3 = self.transform(img_new)
            im_4 = self.transform(img_new)
        else:
            im_3, im_4 = img_new, img_new

        # 现在不再返回 entropy
        return im_1, im_2, im_3, im_4

class MyDataSet_DDT(Dataset):
    def __init__(self, img_root, transform=None, crop_root=None, strict_crop=True, **kwargs):
        """
        img_root: 原始数据根目录（本类用于根据此目录建立类别与相对路径）
        crop_root: 裁剪图根目录（目录结构与 img_root 保持一致；若为 None 则直接读原图）
        strict_crop: True 时若找不到裁剪图就报错；False 时退回读原图
        """
        self.root = os.path.abspath(img_root)
        self.transform = transform
        self.crop_root = os.path.abspath(crop_root) if crop_root is not None else None
        self.strict_crop = strict_crop

        file_paths = []
        labels_list = []
        labels = {}
        label_counter = 0

        # 按类别子目录建立标签映射（与原实现一致）
        for root, dirs, files in os.walk(self.root):
            # 只在第一层目录建立标签；保持与原实现一致（遍历子目录）
            if root == self.root:
                for dir_name in dirs:
                    subdir_path = os.path.join(root, dir_name)
                    if dir_name not in labels:
                        labels[dir_name] = label_counter
                        label_counter += 1
                    # 收集该子目录下的文件
                    for fname in os.listdir(subdir_path):
                        fpath = os.path.join(subdir_path, fname)
                        if os.path.isfile(fpath):
                            file_paths.append(os.path.abspath(fpath))
                            labels_list.append(labels[dir_name])

        self.imgs = file_paths
        self.labels = labels_list

    def _to_crop_path(self, img_abs_path: str) -> str:
        """将原图绝对路径映射到裁剪图绝对路径：<crop_root>/<relpath-from-img_root>"""
        rel = os.path.relpath(os.path.abspath(img_abs_path), start=self.root)
        if rel.startswith(".."):
            # 兜底：理论上不应发生；发生则只用文件名
            rel = os.path.basename(img_abs_path)
        return os.path.join(self.crop_root, rel)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.labels[index]

        # 若提供裁剪根目录，则从裁剪目录读取同名同结构图片
        if self.crop_root is not None:
            crop_path = self._to_crop_path(img_path)
            if os.path.exists(crop_path) and os.path.isfile(crop_path):
                img = default_loader(crop_path)
            else:
                if self.strict_crop:
                    raise FileNotFoundError(
                        f"未找到裁剪图：{crop_path}\n"
                        f"原图：{img_path}\n"
                        f"请确保裁剪图在 crop_root 下与原图相同的相对路径。"
                    )
                # 宽松模式：退回原图
                img = default_loader(img_path)
        else:
            # 未提供裁剪目录：直接读取原图
            img = default_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)
    

class MyPair_entropy_DDT_five(MyDataSet):
    def __init__(self, img_root,transform=None, transform_distill=None, crop_root=None, strict_crop=True):
        """
        crop_root: 裁剪图根目录（与原始 img_root 的相对路径结构一致）
        strict_crop: True 时缺失裁剪图直接报错
        """
        super().__init__(img_root=img_root, transform=transform)
        if not crop_root:
            raise ValueError("必须提供 crop_root（裁剪图根目录，目录结构需与 img_root 一致）")
        self.crop_root = os.path.abspath(crop_root)
        self.strict_crop = strict_crop
        self.distill_transform = transform_distill

    def _to_crop_path(self, img_path: str) -> str:
        # 用相对于 img_root 的路径去映射到 crop_root
        rel = os.path.relpath(os.path.abspath(img_path), start=self.root)
        if rel.startswith(".."):
            rel = os.path.basename(img_path)  # 极端兜底；正常不应触发
        return os.path.join(self.crop_root, rel)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = default_loader(img_path)  

        # 原图两路视图
        im_1 = self.transform(img) if self.transform is not None else img
        im_2 = self.transform(img) if self.transform is not None else img

        # 裁剪图两路视图（只从 crop_root 读取）
        crop_path = self._to_crop_path(img_path)
        if not os.path.exists(crop_path):
            if self.strict_crop:
                raise FileNotFoundError(
                    f"未找到裁剪图：{crop_path}\n"
                    f"原图：{img_path}\n"
                    f"请确保裁剪图在 crop_root 下的相对路径与原图一致。"
                )
            img_new = img
        else:
            img_new = default_loader(crop_path)

        if self.transform is not None:
            im_3 = self.transform(img_new)
            im_4 = self.transform(img_new)
        else:
            im_3, im_4 = img_new, img_new
        
        if self.distill_transform is not None:
            im_distill = self.distill_transform(img_new)
        else:
            # 兜底：若未提供 distill_transform，就复用强增广
            im_distill = self.transform(img_new) if self.transform is not None else img_new

        # 现在不再返回 entropy
        return im_1, im_2, im_3, im_4, im_distill
    

class MyPair_entropy_DDT_four_dift(MyDataSet):
    def __init__(self, img_root, global_transform=None, local_transform=None, crop_root=None, strict_crop=True):
        """
        crop_root: 裁剪图根目录（与原始 img_root 的相对路径结构一致）
        strict_crop: True 时缺失裁剪图直接报错
        """
        super().__init__(img_root=img_root)
        if not crop_root:
            raise ValueError("必须提供 crop_root（裁剪图根目录，目录结构需与 img_root 一致）")
        self.crop_root = os.path.abspath(crop_root)
        self.strict_crop = strict_crop
        self.global_transform=global_transform
        self.local_transform=local_transform

    def _to_crop_path(self, img_path: str) -> str:
        # 用相对于 img_root 的路径去映射到 crop_root
        rel = os.path.relpath(os.path.abspath(img_path), start=self.root)
        if rel.startswith(".."):
            rel = os.path.basename(img_path)  # 极端兜底；正常不应触发
        return os.path.join(self.crop_root, rel)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = default_loader(img_path)

        # 原图两路视图
        im_1 = self.global_transform(img) if self.global_transform is not None else img
        im_2 = self.global_transform(img) if self.global_transform is not None else img

        # 裁剪图两路视图（只从 crop_root 读取）
        crop_path = self._to_crop_path(img_path)
        if not os.path.exists(crop_path):
            if self.strict_crop:
                raise FileNotFoundError(
                    f"未找到裁剪图：{crop_path}\n"
                    f"原图：{img_path}\n"
                    f"请确保裁剪图在 crop_root 下的相对路径与原图一致。"
                )
            img_new = img  # 如果你以后想宽松处理，可以把 strict_crop 设为 False
        else:
            img_new = default_loader(crop_path)

        if self.local_transform is not None:
            im_3 = self.local_transform(img_new)
            im_4 = self.local_transform(img_new)
        else:
            im_3, im_4 = img_new, img_new

        # 现在不再返回 entropy
        return im_1, im_2, im_3, im_4
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
        crop_root: root directory of cropped images (the relative directory structure
                   should be identical to img_root)
        strict_crop: if True, raise an error when a cropped image is missing
        """
        super().__init__(img_root=img_root, transform=transform)
        if not crop_root:
            raise ValueError("crop_root must be provided (directory structure must match img_root)")
        self.crop_root = os.path.abspath(crop_root)
        self.strict_crop = strict_crop

    def _to_crop_path(self, img_path: str) -> str:
        # Map the relative path (w.r.t. img_root) to crop_root
        rel = os.path.relpath(os.path.abspath(img_path), start=self.root)
        if rel.startswith(".."):
            rel = os.path.basename(img_path)  # extreme fallback; should not normally happen
        return os.path.join(self.crop_root, rel)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = default_loader(img_path)

        # Two augmented views from the original image
        im_1 = self.transform(img) if self.transform is not None else img
        im_2 = self.transform(img) if self.transform is not None else img

        # Two augmented views from the cropped image (loaded from crop_root only)
        crop_path = self._to_crop_path(img_path)
        if not os.path.exists(crop_path):
            if self.strict_crop:
                raise FileNotFoundError(
                    f"Cropped image not found: {crop_path}\n"
                    f"Original image: {img_path}\n"
                    f"Please ensure the cropped image has the same relative path under crop_root."
                )
            img_new = img  # fallback if strict_crop is False
        else:
            img_new = default_loader(crop_path)

        if self.transform is not None:
            im_3 = self.transform(img_new)
            im_4 = self.transform(img_new)
        else:
            im_3, im_4 = img_new, img_new

        # Entropy is no longer returned
        return im_1, im_2, im_3, im_4

class MyDataSet_DDT(Dataset):
    def __init__(self, img_root, transform=None, crop_root=None, strict_crop=True, **kwargs):
        """
        img_root: root directory of the original dataset (used to build class labels and relative paths)
        crop_root: root directory of cropped images (must keep the same directory structure as img_root);
                   if None, original images are used directly
        strict_crop: if True, raise an error when a cropped image is missing;
                     if False, fallback to the original image
        """
        self.root = os.path.abspath(img_root)
        self.transform = transform
        self.crop_root = os.path.abspath(crop_root) if crop_root is not None else None
        self.strict_crop = strict_crop

        file_paths = []
        labels_list = []
        labels = {}
        label_counter = 0

        # Build class-to-index mapping from first-level subdirectories
        for root, dirs, files in os.walk(self.root):
            # Only create labels at the first directory level (consistent with the original implementation)
            if root == self.root:
                for dir_name in dirs:
                    subdir_path = os.path.join(root, dir_name)
                    if dir_name not in labels:
                        labels[dir_name] = label_counter
                        label_counter += 1
                    # Collect image files under this class directory
                    for fname in os.listdir(subdir_path):
                        fpath = os.path.join(subdir_path, fname)
                        if os.path.isfile(fpath):
                            file_paths.append(os.path.abspath(fpath))
                            labels_list.append(labels[dir_name])

        self.imgs = file_paths
        self.labels = labels_list

    def _to_crop_path(self, img_abs_path: str) -> str:
        """Map an absolute image path to the corresponding cropped image path:
        <crop_root>/<relative-path-from-img_root>
        """
        rel = os.path.relpath(os.path.abspath(img_abs_path), start=self.root)
        if rel.startswith(".."):
            # Fallback: theoretically should not happen; use filename only
            rel = os.path.basename(img_abs_path)
        return os.path.join(self.crop_root, rel)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.labels[index]

        # If crop_root is provided, try to load the corresponding cropped image
        if self.crop_root is not None:
            crop_path = self._to_crop_path(img_path)
            if os.path.exists(crop_path) and os.path.isfile(crop_path):
                img = default_loader(crop_path)
            else:
                if self.strict_crop:
                    raise FileNotFoundError(
                        f"Cropped image not found: {crop_path}\n"
                        f"Original image: {img_path}\n"
                        f"Please ensure the cropped image has the same relative path under crop_root."
                    )
                # Non-strict mode: fallback to original image
                img = default_loader(img_path)
        else:
            # No crop directory provided: load original image directly
            img = default_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)
    

class MyPair_entropy_DDT_five(MyDataSet):
    def __init__(self, img_root, transform=None, transform_distill=None, crop_root=None, strict_crop=True):
        """
        crop_root: root directory of cropped images (the relative directory structure
                   should be identical to img_root)
        strict_crop: if True, raise an error when a cropped image is missing
        """
        super().__init__(img_root=img_root, transform=transform)
        if not crop_root:
            raise ValueError("crop_root must be provided (directory structure must match img_root)")
        self.crop_root = os.path.abspath(crop_root)
        self.strict_crop = strict_crop
        self.distill_transform = transform_distill

    def _to_crop_path(self, img_path: str) -> str:
        # Map the relative path (w.r.t. img_root) to crop_root
        rel = os.path.relpath(os.path.abspath(img_path), start=self.root)
        if rel.startswith(".."):
            rel = os.path.basename(img_path)  # extreme fallback; should not normally happen
        return os.path.join(self.crop_root, rel)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = default_loader(img_path)  

        # Two augmented views from the original image
        im_1 = self.transform(img) if self.transform is not None else img
        im_2 = self.transform(img) if self.transform is not None else img

        # Two augmented views from the cropped image
        crop_path = self._to_crop_path(img_path)
        if not os.path.exists(crop_path):
            if self.strict_crop:
                raise FileNotFoundError(
                    f"Cropped image not found: {crop_path}\n"
                    f"Original image: {img_path}\n"
                    f"Please ensure the cropped image has the same relative path under crop_root."
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
            # Fallback: reuse the strong augmentation if distill_transform is not provided
            im_distill = self.transform(img_new) if self.transform is not None else img_new

        # Entropy is no longer returned
        return im_1, im_2, im_3, im_4, im_distill
    

class MyPair_entropy_DDT_four_dift(MyDataSet):
    def __init__(self, img_root, global_transform=None, local_transform=None, crop_root=None, strict_crop=True):
        """
        crop_root: root directory of cropped images (the relative directory structure
                   should be identical to img_root)
        strict_crop: if True, raise an error when a cropped image is missing
        """
        super().__init__(img_root=img_root)
        if not crop_root:
            raise ValueError("crop_root must be provided (directory structure must match img_root)")
        self.crop_root = os.path.abspath(crop_root)
        self.strict_crop = strict_crop
        self.global_transform = global_transform
        self.local_transform = local_transform

    def _to_crop_path(self, img_path: str) -> str:
        # Map the relative path (w.r.t. img_root) to crop_root
        rel = os.path.relpath(os.path.abspath(img_path), start=self.root)
        if rel.startswith(".."):
            rel = os.path.basename(img_path)  # extreme fallback; should not normally happen
        return os.path.join(self.crop_root, rel)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = default_loader(img_path)

        # Two global augmented views from the original image
        im_1 = self.global_transform(img) if self.global_transform is not None else img
        im_2 = self.global_transform(img) if self.global_transform is not None else img

        # Two local augmented views from the cropped image
        crop_path = self._to_crop_path(img_path)
        if not os.path.exists(crop_path):
            if self.strict_crop:
                raise FileNotFoundError(
                    f"Cropped image not found: {crop_path}\n"
                    f"Original image: {img_path}\n"
                    f"Please ensure the cropped image has the same relative path under crop_root."
                )
            img_new = img  # fallback if strict_crop is False
        else:
            img_new = default_loader(crop_path)

        if self.local_transform is not None:
            im_3 = self.local_transform(img_new)
            im_4 = self.local_transform(img_new)
        else:
            im_3, im_4 = img_new, img_new

        # Entropy is no longer returned
        return im_1, im_2, im_3, im_4

# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import os.path
import torch
import cv2
import nibabel as nib
from PIL import Image
import numpy as np
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ToTensor,
    RandRotate,
    Resize,
    RandZoom,
    RandFlip,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandShiftIntensity,
    RandCoarseDropout,
    ScaleIntensity,
    NormalizeIntensity,
    RandHistogramShift,
    Rand2DElastic,
    Rand3DElastic, Compose,
    CropForeground, RandomizableTransform
)


class BaseObject:
    def __init__(self, ignore_keys=["patient", "TIC"], prob=1.0):
        self.R = np.random.RandomState()
        self.ignore_keys = ignore_keys
        self.prob = min(max(prob, 0.0), 1.0)

    @property
    def _do_transform(self):
        return self.R.rand() < self.prob

    def operator(self, m, x):
        raise NotImplementedError

    def operate(self, m, x):
        if isinstance(x, list):
            if isinstance(x[0], (int, float)):
            # TIC input
                return torch.tensor(x, dtype=torch.float)
            else:
                return [self.operate(m, p) for p in x]
        out = self.operator(m, x)
        return out

    def __call__(self, imgs):
        imgs = dict(imgs)
        if 'patient' not in imgs.keys():
            p = None
            for p in imgs.values():
                if isinstance(p, str):
                    break
            imgs['patient'] = os.path.dirname(p)
        try:
            for m in imgs.keys():
                if m in self.ignore_keys:
                    continue
                imgs[m] = self.operate(m, imgs[m])
        except Exception as e:
            print(f"INFO | {imgs['patient']} | Modality {m}")
            raise e
        return imgs


class ReadImage(BaseObject):
    def __init__(self, root_dir=None, spacing=None):
        super(ReadImage, self).__init__(ignore_keys=[])
        self.root_dir = root_dir
        self.loader = LoadImage()
        self.first = EnsureChannelFirst(channel_dim='no_channel')
        self.orient = Orientation('RAS')
        self.spacing = Spacing(pixdim=spacing) if spacing is not None else spacing

    def operator(self, m, x):
        if isinstance(x, str):
            if self.root_dir is not None:
                x = os.path.join(self.root_dir, x)
            if m == 'patient':
                return x
            elif x.endswith('.nii.gz'):
                x = self.loader(x)
                x = self.first(x)
                x = self.orient(x)
                if self.spacing is not None:
                    x = self.spacing(x)
            elif x.endswith('.png') or x.endswith('.jpg'):
                x = self.loader(x)
                if x.ndim == 2:
                    x = x.unsqueeze(0).repeat(3, 1, 1)
                elif x.ndim == 3:
                    x = x.permute(2, 0, 1)
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            print(x)
            raise Exception
        return x


class MyResize(BaseObject):
    def __init__(self, img_size, mode='area'):
        super(MyResize, self).__init__()
        self.img_size = img_size
        if isinstance(img_size, int):
            img_size = [img_size] * 3
        self.resize_3d = Resize(spatial_size=img_size[:3], mode=mode)
        self.resize_2d = Resize(spatial_size=img_size[:2], mode=mode)

    def operator(self, m, x):
        if x.ndim == 3:
            x = self.resize_2d(x)
        elif x.ndim == 4:
            x = self.resize_3d(x)
        return x


class ClipTrans(BaseObject):
    def __init__(self, trans=None):
        super(ClipTrans, self).__init__()
        self.trans = trans

    def operator(self, m, x):
        x = Image.open(x)
        return self.trans(x)


class Resized(BaseObject):
    def __init__(self, img_size, mode='area'):
        super(Resized, self).__init__()
        if isinstance(img_size, int):
            self.resize_3d = Resize(spatial_size=[img_size] * 3, mode=mode)
            self.resize_2d = Resize(spatial_size=[img_size] * 2, mode=mode)
        else:
            self.resize_3d = Resize(spatial_size=img_size[:3], mode=mode)
            self.resize_2d = Resize(spatial_size=img_size[:2], mode=mode)

    def operator(self, m, x):
        if x.ndim == 3:
            return self.resize_2d(x)
        return self.resize_3d(x)


class RandFlipd(BaseObject):
    def __init__(self, prob: float = 0.1):
        super().__init__()
        self.flip = RandFlip(prob=prob, spatial_axis=1)
        self.flip_3d = RandFlip(prob=prob, spatial_axis=(1, 2))

    def operator(self, m, x):
        return self.flip_3d(x) if len(x.shape) == 4 else self.flip(x)


class RandZoomd(BaseObject):
    def __init__(self, min_zoom, max_zoom, prob):
        super().__init__()
        self.zoom = RandZoom(min_zoom=min_zoom, max_zoom=max_zoom, prob=prob, padding_mode='constant')

    def operator(self, m, x):
        return self.zoom(x)


class RandCrop(BaseObject):
    """Foreground crop based on a reference mask under the patient's folder."""

    def __init__(self, source_key: str, prob: float = 0.5, dilation_radius: int = 5, cube: bool = False) -> None:
        super().__init__(ignore_keys=["patient", "TIC"], prob=prob)
        self.cropper = CropForeground()
        self.source_key = source_key
        self.cube = cube
        self.dilation_radius = int(dilation_radius)
        self._bbox_cache = {}

    def _load_mask(self, patient_dir) -> torch.Tensor:
        mask_path = os.path.join(patient_dir, f"{self.source_key}.nii.gz")
        if not os.path.exists(mask_path):
            # 兼容 .nii
            alt = patient_dir / f"{self.source_key}.nii"
            if not alt.exists():
                raise FileNotFoundError(f"Mask not found: {mask_path} or {alt}")
            mask_path = alt
        mask = LoadImage(image_only=True)(str(mask_path))
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)  # (1, D, H, W)
        return mask

    def __call__(self, d):
        d = dict(d)
        if not self._do_transform:
            return d

        patient_dir = d["patient"]
        if patient_dir in self._bbox_cache:
            box_start, box_end, full_shape = self._bbox_cache[patient_dir]
        else:
            mask = self._load_mask(patient_dir)
            if torch.abs(mask).sum() == 0:
                raise ValueError(f"Empty mask: shape={tuple(mask.shape)} under {patient_dir}")
            box_start, box_end = self.cropper.compute_bounding_box(img=mask)
            full_shape = mask.shape[1:]  # (D,H,W)
            self._bbox_cache[patient_dir] = (box_start, box_end, full_shape)

        # 可选：调整为 cube
        if self.cube:
            sizes = [int(e - s) for s, e in zip(box_start, box_end)]
            max_size = int(max(sizes))
            centers = [int((e + s) // 2) for s, e in zip(box_start, box_end)]
            half = max_size // 2
            box_start = np.array([c - half for c in centers], dtype=int)
            box_end = np.array([c + half for c in centers], dtype=int)

        # 膨胀并裁剪到有效范围
        start = np.maximum(0, np.array(box_start, dtype=int) - self.dilation_radius)
        end = np.minimum(np.array(full_shape, dtype=int), np.array(box_end, dtype=int) + self.dilation_radius)

        for key in d.keys():
            if key in self.ignore_keys:
                continue
            d[key] = torch.cat([
                d[key],
                Resize(spatial_size=d[key].shape[1:])(self.cropper.crop_pad(img=d[key], box_start=start, box_end=end, mode="constant", lazy=False))
                ], dim=0)
        # d['TIC'] = torch.cat([d['TIC'], d['TIC']], dim=0)
        return d


class RandGaussianNoised(BaseObject):
    def __init__(self, prob):
        super().__init__()
        self.noise = RandGaussianNoise(prob=prob)

    def operator(self, m, x):
        return self.noise(x)


class RandGaussianSmoothd(BaseObject):
    def __init__(self, prob):
        super().__init__()
        self.smooth = RandGaussianSmooth(prob=prob)

    def operator(self, m, x):
        return self.smooth(x)


class RandRotated(BaseObject):
    def __init__(self, range: float = .1, prob=.1):
        super().__init__()
        self.rotate = RandRotate(range_x=range, prob=prob, padding_mode='zeros')
        self.rotate_3d = RandRotate(range_x=range, range_y=range, range_z=range,
                                    prob=prob, padding_mode='zeros')

    def operator(self, m, x):
        return self.rotate(x) if len(x.shape) == 3 else self.rotate_3d(x)


class ScaleIntensityd(BaseObject):
    def __init__(self, minv=0.0, maxv=1.0):
        super().__init__()
        self.scale = ScaleIntensity(minv=minv, maxv=maxv)

    def operator(self, m, x):
        return self.scale(x)


class RandShiftIntensityd(BaseObject):
    def __init__(self, offsets=0.1, prob=1.0):
        super().__init__()
        self.shift = RandShiftIntensity(offsets=offsets, prob=prob)

    def operator(self, m, x):
        return self.shift(x)


class RandCoarseDropoutd(BaseObject):
    def __init__(self, holes, spatial_size, prob=1.0):
        super().__init__()
        self.dropout = RandCoarseDropout(holes=holes, spatial_size=spatial_size, prob=prob)

    def operator(self, m, x):
        return self.dropout(x)


class RandHistogramShiftd(BaseObject):
    def __init__(self, num_control_points=(5, 10), prob=.1):
        super().__init__()
        self.histo_shift = RandHistogramShift(num_control_points=num_control_points, prob=prob)

    def operator(self, m, x):
        return self.histo_shift(x)


class RandElasticd(BaseObject):
    def __init__(self, spacing=(20, 30), magnitude_range=(0, 1), prob=.1):
        super().__init__()
        self.elastic = Rand2DElastic(spacing=spacing, magnitude_range=magnitude_range, padding_mode='zeros', prob=prob)
        self.elastic_3d = Rand3DElastic(sigma_range=(5, 8), magnitude_range=(20, 50), padding_mode='zeros',
                                        prob=prob)

    def operator(self, m, x):
        return self.elastic(x) if len(x.shape) == 3 else self.elastic_3d(x)


class Normalize(BaseObject):
    def __init__(self, normalize=False, low=-1, high=1):
        super().__init__()
        self.z_normalize = NormalizeIntensity() if normalize else None
        self.low = low
        self.high = high
        self.tensor = ToTensor(dtype=torch.float)

    def operator(self, m, x):
        x = self.tensor(x)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        if self.z_normalize is not None:
            x = self.z_normalize(x)
        else:
            x = x * (self.high - self.low) + self.low
        return x

    def inverse(self, x):
        return self.z_normalize.inverse()


#######################################################################################################################
class Transform:
    def __init__(self, root_dir, normalize, img_size, spacing, crop_prob):
        self.root_dir = root_dir
        self.normalize = normalize
        self.img_size = img_size
        self.spacing = spacing
        self.crop_prob = crop_prob

    def image_reader(self):
        return Compose([ReadImage(root_dir=self.root_dir, spacing=self.spacing),
                        MyResize(img_size=self.img_size)])

    def train_transforms(self):
        return Compose(
            [
                ReadImage(root_dir=self.root_dir, spacing=self.spacing),
                RandCrop(source_key='BOX', prob=self.crop_prob),
                MyResize(img_size=self.img_size),
                RandFlipd(prob=.5),
                RandRotated(range=.3, prob=.5),
                RandGaussianSmoothd(prob=.2),
                RandGaussianNoised(prob=.2),
                RandZoomd(min_zoom=0.8, max_zoom=1.2, prob=.5),
                ##############################################
                # RandElasticd(prob=.2),
                # RandShiftIntensityd(prob=.2),
                ##############################################
                Normalize(normalize=self.normalize),
            ]
        )

    def val_transforms(self):
        return Compose(
            [
                ReadImage(root_dir=self.root_dir, spacing=self.spacing),
                RandCrop(source_key='BOX', prob=self.crop_prob),
                MyResize(img_size=self.img_size),
                Normalize(normalize=self.normalize),
            ]
        )

    def test_transforms(self):
        return Compose(
            [
                ReadImage(root_dir=self.root_dir, spacing=self.spacing),
                RandCrop(source_key='BOX', prob=self.crop_prob),
                MyResize(img_size=self.img_size),
                Normalize(normalize=self.normalize),
            ]
        )

    def pure_transforms(self):
        return Compose([ReadImage(root_dir=self.root_dir, spacing=self.spacing),
                        MyResize(img_size=self.img_size)])

    def __getitem__(self, item):
        if item == 'train':
            return self.train_transforms()
        elif item == 'val':
            return self.val_transforms()
        elif item == 'test':
            return self.test_transforms()
        else:
            return self.pure_transforms()

    @property
    def transforms(self):
        return dict(train=self.train_transforms(), val=self.val_transforms(), test=self.test_transforms())

    @staticmethod
    def clip_transforms(clip_trans):
        return ClipTrans(clip_trans)

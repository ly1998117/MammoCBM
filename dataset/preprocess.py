import os

import nibabel

os.environ["OMP_NUM_THREADS"] = "1"
import shutil
from functools import wraps
from subprocess import CalledProcessError

import cv2
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom as dcm
import matplotlib.pyplot as plt
import subprocess
import ants
from pathlib import Path
from ast import literal_eval
from mpire import WorkerPool
from tqdm import tqdm
from PIL import Image
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Orientation, SaveImage, CropForeground, Spacing, Resize, ResampleToMatch
from monai.transforms import LoadImaged, EnsureChannelFirstd, Orientationd, SaveImaged, CropForegroundd, Resized, Spacingd

from shutil import copy2, copytree
from scipy.ndimage import binary_dilation
from monai.transforms import Transform, MapTransform


class ForceAffine(Transform):
    def __init__(self, space=None):
        self.affine = None
        if space is not None:
            self.affine = torch.eye(3)
            for i, s in enumerate(space):
                self.affine[i, i] = s

    def __call__(self, img):
        if not hasattr(img, "meta") or not hasattr(img, "affine"):
            raise KeyError
        if self.affine is not None:
            img.meta["affine"][:3, :3] = self.affine
        img.meta["affine"][:, 3] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        return img


class ForceAffined(MapTransform):
    def __init__(self, keys, space=None, allow_missing_keys: bool = False, lazy: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.force = ForceAffine(space)

    def __call__(self, data, lazy: bool | None = None):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.force(d[key])
        return d


class CropPart(CropForeground):
    def __init__(self, path, path2):
        x = LoadImage(image_only=True)(path).unsqueeze(0)
        assert x.abs().sum() > 0, f"Empty mask {x.shape} {x.meta['filename_or_obj']}"
        mid = x.shape[1] // 2
        indices = torch.zeros_like(x).bool()
        nonzero = torch.where(x > 0)[1]
        center_x = (nonzero.min() + nonzero.max()).float() / 2.0
        if center_x < mid:
            indices[:, :mid, ...] = True
        else:
            indices[:, mid:, ...] = True
        y = LoadImage(image_only=True)(path2).unsqueeze(0)
        y = ((y - y.min()) / (y.max() - y.min()) * 255) > 40
        # y = (y-y.min())/(y.max()-y.min()) * 128 > 
        resample_nn = ResampleToMatch(mode="nearest")
        y_on_x = resample_nn(y, x).bool()
        base_mask = indices & y_on_x 
        def _select_fn(img):
            m = resample_nn(base_mask, img).bool()
            return m
        super().__init__(select_fn=_select_fn)
        
class DilationCropForegroundd(CropForegroundd):
    def __init__(self, dilation_radius=5, cube=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dilation_radius = dilation_radius
        self.cube = cube

    def __call__(self, data, lazy=None):
        d = dict(data)
        self.cropper: CropForeground
        mask = d[self.source_key]
        assert mask.abs().sum() > 0, f"Empty mask {mask.shape} {mask.meta['filename_or_obj']}"
        box_start, box_end = self.cropper.compute_bounding_box(img=mask)
        if self.cube:
            max_size = max([e - s for s, e in zip(box_start, box_end)])
            centers = [(e + s) // 2 for s, e in zip(box_start, box_end)]
            box_start = [c - max_size // 2 for c in centers]
            box_end = [c + max_size // 2 for c in centers]

        box_start = np.array([max(0, s - self.dilation_radius) for s in box_start])
        box_end = np.array([min(sz, e + self.dilation_radius) for e, sz in zip(box_end, mask.shape[1:])])

        if self.start_coord_key is not None:
            d[self.start_coord_key] = box_start  # type: ignore
        if self.end_coord_key is not None:
            d[self.end_coord_key] = box_end  # type: ignore

        lazy_ = self.lazy if lazy is None else lazy
        for key, m in self.key_iterator(d, self.mode):
            d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m, lazy=lazy_)
        return d


def multi_process(num_process):
    def decorator(func):
        @wraps(func)
        def wrapper(params):
            with WorkerPool(num_process) as pool:
                return pool.map(func, params, progress_bar=True)

        return wrapper

    return decorator


def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


@multi_process(16)
def dcm_to_nii(path, npath):
    os.makedirs(os.path.dirname(npath), exist_ok=True)
    if os.path.isdir(path):
        npath = npath + '.nii.gz'
        if os.path.exists(npath):
            return
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(path))
        reader.SetFileNames(dicom_names)
        try:
            image = reader.Execute()
        except RuntimeError as error:
            print(error, str(path))
            return
        array = sitk.GetArrayFromImage(image)
        if 'ADC' in path and array.max() < 1:
            array = (array - array.min()) / (array.max() - array.min()) * 4095 - 1
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        image = sitk.GetImageFromArray(array.astype(np.int16))
        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        sitk.WriteImage(image, npath)
    elif '.dcm' in path:
        npath = npath + '.png'
        if os.path.exists(npath):
            return
        tic = dcm.dcmread(path)
        if tic.pixel_array.max() == 0:
            overlay_data = tic[0x6000, 0x3000].value
            rows = tic[0x6000, 0x0010].value
            columns = tic[0x6000, 0x0011].value

            # 叠加数据通常是位打包的，需要解包
            overlay_array = np.frombuffer(overlay_data, dtype=np.uint8)
            overlay_bits = np.unpackbits(overlay_array, bitorder='little')
            image = overlay_bits.reshape(rows, columns) * 255
        else:
            image = tic.pixel_array
        # black to white
        # image = 255 - image
        if len(image.shape) == 4:
            raise ValueError('4D image')
        image = Image.fromarray(image)
        image.save(npath)
    elif '.nii.gz' in path:
        npath = npath + '.nii.gz'
        if os.path.exists(npath):
            return
        copy2(path, npath)
    elif '.jpg' in path:
        print(path, npath)
        npath = npath + '.jpg'
        if os.path.exists(npath):
            return
        copy2(path, npath)


@multi_process(64)
def monai_preprocess(**kwargs):
    patient_dir = kwargs['dir_path']
    pathology_dir = os.path.dirname(patient_dir)
    output_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(pathology_dir)), kwargs['save_dir'])
    output_patient_dir = os.path.join(output_dataset_dir, os.path.basename(pathology_dir),
                                      os.path.basename(patient_dir))
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim='no_channel'),
        Orientation(axcodes='RAS'),
        Spacing(pixdim=(0.625, 0.625, 1)),
        ForceAffine(),
        SaveImage(output_dir=output_dataset_dir, output_ext='.nii.gz', output_postfix='',
                  data_root_dir=os.path.dirname(pathology_dir),
                  separate_folder=False, output_dtype=np.int16)
    ])
    for data in ['ADC.nii.gz', 'CE.nii.gz', 'DWI.nii.gz', 'T2WI.nii.gz', 'BOX.nii.gz']:
        path = os.path.join(patient_dir, data)
        if not os.path.exists(os.path.join(output_patient_dir, data)):
            transform(path)
    if os.path.exists(f'{patient_dir}/TIC.png'):
        copy2(f'{patient_dir}/TIC.png', f'{output_patient_dir}/TIC.png')
        return
    if os.path.exists(f'{patient_dir}/TIC.jpg'):
        copy2(f'{patient_dir}/TIC.jpg', f'{output_patient_dir}/TIC.jpg')
        return
    raise ValueError(f'No TIC image: {patient_dir}')


@multi_process(64)
def monai_process_crop_breast(**kwargs):
    pos = kwargs['position']
    patient_dir = kwargs['dir_path']
    pathology_dir = os.path.dirname(patient_dir)
    output_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(pathology_dir)), kwargs['save_dir'])
    output_patient_dir = os.path.join(output_dataset_dir, os.path.basename(pathology_dir),
                                      os.path.basename(patient_dir))

    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim='no_channel'),
        CropPart(os.path.join(patient_dir, 'BOX.nii.gz'), os.path.join(patient_dir, 'CE.nii.gz')),
   
        Resize(spatial_size=(128, 256, 128)),
        SaveImage(output_dir=output_dataset_dir, output_ext='.nii.gz', output_postfix='',
                  data_root_dir=os.path.dirname(pathology_dir),
                  separate_folder=False, output_dtype=np.int16)
    ])
    for data in ['ADC.nii.gz', 'CE.nii.gz', 'DWI.nii.gz', 'T2WI.nii.gz', 'BOX.nii.gz']:
        path = os.path.join(patient_dir, data)
        if not os.path.exists(os.path.join(output_patient_dir, data)):
            try:
                transform(path)
            except:
                raise ValueError(path)
    if os.path.exists(f'{patient_dir}/TIC.png'):
        copy2(f'{patient_dir}/TIC.png', f'{output_patient_dir}/TIC.png')
        return
    if os.path.exists(f'{patient_dir}/TIC.jpg'):
        copy2(f'{patient_dir}/TIC.jpg', f'{output_patient_dir}/TIC.jpg')
        return
    raise ValueError(f'No TIC image: {patient_dir}')


@multi_process(64)
def monai_process_crop_box(**kwargs):
    patient_dir = kwargs['dir_path']
    pathology_dir = os.path.dirname(patient_dir)
    output_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(pathology_dir)), kwargs['save_dir'])
    output_patient_dir = os.path.join(output_dataset_dir, os.path.basename(pathology_dir),
                                      os.path.basename(patient_dir))
    print(f'{patient_dir} -> {output_patient_dir}')
    transform = Compose([
        LoadImaged(image_only=True, keys=['image', 'bbox']),
        EnsureChannelFirstd(channel_dim='no_channel', keys=['image', 'bbox']),
        # ForceAffined(keys=['image', 'bbox'], space=(1, 1, 1)),
        Spacingd(keys=['image', 'bbox'], pixdim=(0.625, 0.625, 0.625)),
        DilationCropForegroundd(keys=['image'], source_key='bbox', dilation_radius=15, cube=True),
        SaveImaged(keys=['image'], output_dir=output_dataset_dir, output_ext='.nii.gz', output_postfix='',
                   data_root_dir=os.path.dirname(pathology_dir),
                   separate_folder=False, output_dtype=np.int16)
    ])
    for data in ['ADC.nii.gz', 'CE.nii.gz', 'DWI.nii.gz', 'T2WI.nii.gz', 'BOX.nii.gz']:
        path = dict(image=os.path.join(patient_dir, data), bbox=os.path.join(patient_dir, 'BOX.nii.gz'))
        if not os.path.exists(os.path.join(output_patient_dir, data)):
            transform(path)
    for f in ['CE.nii.gz', 'BOX.nii.gz', 'TIC.png', 'TIC.jpg']:
        if os.path.exists(f'{patient_dir}/{f}') and not os.path.exists(f'{output_patient_dir}/{f}'):
            copy2(f'{patient_dir}/{f}', f'{output_patient_dir}/{f}')


@multi_process(64)
def registration_freesurfer(**kwargs):
    def _mri_reg(mov, dst, lta, out, reg=True):
        if os.path.exists(out):
            _check(out)
            return
        if not os.path.exists(lta) and reg:
            cmd_register = [
                "mri_robust_register",
                "--mov", mov,
                "--dst", dst,
                "--lta", lta,
                "--affine",
                "--iscale",
                "--satit",
                "--subsample", "2",
                # "--noinit"
            ]
            subprocess.run(cmd_register, check=True)

        cmd_convert = [
            "mri_vol2vol",
            "--mov", mov,
            "--targ", dst,
            "--reg", lta,
            "--o", out
        ]
        subprocess.run(cmd_convert, check=True)
        _check(out)

    def _check(out):
        img = nibabel.load(out).get_fdata()
        if np.abs(img).sum() < 1:
            raise ValueError(f"Empty mask {img.shape} {out}")

    patient_dir = kwargs['dir_path']
    pathology_dir = os.path.dirname(patient_dir)
    output_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(pathology_dir)), kwargs['save_dir'])
    output_patient_dir = os.path.join(output_dataset_dir, os.path.basename(pathology_dir),
                                      os.path.basename(patient_dir))
    os.makedirs(output_patient_dir, exist_ok=True)
    print(f'{patient_dir} -> {output_patient_dir}')
    for f in ['CE.nii.gz', 'BOX.nii.gz', 'TIC.png', 'TIC.jpg']:
        if os.path.exists(f'{patient_dir}/{f}') and not os.path.exists(f'{output_patient_dir}/{f}'):
            copy2(f'{patient_dir}/{f}', f'{output_patient_dir}/{f}')

    try:
        dst_path = os.path.join(patient_dir, 'CE.nii.gz')
        _mri_reg(mov=os.path.join(patient_dir, 'DWI.nii.gz'), dst=dst_path,
                 lta=os.path.join(output_patient_dir, 'DWI2CE.lta'),
                 out=os.path.join(output_patient_dir, 'DWI.nii.gz'))
        _mri_reg(mov=os.path.join(patient_dir, 'ADC.nii.gz'), dst=dst_path,
                 lta=os.path.join(output_patient_dir, 'DWI2CE.lta'),
                 out=os.path.join(output_patient_dir, 'ADC.nii.gz'), reg=False)

        # _delete(os.path.join(output_patient_dir, 'DWI2CE.lta'))
        _mri_reg(mov=os.path.join(patient_dir, 'T2WI.nii.gz'), dst=dst_path,
                 lta=os.path.join(output_patient_dir, 'T2WI2CE.lta'),
                 out=os.path.join(output_patient_dir, 'T2WI.nii.gz'))
        # _delete(os.path.join(output_patient_dir, 'T2WI2CE.lta'))
    except Exception as e:
        for f in ['DWI.nii.gz', 'ADC.nii.gz', 'T2WI.nii.gz']:
            if os.path.exists(f'{patient_dir}/{f}'):
                copy2(f'{patient_dir}/{f}', f'{output_patient_dir}/{f}')
        return {'path': output_patient_dir, 'status': e}
    return {'path': output_patient_dir, 'status': 'DONE'}


@multi_process(32)
def registration_ants(**kwargs):
    patient_dir = kwargs['dir_path']
    pathology_dir = os.path.dirname(patient_dir)
    output_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(pathology_dir)), kwargs['save_dir'])
    output_patient_dir: str = os.path.join(output_dataset_dir, os.path.basename(pathology_dir),
                                           os.path.basename(patient_dir))
    os.makedirs(output_patient_dir, exist_ok=True)
    print(f'{patient_dir} -> {output_patient_dir}')

    # 拷贝图片（如存在）
    for tic_name in ['CE.nii.gz', 'BOX.nii.gz', 'TIC.png', 'TIC.jpg']:
        src = os.path.join(patient_dir, tic_name)
        dst = os.path.join(output_patient_dir, tic_name)
        if os.path.exists(src) and not os.path.exists(dst):
            copy2(src, dst)

    def ants_reg(moving, fixed, out_path, reg=None, outprefix='', interp='linear'):
        if os.path.exists(out_path):
            return
        if isinstance(moving, str):
            moving = ants.image_read(moving)
        if isinstance(fixed, str):
            fixed = ants.image_read(fixed)
        if reg is None:
            reg = ants.registration(fixed=fixed, moving=moving,
                                    type_of_transform='SyN',
                                    outprefix=f'{outprefix}/',
                                    # type_of_transform='Affine',
                                    # type_of_transform='antsRegistrationSyN[a]',
                                    )
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=reg['fwdtransforms'],
            interpolator=interp
        )
        warped = ants.from_numpy(
            data=warped.numpy().astype(np.int16),
            origin=warped.origin,
            spacing=warped.spacing,
            direction=warped.direction
        )
        warped.to_filename(out_path)
        return reg

    try:
        fixed = os.path.join(output_patient_dir, 'CE.nii.gz')

        # DWI → CE
        dwi_path = os.path.join(patient_dir, 'DWI.nii.gz')
        dwi_out_path = os.path.join(output_patient_dir, 'DWI.nii.gz')
        reg = ants_reg(moving=dwi_path, fixed=fixed, out_path=dwi_out_path, outprefix=output_patient_dir)

        # ADC → CE using DWI→CE transform
        adc_path = os.path.join(patient_dir, 'ADC.nii.gz')
        adc_out_path = os.path.join(output_patient_dir, 'ADC.nii.gz')
        ants_reg(moving=adc_path, fixed=fixed, out_path=adc_out_path, reg=reg, outprefix=output_patient_dir)

        # T2WI → CE (独立配准)
        t2_path = os.path.join(patient_dir, 'T2WI.nii.gz')
        t2_out_path = os.path.join(output_patient_dir, 'T2WI.nii.gz')
        ants_reg(moving=t2_path, fixed=fixed, out_path=t2_out_path, outprefix=output_patient_dir)


    except Exception as e:
        print(f"[ERROR] {patient_dir}: {e}")
        return {'path': output_patient_dir, 'status': e}

    return {'path': output_patient_dir, 'status': 'DONE'}


class Preprocess:
    def __init__(self, rootdir, modalities):
        self.rootdir = Path(rootdir)
        self.modalities = modalities
        self.data = []
        self.error = []
        for pathology in self.rootdir.iterdir():
            if not pathology.is_dir() or 'CSV' in pathology.name:
                continue
            for patient in pathology.iterdir():
                if not patient.is_dir():
                    continue
                modalities = [m for m in patient.iterdir() if
                              any([tm for tm in self.modalities if
                                   tm in '_'.join(m.name.split('_')[-2:]) and '.lta' not in m.name])]
                for modality in modalities:
                    if '.DS_Store' in str(modality):
                        continue
                    if len(modalities) == len(self.modalities):
                        self.data.append({
                            'pathology': pathology.name,
                            'name': patient.name,
                            'modality': modality.name,
                            'path': str(modality)
                        })
                    else:
                        self.error.append({
                            'pathology': pathology.name,
                            'name': patient.name,
                            'modality': modality.name,
                            'path': str(modality)
                        })

        self.data = pd.DataFrame(self.data)
        self.error = pd.DataFrame(self.error)
        if len(self.error):
            raise FileNotFoundError(self.error)

    def dcm_to_nii(self):
        data = []
        for idx, row in tqdm(self.data.iterrows()):
            path = row['path']
            npath = os.path.join(
                str(self.rootdir).replace('DCM', 'NII_TMP'),
                row['pathology'], row['name'], row['modality']
            )
            data.append((path, npath))
        dcm_to_nii(data)

    def regularize(self, save_dir):
        info = pd.concat([
            pd.read_csv(f'{self.rootdir}/BC.csv'),
            pd.read_csv(f'{self.rootdir}/Benign.csv')
        ]).reset_index()
        info['dir_path'] = info.apply(lambda x: os.path.join(str(self.rootdir), x['pathology'], x['name_en']), axis=1)
        info['save_dir'] = save_dir
        info = info.to_dict(orient='records')
        monai_preprocess(info)
        copy2(f'{self.rootdir}/BC.csv', f'{os.path.join(os.path.dirname(self.rootdir), save_dir)}/BC.csv')
        copy2(f'{self.rootdir}/Benign.csv', f'{os.path.join(os.path.dirname(self.rootdir), save_dir)}/Benign.csv')

    def crop_tic(self):
        for idx, row in tqdm(self.data.iterrows()):
            if 'TIC' in row['modality']:
                path = row['path']
                npath = path.replace(os.path.basename(str(self.rootdir)), f'{os.path.basename(str(self.rootdir))}_TIC')
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # black to white
                image = 255 - image
                image[image < 200] = 0
                # mask = np.all(image == 255, axis=-1)
                # image[~mask] = 0
                # size = str(image.shape[:2]).replace('(', '').replace(')', '').replace(', ', '_')
                # npath = f'TIC/{size}/{npath}'
                os.makedirs(os.path.dirname(npath), exist_ok=True)
                cv2.imwrite(npath, image)
                # os.system(f'cp {path} {npath}')
                # # blue [57, 125, 220],  [140, 255, 255]
                # lower = np.array([57, 125, 220])
                # upper = np.array([60, 130, 230])
                # mask = cv2.inRange(image, lower, upper)
                # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # for contour in contours:
                #     x, y, w, h = cv2.boundingRect(contour)
                #     x0, x1 = (x, y), (x + w, y + h)
                #     cv2.rectangle(image, x0, x1, (255, 0, 0), 2)

                pass

    def save_data_info(self, pathology='Benign', filename='CLIP_sec1'):
        if pathology == 'Benign':
            info = pd.read_excel(f'{self.rootdir}/{filename}_Benign.xlsx')
        else:
            info = pd.read_excel(f'{self.rootdir}/{filename}_BC.xlsx').reset_index(drop=True)

        data = self.data.loc[self.data['pathology'] == pathology].copy()
        data['id'] = data['name'].map(lambda x: '_'.join(x.replace('-', '_').split('_')[2:]).lower())
        info['id'] = info['id'].map(lambda x: str(x).lower().replace('-', '_'))
        patient = data[['id', 'name', 'pathology']].drop_duplicates()
        inner = patient.merge(info, on='id', suffixes=('_en', '_cn'), how='inner')[
            ['id', 'name_en', 'pathology', '备注', 'position',
             'name_cn', 'sex', 'age', '检查室', 'MRI检查时间', '影像表现', '影像诊断', '病理报告时间',
             '病理结果']]
        outer1 = info[~info['id'].isin(inner['id'])]
        print(outer1[['id', 'name']])
        inner.to_csv(f'{self.rootdir}/{pathology}.csv', index=False)
        pass

    def to_datalist(self):
        p_data = []
        for pathology in self.data['pathology'].unique():
            path_data = self.data.loc[self.data['pathology'] == pathology]
            for name in path_data['name'].unique():
                data = path_data.loc[path_data['name'] == name].sort_values(by='modality')
                p_data.append({
                    'name': name,
                    'pathology': pathology,
                    'path': {m: p for m, p in zip(data['modality'], data['path'])},
                    'modality': data['modality'].unique().tolist()
                })
        os.makedirs(f'{self.rootdir}/CSV/data_split', exist_ok=True)
        p_data = pd.DataFrame(data=p_data).reset_index(drop=True)
        p_data['pid'] = p_data.index
        p_data.to_csv(f'{self.rootdir}/CSV/data_split/datalist.csv', index=False)

    def breast_crop(self, save_dir):
        info = pd.concat([
            pd.read_csv(f'{self.rootdir}/BC.csv'),
            pd.read_csv(f'{self.rootdir}/Benign.csv')
        ]).reset_index()
        info['dir_path'] = info.apply(lambda x: os.path.join(str(self.rootdir), x['pathology'], x['name_en']), axis=1)
        info['save_dir'] = save_dir
        info = info.to_dict(orient='records')
        monai_process_crop_breast(info)
        copy2(f'{self.rootdir}/BC.csv', f'{os.path.join(os.path.dirname(self.rootdir), save_dir)}/BC.csv')
        copy2(f'{self.rootdir}/Benign.csv', f'{os.path.join(os.path.dirname(self.rootdir), save_dir)}/Benign.csv')

    def box_crop(self, save_dir):
        info = pd.concat([
            pd.read_csv(f'{self.rootdir}/BC.csv'),
            pd.read_csv(f'{self.rootdir}/Benign.csv')
        ]).reset_index()
        info['dir_path'] = info.apply(lambda x: os.path.join(str(self.rootdir), x['pathology'], x['name_en']), axis=1)
        info['save_dir'] = save_dir
        info = info.to_dict(orient='records')
        monai_process_crop_box(info)
        copy2(f'{self.rootdir}/BC.csv', f'{os.path.join(os.path.dirname(self.rootdir), save_dir)}/BC.csv')
        copy2(f'{self.rootdir}/Benign.csv', f'{os.path.join(os.path.dirname(self.rootdir), save_dir)}/Benign.csv')

    def registration(self, save_dir, ants=False):
        info = pd.concat([
            pd.read_csv(f'{self.rootdir}/BC.csv'),
            pd.read_csv(f'{self.rootdir}/Benign.csv')
        ]).reset_index()
        info['dir_path'] = info.apply(lambda x: os.path.join(str(self.rootdir), x['pathology'], x['name_en']), axis=1)
        info['save_dir'] = save_dir
        info = info.to_dict(orient='records')
        if ants:
            results = registration_ants(info)
        else:
            results = registration_freesurfer(info)
        pd.DataFrame(results).to_csv(f'{save_dir}/error.csv', index=False)
        copy2(f'{self.rootdir}/BC.csv', f'{os.path.join(os.path.dirname(self.rootdir), save_dir)}/BC.csv')
        copy2(f'{self.rootdir}/Benign.csv', f'{os.path.join(os.path.dirname(self.rootdir), save_dir)}/Benign.csv')


def convert(rootdir='CLIP_BC_sec1_DCM'):
    preprocess = Preprocess(rootdir, modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC'])
    preprocess.dcm_to_nii()


def save_data_info(rootdir='CLIP_BC_sec1_NII', filename='CLIP_sec1'):
    preprocess = Preprocess(rootdir, modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC'])
    preprocess.save_data_info('BC', filename)
    preprocess.save_data_info('Benign', filename)


def crop_box(rootdir='CLIP_BC_sec1_NII', save_dir='PreStudy'):
    preprocess = Preprocess(rootdir, modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX'])
    preprocess.box_crop(save_dir)


def crop_tic(rootdir='PreStudy'):
    preprocess = Preprocess(rootdir, modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC'])
    preprocess.crop_tic()


def to_datalist(rootdir='PreStudy'):
    preprocess = Preprocess(rootdir, modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC'])
    preprocess.to_datalist()


def generate_report(rootdir='PreStudy', reportdir='CLIP_BC_sec1_NII'):
    benign = pd.read_csv(f'{reportdir}/Benign.csv')
    bc = pd.read_csv(f'{reportdir}/BC.csv')
    data = pd.concat([benign, bc]).reset_index(drop=True)[['name_en', 'pathology', 'position',
                                                           '影像表现', '影像诊断', '病理结果']]
    os.makedirs(f'{rootdir}/CSV/report', exist_ok=True)
    data.to_csv(f'{rootdir}/CSV/report/report.csv', index=False)


def tic_label(rootdir='PreStudy', ):
    phase2onehot = {
        'Slow': [1, 0, 0],
        'Medium': [0, 1, 0],
        'Fast': [0, 0, 1],
        'Wash-out': [1, 0, 0],
        'Plateau': [0, 1, 0],
        'Persistent': [0, 0, 1]
    }
    data = pd.read_csv(f'{rootdir}/CSV/data_split/datalist_new.csv')
    if 'tic_onehot' in data:
        del data['tic_onehot']
    tic_df = pd.read_csv(f'{rootdir}/CSV/report/tic_gt.csv')
    tic_df['Initial enhancement phase'] = tic_df['Initial enhancement phase'].map(lambda x: phase2onehot[x])
    tic_df['Delayed phase'] = tic_df['Delayed phase'].map(lambda x: phase2onehot[x])
    tic_df['tic_onehot'] = tic_df['Initial enhancement phase'] + tic_df['Delayed phase']
    data = data.merge(tic_df[['pid', 'tic_onehot']], on='pid', how='left')
    data['path'] = data.apply(lambda x: {m: (p if m != 'TIC' else x['tic_onehot'])
                                         for m, p in literal_eval(x['path']).items()}, axis=1)
    data.to_csv(f'{rootdir}/CSV/data_split/datalist_new.csv', index=False)


def concept_label(rootdir='PreStudy'):
    from ast import literal_eval
    def dict_flatten(d, parent_key='', sep='.'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):  # 如果值是字典，递归展平
                items.update(dict_flatten(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    def bool_2_label(d):
        if isinstance(d, dict):
            return {k: bool_2_label(v) for k, v in d.items()}
        if isinstance(d, list):
            return [bool_2_label(l) for l in d]
        if isinstance(d, bool):
            return int(d)
        return d

    concept = pd.read_csv(f'{rootdir}/CSV/report/concepts.csv')
    concept['concept'] = concept['concept'].map(literal_eval).map(bool_2_label).map(dict_flatten)

    concept['concept_key'] = concept['concept'].map(lambda x: list(x.keys()))
    concept['concept_label'] = concept['concept'].map(lambda x: list(x.values()))
    data = pd.read_csv(f'{rootdir}/CSV/data_split/datalist.csv')
    if 'concept_key' in data.columns:
        del data['concept_key']
    if 'concept_label' in data.columns:
        del data['concept_label']

    data = data.merge(concept[['pid', 'concept_key', 'concept_label']], on='pid', how='left')
    data.to_csv(f'{rootdir}/CSV/data_split/datalist_new.csv', index=False)


############################################# Processing #############################################
def prestudy():
    convert('/data_smr/liuy/Project/CLIPCBM/dataset/CLIP_BC_sec1_DCM')
    save_data_info('/data_smr/liuy/Project/CLIPCBM/dataset/CLIP_BC_sec1_DCM', 'CLIP_sec1')
    # crop_breast('/data_smr/liuy/Project/CLIPCBM/dataset/CLIP_BC_sec1_NII', save_dir='PreStudy')
    # crop_tic('PreStudy')
    # to_datalist('PreStudy')
    # generate_report(rootdir='PreStudy', reportdir='CLIP_BC_sec2_NII')
    # concept_label(rootdir='PreStudy')
    # tic_label(rootdir='PreStudy')


def prestudy_box():
    # convert('/data_smr/liuy/Project/CLIPCBM/dataset/CLIP_BC_sec1_DCM')
    # save_data_info('/data_smr/liuy/Project/CLIPCBM/dataset/CLIP_BC_sec1_DCM', 'CLIP_sec1')
    Preprocess('CLIP_BC_sec1_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).breast_crop('PreStudyBreast')
    crop_box('/data_smr/liuy/Project/CLIPCBM/dataset/CLIP_BC_sec1_NII', save_dir='PreStudy')
    # crop_tic('PreStudy')
    # to_datalist('PreStudy')
    # generate_report(rootdir='PreStudy', reportdir='CLIP_BC_sec2_NII')
    # concept_label(rootdir='PreStudy')
    # tic_label(rootdir='PreStudy')


def prestudy2():
    convert('/Users/liuyang/Documents/CLIP_BC_sec2_DCM')
    save_data_info('/Users/liuyang/Documents/CLIP_BC_sec2_DCM', 'CLIP_sec2')
    Preprocess('CLIP_BC_sec2_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC']).breast_crop('PreStudy2')
    # crop_tic('PreStudy2')
    # to_datalist('PreStudy2')
    # generate_report(rootdir='PreStudy2', reportdir='CLIP_BC_sec2_NII')
    # concept_label(rootdir='PreStudy2')
    # tic_label(rootdir='PreStudy2')


def prestudy2_box():
    # convert('/Users/liuyang/Documents/CLIP_BC_sec2_DCM')
    # save_data_info('/Users/liuyang/Documents/CLIP_BC_sec2_DCM', 'CLIP_sec2')
    Preprocess('CLIP_BC_sec2_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).breast_crop('PreStudy2Breast')
    crop_box('/data_smr/liuy/Project/CLIPCBM/dataset/CLIP_BC_sec2_NII', save_dir='PreStudy2')
    # crop_tic('PreStudy2')
    # to_datalist('PreStudy2')
    # generate_report(rootdir='PreStudy2', reportdir='CLIP_BC_sec2_NII')
    # concept_label(rootdir='PreStudy2')
    # tic_label(rootdir='PreStudy2')


def prestudy_all():
    ps = pd.read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/PreStudy/CSV/data_split/datalist.csv')
    ps2 = pd.read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/PreStudy2/CSV/data_split/datalist.csv')
    ps_all = pd.concat([ps, ps2]).drop_duplicates(subset=('name', 'pathology'))
    ps_all.to_csv('/data_smr/liuy/Project/CLIPCBM/dataset/PreStudyALL/CSV/data_split/datalist.csv', index=False)
    concept = pd.read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/PreStudy/CSV/report/concepts.csv')
    concept2 = pd.read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/PreStudy2/CSV/report/concepts.csv')
    concept_all = pd.concat([concept, concept2]).drop_duplicates(subset=('name', 'pathology'))
    concept_all.to_csv('/data_smr/liuy/Project/CLIPCBM/dataset/PreStudyALL/CSV/concept/concepts.csv', index=False)


def prospective():
    # convert('/data_smr/liuy/Project/CLIPCBM/dataset/ProspectiveData_DCM')
    # save_data_info('/data_smr/liuy/Project/CLIPCBM/dataset/ProspectiveData_DCM', 'CLIP_prospective')
    # crop_breast('ProspectiveData_NII', save_dir='ProspectiveData')
    # crop_tic('ProspectiveData')
    # to_datalist('ProspectiveData')
    # generate_report(rootdir='ProspectiveData', reportdir='ProspectiveData_NII')
    concept_label(rootdir='ProspectiveData')
    tic_label(rootdir='ProspectiveData')


def external():
    # convert('/data_smr/liuy/Project/CLIPCBM/dataset/ExternalData_DCM')
    # save_data_info('/data_smr/liuy/Project/CLIPCBM/dataset/ExternalData_DCM', 'CLIP_external')
    # crop_breast('ExternalData_NII', save_dir='ExternalData')
    # crop_tic('ExternalData')
    # to_datalist('ExternalData')
    # generate_report(rootdir='ExternalData', reportdir='ExternalData_NII')
    concept_label(rootdir='ExternalData')
    tic_label(rootdir='ExternalData')


def get_SLN_mask():
    def read_csv(path, t):
        x = pd.read_csv(path)
        x['type'] = t
        return x

    sln = pd.concat([
        read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/raw_data/file/SLN_internal_correct.csv',
                 'internal'),
        read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_external/raw_data/file/SLN_external_correct.csv',
                 'internal')
    ])
    cbm = pd.concat([
        read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/PreStudyALL/CSV/data_split/datalist.csv', 'internal'),
        read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/ProspectiveData/CSV/data_split/datalist.csv', 'Prospective'),
        read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/ExternalData/CSV/data_split/datalist.csv', 'External'),
    ])
    pass


def merge_dataset(root_dir, d1, d2, target):
    def copy_2_t(d):
        for pathology in os.listdir(d):
            if not os.path.isdir(os.path.join(d, pathology)):
                continue
            os.makedirs(os.path.join(target, pathology), exist_ok=True)
            for patient in os.listdir(os.path.join(d, pathology)):
                src = os.path.join(d, pathology, patient)
                if os.path.isdir(src):
                    tp = os.path.join(target, pathology, patient)
                    if not os.path.exists(tp):
                        print(src, tp)
                        copytree(src, tp)
                else:
                    tp = os.path.join(target, pathology.replace('_BOX', ''),
                                      patient.replace('.nii.gz', '/BOX.nii.gz'))
                    if not os.path.exists(tp):
                        print(src, tp)
                        copy2(src, tp)

    target = os.path.join(root_dir, target)
    os.makedirs(target, exist_ok=True)

    if d1:
        d1 = os.path.join(root_dir, d1)
        copy_2_t(d1)
    if d2:
        d2 = os.path.join(root_dir, d2)
        copy_2_t(d2)
    if d1 and d2:
        files = [c for c in os.listdir(d1) if os.path.isfile(os.path.join(d1, c))]
        for f in files:
            if 'csv' in f:
                pd.concat([pd.read_csv(os.path.join(d1, f)), pd.read_csv(os.path.join(d2, f))], axis=0).to_csv(
                    os.path.join(target, f),
                    index=False
                )
            if 'xlsx' in f:
                pd.concat([pd.read_excel(os.path.join(d1, f)), pd.read_excel(os.path.join(d2, f))], axis=0).to_excel(
                    os.path.join(target, f),
                    index=False
                )


def internal_box(reg_method):
    Preprocess('InternalData_NII_TMP',
               modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).regularize('InternalData_NII')
    if reg_method == 'freesurfer':
        Preprocess('InternalData_NII',
                   modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).registration('InternalDataReg_NII')
        Preprocess('InternalDataReg_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).breast_crop(
            'InternalDataRegBreast')
        Preprocess('InternalDataReg_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).box_crop(
            'InternalDataReg')
    else:
        Preprocess('InternalData_NII',
                   modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).registration('InternalDataAnts_NII',
                                                                                       ants=True)
        Preprocess('InternalDataAnts_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).breast_crop(
            'InternalDataAntsBreast')
        Preprocess('InternalDataAnts_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).box_crop(
            'InternalDataAnts')
        # to_datalist('InternalData')
        #
        # datalist = pd.read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/InternalData/CSV/data_split/datalist.csv')[[
        #     'name', 'pid', 'pathology'
        # ]]
        # concept = pd.read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/Old_PreStudy/CSV/report/concepts2.csv')
        # concept2 = pd.read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/Old_PreStudy2/CSV/report/concepts.csv')
        # concept_all = pd.concat([concept, concept2]).drop_duplicates(subset=('name', 'pathology', 'position'))
        # concept_all['path'] = concept_all['path'].map(
        #     lambda x: x.replace('PreStudy2', 'InternalData').replace('PreStudy', 'InternalData'))
        #
        # concept_all = concept_all.drop(columns='pid').merge(datalist, on=['name', 'pathology'], how='left')
        # concept_all.to_csv('/data_smr/liuy/Project/CLIPCBM/dataset/InternalData/CSV/report/concepts.csv', index=False)
        # #
        # tic = pd.read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/Old_PreStudy/CSV/report/tic_gt.csv')
        # tic2 = pd.read_csv('/data_smr/liuy/Project/CLIPCBM/dataset/Old_PreStudy2/CSV/report/tic_gt.csv')
        # tic_all = pd.concat([tic, tic2]).drop_duplicates(subset=('name', 'pathology'))
        # tic_all['path'] = tic_all['path'].map(
        #     lambda x: x.replace('PreStudy2', 'InternalData').replace('PreStudy', 'InternalData'))
        # tic_all = tic_all.drop(columns='pid').merge(datalist, on=['name', 'pathology'], how='left')
        # tic_all.to_csv('/data_smr/liuy/Project/CLIPCBM/dataset/InternalData/CSV/report/tic_gt.csv', index=False)

        # crop_tic('PreStudy2')

        # concept_label(rootdir='InternalData')
        # tic_label(rootdir='InternalData')


def prospective_box(reg_method):
    Preprocess('ProspectiveData_NII_TMP',
               modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).regularize('ProspectiveData_NII')
    if reg_method == 'freesurfer':
        Preprocess('ProspectiveData_NII',
                   modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).registration('ProspectiveDataReg_NII')
        Preprocess('ProspectiveDataReg_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).breast_crop(
            'ProspectiveDataRegBreast')
        Preprocess('ProspectiveDataReg_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).box_crop(
            'ProspectiveDataReg')

    else:
        Preprocess('ProspectiveData_NII',
                   modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).registration('ProspectiveDataAnts_NII',
                                                                                       ants=True)
        Preprocess('ProspectiveDataAnts_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).breast_crop(
            'ProspectiveDataAntsBreast')
        Preprocess('ProspectiveDataAnts_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).box_crop(
            'ProspectiveDataAnts')

    # crop_box('ProspectiveDataReg_NII', save_dir='ProspectiveDataReg')

    # crop_tic('ProspectiveData')
    # to_datalist('ProspectiveData')
    # generate_report(rootdir='ProspectiveData', reportdir='ProspectiveData_NII')
    # concept_label(rootdir='ProspectiveData')
    # tic_label(rootdir='ProspectiveData')


def process_box(name, reg_method):
    Preprocess(f'{name}_NII_TMP',
               modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).regularize(f'{name}_NII')
    if reg_method == 'freesurfer':
        Preprocess(f'{name}_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).registration(f'{name}Reg_NII')
        Preprocess(f'{name}Reg_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).breast_crop(f'{name}RegBreast')
        Preprocess(f'{name}Reg_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).box_crop(f'{name}Reg')
    elif reg_method=='ants':
        Preprocess(f'{name}_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).registration(f'{name}Ants_NII', ants=True)
        Preprocess(f'{name}Ants_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).breast_crop(f'{name}AntsBreast')
        Preprocess(f'{name}Ants_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).box_crop(f'{name}Ants')
    else:
        Preprocess(f'{name}_NII', modalities=['ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'BOX']).breast_crop(f'{name}Breast')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reg", type=str, default='ants')
    args = parser.parse_args()
    print(args)
    process_box(name='InternalData', reg_method=args.reg)
    process_box(name='ProspectiveData', reg_method=args.reg)
    process_box(name='ExternalData', reg_method=args.reg)
    # prestudy_box()
    # prestudy2_box()
    # prestudy2()
    # prospective()
    # external()
    # get_SLN_mask()
    # merge_dataset('/data_smr/liuy/Project/CLIPCBM/dataset', 'CLIP_BC_sec1_NII', None,
    #               'CLIP_BC_sec1_NII')

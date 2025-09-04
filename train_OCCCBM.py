import os

import numpy as np
import nibabel as nib
import torch

from dataset import DataModule
from config.config import Config
from models import get_model_from_config
from utils.trainHelper import TrainHelper as _TrainHelper
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from chatGPT import ChatGPT, Prompt


class TrainHelper(_TrainHelper):
    def set_datamodule(self):
        self.set_encoder()
        self.datamodule = DataModule(self.config, encoder=self.encoder if self.config.pre_encode_to_feature else None)

    def set_model(self):
        self.model = get_model_from_config(self.config, encoder=self.encoder)

    def set_encoder(self):
        self.encoder = self.load_from_configfile(output_dir=f'{self.config.encode_dir}/fold_{self.config.k}',
                                                 device=self.config.device,
                                                 avg_pooling=self.config.avg_pooling)

    def config_callbacks(self):
        callbacks = [RichProgressBar(), LearningRateMonitor(logging_interval='epoch')]
        if self.config.iter_train:
            modalities = {*self.config.modality_to_model[self.config.modality], self.config.modality}
        else:
            modalities = {self.config.modality}
        for modality in modalities:
            callbacks.append(ModelCheckpoint(
                dirpath=self.checkpoint_dir,
                filename='{epoch}-{step}-{' + f'val_{modality}_{self.monitor}' + ':.4f}',
                monitor=f'val_{modality}_{self.monitor}',
                mode='max',
                save_top_k=3,
                every_n_epochs=1))
            callbacks.append(ModelCheckpoint(
                dirpath=self.checkpoint_dir,
                filename='{epoch}-{step}-{val_mse_loss:.4f}',
                monitor=f'val_mse_loss',
                mode='min',
                save_top_k=1,
                every_n_epochs=1))
        return callbacks

    @torch.no_grad()
    def infer(self, path, modality=None, output_dir=None):
        inp = {k: d.unsqueeze(0) if isinstance(d, torch.Tensor) else torch.FloatTensor(d).unsqueeze(0) for k, d in
               self.transform(path).items()}
        output_dir = self.config.output_dir if output_dir is None else output_dir
        self.load_from_configfile(output_dir=output_dir, model=self.model, postfix='val_mse_loss')
        inp_map = self.model.encode_image(inp, fusion=False, grad=False, avg_pooling=False)
        feature_map, occ_map = self.model.compute_occ(inp)
        # occ map 1 C 16 16 16
        concept_score = self.model.compute_concept_score(feature_map, occ_map)
        logits = self.model.classifier(self.model.scale * concept_score)
        os.makedirs(os.path.join(output_dir, 'OCCMAP', 'MAP'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'OCCMAP', 'Feature'), exist_ok=True)
        size = (128, 128, 128)
        for modality in occ_map.keys():
            if 'TIC' in modality:
                continue
            occ = torch.nn.functional.interpolate(occ_map[modality].cpu(), size=size, mode='nearest').numpy()[0]
            data = ((inp[modality] - inp[modality].min()) / (inp[modality].max() - inp[modality].min())).numpy()[0]
            i_map = torch.nn.functional.interpolate(inp_map[modality].cpu(), size=size, mode='nearest').numpy()[0]
            C = occ.shape[0]  # Number of channels in occ_map
            for c in range(C):
                occ_img = nib.Nifti1Image(occ[c], np.eye(4))  # c-th channel of occ_map
                nib.save(occ_img, os.path.join(output_dir, 'OCCMAP', 'MAP', f"{modality}_map_channel_{c:03d}.nii.gz"))
            for i in range(i_map.shape[0]):
                img = nib.Nifti1Image(i_map[i], np.eye(4))
                nib.save(img, os.path.join(output_dir, 'OCCMAP', 'Feature', f"{modality}_map_channel_{i:03d}.nii.gz"))
            # Remove the batch dimension (1) and save it as 3D NIfTI
            data_img = nib.Nifti1Image(data[0], np.eye(4))  # Remove batch dimension for single image
            nib.save(data_img, os.path.join(output_dir, 'OCCMAP', f"{modality}_data.nii.gz"))
        return logits


if __name__ == '__main__':
    config = Config.config(config='config/PreStudy/CBM/occcbm.py')
    helper = TrainHelper(config=config)
    helper.run()
    # helper.infer({'ADC': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/ADC.nii.gz',
    #               'CE': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/CE.nii.gz',
    #               'DWI': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/DWI.nii.gz',
    #               'T2WI': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/T2WI.nii.gz',
    #               'TIC': [0, 0, 1, 1, 0, 0]},
    #              'MM',
    #              'result/PreStudy_Zeros_L1_ValidOnly/OccCBM_LR0.0001_fullshot_ActSigmoid/fold_0')

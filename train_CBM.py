import os
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
        self.datamodule = DataModule(self.config,
                                     encoder=self.encoder if self.config.pre_encode_to_feature else None)

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
                save_top_k=1,
                every_n_epochs=1))
        return callbacks


if __name__ == '__main__':
    config = Config.config(#config='config/PreStudyALL/CBM/mmcbm.py',
                           #encode_dir='result/PreStudyALL_Zeros_TestOnly/efficientnet-b0_pool_iterTrain_LR0.0001_fullshot',
                           #postfix='ProspectiveData',
                           #test_csv='dataset/ProspectiveData/CSV/data_split/datalist.csv',
                           #test_only=True
                           )
    # config.img_size = 256
    # config.device = 'cpu'
    print(config)
    helper = TrainHelper(config=config)
    helper.run()
    helper.test()
    # helper.gradcam({'ADC': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/ADC.nii.gz',
    #                 'CE': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/CE.nii.gz',
    #                 'DWI': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/DWI.nii.gz',
    #                 'T2WI': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/T2WI.nii.gz',
    #                 'TIC': [0, 0, 1, 1, 0, 0]},
    #                'MM')

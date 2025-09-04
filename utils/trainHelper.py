import glob
import os

import pandas as pd
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import DataModule
from config.config import Config
from models import get_model_from_config
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from .logger import Logger

torch.set_float32_matmul_precision('high')


class TrainHelper:
    def __init__(self, config=None, monitor='f1'):
        self.monitor = monitor
        self.config = Config.config() if config is None else config
        Config.save_config(self.config)
        self.logger = Logger(log_file=f'{self.config.output_dir}/trainHelper.log')
        self.config.logger = self.logger
        self.checkpoint_dir = os.path.join(self.config.output_dir, 'checkpoints')

        self.set_datamodule()
        self.set_model()
        self.set_transform()

    def set_datamodule(self):
        self.datamodule = DataModule(self.config)

    def set_model(self):
        self.model = get_model_from_config(self.config)

    def set_transform(self):
        self.transform = self.datamodule.transform['test']

    def run(self):
        if not self.config.test:
            self.train()
        self.test()

    def checkpoint_path(self, checkpoint_dir=None, last=False, modality='*', postfix=''):
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        if postfix:
            checkpoints = glob.glob(os.path.join(checkpoint_dir, f'*{postfix}*.ckpt'))
        else:
            checkpoints = glob.glob(os.path.join(checkpoint_dir, f'*val_{modality}_{self.monitor}*.ckpt'))
        if not checkpoints:
            return None

        def get_val_acc(ckpt):
            filename = os.path.basename(ckpt)
            if postfix:
                val_acc = filename.split(f'{postfix}=')[-1].split('.ckpt')[0].split('-')[0]
            else:
                val_acc = filename.split(f'val_MM_{self.monitor}=')[-1].split('.ckpt')[0].split('-')[0]
            return float(val_acc)

        def get_last_epoch(ckpt):
            filename = os.path.basename(ckpt)
            epoch = filename.split('-')[0].split('=')[-1]
            return int(epoch)

        checkpoint_path = max(checkpoints, key=get_last_epoch if last else get_val_acc)
        print(f'Checkpoint path: {checkpoint_path}')
        return checkpoint_path

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

    def train(self):
        self.logger.info(f'[TrainHelper:train] Starting training')
        seed_everything(42)
        checkpoint_path = self.checkpoint_path(last=True)
        if checkpoint_path is not None:
            self.logger.info(f'[TrainHelper:train] Loading model from {checkpoint_path}')

        trainer = Trainer(devices=[self.config.device],
                          callbacks=self.config_callbacks(),
                          logger=TensorBoardLogger(
                              name=f'{self.config.n_shot}shot_train',
                              save_dir=self.config.output_dir,
                          ),
                          check_val_every_n_epoch=1,
                          default_root_dir=self.config.output_dir,
                          max_epochs=self.config.epochs,
                          )
        trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=checkpoint_path)

    def test(self):
        self.logger.info(f'[TrainHelper:test] Starting testing')
        seed_everything(42)  # seed matches first run of linear probe
        checkpoint_path = self.checkpoint_path(modality='MM')
        self.logger.info(f'[TrainHelper:test] Loading model from {checkpoint_path}')
        trainer = Trainer(devices=[self.config.device],
                          callbacks=[RichProgressBar()],
                          default_root_dir=self.config.output_dir, )
        trainer.validate(self.model, datamodule=self.datamodule, ckpt_path=checkpoint_path)
        trainer.test(self.model, datamodule=self.datamodule, ckpt_path=checkpoint_path)
        return

    def load_from_configfile(self, output_dir, model=None, device=None, encoder=None, postfix='', **kwargs):
        device = torch.device(device if device is not None else self.config.device)
        config = Config.load_config(output_dir)
        config.logger = None
        for k, v in kwargs.items():
            setattr(config, k, v)
        if model is None:
            model = get_model_from_config(config=config, encoder=encoder)
        model.load_state_dict(
            torch.load(
                self.checkpoint_path(checkpoint_dir=os.path.join(config.output_dir, 'checkpoints'),
                                     modality='MM', postfix=postfix),
                map_location=device,
                weights_only=False
            )['state_dict']
        )
        model.eval()
        model.to(device)
        return model

    def infer(self, path, modality=None):
        path = self.transform(path)
        return self.model(path, modality=modality)

    @staticmethod
    def print_logs(output_dir):
        import re
        metrics_list = []
        for subdir in os.listdir(output_dir):
            with open(f'{output_dir}/{subdir}/trainHelper.log', 'r') as f:
                log_data = f.readlines()[-12:]
            for log_str in log_data:
                metrics_str = re.search(r"Metrics: ({.*})", log_str).group(1)
                metrics_str = re.sub(r"(metatensor|tensor)\(([\d\.]+)(?:, device='[^']*')?\)", r"\2", metrics_str)
                metrics = eval(metrics_str)  # 将字符串转为字典
                modality = list(metrics.keys())[0].split('_')[1]
                stage = list(metrics.keys())[0].split('_')[0]
                metrics = {k.split('_')[-1]: v for k, v in metrics.items()}
                metrics['stage'] = stage
                metrics['modality'] = modality
                metrics['fold'] = subdir
                metrics_list.append(metrics)

        # 查看结果字典
        pd.DataFrame(metrics_list).to_csv(f'{output_dir}/allmetrics.csv', index=False)


if __name__ == '__main__':
    helper = TrainHelper()
    helper.run()

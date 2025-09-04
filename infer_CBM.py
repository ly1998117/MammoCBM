import os
import torch

from dataset import DataModule
from config.config import Config
from models import get_model_from_config
from utils.trainHelper import TrainHelper as _TrainHelper
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from chatGPT import ChatGPT, Prompt
from train_CBM import TrainHelper as _TrainHelper
from reportProcess import ReportProcess

class TrainHelper(_TrainHelper):
    def set_datamodule(self):
        self.set_encoder()
        self.datamodule = DataModule(self.config, encoder=None)
        
    def generate_report(self, data, modality=None):
        data = {k: v.unsqueeze(0) for k,v in data.items()}
        data = self.encoder.encode_image(data, modality=modality)
        logit, *_ = self.model(data, modality=modality)
        logit = logit.flatten()
        logit = torch.softmax(logit, dim=-1)[1].item()
        birads = self.model.bi_rads(data, modality, 0.5)
        return logit, birads
        
    def infer_from_path(self, path, modality=None):
        import time
        path = {k: d.unsqueeze(0) if isinstance(d, torch.Tensor) else torch.FloatTensor(d).unsqueeze(0) for k, d in
                self.transform(path).items()}

    @torch.no_grad()
    def infer_from_module(self, dirpath, modality=None, stage='test'):
        dataset = self.datamodule.get_datasets(stage)[0]
        self.load_from_configfile(output_dir=self.config.output_dir, model=self.model)
        logits, birads, pathes, diagnoses = [], [], [], []
        for ind in range(len(dataset)):
            data = dataset.get_preprocessed_img(ind)
            logit, birad = self.generate_report(data['data'], modality)
            logits.append(logit)
            birads.append(birad)
            pathes.append(os.path.join(dirpath, f"{data['pathology']}-{data['name']}.md"))
            diagnoses.append(['Benign', 'BC'][0 if logit<0.5 else 1])
        logits = torch.tensor(logits)
        logits[logits<0.5] = 0.02*(logits[logits<0.5] - logits[logits<0.5].min()) / (logits[logits<0.5].max()-logits[logits<0.5].min())
        logits[logits>0.5] = 0.02 + 0.98 * (logits[logits>0.5] - logits[logits>0.5].min()) / (logits[logits>0.5].max()-logits[logits>0.5].min())
        print(logits)
        for logit, birad, filepath, diagnose in zip(logits, birads, pathes, diagnoses):
            rp = ReportProcess(
                data_dir='ProspectiveData',
                api_base='https://c-z0-api-01.hash070.com/v1',
                api_key='sk-E3LDC8hj6e07fd77986aT3BLbKFJCd97ac6fA09E452794cc',
                model='gpt-5-chat-latest'
            )
            rp.structure_to_report(birad, logit, filepath, diagnose)
            

    def gradcam(self, path, modality=None):
        from utils.grad_cam import GradCAM
        path = {k: d.unsqueeze(0) if isinstance(d, torch.Tensor) else torch.FloatTensor(d).unsqueeze(0) for k, d in
                self.transform(path).items()}
        self.load_from_configfile(output_dir=self.config.output_dir, model=self.model)

        class Module(torch.nn.Module):
            def __init__(self, encoder, predictor):
                super().__init__()
                self.encoder = encoder
                self.predictor = predictor

            def forward(self, x):
                x = self.encoder.encode_image(x, modality=modality, fusion=True)[modality]
                return self.predictor(x)

        def getcam(modality, target_layers, class_idx=None):
            cam = GradCAM(nn_module=Module(self.encoder, self.model.predictor),
                          target_layers=target_layers)
            result = cam(x=path, modality=modality, class_idx=class_idx)
            return result

        x = getcam('DWI', "encoder.encoder.encoder.DWI._bn1", class_idx=1)
        y = getcam('DWI', "encoder.encoder.encoder.DWI._bn1", class_idx=10)
        pass


if __name__ == '__main__':
    config = Config.config(config='config/InternalData/CBM/mmcbm.py',
                           encode_dir='result/InternalData_Zeros_TestOnly_TestCSVdataset/efficientnet-b0_spool_iterTrain_LR0.0001_fullshot',
                           postfix='ProspectiveData',
                           test_csv='dataset/ProspectiveData/CSV/data_split/datalist.csv',
                           test_only=True,
                           cache=False,
                           device='cpu')
    # config.img_size = 256
    # config.device = 'cpu'
    print(config)
    helper = TrainHelper(config=config)
    helper.infer_from_module('dataset/ProspectiveData/GeneratedReport', 'MM')
    # helper.gradcam({'ADC': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/ADC.nii.gz',
    #                 'CE': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/CE.nii.gz',
    #                 'DWI': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/DWI.nii.gz',
    #                 'T2WI': 'PreStudy/BC/LONGSHIQUN_42Y_0015431853/T2WI.nii.gz',
    #                 'TIC': [0, 0, 1, 1, 0, 0]},
    #                'MM')

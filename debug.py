import torch
from lightning import seed_everything
from dataset import DataModule
from dataset.dataloader import ConceptDataset, Transform
from config.config import Config
from models import get_model_from_config, MMAttnSCLSEfficientNet
from models.cbank.conceptLearner import ConceptsLearner
from tqdm import tqdm


def cache():
    config = Config.config()
    config.logger = None
    config.num_worker = 0
    config.use_redis = False
    config.output_dir = 'result/PreStudy_Zeros_ValidOnly/efficientnet-b0_pool-LR0.0001-fullshot/fold_0'
    config.modality_to_model['MM'] = ('ADC', 'CE', 'DWI', 'T2WI')
    config.fusion_method = 'pool'
    model = get_model_from_config(config)
    MMAttnSCLSEfficientNet.load_from_checkpoint(
        'result/PreStudy_Zeros_ValidOnly/efficientnet-b0_pool-LR0.0001-fullshot/fold_0/checkpoints/'
        'epoch=122-step=9840-val_MM_f1=0.9091.ckpt'
    )
    # model.load_state_dict(
    #     torch.load('result/PreStudy_Zeros_ValidOnly/efficientnet-b0_pool-LR0.0001-fullshot/fold_0/checkpoints/'
    #                'epoch=122-step=9840-val_MM_f1=0.9091.ckpt', map_location='cpu')['state_dict'])
    model.to(6)
    datamodule = DataModule(config, encoder=model)
    datamodule.prepare_data()
    datamodule.setup(None)
    train_loader = datamodule.train_dataloader()


def concept():
    config = Config.config()
    config.output_dir = 'result/PreStudy_Zeros_ValidOnly/efficientnet-b0_pool-LR0.0001-fullshot/fold_0'
    config.fusion_method = 'pool'
    model = get_model_from_config(config)
    model.load_state_dict(
        torch.load(
            'result/PreStudy_Zeros_ValidOnly/efficientnet-b0_pool-LR0.0001-fullshot/fold_0/checkpoints/epoch=116-step=9477-val_MM_f1=0.8675.ckpt',
            map_location='cpu')['state_dict'])
    model.to(1)
    dataset = ConceptDataset(encoder=model,
                             data_path=f'dataset/{config.dataset}',
                             transform=Transform(root_dir='./dataset',
                                                 normalize=config.z_normalize,
                                                 img_size=config.img_size,
                                                 spacing=config.spacing).test_transforms(),
                             pathology_labels=config.pathology_labels)
    ConceptsLearner()


def dataset_debug():
    config = Config.config(config='config/PreStudyALL/BlackBox/efficient.py')
    config.output_dir = 'result/PreStudyALL_Zeros/efficientnet-b0_pool-LR0.0001-fullshot/fold_0'
    config.fusion_method = 'pool'
    datamodule = DataModule(config, test_csv='dataset/ProspectiveData/CSV/data_split/datalist.csv')
    datamodule.prepare_data()
    datamodule.setup(None)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    # for data in tqdm(val_loader):
    #     pass
    for data in tqdm(test_loader):
        pass


def redis():
    config = Config.config()
    config.fusion_method = 'pool'
    config.use_redis = False
    datamodule = DataModule(config)
    datamodule.prepare_data()
    datamodule.setup(None)
    train_loader = datamodule.train_dataloader()
    for d in train_loader:
        print(d['data']['TIC'].shape)
        break


def model():
    from models.backbone.networks.efficientnet import EfficientNet
    print(EfficientNet(
        model_name='efficientnet-b0',
        spatial_dims=3,
        in_channels=1,
        depth_divisor=8,
        modality=None,
        encoders=None,
        pretrained=False,
        avg_pooling=False,
        blocks_args_str=[
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s11_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s11_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ],
        efficientnet_params={
            "efficientnet-b0": (1.0, 1.0, 224, 0.2, 0.2),
        }
    ))
    print(EfficientNet(
        model_name='efficientnet-b0',
        spatial_dims=3,
        in_channels=1,
        depth_divisor=8,
        modality=None,
        encoders=None,
        pretrained=False,
        avg_pooling=False,
        blocks_args_str=[
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s11_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s11_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ],
        efficientnet_params={
            "efficientnet-b0": (1.0, 1.0, 512, 0.2, 0.2),
        }
    ))


def printlog():
    from utils.trainHelper import TrainHelper
    TrainHelper.print_logs(
        'result/PreStudyALL_Zeros_L1/mm2cbm_iterTrain_LR5e-05_fullshot/Efficientnet-b0_rnpool_itertrain_lr0.0001_fullshot')
        # 'result/PreStudyALL_Zeros/efficientnet-b0_pool_iterTrain_LR0.0001_fullshot')


if __name__ == '__main__':
    dataset_debug()
    # printlog()

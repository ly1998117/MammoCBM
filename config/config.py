import argparse
import os

from mmengine import DictAction
from mmengine.config import Config as BaseConfig


def get_args(config='config/PreStudy/BlackBox/efficient.py'):
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    ######################################### dataset params #########################################
    parser.add_argument('--config', default=config, help='path to config file')
    parser.add_argument("--modality", default='MM', type=str, help="MRI contrast(default, normal)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--k", "-k", type=int, default=0)
    parser.add_argument("--crop_prob", type=float, default=0)
    parser.add_argument('--test', action='store_true', default=False)

    ######################################### train params #########################################
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--no_data_aug', action='store_true', default=False)
    parser.add_argument("--idx", default=None, type=int)
    parser.add_argument("--bidx", default=180, type=int)
    parser.add_argument('--ignore', action='store_true', default=False)
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--cache_data', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--plot_curve', action='store_true', default=False)
    parser.add_argument('--plot_after_train', action='store_true', default=False)
    parser.add_argument('--cudnn_nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')

    ######################################### cocnept params #########################################

    parser.add_argument('--bias', action='store_true', default=False)
    parser.add_argument('--weight_norm', action='store_true', default=False)
    parser.add_argument('--occ_act', default='abs', type=str, help='sigmoid, softmax')
    parser.add_argument('--options', nargs='+', action=DictAction,
                        help='overwrite parameters in cfg from commandline')
    args = parser.parse_args()
    return args


################################################### Configuration ##################################################
def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


class Config(BaseConfig):
    @staticmethod
    def config(config='config/PreStudy/BlackBox/efficient.py', **kwargs):
        args = get_args(config)
        config = Config.from_args(args, **kwargs)
        return config

    @staticmethod
    def from_args(args, generate_config=True, **kwargs):
        config = Config.fromfile(args.config)
        config.merge_from_dict(vars(args))
        for k, v in kwargs.items():
            setattr(config, k, v)
        if args.options is not None:
            print(args.options)
            for item in args.options:
                setattr(config, item, args.options[item])
        if generate_config:
            config = Config.generate_config(config)
        return config

    @staticmethod
    def load_config(output_dir):
        files = [name for name in os.listdir(output_dir) if '.py' in name]
        if len(files) == 0:
            raise FileNotFoundError(f"No config file found in {output_dir}")
        files = files[0]
        config = Config.fromfile(os.path.join(output_dir, files))
        config.output_dir = output_dir
        return config

    @staticmethod
    def save_config(config, force=False):
        path = os.path.join(config.output_dir, os.path.basename(config.config))
        if not os.path.exists(path) or force:
            mkdir_or_exist(config.output_dir)
            config.dump(file=path)

    @staticmethod
    def generate_config(config):
        data_dir = f'datasets/{config.dataset}'
        setattr(config, 'data_dir', data_dir)
        if hasattr(config, 'encoder_path'):
            setattr(config, 'encoder_path', os.path.join(config.encoder_path, f'fold_{config.k}'))
        # if config.output_dir.split('/').__len__() < 2:
        config.output_dir = Config.config_to_dir(config)
        return config

    @staticmethod
    def config_to_dir(config):
        output_dir = config.output_dir
        model_name = config.model_name.replace("/", "-")
        # dirname = '/'.join(output_dir.split('/')[:2])
        dirname = os.path.join(output_dir, f'{config.dataset}_{config.weight_init_method.capitalize()}')
        if config.lambda_l1 > 0:
            dirname += f'_L1'
        if config.use_normalize:
            dirname += '_Norm'
        if config.test_only:
            dirname += '_TestOnly'
        if config.valid_only:
            dirname += '_ValidOnly'
        if config.test_csv is not None:
            dirname += f'_TestCSV{config.test_csv.split("/")[1]}'
        exp_name = model_name
        if hasattr(config, 'fusion_method'):
            exp_name += f'_{config.fusion_method}'
        if hasattr(config, 'iter_train') and config.iter_train:
            exp_name += f'_iterTrain'
        exp_name = f'{exp_name}_LR{config.lr}_{config.n_shot}shot'
        if hasattr(config, 'lambda_transform') and config.lambda_transform > 0:
            exp_name += f'_Transform{config.lambda_transform}'
        if hasattr(config, 'lambda_occnorm') and config.lambda_occnorm > 0:
            exp_name += f'_OccNorm{config.lambda_occnorm}'
        if hasattr(config, 'map_activation'):
            exp_name += f'_Act{config.map_activation.capitalize()}'
        if config.crop_prob != 0:
            exp_name += f'_CropP{config.crop_prob}'
        if config.postfix != '':
            exp_name += f'_{config.postfix}'
        if hasattr(config, 'encode_dir'):
            exp_name += f'/{os.path.basename(config.encode_dir).capitalize()}'
        if hasattr(config, 'iter_train') and config.iter_train:
            output_dir = os.path.join(dirname, exp_name, f'fold_{config.k}')
        else:
            output_dir = os.path.join(dirname, exp_name, f'{config.modality}_fold_{config.k}')
        return output_dir

    @staticmethod
    def dir_to_config(output_dir):
        config = Config(dict(output_dir=output_dir))

        # Parse the dataset and weight_init_method
        parts = output_dir.split('/')
        dataset_weight = parts[2].split('_')
        config.dataset = dataset_weight[0]
        config.weight_init_method = dataset_weight[1].lower()
        if 'L1' in parts[2]:
            config.lambda_l1 = 0.001
        if 'NoNorm' in parts[2]:
            config.use_normalize = False
        # Extract static and dynamic concepts from the bottleneck part
        bottleneck_info = parts[3].split('=')[1].split('-')[0]
        lr = parts[3].split(bottleneck_info + '-')[-1]
        config.lr = float(lr[2:])
        static_dynamic = bottleneck_info.split('+')
        config.num_static_concept = int(static_dynamic[0][1:])
        config.num_dynamic_concept = int(static_dynamic[1][1:])

        # Parse n_shots and clip_model
        shot_info, clip_model = parts[4].split('_')[:2]
        config.n_shots = int(shot_info[:-4]) if 'all' not in shot_info else 'all'
        config.clip_model = 'ViT-' + '/'.join(clip_model.split('-')[1:])
        return config

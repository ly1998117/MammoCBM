import os

from config import Config
from utils.multi_run import MultiRunner, UpdatableCommand


class GridSearch:
    def __init__(self, devices=(0, 1, 2, 3, 4, 5, 6, 7), n_jobs_per_gpu=3, tail_command='',
                 script='train_efficientNet.py', **kwargs):
        if isinstance(devices, str) and ',' in devices:
            devices = [int(d) for d in devices.split(',')]
        else:
            devices = [devices]
        self.devices = devices
        self.n_jobs_per_gpu = n_jobs_per_gpu
        self.tail_command = tail_command
        self.commands = [UpdatableCommand(script=script, **kwargs)]
        self.searched = False

    def __str__(self):
        return '\n'.join([str(c) for c in self.commands])

    def update(self, **kwargs):
        commands = []
        for command in self.commands:
            command.update(**kwargs)
            commands.append(command)
        self.commands = commands

    def set_script(self, script):
        commands = []
        for command in self.commands:
            command.update(script=script)
            commands.append(command)
        self.commands = commands

    def run(self):
        if not self.searched:
            raise ValueError('You must search before running')

        self.commands = [c() for c in self.commands]
        print(self.commands)
        runner = MultiRunner(self.commands, devices=self.devices, n_workers_per_device=self.n_jobs_per_gpu,
                             tail_command=self.tail_command)
        runner.run()

    def save(self, path, n_files=1):
        os.makedirs('commands', exist_ok=True)
        path = os.path.join('commands', path)
        n = len(self.commands) // n_files
        for i in range(n_files):
            with open(f'{path}_{i}.sh', 'w') as f:
                for command in self.commands[i * n: (i + 1) * n]:
                    f.write(command() + '\n')

    def load(self, path):
        path = os.path.join('commands', path)
        with open(path, 'r') as f:
            commands = [command.replace('\n', '') for command in f.readlines()]
        runner = MultiRunner(commands, devices=self.devices, n_workers_per_device=self.n_jobs_per_gpu,
                             tail_command=self.tail_command)
        runner.run()

    def search_n_fold(self, folds=(0, 1, 2, 3, 4)):
        self.searched = True
        commands = []
        for command in self.commands:
            for n in folds:
                commands.append(UpdatableCommand(command=command, k=n))
        self.commands = commands

    def search_parameters(self, transform=None, occnorm=None):
        if transform is None:
            transform = [1e-1, 1e-2, 1e-3]
        if occnorm is None:
            occnorm = [1e-1, 1e-2, 1e-3]
        self.searched = True
        commands = []
        for command in self.commands:
            commands.append(UpdatableCommand(command=command, lambda_transform=0, lambda_occnorm=0))
            for trans in transform:
                for norm in occnorm:
                    commands.append(UpdatableCommand(command=command, lambda_transform=trans, lambda_occnorm=norm))
        self.commands = commands

    def search_norm(self, norms=(True, False)):
        self.searched = True
        commands = []
        for command in self.commands:
            for norm in norms:
                commands.append(UpdatableCommand(command=command, use_normalize=norm))
        self.commands = commands

    def search_shot(self, shots=(1, 2, 4, 8, 16, 'all')):
        self.searched = True
        commands = []
        for command in self.commands:
            for shot in shots:
                commands.append(UpdatableCommand(command=command, n_shots=shot))
        self.commands = commands

    def search_dataset(self, datasets=('CUB', 'CIFAR10')):
        if isinstance(datasets, str):
            if ',' in datasets:
                datasets = [d.strip() for d in datasets.split(',')]
            else:
                datasets = [datasets]
        self.searched = True
        commands = []
        for command in self.commands:
            for dataset in datasets:
                commands.append(UpdatableCommand(command=command, dataset=dataset))
        self.commands = commands

    def search_select_fn(self, select_fns=('submodular', 'random')):
        self.searched = True
        commands = []
        for command in self.commands:
            for select_fn in select_fns:
                commands.append(UpdatableCommand(command=command, concept_select_fn=select_fn))
        self.commands = commands

    def search_lr(self, lrs=(1e-4, 5e-5)):
        self.searched = True
        commands = []
        for command in self.commands:
            for lr in lrs:
                commands.append(UpdatableCommand(command=command, lr=lr))
        self.commands = commands

    def search_scale(self, scales=(.1, 1)):
        self.searched = True
        commands = []
        for command in self.commands:
            for scale in scales:
                commands.append(UpdatableCommand(command=command, scale=scale))
        self.commands = commands

    def search_neck_size(self, ratios=(0, 0.5, 1), pre_class_list=None):
        if isinstance(ratios, str) and ',' in ratios:
            ratios = [float(r) for r in ratios.split(',')]
        elif not isinstance(ratios, list):
            ratios = [float(ratios)]

        self.searched = True
        commands = []
        for command in self.commands:
            if pre_class_list is None:
                max_per_class = Config.fromfile(command.config_to_str())['num_concept_per_class']
                pre_class_list = range(10, max_per_class + 1, 10)
            for per_class in pre_class_list:
                for ratio in ratios:
                    commands.append(UpdatableCommand(command=command,
                                                     num_concept_per_class=per_class,
                                                     dynamic_concept_ratio=ratio))
        self.commands = commands

    def test(self, dirpath):
        self.searched = True
        commands = []
        for exp_root in os.listdir(dirpath):
            exp_root = os.path.join(dirpath, exp_root)
            config = Config.root_to_config(exp_root)
            commands.append(UpdatableCommand(script='trainLinear.py', **config))
        self.commands = commands


def search_fold(grid):
    grid.search_n_fold()
    return grid


def search_attn_fold(grid):
    grid.update(fusion_method='pool')
    grid.search_n_fold()
    return grid


def search_param_loss(grid):
    grid.search_parameters()
    return grid


if __name__ == '__main__':
    import argparse
    from mmengine import DictAction

    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model', type=str, default='ViT-L/14')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--lr', action='store_true', default=False)
    parser.add_argument('--print', action='store_true', default=False)
    parser.add_argument('--script', type=str, default='train_efficientNet.py')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--folds', type=str, default=None)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--n_jobs', type=int, default=3)
    parser.add_argument('--n_files', type=int, default=1)
    parser.add_argument('--func', type=str, default=None)
    parser.add_argument('--options', nargs='+', action=DictAction, default={})
    args = parser.parse_args()
    tail_command = ''
    if args.test:
        tail_command = '--test'
    kwargs = dict(script=args.script, devices=args.device, n_jobs_per_gpu=args.n_jobs, tail_command=tail_command,
                  **args.options)
    grid = GridSearch(**kwargs)
    if isinstance(args.folds, str):
        args.folds = [int(d) for d in args.folds.split(',')]
        grid.search_n_fold(args.folds)

    if args.func is not None:
        grid = eval(args.func)(grid)

    if args.lr:
        grid.search_lr()
    if args.save:
        grid.save(args.save, n_files=args.n_files)
        exit()
    if args.load:
        grid.load(args.load)
        exit()
    if args.print:
        print(grid)
        exit()
    grid.run()

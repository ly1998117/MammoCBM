import torch
import torch.nn as nn
import lightning as L
import torchmetrics
import pandas as pd
from collections import defaultdict


def freeze(net):
    if not isinstance(net, (list, tuple)):
        net = [net]
    for n in net:
        for param in n.parameters():
            param.requires_grad = False


def unfreeze(net):
    if not isinstance(net, (list, tuple)):
        net = [net]
    for n in net:
        for param in n.parameters():
            param.requires_grad = True


def l1_norm(weight):
    row_l1_norm = torch.linalg.vector_norm(weight, ord=1, dim=-1).mean()
    return row_l1_norm


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class MultiModalModule(L.LightningModule):
    """
    Base class for MultiModal Model, which can handle different modalities.
    """

    def __init__(self, config, label_key='label'):
        super().__init__()
        self.label_key = label_key
        self.config = config
        self.model_name = config.model_name
        self.modalities = config.modality_to_model[config.modality]
        self.iter_train = config.iter_train
        self.loss = Dict(self.configure_loss())
        # self.save_hyperparameters(ignore=['logger', 'predictor'])
        self.filelogger = config.logger
        self.automatic_optimization = False
        self.current_stage = None
        self.train_metrics = self.configure_metric(prefix='train_', num_classes=config.num_class)
        self.val_metrics = self.configure_metric(prefix='val_', num_classes=config.num_class)
        self.test_metrics = self.configure_metric(prefix='test_', num_classes=config.num_class)
        if self.filelogger is not None:
            self.filelogger.info(f"initialized {self.__class__.__name__} ModelName: {self.model_name} "
                                 f"Modality: {self.modalities}[{config.modality}]")

    def configure_metric(self, prefix, num_classes):
        task = 'binary' if num_classes == 2 else 'multiclass'
        metric_d = nn.ModuleDict()
        for modality in [*self.modalities, 'MM']:
            metrics = torchmetrics.MetricCollection({
                f'{modality}_acc': torchmetrics.Accuracy(task=task, num_classes=num_classes),
                f'{modality}_auc': torchmetrics.AUROC(task=task, num_classes=num_classes),
                f'{modality}_recall': torchmetrics.Recall(task=task, num_classes=num_classes),
                f'{modality}_precision': torchmetrics.Precision(task=task, num_classes=num_classes),
                f'{modality}_f1': torchmetrics.F1Score(task=task, num_classes=num_classes)
            }, prefix=prefix)
            metric_d.update({modality: metrics})
        return metric_d

    def configure_loss(self) -> dict:
        return {
            'cls_loss': nn.CrossEntropyLoss(),
            'l1_norm': l1_norm
        }

    def compute_loss(self, logits, label, concept_label, *output):
        loss = self.loss.cls_loss(logits, label)
        return loss

    def configure_optimizers(self):
        op = torch.optim.Adam(params=self.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=op,
                                                      lr_lambda=lambda epoch: (1 - epoch / self.config.epochs) ** 0.9)
        return {'optimizer': op, 'lr_scheduler': scheduler}

    @property
    def name(self):
        return self.model_name

    def forward(self, inp, modality, *args, **kwargs):
        raise NotImplementedError

    def logits_to_prob(self, logits):
        return logits.detach().softmax(dim=-1)[:, -1]

    def label_to_target(self, label):
        return label

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers()
        data = train_batch['data']
        label = train_batch[self.label_key]
        modality = train_batch['modality'][0]
        concept_label = train_batch['concept_label'] if 'concept_label' in train_batch else None
        if modality == 'MM':
            iter_modality = [*self.modalities, modality]
        else:
            iter_modality = [modality]
        for m in iter_modality:
            if not self.iter_train and m != 'MM':
                continue
            logits = self.forward(inp=data, modality=m, train_classifier=m == 'MM')
            if isinstance(logits, (list, tuple)):
                logits, *output = logits
            else:
                output = []
            self.train_metrics[m].update(self.logits_to_prob(logits), self.label_to_target(label))
            loss = self.compute_loss(logits, label, concept_label, *output)
            self.manual_backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            self.log(f'train_{m}_loss', loss.detach())

    def on_train_epoch_start(self) -> None:
        optimizer = self.optimizers()
        self.current_stage = 'train'
        if self.filelogger is not None:
            self.filelogger.info(f'Epoch: {self.current_epoch} LR = {optimizer.param_groups[0]["lr"]}')

    def on_train_epoch_end(self) -> None:
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        for m, metric_collection in self.train_metrics.items():
            try:
                result = metric_collection.compute()
            except ValueError:
                continue
            self.log_dict(result, on_epoch=True, prog_bar=True)
            metric_collection.reset()

    def validation_step(self, batch, batch_idx):
        data = batch['data']
        label = batch[self.label_key]
        modality = batch['modality'][0]
        concept_label = batch['concept_label'] if 'concept_label' in batch else None
        if modality == 'MM':
            iter_modality = [*self.modalities, modality]
        else:
            iter_modality = [modality]
        for m in iter_modality:
            logits = self.forward(inp=data, modality=m)
            if isinstance(logits, (list, tuple)):
                logits, *output = logits
            else:
                output = []
            loss = self.compute_loss(logits, label, concept_label, *output)
            self.log(f'val_{m}_loss', loss)
            self.val_metrics[m].update(self.logits_to_prob(logits),  self.label_to_target(label))

    def on_validation_epoch_start(self) -> None:
        self.current_stage = 'val'

    def on_validation_epoch_end(self) -> None:
        for m, metric_collection in self.val_metrics.items():
            result = metric_collection.compute()
            self.log_dict(result, on_epoch=True, prog_bar=True)
            if self.filelogger is not None:
                self.filelogger.info(f'Validation {m} Metrics: {result}')
            metric_collection.reset()

    def test_step(self, batch, batch_idx):
        data = batch['data']
        label = batch[self.label_key]
        names, pathologies, modalities = batch['name'], batch['pathology'], batch['modality']
        modality = modalities[0]
        concept_label = batch['concept_label'] if 'concept_label' in batch else None
        if modality == 'MM':
            iter_modality = [*self.modalities, modality]
        else:
            iter_modality = [modality]
        for m in iter_modality:
            logits = self.forward(inp=data, modality=m)
            if isinstance(logits, (list, tuple)):
                logits, *output = logits
            else:
                output = []
            loss = self.compute_loss(logits, label, concept_label, *output)
            self.all_y[m].append(label.cpu())
            self.all_pred[m].append(logits.cpu())
            self.log(f'test_{m}_loss', loss.detach())

            for name, pathology, l, p in zip(names, pathologies,  self.label_to_target(label).cpu().numpy(),
                                            self.logits_to_prob(logits).cpu().numpy()):
                self.results.append({
                    'name': name,
                    'pathology': pathology,
                    'modality': m,
                    'label': l,
                    'probability': p,
                    'pred': self.config.id_to_labels[int(p>0.5)]
                })

    def on_test_epoch_start(self) -> None:
        self.all_y = defaultdict(list)
        self.all_pred = defaultdict(list)
        self.results = []
        self.current_stage = 'test'

    def on_test_epoch_end(self):
        test_metrics = []
        self.test_metrics = self.test_metrics.cpu()
        for m in self.all_y.keys():
            self.all_y[m] = torch.cat(self.all_y[m], dim=0).cpu()
            self.all_pred[m] = torch.cat(self.all_pred[m], dim=0).cpu()
            metrics = self.test_metrics[m](self.logits_to_prob(self.all_pred[m]), self.label_to_target(self.all_y[m]))
            if self.filelogger is not None:
                self.filelogger.info(f'Test {m} Metrics: {metrics}')
            test_metrics.append(metrics)
            self.test_metrics[m].reset()
        torch.save({
            'y': self.all_y,
            'pred': self.all_pred
        }, f'{self.config.output_dir}/test_pred.pth')
        results = pd.DataFrame(self.results)
        results.to_csv(f'{self.config.output_dir}/test_pred.csv', index=False)
        pd.DataFrame(test_metrics).to_csv(f'{self.config.output_dir}/test_pred_metrics.csv', index=False)
    
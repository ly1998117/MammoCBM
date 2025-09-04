# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117
@Contact :  liu.yang.mine@gmail.com
"""
import math
import torch.nn as nn
import torch
import torchmetrics
import pandas as pd
from collections import defaultdict
from models.backbone.networks import RNNClassifier, AttnPoolClassifier, MaxPoolClassifier, TransformerClassifier, \
    AttnConceptClassifier, TransformerPoolClassifier, TransformerRoutingPoolingClassifier
from models.backbone.networks import MMEfficientEncoder, MMViTEncoder, MMSwinTransformerEncoder
from models.module import MultiModalModule

efficientnet_params = {
    "b0": "efficientnet-b0",
    "b1": "efficientnet-b1",
    "b2": "efficientnet-b2",
    "b3": "efficientnet-b3",
    "b4": "efficientnet-b4",
    "b5": "efficientnet-b5",
    "b6": "efficientnet-b6",
    "b7": "efficientnet-b7",
    "b8": "efficientnet-b8",
    "l2": "efficientnet-l2",
}


######################################################## network ####################################################


class SingleBaseNet(MultiModalModule):
    """
    Single FusionEncoder for all modalities
    ignore the modality difference in vision information encoding
    """

    def __init__(self, Encoder_fn, Classifier_fn, config, label_key='label'):
        super().__init__(config, label_key=label_key)
        self.avg_pooling = config.avg_pooling
        self.encoder = Encoder_fn()
        self.classifier = Classifier_fn(out_channels=self.out_channels)

    @property
    def predictor(self):
        return self.classifier.classifier

    def configure_optimizers(self):
        if self.config.warmup_ratio == 0:
            return super().configure_optimizers()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        def lr_lambda(current_step):
            warmup_steps = int(self.config.warmup_ratio * self.config.epochs)
            total_steps = self.config.epochs

            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))  # Linear warmup
            if self.config.scheduler == 'poly':
                progress = (current_step - warmup_steps) / float(total_steps - warmup_steps)
                return (1 - progress) ** 0.9
            else:
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Cosine decay

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda),
            'interval': 'epoch',
            'frequency': 1,
            'name': 'lr_scheduler'
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @property
    def out_channels(self):
        return self.encoder.out_channels

    def fusion(self, x):
        return self.classifier.fusion(x)

    def forward(self, inp, modality, train_classifier=True, **kwargs):
        # if train_classifier:
        #     self.classifier.unfreeze()
        # else:
        #     self.classifier.freeze()

        inp = {k: v for k, v in inp.items() if k in self.modalities}
        features = self.encode_image(inp, modality=modality, fusion=False)
        return self.classifier(features)

    def encode_image(self, image, modality=None, fusion=False, avg_pooling=None) -> dict:
        if modality is None and not isinstance(image, dict):
            raise ValueError('modality is required when image is not dict')
        if not isinstance(image, dict):
            image = {modality: image}
        if modality != 'MM' and modality is not None:
            image = {modality: image[modality]}
        image = {k: v.to(self.device) for k, v in image.items()}
        if avg_pooling is None:
            avg_pooling = self.avg_pooling
        image = self.encoder(image, avg_pooling)
        if fusion and avg_pooling:
            image = self.fusion(image)
            return {'MM': image}
        return image

    def classify(self, inp):
        out = {}
        for m, x in inp.items():
            out.update(self.encoder(x))
        return self.classifier(*out)

    def get_classifier(self):
        return self.classifier

    def __getitem__(self, modality):
        '''
        :param modality: 根据输入 modality 返回模型不同部分，作为 Optimizer 输入
        :return:
        '''
        if modality == 'classifiers':
            return self.classifier
        if modality == 'encoders':
            return self.encoder
        else:
            return nn.ModuleDict({
                'encoder': self.encoder[modality],
                'classifier': self.classifier[modality]
            })


########################################## Single CLS ##########################################


class MMAttnSCLSEfficientNet(SingleBaseNet):
    def __init__(self, config):
        modalities = config.modality_to_model[config.modality]
        modality_to_encoder = config.modality_to_encoder

        def get_cls(out_channels):
            if config.fusion_method == 'pool':
                return AttnPoolClassifier(in_features=out_channels,
                                          out_features=config.num_class,
                                          use_norm=False,
                                          use_residual=False,
                                          use_proj=False,
                                          linearcls=False)
            elif config.fusion_method == 'spool':
                return AttnPoolClassifier(in_features=out_channels,
                                          out_features=config.num_class,
                                          use_norm=False,
                                          use_residual=False,
                                          use_proj=True,
                                          linearcls=True)
            elif config.fusion_method == 'rpool':
                return AttnPoolClassifier(in_features=out_channels,
                                          out_features=config.num_class,
                                          use_norm=False,
                                          use_residual=True,
                                          use_proj=True,
                                          linearcls=True)
            elif config.fusion_method == 'npool':
                return AttnPoolClassifier(in_features=out_channels,
                                          out_features=config.num_class,
                                          use_norm=True,
                                          use_residual=False,
                                          use_proj=True,
                                          linearcls=True)
            elif config.fusion_method == 'rnpool':
                return AttnPoolClassifier(in_features=out_channels,
                                          out_features=config.num_class,
                                          use_norm=True,
                                          use_residual=True,
                                          use_proj=True,
                                          linearcls=True)
            elif config.fusion_method == 'mattn':
                return TransformerPoolClassifier(modalities=modalities,
                                                 in_features=out_channels,
                                                 out_features=config.num_class)
            elif config.fusion_method == 'route':
                return TransformerRoutingPoolingClassifier(modalities=modalities,
                                                           in_features=out_channels,
                                                           out_features=config.num_class)
            elif config.fusion_method == 'max':
                return MaxPoolClassifier(in_features=out_channels,
                                         out_features=config.num_class)
            elif config.fusion_method == 'transformer':
                return TransformerClassifier(modalities=modalities, in_features=out_channels,
                                             out_features=config.num_class)
            else:
                raise ValueError(f'No such fusion_method: {config.fusion_method}')

        super(MMAttnSCLSEfficientNet, self).__init__(
            Encoder_fn=lambda **kwargs: MMEfficientEncoder(modalities=modalities,
                                                           modality_to_encoder=modality_to_encoder,
                                                           model_name=config.model_name,
                                                           spatial_dims=config.spatial_dims,
                                                           input_channels=config.input_channels,
                                                           pretrained=config.pretrained,
                                                           avg_pooling=config.avg_pooling,
                                                           blocks_args_str=config.blocks_args_str,
                                                           efficientnet_params=config.efficientnet_params
                                                           ),
            Classifier_fn=get_cls,
            config=config
        )

#######################################################################################################################################
class MMConcepOnlytEfficientNet(SingleBaseNet):
    def __init__(self, config, classifier):
        modalities = config.modality_to_model[config.modality]
        modality_to_encoder = config.modality_to_encoder
        self.concept_num = len(classifier.concepts)

        def get_cls(out_channels):
            return AttnPoolClassifier(in_features=out_channels,
                                        out_features=config.num_class,
                                        use_norm=False,
                                        use_residual=False,
                                        use_proj=True,
                                        linearcls=True,
                                        classifier=classifier)

        super(MMConcepOnlytEfficientNet, self).__init__(
            Encoder_fn=lambda **kwargs:  MMEfficientEncoder(modalities=modalities,
                                                           modality_to_encoder=modality_to_encoder,
                                                           model_name=config.model_name,
                                                           spatial_dims=config.spatial_dims,
                                                           input_channels=config.input_channels,
                                                           pretrained=config.pretrained,
                                                           avg_pooling=config.avg_pooling,
                                                           blocks_args_str=config.blocks_args_str,
                                                           efficientnet_params=config.efficientnet_params
                                                           ),
            Classifier_fn=get_cls,
            config=config,
            label_key='concept_label'
        )

    def configure_metric(self, prefix, num_classes):
        metric_d = nn.ModuleDict()
        for modality in [*self.modalities, 'MM']:
            metrics = torchmetrics.MetricCollection({
                f'{modality}_acc': torchmetrics.Accuracy(task='multilabel', num_labels=self.concept_num),
                f'{modality}_auc': torchmetrics.AUROC(task='multilabel', num_labels=self.concept_num),
                f'{modality}_recall': torchmetrics.Recall(task='multilabel', num_labels=self.concept_num),
                f'{modality}_precision': torchmetrics.Precision(task='multilabel', num_labels=self.concept_num),
                f'{modality}_f1': torchmetrics.F1Score(task='multilabel', num_labels=self.concept_num)
            }, prefix=prefix)
            metric_d.update({modality: metrics})
        return metric_d

    def configure_loss(self) -> dict:
        return {
            'cls_loss': nn.BCEWithLogitsLoss()
        }

    def compute_loss(self, logits, label, concept_label, *output):
        loss = self.loss.cls_loss(logits, concept_label)
        return loss
    
    def logits_to_prob(self, logits):
        return logits.detach().sigmoid()

    def label_to_target(self, label):
        return label.long()

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

            for name, pathology, l, p in zip(names, pathologies, label.cpu().numpy(),
                                            self.logits_to_prob(logits).cpu().numpy()):
                self.results.append({
                    'name': name,
                    'pathology': pathology,
                    'modality': m,
                    'label': l,
                    'probability': p,
                })


########################################################################################################################

class MMConceptBaseNet(SingleBaseNet):
    def __init__(self, config):
        modalities = config.modality_to_model[config.modality]
        modality_to_encoder = config.modality_to_encoder

        def get_cls(out_channels):
            return AttnConceptClassifier(in_features=out_channels,
                                         out_features=config.num_class,
                                         concept_num=56)

        super(MMConceptBaseNet, self).__init__(
            Encoder_fn=lambda **kwargs: None,
            Classifier_fn=get_cls,
            config=config
        )
        self.train_multi_metrics = self.config_multi_metric(prefix='train_')
        self.val_multi_metrics = self.config_multi_metric(prefix='val_')
        self.test_multi_metrics = self.config_multi_metric(prefix='test_')

    def configure_metric(self, prefix, num_classes):
        task = 'binary' if num_classes == 2 else 'multiclass'
        metric_d = nn.ModuleDict()
        for modality in [*self.modalities, 'MM']:
            metrics = torchmetrics.MetricCollection({
                f'{modality}_acc': torchmetrics.Accuracy(task=task, num_classes=num_classes),
                f'{modality}_auc': torchmetrics.AUROC(task=task, num_classes=num_classes),
                f'{modality}_recall': torchmetrics.Recall(task=task, num_classes=num_classes),
                f'{modality}_precision': torchmetrics.Precision(task=task, num_classes=num_classes),
                f'{modality}_f1': torchmetrics.F1Score(task=task, num_classes=num_classes),
            }, prefix=prefix)
            metric_d.update({modality: metrics})
        return metric_d

    def config_multi_metric(self, prefix):
        metric_d = nn.ModuleDict()
        for modality in [*self.modalities, 'MM']:
            metrics = torchmetrics.MetricCollection({
                f'{modality}_multilabel_acc': torchmetrics.Accuracy(task='multilabel', num_labels=self.concept_num),
                f'{modality}_multilabel_auc': torchmetrics.AUROC(task='multilabel', num_labels=self.concept_num),
                f'{modality}_multilabel_recall': torchmetrics.Recall(task='multilabel', num_labels=self.concept_num),
                f'{modality}_multilabel_precision': torchmetrics.Precision(task='multilabel',
                                                                           num_labels=self.concept_num),
                f'{modality}_multilabel_f1': torchmetrics.F1Score(task='multilabel', num_labels=self.concept_num)
            }, prefix=prefix)
            metric_d.update({modality: metrics})
        return metric_d

    def configure_loss(self) -> dict:
        return {
            'cls_loss': nn.CrossEntropyLoss(),
            'concept_loss': nn.BCEWithLogitsLoss()
        }

    def compute_loss(self, logits, label, concept_logits, concept_label):
        return self.loss.cls_loss(logits, label) + self.loss.concept_loss(concept_logits, concept_label)

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers()
        data = train_batch['data']
        label = train_batch['label']
        concept_label = train_batch['concept_label']
        modality = train_batch['modality'][0]
        if modality == 'MM':
            iter_modality = [*self.modalities, modality]
        else:
            iter_modality = [modality]
        for m in iter_modality:
            if not self.iter_train and m != 'MM':
                continue
            logits, concept_logits = self.forward(inp=data, modality=m, train_classifier=m == 'MM')
            self.train_metrics[m].update(logits.detach().softmax(dim=-1)[:, -1], label)
            self.train_multi_metrics[m].update(concept_logits.detach().sigmoid(), concept_label.long())
            loss = self.compute_loss(logits, label, concept_logits, concept_label)
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
        for m, metric_collection in self.train_multi_metrics.items():
            try:
                result = metric_collection.compute()
            except ValueError:
                continue
            self.log_dict(result, on_epoch=True, prog_bar=True)
            metric_collection.reset()

    def validation_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        modality = batch['modality'][0]
        concept_label = batch['concept_label']

        if modality == 'MM':
            iter_modality = [*self.modalities, modality]
        else:
            iter_modality = [modality]
        for m in iter_modality:
            logits, concept_logits = self.forward(inp=data, modality=m)
            loss = self.compute_loss(logits, label, concept_logits, concept_label)
            self.log(f'val_{m}_loss', loss)
            self.val_metrics[m].update(logits.softmax(dim=-1)[:, -1], label)
            self.val_multi_metrics[m].update(concept_logits.detach().sigmoid(), concept_label.long())

    def on_validation_epoch_start(self) -> None:
        self.current_stage = 'val'

    def on_validation_epoch_end(self) -> None:
        for m, metric_collection in self.val_metrics.items():
            result = metric_collection.compute()
            self.log_dict(result, on_epoch=True, prog_bar=True)
            if self.filelogger is not None:
                self.filelogger.info(f'Validation {m} Metrics: {result}')
            metric_collection.reset()
        for m, metric_collection in self.val_multi_metrics.items():
            result = metric_collection.compute()
            self.log_dict(result, on_epoch=True, prog_bar=True)
            if self.filelogger is not None:
                self.filelogger.info(f'Validation {m} Metrics: {result}')
            metric_collection.reset()

    def test_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        concept_label = batch['concept_label']

        names, pathologies, modalities = batch['name'], batch['pathology'], batch['modality']
        modality = modalities[0]
        if modality == 'MM':
            iter_modality = [*self.modalities, modality]
        else:
            iter_modality = [modality]
        for m in iter_modality:
            logits, concept_logits = self.forward(inp=data, modality=m)
            loss = self.compute_loss(logits, label, concept_logits, concept_label)
            self.all_y[m].append(label.cpu())
            self.all_multi_y[m].append(concept_label.cpu())
            self.all_pred[m].append(logits.cpu())
            self.all_multi_pred[m].append(concept_logits.cpu())
            self.log(f'test_{m}_loss', loss.detach())

            for name, pathology, l, p in zip(names, pathologies, label.cpu().numpy(),
                                             logits.softmax(-1).cpu().numpy()):
                self.results.append({
                    'name': name,
                    'pathology': pathology,
                    'modality': m,
                    'label': l,
                    'probability': p[-1],
                    'pred': self.config.id_to_labels[p.argmax(-1)]
                })

    def on_test_epoch_start(self) -> None:
        self.all_y = defaultdict(list)
        self.all_multi_y = defaultdict(list)
        self.all_pred = defaultdict(list)
        self.all_multi_pred = defaultdict(list)
        self.results = []
        self.current_stage = 'test'

    def on_test_epoch_end(self):
        super().on_test_batch_end()
        self.test_multi_metrics = self.test_multi_metrics.cpu()
        for m in self.all_multi_y.keys():
            self.all_multi_y[m] = torch.cat(self.all_multi_y[m], dim=0).cpu()
            self.all_multi_pred[m] = torch.cat(self.all_multi_pred[m], dim=0).cpu()
            multi_metrics = self.test_multi_metrics[m](self.all_multi_pred[m].sigmoid(),
                                                       self.all_multi_y[m].long())
            if self.filelogger is not None:
                self.filelogger.info(multi_metrics)
            self.test_multi_metrics[m].reset()

        torch.save({
            'y': self.all_multi_y,
            'pred': self.all_multi_pred
        }, f'{self.config.output_dir}/test_concept_pred.pth')
        results = pd.DataFrame(self.results)
        results.to_csv(f'{self.config.output_dir}/test_concept_pred.csv', index=False)


class MMConceptEfficientNet(MMConceptBaseNet):
    def __init__(self, config):
        modalities = config.modality_to_model[config.modality]
        modality_to_encoder = config.modality_to_encoder
        self.concept_num = 47

        def get_cls(out_channels):
            return AttnConceptClassifier(in_features=out_channels,
                                         out_features=config.num_class,
                                         concept_num=self.concept_num)

        super(MMConceptBaseNet, self).__init__(
            Encoder_fn=lambda **kwargs: MMEfficientEncoder(modalities=modalities,
                                                           modality_to_encoder=modality_to_encoder,
                                                           model_name=config.model_name,
                                                           spatial_dims=config.spatial_dims,
                                                           input_channels=config.input_channels,
                                                           pretrained=config.pretrained,
                                                           avg_pooling=config.avg_pooling,
                                                           blocks_args_str=config.blocks_args_str,
                                                           efficientnet_params=config.efficientnet_params
                                                           ),
            Classifier_fn=get_cls,
            config=config
        )
        self.train_multi_metrics = self.config_multi_metric(prefix='train_')
        self.val_multi_metrics = self.config_multi_metric(prefix='val_')
        self.test_multi_metrics = self.config_multi_metric(prefix='test_')



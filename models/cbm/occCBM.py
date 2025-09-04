import torch
from copy import deepcopy
from .baseCBM import _CBM
from models.backbone.networks.occurrencenet import MMOccurrenceNet
from monai.transforms import RandAffine


class OccNorm:
    def __init__(self, p=2, reduction="mean"):
        self.p = p
        self.reduction = reduction

    def __call__(self, x, dim=None):
        if isinstance(x, dict):
            loss = 0
            for k in x.keys():
                if 'TIC' in k:
                    continue
                loss += self(x[k], dim)
            return loss
        loss = x.norm(p=self.p, dim=dim)
        if self.reduction == "mean":
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class TransformLoss:
    """
    the loss applied on generated ROIs!
    """

    def __init__(self, device, reduction="mean"):
        self.criterion = torch.nn.L1Loss(reduction="mean")
        self.reduction = reduction
        self.transform = RandAffine(prob=1,
                                    rotate_range=20,
                                    translate_range=0,
                                    scale_range=(0.6, 1.5),
                                    device=torch.device(device))

    def __call__(self, x, occurrence_map, compute_occ_map_fn, m=None):
        if isinstance(x, dict):
            loss = 0
            for k in x.keys():
                if 'TIC' in k:
                    continue
                loss += self(x[k], occurrence_map[k], compute_occ_map_fn, k)
            return loss
        B, C, *spatial_dims = x.shape
        transformed_x = self.transform(x.reshape(B * C, *spatial_dims)).reshape(B, C, *spatial_dims)
        occurrence_map_transformed = compute_occ_map_fn({m: transformed_x})[m]
        B, C, *spatial_dims = occurrence_map_transformed.shape
        transformed_occurrence_map = self.transform(
            occurrence_map.reshape(B * C, *spatial_dims),
            randomize=False
        ).reshape(B, C, *spatial_dims)
        # compute L1 loss
        loss = self.criterion(occurrence_map_transformed, transformed_occurrence_map)
        return loss


class OccurrenceCBM(_CBM):
    def config_classifier(self):
        classifier = torch.nn.Linear(len(self.concepts), self.config.num_class)
        classifier.weight.data = self.init_weight_matrix()
        classifier.bias.data.zero_()
        return classifier

    def config_module(self):
        self.predictor.freeze()
        self.occmap = MMOccurrenceNet(
            modalities=self.config.modality_to_model[self.config.modality],
            modality_to_encoder=self.config.modality_to_encoder,
            input_channels=self.encoder.out_channels,
            out_channels=len(self.concepts),
            spatial_dims=self.config.spatial_dims,
            act=self.config.map_activation,
            pool=self.config.map_pool
        )
        self.occ_predictor = deepcopy(self.predictor)
        self.occ_predictor.unfreeze()

    def configure_loss(self) -> dict:
        base_loss = super().configure_loss()
        base_loss.update({
            'transform': TransformLoss(self.config.device),
            'mse': torch.nn.MSELoss(),
            'occnorm': OccNorm(),
        })
        return base_loss

    def compute_occ(self, inp):
        inp = self.encode_image(inp, fusion=False, grad=False, avg_pooling=False)
        feature_map = self.occmap.feature_map(inp)
        occ_map = self.occmap.occ_map(inp)
        return feature_map, occ_map

    def compute_concept_score(self, feature_map, occ_map):
        img_map = self.occmap.output(feature_map, occ_map)
        img_map = {k: v.reshape(-1, self.encoder.out_channels) for k, v in img_map.items()}
        img_feat = self.encoder.fusion(img_map).reshape(-1, len(self.concepts), self.encoder.out_channels)
        concept_score = torch.cat([self.occ_predictor[i](img_feat[:, i, :]) for i in range(len(self.concepts))], dim=-1)
        return concept_score

    def compute_loss(self, logits, label, concept_label, *args):
        inp, occ_map, concept_score, concept_score_gt = args
        # classification accuracy
        loss = self.loss.cls_loss(logits, label)
        if self.lambda_l1 > 0:
            loss += self.lambda_l1 * self.loss.l1_norm(self.classifier.weight)
        if self.config.lambda_transform > 0:
            transform_loss = self.config.lambda_transform * self.loss.transform(inp, occ_map,
                                                                                lambda x: self.compute_occ(x)[-1])
            if self.current_stage == 'train' and transform_loss.grad_fn is None:
                raise ValueError('transform loss is None')
            self.log(f'{self.current_stage}_transform_loss', transform_loss)
            loss += transform_loss
        if self.config.lambda_mse > 0:
            mse_loss = self.config.lambda_mse * self.loss.mse(concept_score, concept_score_gt)
            if self.current_stage == 'train' and mse_loss.grad_fn is None:
                raise ValueError('mse loss is None')
            self.log(f'{self.current_stage}_mse_loss', mse_loss)
            loss += mse_loss
        if self.config.lambda_occnorm > 0:
            occnorm_loss = self.config.lambda_occnorm * self.loss.occnorm(occ_map)
            if self.current_stage == 'train' and occnorm_loss.grad_fn is None:
                raise ValueError('occl1 loss is None')
            self.log(f'{self.current_stage}_occnorm_loss', occnorm_loss)
            loss += occnorm_loss
        return loss

    def forward(self, inp, modality, **kwargs):
        if isinstance(inp, dict):
            inp = {k: v for k, v in inp.items() if k in self.modalities}
            feature_map, occ_map = self.compute_occ(inp)
            concept_score = self.compute_concept_score(feature_map, occ_map)
        else:
            raise TypeError
        with torch.no_grad():
            concept_score_gt = self.predictor(self.encode_image(inp, fusion=True, grad=False, avg_pooling=True)['MM'])
        logits = self.classifier(self.scale * concept_score)
        return logits, inp, occ_map, concept_score, concept_score_gt

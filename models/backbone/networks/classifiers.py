# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import random

import numpy as np
import torch
import torch.nn as nn
from .blocks import LinearSelfAttention, SelfAttentionPooling, MultiModalTransformer, TransformerPooling, \
    TransformerRoutingPooling


@torch.no_grad()
def cuboid_mask_to_gaussian_heatmap(mask3d: torch.Tensor, k_sigma: float = 0.25):
    # mask3d: [B,1,D,H,W] 或 [D,H,W]
    m = mask3d
    if m.ndim == 5:
        m = m[0, 0]
    elif m.ndim == 4:
        m = m[0]
    assert m.ndim == 3
    idx = m.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return torch.zeros_like(m, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    z1y1x1 = idx.min(0).values;
    z2y2x2 = idx.max(0).values + 1
    z1, y1, x1 = z1y1x1.tolist();
    z2, y2, x2 = z2y2x2.tolist()
    zc = (z1 + z2 - 1) / 2.0;
    yc = (y1 + y2 - 1) / 2.0;
    xc = (x1 + x2 - 1) / 2.0
    sz = max(z2 - z1, 1);
    sy = max(y2 - y1, 1);
    sx = max(x2 - x1, 1)
    sigma_z = max(k_sigma * sz, 1.0);
    sigma_y = max(k_sigma * sy, 1.0);
    sigma_x = max(k_sigma * sx, 1.0)
    D, H, W = m.shape
    z = torch.arange(D, device=m.device).view(D, 1, 1)
    y = torch.arange(H, device=m.device).view(1, H, 1)
    x = torch.arange(W, device=m.device).view(1, 1, W)
    g = torch.exp(-0.5 * (((z - zc) / sigma_z) ** 2 + ((y - yc) / sigma_y) ** 2 + ((x - xc) / sigma_x) ** 2))
    return g.clamp_(0, 1).unsqueeze(0).unsqueeze(0)  # -> [1,1,D,H,W]


class SpatialAttention3D(nn.Module):
    """
    输入: F ∈ [B, C, D, H, W]
    输出: A ∈ [B, 1, D, H, W]，范围 [0,1]
    """

    def __init__(self, in_channels, mid_channels=32, sharpness=4.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, 1, 1)  # -> 1 通道注意力 logits
        )
        self.sharpness = nn.Parameter(torch.tensor(float(sharpness)))  # 可学习的 β

    def forward(self, feat):  # feat: [B,C,D,H,W]
        logits = self.conv(feat)  # [B,1,D,H,W]
        A = torch.sigmoid(self.sharpness * logits)
        return A, logits


class AttnGate3D(nn.Module):
    """
    将注意力应用在特征上，支持残差门控
    """

    def __init__(self, in_channels, mid_channels=32, sharpness=4.0, residual=True, alpha=1.0):
        super().__init__()
        self.attn = SpatialAttention3D(in_channels, mid_channels, sharpness)
        self.residual = residual
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))  # 可学习的 α

    def forward(self, feat):  # feat: [B,C,D,H,W]
        A, logits = self.attn(feat)  # A: [B,1,D,H,W]
        if self.residual:
            gated = (1.0 + self.alpha * A) * feat
        else:
            gated = A * feat
        return gated, A, logits


##############################################################
class BaseClassifier(nn.Module):
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class AvgMaxPool(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.down_sample = nn.Conv1d(in_channels=2,
                                     out_channels=1,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False)

    def forward(self, x):
        avg = x.mean(dim=1)
        max = x.max(dim=1)[0]

        return self.down_sample(torch.stack([avg, max], dim=1)).squeeze(1)


class SimpleClassifier(BaseClassifier):
    def __init__(self, in_features, out_features, reduction=16):
        super().__init__()
        hidden_dim = max(in_features // reduction, 64)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.mlp(x)


class RNNClassifier(BaseClassifier):
    def __init__(self, in_features, out_features, num_layers=1, reduction=8, mask_prob=0.5):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_features, hidden_size=in_features // 2,
                           num_layers=num_layers, bidirectional=True, batch_first=True)
        self.attn = LinearSelfAttention(in_features=in_features, bias=False)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.classifier = SimpleClassifier(in_features=in_features,
                                           out_features=out_features,
                                           reduction=reduction)
        self.R = np.random.RandomState()
        self.mask_prob = mask_prob

    def __getitem__(self, item):
        return self

    def do(self):
        return self.R.rand() < self.mask_prob

    def random_mask(self, x: list):
        if isinstance(x, list) and self.do and len(x) > 1:
            x.pop(random.randint(0, len(x) - 1))
        return x

    def random_shuffle(self, x: list):
        if isinstance(x, list) and self.do():
            np.random.shuffle(x)
        return x

    def fusion(self, x):
        x = list(x.values()) if isinstance(x, dict) else x
        x = torch.concat(x, dim=1)
        x = self.rnn(x)[0][:, -1]
        return x

    def forward(self, x, modality=None):
        x = self.fusion(x)
        x = self.attn(x)
        x = self.relu(x)
        return self.classifier(x)


class AttnConceptClassifier(BaseClassifier):
    def __init__(self, in_features, out_features, concept_num=8):
        super().__init__()
        self.pool = SelfAttentionPooling(input_dim=in_features, use_residual=True, use_proj=True, use_norm=True)
        self.attn = LinearSelfAttention(in_features=in_features, bias=False, use_residual=True, use_proj=True,
                                        use_norm=True)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.concept_cls = nn.Linear(in_features, concept_num)
        self.classifier = nn.Linear(in_features, out_features)

    def __getitem__(self, item):
        return self

    def fusion(self, x):
        x = list(x.values()) if isinstance(x, dict) else x
        x = torch.stack(x, dim=1)
        x = self.pool(x)
        x = self.attn(x)
        return x

    def forward(self, x, modality=None):
        x = self.fusion(x)
        x = self.relu(x)
        concept = self.concept_cls(x)
        x = self.classifier(x)
        return x, concept


class AttnPoolClassifier(BaseClassifier):
    def __init__(self, in_features, out_features, reduction=8, use_norm=False, use_residual=False, linearcls=False,
                 use_proj=False, classifier=None):
        super().__init__()
        self.use_proj = use_proj
        self.pool = SelfAttentionPooling(input_dim=in_features, use_norm=use_norm, use_residual=use_residual,
                                         use_proj=use_proj)
        self.attn = LinearSelfAttention(in_features=in_features, bias=False, use_norm=use_norm,
                                        use_residual=use_residual, use_proj=use_proj)
        if classifier is None:
            self.classifier = SimpleClassifier(in_features=in_features,
                                            out_features=out_features,
                                            reduction=reduction) if not linearcls else nn.Linear(in_features=in_features,
                                                                                                    out_features=out_features)
        else:
            self.classifier = classifier
        if not use_proj:
            self.relu = nn.LeakyReLU(0.2, inplace=True)

    def __getitem__(self, item):
        return self

    def fusion(self, x):
        x = list(x.values()) if isinstance(x, dict) else x
        x = torch.stack(x, dim=1)
        x = self.pool(x)
        x = self.attn(x)
        return x

    def forward(self, x, modality=None):
        x = self.fusion(x)
        if not self.use_proj:
            x = self.relu(x)
        x = self.classifier(x)
        return x


class TransformerPoolClassifier(BaseClassifier):
    def __init__(self, modalities, in_features, out_features, reduction=8):
        super().__init__()
        self.pool = TransformerPooling(modalities=modalities, input_dim=in_features)
        self.classifier = SimpleClassifier(in_features=in_features,
                                           out_features=out_features,
                                           reduction=reduction)

    def __getitem__(self, item):
        return self

    def fusion(self, x):
        x = self.pool(x)
        return x

    def forward(self, x, modality=None):
        x = self.fusion(x)
        x = self.classifier(x)
        return x


class TransformerRoutingPoolingClassifier(BaseClassifier):
    def __init__(self, modalities, in_features, out_features, reduction=8):
        super().__init__()
        self.pool = TransformerRoutingPooling(modalities=modalities, input_dim=in_features)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.classifier = SimpleClassifier(in_features=in_features,
                                           out_features=out_features,
                                           reduction=reduction)

    def __getitem__(self, item):
        return self

    def fusion(self, x):
        x = self.pool(x)
        return x

    def forward(self, x, modality=None):
        x = self.fusion(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


class TransformerClassifier(BaseClassifier):
    def __init__(self, modalities, in_features, out_features):
        super().__init__()
        self.transformer = MultiModalTransformer(embed_dim=in_features, modalities=modalities,
                                                 num_heads=8, num_layers=1)
        self.fn = nn.Linear(in_features, out_features)

    def __getitem__(self, item):
        return self

    def fusion(self, x):
        return self.transformer(x)

    def forward(self, x, *args, **kwargs):
        assert isinstance(x, dict), 'input must be a dict'
        x = self.transformer(x)
        x = self.fn(x)
        return x


class MaxPoolClassifier(BaseClassifier):
    def __init__(self, in_features, out_features, reduction=8, mask_prob=0.5):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.attn = LinearSelfAttention(in_features=in_features, bias=False)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.classifier = SimpleClassifier(in_features=in_features,
                                           out_features=out_features,
                                           reduction=reduction)
        self.mask_prob = mask_prob

    def __getitem__(self, item):
        return self

    def fusion(self, x):
        x = list(x.values()) if isinstance(x, dict) else x
        x = torch.concat(x, dim=1)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(2)
        return x

    def forward(self, x, modality=None):
        x = self.fusion(x)
        x = self.attn(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


class MMDictClassifier(BaseClassifier):
    def __init__(self, in_features, out_features, num_layers, reduction=8, mask_prob=0.5):
        super().__init__()
        if num_layers > 0:
            self.rnn = nn.GRU(input_size=in_features, hidden_size=in_features // 2,
                              num_layers=num_layers, bidirectional=True, batch_first=True)
        self.relu = nn.LeakyReLU(.2, inplace=True)
        self.classifiers = nn.ModuleDict({
            modality: SimpleClassifier(
                in_features=in_features,
                out_features=out_features,
                reduction=reduction
            ) for modality in ['FA', 'ICGA', 'US', 'MM']
        })
        self.R = np.random.RandomState()
        self.mask_prob = mask_prob

    def __getitem__(self, item):
        modules = nn.ModuleList()
        if hasattr(self, 'rnn'):
            modules.append(self.rnn)
        modules.append(self.classifiers[item])
        return modules

    def do(self):
        return self.R.rand() < self.mask_prob

    def random_shuffle(self, x: list):
        if self.do():
            np.random.shuffle(x)
        return x

    def forward(self, x: dict, modality=None):
        if len(x) == 1:
            modality = list(x.keys())[0]
        else:
            modality = 'MM'
        x = torch.concat(list(x.values()), dim=1)
        if hasattr(self, 'rnn'):
            x = self.rnn(x)[0][:, -1]
        x = self.relu(x)
        return self.classifiers[modality](x)

import torch
import torch.nn as nn

from monai.networks.nets.vit import ViT as _VIT
from monai.networks.nets.swin_unetr import MERGING_MODE
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from collections.abc import Sequence
from typing_extensions import Final
from .BaseNet import MMBaseEncoder, BackBone
from .blocks import MLP


class VisionTransformer(BackBone):
    def __init__(self, in_channels: int,
                 img_size: Sequence[int] | int,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 num_heads: int = 12,
                 proj_type: str = "conv",
                 dropout_rate: float = 0.0,
                 spatial_dims: int = 3,
                 qkv_bias: bool = False,
                 save_attn: bool = False,
                 avg_pooling=True):
        super().__init__()
        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.vit = _VIT(in_channels=in_channels,
                        img_size=img_size,
                        patch_size=self.patch_size,
                        hidden_size=hidden_size,
                        mlp_dim=mlp_dim,
                        num_layers=self.num_layers,
                        num_heads=num_heads,
                        proj_type=proj_type,
                        classification=False,
                        dropout_rate=dropout_rate,
                        spatial_dims=spatial_dims,
                        qkv_bias=qkv_bias,
                        save_attn=save_attn,
                        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pooling = avg_pooling
        self.out_channels = hidden_size

    def head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embedding(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        return x

    def output(self, x: torch.Tensor, pooling=None) -> torch.Tensor:
        if pooling is None:
            pooling = self.avg_pooling
        if pooling:
            x = x[:, 0]
        return x

    def get_layers(self) -> list:
        return self.vit.blocks


class MMViTEncoder(MMBaseEncoder):
    def __init__(self, img_size, modalities, modality_to_encoder, input_channels, spatial_dims=2, avg_pooling=True):
        super(MMViTEncoder, self).__init__(out_channels=0, spatial_dims=spatial_dims)
        self.encoder = nn.ModuleDict({
            m: VisionTransformer(spatial_dims=spatial_dims, in_channels=input_channels[m], avg_pooling=avg_pooling,
                                 img_size=img_size)
            for m in modalities
        })
        for v in self.encoder.values():
            self.out_channels = v.out_channels
            break
        for m in modalities:
            if modality_to_encoder[m] == 'MLP':
                self.encoder[m] = MLP(input_channels[m], self.out_channels)

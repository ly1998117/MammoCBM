import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as _SwinTransformer
from monai.networks.nets.swin_unetr import MERGING_MODE
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from collections.abc import Sequence
from typing_extensions import Final
from .BaseNet import MMBaseEncoder, BackBone
from .blocks import MLP


class SwinTransformer(BackBone):
    patch_size: Final[int] = 2

    def __init__(self, in_channels: int,
                 depths: Sequence[int] = (2, 2, 2, 2),
                 num_heads: Sequence[int] = (3, 6, 12, 24),
                 embed_dim: int = 24,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 dropout_path_rate: float = 0.0,
                 normalize: bool = True,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3,
                 downsample="merging",
                 avg_pooling=True):
        super().__init__()
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.swinViT = _SwinTransformer(in_chans=in_channels,
                                        embed_dim=embed_dim,
                                        window_size=window_size,
                                        patch_size=patch_sizes,
                                        depths=depths,
                                        num_heads=num_heads,
                                        mlp_ratio=4.0,
                                        qkv_bias=True,
                                        drop_rate=drop_rate,
                                        attn_drop_rate=attn_drop_rate,
                                        drop_path_rate=dropout_path_rate,
                                        norm_layer=nn.LayerNorm,
                                        use_checkpoint=use_checkpoint,
                                        spatial_dims=spatial_dims,
                                        downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample,
                                                                                                          str) else downsample
                                        )
        self.normalize = normalize
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pooling = avg_pooling
        self.out_channels = embed_dim * 2 ** len(depths)

    def head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.swinViT.patch_embed(x)
        x = self.swinViT.pos_drop(x)
        return x

    def iter_block(self, x, layer):
        return layer(x.contiguous())

    def output(self, x: torch.Tensor, pooling=None) -> torch.Tensor:
        x = self.swinViT.proj_out(x, self.normalize)
        if pooling is None:
            pooling = self.avg_pooling
        if pooling:
            x = x.flatten(2)  # (B, C, N)
            x = self.global_pool(x).squeeze(-1)  # (B, C)
        return x

    def get_layers(self) -> list:
        return [self.swinViT.layers1[0],
                self.swinViT.layers2[0],
                self.swinViT.layers3[0],
                self.swinViT.layers4[0]]


class MMSwinTransformerEncoder(MMBaseEncoder):
    def __init__(self, modalities, modality_to_encoder, input_channels, spatial_dims=2, avg_pooling=True, **kwargs):
        super(MMSwinTransformerEncoder, self).__init__(out_channels=0, spatial_dims=spatial_dims)
        self.encoder = nn.ModuleDict({
            m: SwinTransformer(spatial_dims=spatial_dims, in_channels=input_channels[m], avg_pooling=avg_pooling, )
            for m in modalities
        })
        for v in self.encoder.values():
            self.out_channels = v.out_channels
            break
        for m in modalities:
            if modality_to_encoder[m] == 'MLP':
                self.encoder[m] = MLP(input_channels[m], self.out_channels)

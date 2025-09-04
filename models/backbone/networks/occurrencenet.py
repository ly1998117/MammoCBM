import torch
import torch.nn as nn
from monai.networks.layers import Norm, Conv
from .BaseNet import MMBaseEncoder


class OccurrencePool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, spatial_dims: int = 3, act='abs', pool='avg'):
        super().__init__()
        conv_type = Conv[Conv.CONV, spatial_dims]
        self.add_on_layer = nn.Sequential(
            conv_type(in_channels, in_channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            conv_type(in_channels, in_channels, kernel_size=kernel_size),
        )
        self.occ = nn.Sequential(
            conv_type(in_channels, in_channels // 4, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            conv_type(in_channels // 4, in_channels // 8, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            conv_type(in_channels // 8, out_channels, kernel_size=kernel_size, bias=False)
        )
        self._initialize_weights(self.add_on_layer)
        self._initialize_weights(self.occ)
        self.act = act
        self.pool = pool

    def _initialize_weights(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def occ_map(self, x):
        x = self.occ(x)  # B, C, D, H, W
        if self.act == 'abs':
            return torch.abs(x)
        elif self.act == 'relu':
            return torch.relu(x)
        elif self.act == 'sigmoid':
            return torch.sigmoid(x)
        elif self.act == 'tanh':
            return torch.tanh(x)
        else:
            raise NotImplementedError

    def feature_map(self, x):
        x = self.add_on_layer(x)  # B, N, D, H, W
        return x

    def output(self, feature_map, occ_map):
        out = (occ_map.unsqueeze(2) * feature_map.unsqueeze(1))  # B, C, N, D, H, W
        if self.pool == 'avg':
            out = out.mean(dim=[3, 4, 5])
        elif self.pool == 'max':
            out, _ = out.max(dim=3, keepdim=False)  # 返回最大值，按 D 维度池化
            out, _ = out.max(dim=3, keepdim=False)  # 按 H 维度池化
            out, _ = out.max(dim=3, keepdim=False)  # 按 W 维度池化
        elif self.pool == 'sum':
            out = out.flatten(start_dim=3).sum(dim=3)
        else:
            raise NotImplementedError
        return out

    def forward(self, x):
        """
        :param x: (B, N, T, H, W)
        :return:
        """
        feature_map = self.feature_map(x)  # (B, 1, N, D, H, W)
        occ_map = self.occ_map(x)  # (B, C, 1, D, H, W)
        return self.output(feature_map, occ_map)  # shape (B, C, N)


class Repeat(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

    def feature_map(self, x):
        return x

    def occ_map(self, x):
        return x

    def output(self, x, y):
        return self.forward(x)

    def forward(self, x):
        """
        :param x: (B, L)
        :return:
        """
        x = x.unsqueeze(dim=1)  # (B, 1, L)
        x = x.repeat(1, self.out_channels, 1)
        return x  # (B, C, L)


class MMOccurrenceNet(MMBaseEncoder):
    def __init__(self, modalities, modality_to_encoder, input_channels, out_channels, spatial_dims=2, act='abs',
                 pool='avg'):
        super(MMOccurrenceNet, self).__init__(out_channels=out_channels, spatial_dims=spatial_dims)
        self.input_channels = input_channels
        self.encoder = nn.ModuleDict({
            m: OccurrencePool(in_channels=input_channels, out_channels=out_channels, spatial_dims=spatial_dims,
                              act=act, pool=pool)
            for m in modalities if modality_to_encoder[m] == 'EfficientNet'
        })
        for m in modalities:
            if modality_to_encoder[m] != 'EfficientNet':
                self.encoder[m] = Repeat(out_channels=out_channels)

    def feature_map(self, x, m=None):
        if isinstance(x, dict):
            return {k: self.feature_map(v, k) for k, v in x.items()}
        return self.encoder[m].feature_map(x)

    def occ_map(self, x, m=None):
        if isinstance(x, dict):
            return {k: self.occ_map(v, k) for k, v in x.items()}
        return self.encoder[m].occ_map(x)

    def output(self, feature_map, occ_map, m=None):
        if isinstance(feature_map, dict) and isinstance(occ_map, dict):
            return {k: self.output(feature_map[k], occ_map[k], k) for k in feature_map.keys()}
        if m is None:
            for m in self.encoder:
                try:
                    return self.output(feature_map, occ_map, m)
                except:
                    continue
        return self.encoder[m].output(feature_map, occ_map)

    def __getitem__(self, modality):
        return self.encoder[modality]


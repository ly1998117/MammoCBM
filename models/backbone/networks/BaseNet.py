# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import torch
import torch.nn as nn


class BackBone(nn.Module):
    def __init__(self, modality=None, *args, **kwargs):
        super().__init__()
        self.modality = modality

    def head(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def output(self, x: torch.Tensor, pooling) -> torch.Tensor:
        return x

    def get_layers(self) -> list or nn.ModuleList:
        raise NotImplementedError

    def iter_block(self, x, layer):
        x = layer(x)
        return x

    def forward(self, x: dict, pooling=None):
        x = x[self.modality] if isinstance(x, dict) else x
        x = self.head(x)
        for idx, layer in enumerate(self.get_layers()):
            x = self.iter_block(x, layer)
        x = self.output(x, pooling)
        return x


class SingleBaseEncoder(nn.Module):
    def __init__(self, out_channels, spatial_dims=2, enc_no_grad=False):
        super().__init__()
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.encoder = None
        self.enc_no_grad = enc_no_grad

    def to_B(self, x):
        if self.spatial_dims == 2:
            self.b = x.shape[0]
            x = x.reshape(-1, *x.shape[2:])
        return x

    def to_T(self, x):
        if self.spatial_dims == 2:
            x = x.reshape(self.b, -1, *x.shape[1:])
        return x

    def encode(self, x, m, pooling):
        x = self.encoder(x, pooling)
        return x

    def forward(self, x: dict, pooling=None):
        out = {}
        for m, v in x.items():
            v = self.to_B(v)
            if self.enc_no_grad and len(x) > 1:
                with torch.no_grad():
                    v = self.encode(v, m, pooling)
            else:
                v = self.encode(v, m, pooling)
            v = self.to_T(v)
            out[m] = v
        return out

    def __getitem__(self, item):
        return self


class MMBaseEncoder(SingleBaseEncoder):
    def encode(self, x, m, pooling):
        return self.encoder[m](x, pooling)

    def __getitem__(self, item):
        return self.encoder[item]


class BaseEncoder(nn.Module):
    def __init__(self, out_channels, concat=False, bidirectional=False, spatial_dims=2):
        super().__init__()
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.encoders = nn.ModuleList()
        self.rnns = nn.ModuleList()
        if concat:
            import pdb
            pdb.set_trace()
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=concat, out_channels=12, kernel_size=1),
                nn.BatchNorm1d(12),
                nn.ReLU(),
                nn.Conv1d(in_channels=12, out_channels=1, kernel_size=1)
            )
        if bidirectional:
            self.rnns.append(nn.LSTM(input_size=out_channels, hidden_size=out_channels // 2, bidirectional=True))

    def to_B(self, x):
        if self.spatial_dims == 2:
            self.time_step = x.shape[1]
            x = x.reshape(-1, *x.shape[2:])
        return x

    def to_T(self, x):
        if self.spatial_dims == 2:
            x = x.reshape(-1, self.time_step, x.shape[-1])
        return x

    def to_rnn(self, x, idx):
        if self.spatial_dims == 2:
            if len(self.rnns) > 0:
                x = self.rnns[idx](x)[0][:, -1, :]

            elif self.time_step > 1:
                x = self.downsample(x).reshape(-1, x.shape[-1])
            else:
                x = x.reshape(-1, x.shape[-1])
        return x

    def forward(self, *x):
        out = []
        for i, data in enumerate(x):
            x_i = self.to_B(data)
            x_i = self.encoders[i](x_i)
            x_i = self.to_T(x_i)

            x_i = self.to_rnn(x_i, i)
            out.append(x_i)
        out = self.attn(*out)
        return out

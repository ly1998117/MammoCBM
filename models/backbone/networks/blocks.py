# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import torch
import torch.nn as nn

from monai.utils import ensure_tuple_rep
from monai.networks.nets.swin_unetr import SwinTransformer
from .BaseNet import BaseEncoder, BackBone


class MultiModalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, dropout=0.1, modalities=("text", "image", "audio")):
        super(MultiModalTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.modality_embeddings = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, embed_dim)) for modality in modalities
        })

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, xes: dict, modality_masks=None):
        """
        xes: List of tensors, each with shape (batch_size, seq_len, embed_dim), representing different modalities.
        modality_names: List of strings, indicating the modality of each tensor in xes.
        modality_masks: Optional list of tensors, each with shape (batch_size, seq_len), indicating valid tokens for each modality.
        """
        # Get modality-aware position embeddings
        # Add modality embeddings to input features
        combined_features = [x + self.modality_embeddings[key] for key, x in xes.items()]

        # Concatenate all modalities along sequence dimension
        combined_features = torch.stack(combined_features, dim=1)  # Shape: (batch_size, total_seq_len, embed_dim)
        batch_size = combined_features.size(0)
        query = self.query.expand(-1, batch_size, -1)  # (1, batch_size, embed_dim)
        combined_features = torch.cat([query.transpose(0, 1), combined_features], dim=1)

        # Create combined mask
        if modality_masks is not None:
            assert len(modality_masks) == len(xes), "Each modality must have a corresponding mask"
            combined_mask = torch.cat(modality_masks, dim=1)  # Shape: (batch_size, total_seq_len)
            combined_mask = combined_mask.unsqueeze(1).unsqueeze(2)  # For compatibility with Transformer (B, 1, 1, L)
            combined_mask = (1.0 - combined_mask) * -1e9  # Invert mask for Transformer
        else:
            combined_mask = None

        # Transpose for Transformer (L, B, D)
        combined_features = combined_features.transpose(0, 1)  # Shape: (total_seq_len, batch_size, embed_dim)

        # Apply Transformer Encoder
        encoded = self.encoder(combined_features, src_key_padding_mask=combined_mask)

        # Transpose back (B, L, D)
        encoded = encoded.transpose(0, 1)

        return encoded[:, 0, :]


class ReductionLinear(nn.Module):
    def __init__(self, in_features, out_features, reduction):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // reduction),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(in_features=in_features // reduction, out_features=out_features)
        )

    def forward(self, x):
        return self.mlp(x)


class AttentionRnnBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.wq = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.wk = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.wv = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)

    def forward(self, x, hidden):
        """
        :param x: (N, T, H)
        :param hidden_t1: (N, 1, H)
        :return:
        """
        if hidden is None:
            hidden = torch.zeros((x.shape[0], x.shape[2])).to(x.device)
        hiddens = []
        for t in range(x.shape[1]):
            hiddens.append(hidden)
            query = self.wq(torch.cat([x[:, t, :], hidden], dim=1))
            key = self.wk(torch.cat([x[:, t, :], hidden], dim=1))
            value = self.wv(torch.cat([x[:, t, :], hidden], dim=1))
            w = torch.sigmoid(query * key)
            hidden = (1 - w) * hidden + w * torch.tanh(value)
        return torch.stack(hiddens, dim=1), hidden


class AttentionRnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.prob = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.attentions = nn.ModuleList([AttentionRnnBlock(hidden_size) for _ in range(num_layers)])

    def forward(self, x, hidden):
        x = self.prob(x)
        for attention in self.attentions:
            x, hidden = attention(x, hidden)
        return x, hidden


class TimeRNNAttentionPooling(nn.Module):
    """
    Implementation of TimeRNNAttentionPooling
    任意时间步的 Pooling
    """

    def __init__(self, input_size, num_layers=2, pure_out=False):
        super().__init__()
        # self.time_weight = AttentionRnn(input_size=input_size, hidden_size=1, num_layers=num_layers)
        # self.spatial_weight = AttentionRnn(input_size=input_size, hidden_size=input_size, num_layers=num_layers)
        self.time_weight = nn.GRU(input_size=input_size, hidden_size=1, num_layers=num_layers,
                                  batch_first=True, bidirectional=True)
        self.prob = nn.Linear(2, 1)
        self.pure_out = pure_out

    def forward(self, x):
        """
            input:
                batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
            attention_weight:
                time_weight : size (N, T, 1)
                spatial_weight : size (N, T, H)
            return:
                utter_rep: size (N, H)
        """

        ta = torch.softmax(self.prob(self.time_weight(x)[0]), dim=1)
        if self.pure_out:
            return (x * ta).sum(1)
        x = (x * ta).sum(1, keepdim=True)
        return x,


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    code from https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    """

    def __init__(self, input_dim, use_residual=False, use_norm=False, use_proj=True):
        super(SelfAttentionPooling, self).__init__()
        self.weight = ReductionLinear(input_dim, input_dim, reduction=8)
        if use_proj:
            self.proj = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                      nn.Linear(input_dim, input_dim))
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.use_proj = use_proj
        if use_norm:
            self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        attention_weight:
            att_w : size (N, T, 1)
        return:
            utter_rep: size (N, H)
        """

        # (N, T, H) -> (N, T) -> (N, T, 1)
        att_w = nn.functional.softmax(self.weight(x), dim=1)
        x = torch.sum(x * att_w, dim=1)
        if not self.use_proj:
            return x
        out = self.proj(x)
        if self.use_residual:
            out = out + x
        if self.use_norm:
            out = self.norm(out)
        return out


class TransformerRoutingPooling(nn.Module):
    def __init__(self, modalities, input_dim, num_seeds=None, num_heads=4, dropout=0.1):
        super().__init__()
        num_seeds = num_seeds if num_seeds else len(modalities)
        self.modalities = modalities
        self.input_dim = input_dim
        self.modality_embed = nn.Embedding(len(modalities), input_dim)
        self.seed_tokens = nn.Parameter(torch.empty(1, num_seeds, input_dim))
        nn.init.xavier_uniform_(self.seed_tokens)
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.weight_mlp = nn.Linear(input_dim, 1)

    def forward(self, x):
        assert isinstance(x, dict), "Expected input as dict {modality: (B, D)}"
        B = next(iter(x.values())).size(0)
        D = self.input_dim
        device = next(iter(x.values())).device

        # Collect tokens and mask
        tokens, mask = zip(*[
            (x[name].unsqueeze(1), torch.zeros(B, 1, dtype=torch.bool, device=device)) if name in x else
            (torch.zeros(B, 1, D, device=device), torch.ones(B, 1, dtype=torch.bool, device=device))
            for name in self.modalities
        ])
        modality_ids = torch.arange(len(self.modalities), device=device).unsqueeze(0).expand(B, -1)  # (B, N)

        x = torch.cat(tokens, dim=1)  # (B, N, D)
        x = x + self.modality_embed(modality_ids)  # Add modality embedding
        key_padding_mask = torch.cat(mask, dim=1)  # (B, N)

        # Routing attention: seed ← x
        seeds = self.seed_tokens.expand(B, -1, -1)  # (B, R, D)
        out, _ = self.attn(query=seeds, key=x, value=x, key_padding_mask=key_padding_mask)  # (B, R, D)
        out = self.norm(out + seeds)
        out = out + self.ffn(out)

        # Weighted token fusion
        weights = torch.nn.functional.softmax(self.weight_mlp(out), dim=1)  # (B, R, 1)
        out = (weights * out).sum(dim=1)  # (B, D)
        return out


class TransformerPooling(nn.Module):
    """
    Self-Attention based pooling module to fuse multiple modality embeddings
    into a single unified representation.
    Input shape: (B, N, D)
    Output shape: (B, D)
    """

    def __init__(self, modalities, input_dim, num_heads=4, ffn_ratio=2, dropout=0.1):
        super(TransformerPooling, self).__init__()
        self.modalities = modalities
        self.modality_to_id = {m: i for i, m in enumerate(modalities)}
        self.modality_embed = nn.Embedding(len(modalities), input_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))  # Learnable CLS
        nn.init.xavier_uniform_(self.cls_token)
        # Normalization after embedding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * ffn_ratio,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Crucial for (B, N, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        """
        x: dict {modality_name: (B, D)}
        Returns: (B, D)
        """
        # mask: 1=valid, 0=missing
        assert isinstance(x, dict), "Expected dict input"

        batch_size = next(iter(x.values())).size(0)
        device = next(iter(x.values())).device
        key_padding_mask = []
        tokens = []
        modality_ids = []
        for name in self.modalities:
            modality_ids.append(torch.full((batch_size, 1), self.modality_to_id[name], device=device))
            if name in x:
                tokens.append(x[name].unsqueeze(1))  # (B, 1, D)
                key_padding_mask.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=device))  # valid = 0
            else:
                tokens.append(torch.zeros(batch_size, 1, self.modality_embed.embedding_dim, device=device))  # pad
                key_padding_mask.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))  # missing = 1

        x = torch.cat(tokens, dim=1)  # (B, N, D)
        key_padding_mask = torch.cat(key_padding_mask, dim=1)  # (B, N)
        modality_ids = torch.cat(modality_ids, dim=1)  # (B, N)

        x = x + self.modality_embed(modality_ids)  # (B, N, D)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, N, D)
        # mean pooling over valid positions
        valid_mask = ~key_padding_mask  # (B, N)
        lengths = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        x = (x * valid_mask.unsqueeze(-1)).sum(dim=1) / lengths  # (B, D)
        return x


class SelfLinearAttentionPooling(SelfAttentionPooling):
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.weight = nn.Linear(input_dim, input_dim, bias=False)


class LinearChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.shape[-1] > 1:
            max_result = self.maxpool(x)
            avg_result = self.avgpool(x)
            output = self.se(max_result) + self.se(avg_result)
        else:
            output = self.se(x)
        output = self.sigmoid(output)
        return output * x


class ReductionLinearSelfAttention(nn.Module):
    def __init__(self, in_features, reduction):
        super().__init__()
        self.weight = ReductionLinear(in_features, in_features, reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.weight(x)) * x


class LinearSelfAttention(nn.Module):
    def __init__(self, in_features, bias=True, use_residual=False, use_norm=False, use_proj=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.use_proj = use_proj
        self.linear_q = nn.Linear(in_features, in_features, bias=bias)
        self.linear_k = nn.Linear(in_features, in_features, bias=bias)
        self.linear_v = nn.Linear(in_features, in_features, bias=bias)
        self.sigmoid = nn.Sigmoid()
        if use_proj:
            self.proj = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                      nn.Linear(in_features, in_features))
        if use_norm:
            self.norm = nn.LayerNorm(normalized_shape=in_features)

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        attn = self.sigmoid(q * k)
        v = attn * v
        if not self.use_proj:
            return v
        v = self.proj(v)
        if self.use_residual:
            v = v + x
        if self.use_norm:
            v = self.norm(v)
        return v


class LinearSpacialAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.weight = ReductionLinear(in_features, in_features, reduction=8)
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                  nn.Linear(in_features, in_features))

    def forward(self, x):
        attn = self.sigmoid(self.weight(x))
        return self.proj(attn * x)


class LinearDotSelfAttention(nn.Module):
    def __init__(self, in_features, bias=True):
        super().__init__()
        self.linear_q = nn.Linear(in_features, in_features, bias=bias)
        self.linear_k = nn.Linear(in_features, in_features, bias=bias)
        self.linear_v = nn.Linear(in_features, in_features, bias=bias)

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        q = q.unsqueeze(-1)
        k = k.unsqueeze(-2)
        attention = torch.softmax(torch.matmul(q, k), dim=-1)
        return torch.matmul(attention, v.unsqueeze(-1)).squeeze(-1)


class CycleAttention(nn.Module):
    def __init__(self, in_features, dot=True):
        super().__init__()
        self.linear_q = nn.Linear(in_features, in_features, bias=False)
        self.linear_k = nn.Linear(in_features, in_features, bias=False)
        self.linear_v = nn.Linear(in_features, in_features, bias=False)
        self.dot = dot

    def forward(self, value, query=None, key=None):
        if query is None and key is None:
            return value
        if query is None:
            query = value
        if key is None:
            key = value

        value = self.linear_v(value)
        query = self.linear_q(query)
        key = self.linear_k(key)
        if self.dot:
            value = value.unsqueeze(-1)
            query = query.unsqueeze(-1)
            key = key.unsqueeze(-2)
            attention = torch.softmax(torch.einsum('b j i, b i k -> b j k', query, key), dim=-1)
            return torch.einsum('b j k, b k i -> b j i', attention, value).squeeze(-1)
        return torch.sigmoid(query * key) * value


class MAModule(nn.Module):
    """ Channel attention module"""

    def __init__(self):
        super(MAModule, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X M X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, M, C, height, width = x.size()
        qkv = x.view(m_batchsize, M, -1)

        energy = torch.bmm(qkv, qkv.permute(0, 2, 1))  # (B X M X M)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        out = torch.bmm(attention, qkv)  # (B X M X M) * (B X M X N) = (B X M X N)
        out = out.view(m_batchsize, M, C, height, width).sum(1)
        return out


class MLP(BackBone):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.ModuleList([
            nn.Linear(in_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, out_channels),
        ])

    def get_layers(self) -> list or nn.ModuleList:
        return self.fc


if __name__ == "__main__":
    import torch

    image = torch.randn(8, 256)
    text = torch.randn(8, 256)
    a = torch.randn(8, 256)
    pooling = MultiModalTransformer(embed_dim=256, num_heads=8, num_layers=1)
    out = pooling({
        "image": image,
        "text": text,
        "audio": a
    })
    pass

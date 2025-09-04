# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import torch
import torch.nn as nn

from monai.networks.layers import Conv
from monai.networks.layers import Pool
from monai.networks.nets.efficientnet import (_round_filters, BlockArgs, _make_same_padder, get_norm_layer,
                                              _calculate_output_image_size, _round_repeats, MBConvBlock, Act, reduce,
                                              operator, math, _load_state_dict)
from .blocks import TimeRNNAttentionPooling
from .BaseNet import MMBaseEncoder, SingleBaseEncoder, BaseEncoder, BackBone
from .blocks import MLP

##################################################### Efficient #######################################################


_efficientnet_params = {
    # model_name: (width_mult, depth_mult, image_size, dropout_rate, dropconnect_rate)
    "efficientnet-b0": (1.0, 1.0, 224, 0.2, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3, 0.2),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3, 0.2),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4, 0.2),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4, 0.2),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5, 0.2),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5, 0.2),
    "efficientnet-b8": (2.2, 3.6, 672, 0.5, 0.2),
    "efficientnet-l2": (4.3, 5.3, 800, 0.5, 0.2),
}
_blocks_args_str = [
    "r1_k3_s11_e1_i32_o16_se0.25",
    "r2_k3_s22_e6_i16_o24_se0.25",
    "r2_k5_s22_e6_i24_o40_se0.25",
    "r3_k3_s22_e6_i40_o80_se0.25",
    "r3_k5_s11_e6_i80_o112_se0.25",
    "r4_k5_s22_e6_i112_o192_se0.25",
    "r1_k3_s11_e6_i192_o320_se0.25",
]


class EfficientNet(BackBone):
    def __init__(
            self,
            model_name,
            spatial_dims: int = 2,
            in_channels: int = 3,
            norm=("batch", {"eps": 1e-3, "momentum": 0.01}),
            depth_divisor: int = 8,
            modality=None,
            encoders=None,
            pretrained=True,
            avg_pooling=True,
            efficientnet_params=None,
            blocks_args_str=None
    ) -> None:
        """
        EfficientNet based on `Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/pdf/1905.11946.pdf>`_.
        Adapted from `EfficientNet-PyTorch <https://github.com/lukemelas/EfficientNet-PyTorch>`_.

        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            width_coefficient: width multiplier coefficient (w in paper).
            depth_coefficient: depth multiplier coefficient (d in paper).
            dropout_rate: dropout rate for dropout layers.
            image_size: input image resolution.
            norm: feature normalization type and arguments.
            drop_connect_rate: dropconnect rate for drop connection (individual weights) layers.
            depth_divisor: depth divisor for channel rounding.

        """
        super().__init__(modality=modality, encoders=encoders)
        if efficientnet_params is None:
            # check if model_name is valid model
            if model_name not in _efficientnet_params:
                model_name_string = ", ".join(_efficientnet_params.keys())
                raise ValueError(f"invalid model_name {model_name} found, must be one of {model_name_string} ")

            # get network parameters
            width_coefficient, depth_coefficient, image_size, dropout_rate, drop_connect_rate = _efficientnet_params[
                model_name]
        else:
            width_coefficient, depth_coefficient, image_size, dropout_rate, drop_connect_rate = efficientnet_params[
                model_name]
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims can only be 1, 2 or 3.")

        # select the type of N-Dimensional layers to use
        # these are based on spatial dims and selected from MONAI factories
        conv_type = Conv["conv", spatial_dims]
        adaptivepool_type = Pool[
            "adaptiveavg", spatial_dims
        ]
        if blocks_args_str is None:
            # decode blocks args into arguments for MBConvBlock
            blocks_args = [BlockArgs.from_string(s) for s in _blocks_args_str]
        else:
            blocks_args = [BlockArgs.from_string(s) for s in blocks_args_str]

        # checks for successful decoding of blocks_args_str
        if not isinstance(blocks_args, list):
            raise ValueError("blocks_args must be a list")

        if blocks_args == []:
            raise ValueError("block_args must be non-empty")

        self._blocks_args = blocks_args
        self.in_channels = in_channels
        self.drop_connect_rate = drop_connect_rate

        # expand input image dimensions to list
        current_image_size = [image_size] * spatial_dims

        # Stem
        stride = 2
        out_channels = _round_filters(32, width_coefficient, depth_divisor)  # number of output channels
        self._conv_stem = conv_type(self.in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self._conv_stem_padding = _make_same_padder(self._conv_stem, current_image_size)
        self._bn0 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)
        current_image_size = _calculate_output_image_size(current_image_size, stride)

        # build MBConv blocks
        num_blocks = 0
        self._blocks = nn.Sequential()

        self.extract_stacks = []

        # update baseline blocks to input/output filters and number of repeats based on width and depth multipliers.
        for idx, block_args in enumerate(self._blocks_args):
            block_args = block_args._replace(
                input_filters=_round_filters(block_args.input_filters, width_coefficient, depth_divisor),
                output_filters=_round_filters(block_args.output_filters, width_coefficient, depth_divisor),
                num_repeat=_round_repeats(block_args.num_repeat, depth_coefficient),
            )
            self._blocks_args[idx] = block_args

            # calculate the total number of blocks - needed for drop_connect estimation
            num_blocks += block_args.num_repeat

            if block_args.stride > 1:
                self.extract_stacks.append(idx)

        self.extract_stacks.append(len(self._blocks_args))

        # create and add MBConvBlocks to self._blocks
        idx = 0  # block index counter
        for stack_idx, block_args in enumerate(self._blocks_args):
            blk_drop_connect_rate = self.drop_connect_rate

            # scale drop connect_rate
            if blk_drop_connect_rate:
                blk_drop_connect_rate *= float(idx) / num_blocks

            sub_stack = nn.Sequential()
            # the first block needs to take care of stride and filter size increase.
            sub_stack.add_module(
                str(idx),
                MBConvBlock(
                    spatial_dims=spatial_dims,
                    in_channels=block_args.input_filters,
                    out_channels=block_args.output_filters,
                    kernel_size=block_args.kernel_size,
                    stride=block_args.stride,
                    image_size=current_image_size,
                    expand_ratio=block_args.expand_ratio,
                    se_ratio=block_args.se_ratio,
                    id_skip=block_args.id_skip,
                    norm=norm,
                    drop_connect_rate=blk_drop_connect_rate,
                ),
            )
            idx += 1  # increment blocks index counter

            current_image_size = _calculate_output_image_size(current_image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)

            # add remaining block repeated num_repeat times
            for _ in range(block_args.num_repeat - 1):
                blk_drop_connect_rate = self.drop_connect_rate

                # scale drop connect_rate
                if blk_drop_connect_rate:
                    blk_drop_connect_rate *= float(idx) / num_blocks

                # add blocks
                sub_stack.add_module(
                    str(idx),
                    MBConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_args.input_filters,
                        out_channels=block_args.output_filters,
                        kernel_size=block_args.kernel_size,
                        stride=block_args.stride,
                        image_size=current_image_size,
                        expand_ratio=block_args.expand_ratio,
                        se_ratio=block_args.se_ratio,
                        id_skip=block_args.id_skip,
                        norm=norm,
                        drop_connect_rate=blk_drop_connect_rate,
                    ),
                )
                idx += 1  # increment blocks index counter

            self._blocks.add_module(str(stack_idx), sub_stack)

        # sanity check to see if len(self._blocks) equal expected num_blocks
        if idx != num_blocks:
            raise ValueError("total number of blocks created != num_blocks")

        # Head
        head_in_channels = block_args.output_filters
        out_channels = _round_filters(1280, width_coefficient, depth_divisor)
        self._conv_head = conv_type(head_in_channels, out_channels, kernel_size=1, bias=False)
        self._conv_head_padding = _make_same_padder(self._conv_head, current_image_size)
        self._bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels)

        # final linear layer
        self.avg_pooling = avg_pooling
        self._avg_pooling = adaptivepool_type(1)
        self._dropout = nn.Dropout(dropout_rate)
        # self._fc = nn.Linear(out_channels, self.num_classes)
        self.out_channels = out_channels
        # swish activation to use - using memory efficient swish by default
        # can be switched to normal swish using self.set_swish() function call
        self._swish = Act["memswish"]()

        # initialize weights using Tensorflow's init method from official impl.
        self._initialize_weights()

        # only pretrained for when `spatial_dims` is 2
        if pretrained and (spatial_dims == 2):
            _load_state_dict(self, model_name, True, False)

    def set_swish(self, memory_efficient: bool = True) -> None:
        """
        Sets swish function as memory efficient (for training) or standard (for JIT export).

        Args:
            memory_efficient: whether to use memory-efficient version of swish.

        """
        self._swish = Act["memswish"]() if memory_efficient else Act["swish"](alpha=1.0)
        for sub_stack in self._blocks:
            for block in sub_stack:
                block.set_swish(memory_efficient)

    def head(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))
        return x

    def output(self, x: torch.Tensor, pooling=None) -> torch.Tensor:
        if pooling is None:
            pooling = self.avg_pooling
        x = self._conv_head(self._conv_head_padding(x))
        x = self._swish(self._bn1(x))
        if pooling:
            # Pooling and final linear layer
            x = self._avg_pooling(x)

            x = x.flatten(start_dim=1)
            x = self._dropout(x)
        return x

    def get_layers(self) -> list:
        return list(self._blocks)

    def _initialize_weights(self) -> None:
        """
        Args:
            None, initializes weights for conv/linear/batchnorm layers
            following weight init methods from
            `official Tensorflow EfficientNet implementation
            <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py#L61>`_.
            Adapted from `EfficientNet-PyTorch's init method
            <https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py>`_.
        """
        for _, m in self.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                fan_out = reduce(operator.mul, m.kernel_size, 1) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                fan_in = 0
                init_range = 1.0 / math.sqrt(fan_in + fan_out)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()


class SingleEfficientEncoder(SingleBaseEncoder):
    def __init__(self, model_name, spatial_dims=2, input_channels=3,
                 pretrained: bool = True):
        super(SingleEfficientEncoder, self).__init__(out_channels=0, spatial_dims=spatial_dims)
        self.encoder = EfficientNet(model_name=model_name, spatial_dims=spatial_dims,
                                    in_channels=input_channels, pretrained=pretrained)
        self.out_channels = self.encoder.out_channels


############################################## Multi FusionEncoder ##############################################


class MMEfficientEncoder(MMBaseEncoder):
    def __init__(self, modalities, modality_to_encoder, model_name, input_channels, spatial_dims=2,
                 blocks_args_str=None, efficientnet_params=None, pretrained: bool = True, avg_pooling=True):
        super(MMEfficientEncoder, self).__init__(out_channels=0, spatial_dims=spatial_dims)
        self.encoder = nn.ModuleDict({
            m: EfficientNet(model_name=model_name, spatial_dims=spatial_dims,
                            in_channels=input_channels[m], pretrained=pretrained, avg_pooling=avg_pooling,
                            blocks_args_str=blocks_args_str, efficientnet_params=efficientnet_params)
            for m in modalities if modality_to_encoder[m] == 'EfficientNet'
        })
        for v in self.encoder.values():
            self.out_channels = v.out_channels
            break
        for m in modalities:
            if modality_to_encoder[m] != 'EfficientNet':
                self.encoder[m] = MLP(input_channels[m], self.out_channels)


class MMOccEfficientEncoder(MMEfficientEncoder):
    def __init__(self, modalities, modality_to_encoder, model_name, input_channels, out_channels, spatial_dims=2,
                 blocks_args_str=None, efficientnet_params=None, pretrained: bool = True, act='abs', pool='avg'):
        from .occurrencenet import MMOccurrenceNet
        super(MMOccEfficientEncoder, self).__init__(
            modalities=modalities,
            modality_to_encoder=modality_to_encoder,
            model_name=model_name,
            input_channels=input_channels,
            spatial_dims=spatial_dims,
            blocks_args_str=blocks_args_str,
            efficientnet_params=efficientnet_params,
            pretrained=pretrained,
            avg_pooling=False
        )
        self.occ = MMOccurrenceNet(modalities, modality_to_encoder, self.out_channels,
                                   out_channels, spatial_dims, act, pool)

    def forward(self, x: dict, pooling=None):
        x = super().forward(x, pooling=pooling)
        x = self.occ.forward(x)
        return x

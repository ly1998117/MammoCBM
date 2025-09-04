_base_ = './base.py'
model_name = 'efficientnet-b0'
fusion_method = 'spool'
input_channels = {
    'ADC': 1,
    'CE': 1,
    'DWI': 1,
    'T2WI': 1,
    'TIC': 6,
}
efficientnet_params = {
    "efficientnet-b0": (1.0, 1.0, 128, 0.2, 0.2),
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
blocks_args_str = None
avg_pooling = True
pretrained = True

_base_ = './base.py'
model_name = 'efficientnet-map-b0'
epochs = 500
batch_size = 2
img_size = 128
postfix = '128'
fusion_method = 'pool'
input_channels = {
    'ADC': 1,
    'CE': 1,
    'DWI': 1,
    'T2WI': 1,
    'TIC': 6,
}
efficientnet_params = {
    "efficientnet-map-b0": (1.0, 1.0, 128, 0.2, 0.2),
    "efficientnet-map-b1": (1.0, 1.1, 240, 0.2, 0.2),
    "efficientnet-map-b2": (1.1, 1.2, 260, 0.3, 0.2),
    "efficientnet-map-b3": (1.2, 1.4, 300, 0.3, 0.2),
    "efficientnet-map-b4": (1.4, 1.8, 380, 0.4, 0.2),
    "efficientnet-map-b5": (1.6, 2.2, 456, 0.4, 0.2),
    "efficientnet-map-b6": (1.8, 2.6, 528, 0.5, 0.2),
    "efficientnet-map-b7": (2.0, 3.1, 600, 0.5, 0.2),
    "efficientnet-map-b8": (2.2, 3.6, 672, 0.5, 0.2),
    "efficientnet-map-l2": (4.3, 5.3, 800, 0.5, 0.2),
}
blocks_args_str = [
    "r1_k3_s11_e1_i32_o16_se0.25",
    "r2_k3_s22_e6_i16_o24_se0.25",
    "r2_k5_s22_e6_i24_o40_se0.25",
    "r3_k3_s11_e6_i40_o80_se0.25",
    "r3_k5_s11_e6_i80_o112_se0.25",
    "r4_k7_s11_e6_i112_o192_se0.25",
    "r1_k3_s11_e6_i192_o320_se0.25",
]
avg_pooling = True
pretrained = True

_base_ = './base.py'
model_name = 'ViT'
fusion_method = 'mattn'
input_channels = {
    'ADC': 1,
    'CE': 1,
    'DWI': 1,
    'T2WI': 1,
    'TIC': 6,
}
modality_to_encoder = {
    'ADC': model_name,
    'CE': model_name,
    'DWI': model_name,
    'T2WI': model_name,
    'TIC': 'MLP'
}
blocks_args_str = None
avg_pooling = True
pretrained = True
batch_size = 8
epochs = 500
lr = 3e-4
weight_decay = 0.05
warmup_ratio = 0.1  # 即前 10% epoch 用于 warmup
scheduler = 'cosine'

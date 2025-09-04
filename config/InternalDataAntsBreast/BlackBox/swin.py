_base_ = './base.py'
model_name = 'SwinTransformer'
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

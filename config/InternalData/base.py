_base_ = '../param.py'
# dataset
dataset = 'InternalData'
imbalance = False
valid_only = False
test_only = False
same_valid = False
test_csv = None
under_sample = False
epochs = 300
batch_size = 4
lr = 1e-4
num_worker = 4
iter_train = True

spacing = None
img_size = 128
z_normalize = True
spatial_dims = 3
num_class = 2
postfix = ''
pathology_labels = {'Benign': 0, 'BC': 1}
id_to_labels = {0: 'Benign', 1: 'BC'}
modality_to_dataset = {
    'ADC': ('ADC', 'MM'),
    'CE': ('CE', 'MM'),
    'DWI': ('DWI', 'MM'),
    'T2WI': ('T2WI', 'MM'),
    'TIC': ('TIC', 'MM'),
    'MM': ('ADC', 'CE', 'DWI', 'T2WI', 'TIC', 'MM')
}
modality_to_encoder = {
    'ADC': 'EfficientNet',
    'CE': 'EfficientNet',
    'DWI': 'EfficientNet',
    'T2WI': 'EfficientNet',
    'TIC': 'MLP'
}
modality_to_model = {
    'ADC': 'ADC',
    'CE': 'CE',
    'DWI': 'DWI',
    'T2WI': 'T2WI',
    'MM': ('ADC', 'CE', 'DWI', 'T2WI', 'TIC')
}
n_shot = 'full'
weight_init_method = 'zeros'
use_normalize = False

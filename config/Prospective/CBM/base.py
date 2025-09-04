_base_ = '../base.py'
lambda_l1 = 0.001
encode_dir = 'result/PreStudyALL_Zeros/efficientnet-b0_pool_iterTrain_LR0.0001_fullshot'  # help='clip model name: RN50, RN101, RN50x4, RN50x16, backbone')
test_csv='dataset/ProspectiveData/CSV/data_split/datalist.csv'
modality_mask = False
scale = 1
lr = 5e-5
pre_encode_to_feature = True
include_concept = False
avg_pooling = True
epochs = 1000
batch_size = 32
cbm_model = 'mm'
init_method = 'zero'
report_shot = 1.
n_samples = 50  # help='50, 100'
activation = None  # help='sigmoid, softmax'
analysis_top_k = None
analysis_threshold = None
num_worker = 1
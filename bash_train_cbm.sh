python multiRun.py --script train_CBM.py --options config=config/InternalData/CBM/cbm.py --options encode_dir='result/InternalData_Zeros/efficientnet-b0_spool_iterTrain_LR0.0001_fullshot' --func search_fold --device 3,4,5,6,7 --n_jobs 1

python multiRun.py --script train_CBM.py --options config=config/InternalData/CBM/mmcbm.py --options encode_dir='result/InternalData_Zeros/efficientnet-b0_spool_iterTrain_LR0.0001_fullshot' --func search_fold --device 3,4,5,6,7 --n_jobs 1

###############################################

python multiRun.py --script train_CBM.py --options config=config/InternalDataAntsBreast/CBM/mmcbm.py --options encode_dir='result/InternalDataAntsBreast_Zeros_ValidOnly/efficientnet-b0_spool_iterTrain_LR0.0001_fullshot' --func search_fold --device 0,1,2,3,4 --n_jobs 1

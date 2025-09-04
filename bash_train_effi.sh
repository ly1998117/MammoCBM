python multiRun.py --script train_efficientNet.py --options config=config/InternalDataBreast/BlackBox/efficient.py --func search_fold --device 4,5,6,7 --n_jobs 1
python multiRun.py --script train_CBM.py --config config/InternalData/CBM/occcbm.py --func search_param_loss --device 4,5,6,7 --n_jobs 1
python multiRun.py --script train_efficientNet.py --config config/InternalData/BlackBox/efficient.py --fold 1,2,3,4 --device 4,5,6,7 --n_jobs 1 --options fusion_method=rnpool


####################################################################
python multiRun.py --script train_efficientNet.py --options config=config/InternalDataAntsBreast/BlackBox/efficient_c2.py --fold 0,1,2,3,4  --n_jobs 1 --options include_concept=True --options crop_prob=1 --options postfix=concept_only --device 3,4,5,6,7
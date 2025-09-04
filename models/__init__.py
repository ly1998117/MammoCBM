import os
import pandas as pd
from ast import literal_eval
from dataset.dataloader import ConceptDataset, Transform
from .backbone.MultiModels import MMAttnSCLSEfficientNet, MMConceptEfficientNet, MMConcepOnlytEfficientNet
from .cbm import LinearCBM, OccurrenceCBM, CBM
from .cbank.conceptLearner import ConceptsLearner, ConceptPredictor, ConceptPredictor2


def get_model_from_config(config, encoder=None):
    if 'efficient' in config.model_name:
        if 'concept' in config.postfix:
            if 'only' in config.postfix:
                concepts = literal_eval(pd.read_csv(os.path.join(f'dataset/{config.dataset}', 'CSV', 'data_split', 'datalist.csv')).iloc[0]['concept_key'])
                print(f'CONCEPTS: {len(concepts)} {concepts}')
                predictor = ConceptPredictor2(n_dims=1280, concepts=concepts)
                return MMConcepOnlytEfficientNet(config=config, classifier=predictor)
            else:
                return MMConceptEfficientNet(config=config)
        else:
            return MMAttnSCLSEfficientNet(config=config)
    if 'cbm' in config.model_name.lower():
        if config.model_name == 'cbm':
            # concepts = \
            #     pd.read_csv(os.path.join(f'dataset/{config.dataset}', 'CSV', 'data_split', 'datalist.csv')).iloc[0][
            #         'concept_key']
            # concepts = [c for c in literal_eval(concepts) if 'BI-RADS_category' not in c]
            # print(f'CONCEPTS: {len(concepts)} {concepts}')
            # predictor = ConceptPredictor(n_dims=1280, concepts=concepts)
            return CBM(config=config, encoder=encoder, predictor=encoder.predictor)
        dataset = ConceptDataset(encoder=encoder,
                                 fold=config.k,
                                 data_path=f'dataset/{config.dataset}',
                                 transform=Transform(root_dir='./dataset',
                                                     normalize=config.z_normalize,
                                                     img_size=config.img_size,
                                                     spacing=config.spacing,
                                                     crop_prob=config.crop_prob).test_transforms(),
                                 pathology_labels=config.pathology_labels,
                                 encoder_name=config.encode_dir.replace('/', '-') + config.postfix)
        learner = ConceptsLearner(output_dir=config.output_dir,
                                  conceptDataset=dataset,
                                  device=config.device,
                                  report_shot=config.report_shot,
                                  n_samples=config.n_samples,
                                  batch_size=8,
                                  learning_rate=5e-4,
                                  epochs=2000,
                                  layer_num=2 if config.model_name == 'mm2cbm' else 1)
        learner.train_predictor()
        if config.model_name == 'mmcbm' or config.model_name == 'mm2cbm':
            return LinearCBM(config=config, encoder=encoder, predictor=learner.predictor)
        if config.model_name == 'OccCBM':
            return OccurrenceCBM(config=config, encoder=encoder, predictor=learner.predictor)
        raise
    return None

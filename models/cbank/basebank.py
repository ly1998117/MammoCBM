# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import os
import torch
import numpy as np
import pandas as pd

from utils.logger import char_color
from collections import defaultdict
from .conceptLearner import ConceptsLearner


##################################################### Concept Bank ################################################


class ConceptBank:
    def __init__(self, device, clip_name, location, encoder,
                 n_samples=50, neg_samples=0, single_modality_score=False,
                 bank_dir='', report_shot=1., concept_shot=1.,  sort=True, save=True,
                 language='en'):
        self.__name__ = 'ConceptBank'
        self.device = device
        self.save = save
        self.path = os.path.join(bank_dir, 'ConceptBank', f"concept_bank.pkl")
        self.clip_name = clip_name
        self.sort = sort
        self.single_modality_score = single_modality_score
        self.encoder = encoder
        self.concept_leaner = ConceptsLearner(device,
                                              clip_name=clip_name,
                                              bank_dir=bank_dir,
                                              location=location,
                                              encoder=encoder,
                                              report_shot=report_shot,
                                              concept_shot=concept_shot,
                                              n_samples=n_samples,
                                              neg_samples=neg_samples,
                                              )
        self.clip_model = self.concept_leaner.get_clip_model()
        self.transform = self.concept_leaner.get_clip_trans()
        # used for modality split. e.g. FA_ICGA: FA, ICGA
        self.modality_map = {'FA': 'FA', 'ICGA': 'ICGA', 'US': 'US', 'MM': 'FA_ICGA_US'}
        if os.path.exists(self.path) and self.save:
            print(char_color(f"Loading from {self.path}", color='green'))
            data = torch.load(self.path, map_location=torch.device(self.device))
            self.modality_mask = data['modality_mask']
            self._vectors = data['vectors']
            self.cls_weight = data['cls_weight']
            self._norms = data['norms']
            self._scales = data['scales']
            self._intercepts = data['intercepts']
            self._concept_names = data['concept_names']
            self.hidden_dim = data['hidden_dim']
        else:
            self.build()
        translate_file = 'CSV/concept/concepts-translation.csv'
        self.language = language
        if os.path.exists(translate_file):
            translation_dict = pd.read_csv(translate_file).set_index('concept')['translation'].to_dict()
            self._concepts_en = pd.DataFrame({'concept': self._concept_names})['concept'].map(
                lambda x: x.split(', ')[0] + ', ' + translation_dict[x.split(', ')[-1]]
                if x.split(', ')[-1] in translation_dict.keys() else x
            ).tolist()
        else:
            self.concepts_en = self._concept_names

    def build(self):
        all_vectors, all_cls, concept_names, all_scales, all_intercepts = [], [], [], [], []
        all_margin_info = defaultdict(list)

        self.concepts_dict = self.concept_leaner()
        self.modality_mask = {
            m: torch.zeros(len(self.concepts_dict), dtype=torch.float, device=self.device) for m in
            ['FA', 'ICGA', 'US']
        }

        for idx, (concept, (tensor, cls, scale, intercept, margin_info)) in enumerate(
                sorted(self.concepts_dict.items()) if self.sort else self.concepts_dict.items()
        ):
            try:
                for modality in self.modality_mask.keys():
                    if modality in concept or {'FA': 'Fluorescein Angiography',
                                               'ICGA': 'Indocyanine Green Angiography',
                                               'US': 'Ultrasound'}[modality] in concept:
                        self.modality_mask[modality][idx] = 1
            except:
                import pdb
                pdb.set_trace()
            all_vectors.append(tensor)
            all_cls.append(cls)
            concept_names.append(concept)
            all_intercepts.append(torch.tensor(intercept).reshape(1, 1))
            all_scales.append(torch.tensor(scale).reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    all_margin_info[key].append(value.reshape(1, 1))

        for key, val_list in all_margin_info.items():
            all_margin_info[key] = torch.tensor(
                torch.concat(val_list, dim=0), requires_grad=False
            ).float().to(self.device)

        self._vectors = torch.concat(all_vectors, dim=0).float().to(self.device)
        self.cls_weight = torch.stack(all_cls, dim=1).float().to(self.device)
        self._norms = torch.norm(self._vectors, p=2, dim=1, keepdim=True).detach()
        self._scales = torch.concat(all_scales, dim=0).float().to(self.device)
        self._intercepts = torch.concat(all_intercepts, dim=0).float().to(self.device)
        self._concept_names = concept_names
        self.hidden_dim = self._vectors.shape[1]
        print("Concept Bank is initialized.")
        torch.save({
            'modality_mask': self.modality_mask,
            'vectors': self._vectors,
            'cls_weight': self.cls_weight,
            'norms': self._norms,
            'scales': self._scales,
            'intercepts': self._intercepts,
            'concept_names': self._concept_names,
            'hidden_dim': self.hidden_dim,
        }, self.path)

    def set_modality_map(self, modality_map):
        self.modality_map.update(modality_map)

    def set_single_modality_score(self, single_modality_score):
        self.single_modality_score = single_modality_score

    @property
    def backbone(self):
        return self.clip_model

    @property
    def n_concepts(self):
        return self.vectors.shape[0]

    @property
    def vectors(self):
        if 'MM' not in self.modality:
            mask = self.get_mask_from_modality()
            return self._vectors[mask > 0]
        return self._vectors

    @property
    def norms(self):
        if 'MM' not in self.modality:
            mask = self.get_mask_from_modality()
            return self._norms[mask > 0]
        return self._norms

    @property
    def scales(self):
        if 'MM' not in self.modality:
            mask = self.get_mask_from_modality()
            return self._scales[mask > 0]
        return self._scales

    @property
    def intercepts(self):
        if 'MM' not in self.modality:
            mask = self.get_mask_from_modality()
            return self._intercepts[mask > 0]
        return self._intercepts

    @property
    def concept_names(self):
        if 'MM' not in self.modality:
            mask = self.get_mask_from_modality()
            if self.language == 'en':
                return np.array(self._concepts_en)[mask > 0].tolist()
            return np.array(self._concept_names)[mask > 0].tolist()
        if self.language == 'en':
            return self._concepts_en
        return self._concept_names

    def __getitem__(self, modality):
        self.modality = self.modality_map[modality] if modality in self.modality_map.keys() else modality
        return self

    def get_mask_from_modality(self):
        mask = 0
        for m in self.modality_mask.keys():
            if m in self.modality:
                mask = mask + self.modality_mask[m].flatten()
        return mask.cpu()

    def modality_score_reshape(self, score):
        """
        deprecated function: Obtain the score for the corresponding mode.
        :return: value
        """
        import warnings
        warnings.warn("This method has been deprecated and is not recommended for use.", DeprecationWarning)
        if score.ndim == 1:
            score = score.unsqueeze(dim=0)
        mask = 0
        for m in self.modality_mask.keys():
            if m in self.modality:
                mask = mask + self.modality_mask[m].flatten()
        if score.shape[-1] == self.vectors.shape[0]:
            indexes = mask.nonzero().flatten()
            return score[:, indexes]
        else:
            new_score = torch.ones((score.shape[0], self.vectors.shape[0]), device=score.device) * -10
            indexes = mask.nonzero().flatten()
            new_score[:, indexes] = score
            return new_score

    def compute_dist(self, emb, m_mask=False):
        if 'clip' in self.clip_name and 'cav' not in self.clip_name:
            margins = emb @ self.vectors.T
        elif 'cav' in self.clip_name:
            # Computing the geometric margin to the decision boundary specified by CAV.
            # margins = (self.scales * (torch.matmul(self.vectors, emb.T) + self.intercepts) / (self.norms)).T
            if emb.ndim == 3:
                margins = ((self.scales * (self.vectors * emb).sum(-1).T + self.intercepts) / self.norms).T
            else:
                margins = ((self.scales * (self.vectors @ emb.T) + self.intercepts) / self.norms).T
        else:
            raise KeyError(f"Unknown mode: {self.clip_name}")

        if 'softmax' in self.clip_name:
            margins = margins.softmax(dim=-1)
        elif 'tanh' in self.clip_name:
            margins = torch.tanh(margins)

        if self.single_modality_score:
            return self.modality_score_reshape(margins)

        if m_mask:
            mask = self.modality_mask[self.modality].repeat(margins.shape[0], 1)
            return margins * mask + (1 - mask) * margins.min(dim=1, keepdim=True)[0]

        return margins

    def get_concept_from_threshold(self, attn_score: torch.Tensor, threshold, modality=None):
        scores, concepts = [], []
        if self.single_modality_score:
            attn_score = self.modality_score_reshape(attn_score).flatten()
        if isinstance(attn_score, torch.Tensor):
            attn_score = attn_score.cpu().numpy()
        for idx, (score, concept) in enumerate(zip(attn_score, self.concept_names)):
            if modality is not None and modality not in concept and modality != 'MM':
                continue
            if threshold is None or score > threshold:
                scores.append(score)
                concepts.append(concept)
        return scores, concepts

    def get_topk_concepts(self, attn_score, k=5, sign=1, modality=None):
        if modality is not None and modality != 'MM':
            if self.single_modality_score:
                attn_score = self.modality_score_reshape(attn_score)
            for indice, concept in enumerate(self.concept_names):
                if modality not in concept:
                    attn_score[indice] = 0

        topk_scores, topk_indices = torch.topk(attn_score, k=k)
        topk_scores = topk_scores.cpu().numpy()
        topk_scores = topk_scores * sign
        topk_indices = topk_indices.detach().cpu().numpy()
        scores, concepts = [], []
        for j in topk_indices:
            concepts.append(self.concept_names[j % len(self.concept_names)])
        return topk_scores, concepts

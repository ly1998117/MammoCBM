import os.path as opath
import pandas as pd
import os
import fire
from ast import literal_eval

def dict_flatten(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        k = k.strip()
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):  # 如果值是字典，递归展平
            items.update(dict_flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def dict_fold(d, sep='.'):
    items = {}
    for k, v in d.items():
        k = k.split(sep)
        cur = items
        for i in k[:-1]:
            if i not in cur:
                cur[i.strip()] = {}
            cur = cur[i.strip()]
        cur[k[-1].strip()] = v
    return items


class Tools:
    def __init__(self, dataset):
        self.dataset = dataset

    def run_path(self, func):
        for name in os.listdir(opath.join(self.dataset, 'CSV', 'data_split')):
            func(opath.join(self.dataset, 'CSV', 'data_split', name))
        func(opath.join(self.dataset, 'CSV', 'report', 'concepts.csv'))
        func(opath.join(self.dataset, 'CSV', 'report', 'tic_gt.csv'))

    def correct_path(self):
        def _correct_path(datapath):
            try:
                data = pd.read_csv(datapath)
            except:
                print(f'{datapath} NOT EXISTS.')
                return
            data['path'] = data['path'].map(literal_eval).map(
                lambda x: {m: opath.join(self.dataset, '/'.join(p.split('/')[1:])) if isinstance(p, str) else p for m, p in
                        x.items()})
            data.to_csv(datapath, index=False)
        self.run_path(_correct_path)

    def correct_concept(self):
        def _correct_path(datapath):
            def _func(item):
                tumor_keys = ['masses', 'non_mass_enhancement', 'Kinetic curve assessment']
                other_keys = ['non_enhancing_lesions', 'associated_features', 'Fat-containing lesions']
                sep_key = 'BI-RADS_category'
                if 'concept' in item:
                    concepts = literal_eval(item['concept'])
                    if not isinstance(concepts['masses']['margin']['Circumscribed'], dict):
                        concepts['masses']['margin']['Circumscribed'] = {'Circumscribed': concepts['masses']['margin']['Circumscribed']}
                    concepts = {**{k: v for k, v in concepts.items() if k in tumor_keys}, **{k: v for k, v in concepts.items() if k in other_keys}}
                    item['concept'] = concepts
                if 'concept_key' in item:
                    item['concept_key'] = literal_eval(item['concept_key'])
                    item['concept_label'] = literal_eval(item['concept_label'])
                    concepts = dict_fold(dict(zip(item['concept_key'], item['concept_label'])))
                    birads = concepts.pop(sep_key)
                    if not isinstance(concepts['masses']['margin']['Circumscribed'], dict):
                        concepts['masses']['margin']['Circumscribed'] = {'Circumscribed': concepts['masses']['margin']['Circumscribed']}
                    concepts = {**{k: v for k, v in concepts.items() if k in tumor_keys}, **{k: v for k, v in concepts.items() if k in other_keys}}
                    item['concept_key'] = list(dict_flatten(concepts).keys())
                    item['concept_label'] = list(dict_flatten(concepts).values())
                    item['birads_key'] = list(dict_flatten({sep_key: birads}).keys())
                    item['birads_label'] = list(dict_flatten({sep_key: birads}).values())
                return item
            try:
                data = pd.read_csv(datapath)
            except Exception as e:
                print(f'{datapath} Error. {e}')
                return
            print(datapath)
            data = data.apply(_func, axis=1)
            data.to_csv(datapath, index=False)
        self.run_path(_correct_path)


if __name__ == '__main__':
  fire.Fire(Tools)

# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""

import os.path
import random

import pandas as pd
from ast import literal_eval
from utils.logger import PrintColor


class DataSplit:
    def __init__(self, data_path="../dataset/PreStudy",
                 valid_only=False, test_only=False, same_valid=False,
                 under_sample=False, exist_ok=False, save_csv=True,
                 print_info=True, val_csv=None, test_csv=None):
        self.valid_only = valid_only
        self.test_only = test_only
        self.same_valid = same_valid
        self.under_sample = under_sample
        self.exist_ok = exist_ok
        self.save_csv = save_csv
        self.print_info = print_info
        self.data_path = data_path
        self.csv_path = os.path.join(data_path, 'CSV', 'data_split')
        self.val_csv = val_csv
        self.test_csv = test_csv  # extra_data
        os.makedirs(self.csv_path, exist_ok=True)
        self.dataSplit = {}

    @property
    def df(self):
        if hasattr(self, '_df'):
            return self._df
        self._df = self._read_df(path=f'{self.csv_path}/datalist.csv')
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    @property
    def test(self):
        if hasattr(self, '_test'):
            return self._test
        if self.test_csv is not None:
            self._test = self._read_df(path=self.test_csv, autoname=False)
            self._test['modality'] = 'MM'
        else:
            self._test = self._read_df('test')
        return self._test

    @test.setter
    def test(self, value):
        self._test = value

    @property
    def train(self):
        if hasattr(self, '_train'):
            return self._train
        self._train = self._read_df('train')
        return self._train

    @train.setter
    def train(self, value):
        self._train = value

    @property
    def k_fold(self):
        if hasattr(self, '_k_fold'):
            return self._k_fold
        self._k_fold = self._read_df('k_fold')
        if self._k_fold is not None:
            self._k_fold['k'] = self._k_fold['k'].astype(int)
        return self._k_fold

    @k_fold.setter
    def k_fold(self, value):
        self._k_fold = value

    @property
    def modality(self):
        if hasattr(self, '_modality'):
            return self._modality
        modality = []
        for ms in self.df['modality'].tolist():
            if isinstance(ms, list):
                for m in ms:
                    if m not in modality:
                        modality.append(m)
        modality.append('MM')
        self._modality = modality
        return modality

    def _file_name(self, name):
        if self.valid_only and name is not None:
            name = f'{name}_valid_only'
        if self.same_valid and name is not None:
            name = f'{name}_same_valid'
        if self.under_sample and name is not None:
            name = f'{name}_under_sample'
        return name

    def _under_sample(self):
        by = 'pathology'

        def _sample(x):
            num = x.groupby(by).count().min()[0]
            x = x.groupby(by).apply(lambda x: x.sample(n=num)).reset_index(drop=True)
            return x

        self.df = self.df.groupby('modality').apply(_sample).reset_index(drop=True)

    def _read_df(self, name=None, path=None, autoname=True):
        if autoname:
            name = self._file_name(name)
        if path is None:
            path = f'{self.csv_path}/{name}.csv'
        if os.path.exists(path):
            if self.print_info:
                PrintColor(f'load {name} from {path}', color='green')
            df = pd.read_csv(path, dtype=str).agg(
                lambda x: x.map(literal_eval) if '{' in str(x.iloc[0]) or '[' in str(x.iloc[0]) else x
            )
            return df
        PrintColor(f'No {name} file: {path}', color='red')
        return None

    def _save_df(self, name):
        df = eval(f'self.{name}')
        name = self._file_name(name)
        if self.save_csv:
            df.to_csv(f'{self.csv_path}/{name}.csv', index=False)

    def _select(self, modality, df=None):
        def is_modality(modality_list):
            if modality == 'MM':
                return len(modality_list) == len(self.modality) - 1
            return modality in modality_list and len(modality_list) != len(self.modality) - 1

        if df is None:
            df = self.df
        df = df[df['modality'].map(lambda x: is_modality(x))].copy()
        if modality != 'MM':
            # 保证单一模态
            df = df.agg(lambda x: x.map(lambda y: {modality: y[modality]}) if isinstance(x.iloc[0], dict) else x)
        df['modality'] = modality
        return df

    def get_train_test_data(self):
        if self.test is not None and self.train is None:
            self.train = self.df[~(self.df['name'].isin(self.test['name']) & self.df['pathology'].isin(
                self.test['pathology']))]
        elif self.train is not None and self.test is None:
            self.test = self.df[~(self.df['name'].isin(self.train['name']) & self.df['pathology'].isin(
                self.train['pathology']))]
        elif self.train is None and self.test is None:
            self.test = self._select('MM').groupby(['pathology']).apply(
                lambda x: x.sample(frac=.2, replace=False, axis=0)).reset_index(level=[0], drop=True)
            self.train = self.df[~self.df.index.isin(self.test.index)]

        self._save_df('test')
        self._save_df('train')

    def k_fold_split(self, modality, k):
        def fn(x):
            if not self.same_valid or modality == 'MM':
                fold_id = [i for i in range(k) for j in range(0, (len(x) - 1) // k + 1) if j * k + i < len(x)]
                random.shuffle(fold_id)
            else:
                fold_id = -1
            x['k'] = fold_id
            return x

        data_frame = self._select(modality, self.train).groupby('pathology').apply(fn)
        return data_frame

    def get_5fold_data(self, k=5):
        if self.k_fold is not None and not self.exist_ok:
            return self.k_fold

        if self.under_sample:
            self._under_sample()

        if not self.valid_only:
            self.get_train_test_data()
        else:
            self.train = self.df
        modalities = []
        for modality in self.modality:
            modalities.append(self.k_fold_split(modality, k))
        self.k_fold = pd.concat(modalities, axis=0).reset_index(drop=True)
        self.k_fold['k'] = self.k_fold['k'].astype(int)
        self._save_df('k_fold')

    def get_data_split(self, modality, k):
        if f'{modality}-{k}' in self.dataSplit.keys():
            return self.dataSplit[f'{modality}-{k}']
        self.get_5fold_data()
        train = self.k_fold[self.k_fold['k'] != k].drop(columns='k').reset_index(drop=True)
        val = self.k_fold[self.k_fold['k'] == k].drop(columns='k').reset_index(drop=True)
        if self.test_csv is not None:
            train = pd.concat([train, val], axis=0).reset_index(drop=True)
            val = self._read_df('test')
        if self.val_csv is not None:
            val = self._read_df(os.path.basename(self.val_csv).split('.')[0], autoname=False)
        if self.test_only:
            PrintColor(f"{modality} test only")
            train = pd.concat([train, val], axis=0).reset_index(drop=True)
            val = None
        elif self.valid_only:
            PrintColor(f"{modality} valid only")
            test = None
        else:
            test = self.test

        train = train[train['modality'] == modality]
        if val is not None:
            val = val[val['modality'] == modality]
        self.dataSplit[f'{modality}-{k}'] = dict(train=train, val=val, test=test)
        return self.dataSplit[f'{modality}-{k}']

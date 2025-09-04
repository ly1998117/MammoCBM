import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import Dataset as _Dataset
from torch.utils.data import DataLoader


class ConceptPredictor(nn.Module):
    def __init__(self, n_dims, concepts, layer_num=1):
        super(ConceptPredictor, self).__init__()
        self.concepts = concepts
        self.layer_num = layer_num
        self.num_concepts = len(concepts)
        self.predictors = nn.ModuleList([self._create_predictor(n_dims, n_dims) for _ in range(self.num_concepts)])

    def _create_predictor(self, n_dims, hidden, o_dims=1):
        # A dual-layer neural network with ReLU activations
        layers = []
        for i in range(self.layer_num - 1):
            if i == 0:
                layers.append(nn.Linear(n_dims, hidden))
            else:
                layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, o_dims))  # Output layer for binary classification
        if self.layer_num == 1:
            return layers[0]
        return nn.Sequential(*layers)

    def forward(self, x, idx=None):
        if idx is None:
            return torch.cat([predictor(x) for predictor in self.predictors], dim=-1)
        if isinstance(idx, str):
            idx = self.concepts.index(idx)
        return self.predictors[idx](x)

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self.concepts.index(item)
        if item < 0 or item >= self.num_concepts:
            raise IndexError("Concept index out of range")
        return self.predictors[item]

    def unfreeze(self):
        for p in self.predictors.parameters():
            p.requires_grad = True

    def freeze(self):
        for p in self.predictors.parameters():
            p.requires_grad = False

    def grouped_concepts(self):
        grouped = {}
        for ind, concept in enumerate(self.concepts):
            if 'masses.margin' in concept:
                key = 'masses.margin'
                subkey = concept.split(f'{key}.')[-1]
            else:
                key = '.'.join(concept.split('.')[:-1])
                subkey = concept.split('.')[-1]
            if key in grouped:
                grouped[key].update({subkey: ind})
            else:
                grouped[key] = {subkey: ind}
        return grouped


class ConceptPredictor2(ConceptPredictor):
    def __init__(self, n_dims, concepts, layer_num=1):
        super(ConceptPredictor, self).__init__()
        self.concepts = concepts
        self.layer_num = layer_num
        self.num_concepts = len(concepts)
        self.predictors = self._create_predictor(n_dims, n_dims, self.num_concepts)

    def forward(self, x):
        return self.predictors(x)


class Dataset(_Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


class ConceptsLearner:
    @staticmethod
    def _resample(pos, neg):
        # resample for data balance
        if pos.size(0) > neg.size(0):
            neg = torch.cat([neg, neg[:pos.size(0) - neg.size(0)]])
        else:
            pos = torch.cat([pos, pos[:neg.size(0) - pos.size(0)]])
        return pos, neg

    @staticmethod
    def _split_data(data):
        val_size = int(0.1 * data.size(0))
        return data[val_size:], data[:val_size]

    def __init__(self,
                 output_dir,
                 conceptDataset,
                 device,
                 report_shot=1.,
                 n_samples=0,
                 batch_size=32,
                 learning_rate=1e-3,
                 epochs=100,
                 layer_num=1
                 ):
        self.output_dir = output_dir
        self.report_shot = report_shot
        self.device = device
        self.conceptDataset = conceptDataset
        self.encoder = conceptDataset.encoder
        self.n_samples = n_samples
        self.n_dims = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.concept_datasets: pd.DataFrame = self.init_datasets()
        self.predictor = ConceptPredictor(n_dims=self.n_dims, concepts=self.concept_datasets['concept'].tolist(),
                                          layer_num=layer_num)
        self.predictor.to(self.device)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def init_datasets(self):
        data = [d for d in tqdm(self.conceptDataset, desc='Loading Concepts')]
        concepts = pd.DataFrame([d['concept'] for d in data]).reset_index(drop=True)
        features = torch.stack([d['data'] for d in data], dim=0)
        self.n_dims = features.shape[1]
        concept_datasets = []

        for concept in concepts.columns:
            idx_pos = concepts.index[concepts[concept] > 0.5]
            idx_neg = concepts.index[concepts[concept] < 0.5]

            pos_features = features[idx_pos][torch.randperm(len(idx_pos))]
            neg_features = features[idx_neg][torch.randperm(len(idx_neg))]

            pos_train, pos_val = self._split_data(pos_features)
            neg_train, neg_val = self._split_data(neg_features)

            pos_train, neg_train = self._resample(pos_train, neg_train)
            pos_val, neg_val = self._resample(pos_val, neg_val)

            if 0 < self.n_samples < pos_train.size(0):
                pos_train = pos_train[:self.n_samples]
                neg_train = neg_train[:self.n_samples]

            def prepare_dataset(pos, neg):
                x = torch.cat([pos, neg])
                y = torch.cat([torch.ones(pos.size(0)), torch.zeros(neg.size(0))])
                assert len(x) > 0, "dataset Empty"
                return Dataset(x.to(self.device), y.to(self.device))

            concept_datasets.append({
                'concept': concept,
                'train': prepare_dataset(pos_train, neg_train),
                'val': prepare_dataset(pos_val, neg_val)
            })
        return pd.DataFrame(concept_datasets)

    def train_predictor(self):
        save_path = os.path.join(self.output_dir, 'concept', 'predictor.pth')
        log_path = os.path.join(self.output_dir, 'concept', 'predictor.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        logs = []
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, weights_only=True, map_location=torch.device(self.device))
            self.predictor.load_state_dict(checkpoint)
            return self.predictor

        for idx, row in self.concept_datasets.iterrows():
            concept = row['concept']
            train_loader = DataLoader(row['train'], batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(row['val'], batch_size=self.batch_size, shuffle=False)
            best_val_acc = 0.0
            best_model = None
            optimizer = torch.optim.Adam(
                params=self.predictor[idx].parameters(),
                lr=self.learning_rate
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                          lr_lambda=lambda epoch: (1 - epoch / self.epochs) ** 0.9)
            patience = int(0.1 * self.epochs)  # Early Stop
            counter = 0
            tqdmer = tqdm(range(1, self.epochs + 1))
            for epoch in tqdmer:
                self.predictor[idx].train()
                correct = 0
                total = 0
                for data, label in train_loader:
                    label = label.view(-1, 1)

                    optimizer.zero_grad()
                    output = self.predictor[idx](data)
                    loss = self.loss_fn(output, label)
                    loss.backward()
                    optimizer.step()
                    preds = (torch.sigmoid(output.detach()) > 0.5).float().view(-1)
                    correct += (preds == label.view(-1)).sum().item()
                    total += label.view(-1).size(0)

                train_acc = correct / total
                scheduler.step()
                with torch.no_grad():
                    self.predictor[idx].eval()
                    correct = 0
                    total = 0
                    for data, label in val_loader:
                        output = self.predictor[idx](data)
                        preds = (torch.sigmoid(output) > 0.5).float().view(-1)
                        correct += (preds == label).sum().item()
                        total += label.size(0)
                    val_acc = correct / total
                    tqdmer.set_description(
                        desc=f'Training Concept: {concept}, Epoch: {epoch} LR: {optimizer.param_groups[0]["lr"]}')
                    tqdmer.set_postfix({'train_acc': train_acc, 'val acc': val_acc})
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = {k: v.clone() for k, v in self.predictor[idx].state_dict().items()}
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience and val_acc > 0.9:
                            print(f'Early stopping for concept {concept} at epoch {epoch}')
                            break

            if best_model is not None:
                with torch.no_grad():
                    self.predictor[idx].load_state_dict(best_model)
                    logs.append({
                        'concept': concept,
                        'val_acc': best_val_acc
                    })
        logs = pd.DataFrame(logs)
        logs.to_csv(log_path, index=False)
        torch.save(self.predictor.state_dict(), save_path)
        return self.predictor

# -*- coding: utf-8 -*-
import torch
from models.module import MultiModalModule
from utils import dict_flatten, dict_unflatten
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, average_precision_score


class _CBM(MultiModalModule):
    def __init__(self, config, encoder, predictor) -> None:
        super().__init__(config)
        self.iter_train = False
        self.lambda_l1 = config.lambda_l1
        self.predictor = predictor
        self.get_encoder = lambda: encoder
        self.concepts = self.predictor.concepts
        # this is the scale factor for the concept score, it can impact the speed of convergence
        self.scale = torch.nn.Parameter(torch.tensor(config.scale).float(), requires_grad=False)
        self.classifier = self.config_classifier()
        self.config_module()
        self.filelogger.info(f"scale: {self.scale.data.item()}")

    @property
    def encoder(self):
        return self.get_encoder()

    def encode_image(self, inp, modality=None, fusion=True, avg_pooling=None, grad=False):
        if not grad:
            with torch.no_grad():
                return self.encode_image(inp, modality, fusion, avg_pooling, True)
        if modality in inp:
            inp = {modality: inp[modality]}
        if isinstance(inp, dict):
            if not self.config.pre_encode_to_feature:
                return self.encoder.encode_image(inp, modality=modality, fusion=fusion, avg_pooling=avg_pooling)
            if len(inp) > 1:
                inp = self.encoder.fusion(inp)
            else:
                inp = inp[modality]
        return inp

    def config_classifier(self):
        """
        configure the classifier, including the weight matrix
        :return: classifier model
        """
        raise NotImplementedError

    def config_module(self):
        pass

    def get_weight_matrix(self):
        """
        get weight matrix, used for interpretability.
        if activation is needed, overwrite this function
        :return:
        """
        return self.classifier.weight

    def init_weight_matrix(self, init_weight=None):
        if init_weight is None:
            init_weight = torch.zeros((self.config.num_class, len(self.concepts)))
        if self.config.weight_init_method == 'zero':
            init_weight.data.zero_()
        elif self.config.weight_init_method == 'rand':
            torch.nn.init.kaiming_normal_(init_weight)
        return init_weight

    def predict_score(self, inp, modality):
        if isinstance(inp, dict):
            inp = {k: v for k, v in inp.items() if k in self.modalities}
            img_feat = self.encode_image(inp, modality=modality)
        else:
            img_feat = inp
        sim_score = self.scale * self.predictor(img_feat)
        return sim_score

    def forward(self, inp, modality, **kwargs):
        sim_score = self.predict_score(inp, modality)
        logits = self.classifier(sim_score)
        return logits

    def compute_loss(self, logits, label, concept_label, *_):
        # classification accuracy
        loss = self.loss.cls_loss(logits, label)
        if self.lambda_l1 > 0:
            loss += self.lambda_l1 * self.loss.l1_norm(self.classifier.weight)
        return loss

    def bi_rads(self, inp, modality, threshold=0.5):
        inp = self.predict_score(inp, modality)
        score = torch.sigmoid(inp)
        birads = {}
        for key, group_d in self.predictor.grouped_concepts().items():
            if threshold is not None:
                group_score = score[0, list(group_d.values())]
                if 'shape' in key or 'margin' in key or 'internal_enhancement_characteristics' in key or 'Initial enhancement phase' in key or 'Delayed phase' in key:
                    birads[key] = {k: False for k, v in group_d.items()}
                    birads[key][max(group_d, key=lambda i: group_d[i])] = True
                else:
                    birads[key] = {k: group_score[i].item() > threshold for i, k in enumerate(group_d.keys())}
            else:
                birads[key] = {k: group_score[i].item() for i, k in enumerate(group_d.keys())}
        birads = dict_unflatten(dict_flatten(birads))
        return birads

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.results_concepts = []
    

    def test_step(self, batch, batch_idx):
        data = batch['data']
        label = batch['label']
        names, pathologies, modalities = batch['name'], batch['pathology'], batch['modality']
        modality = modalities[0]
        concept_label = batch['concept_label'] if 'concept_label' in batch else None
        if modality == 'MM':
            iter_modality = [*self.modalities, modality]
        else:
            iter_modality = [modality]
        for m in iter_modality:
            sim_score = self.predict_score(inp=data, modality=m)
            logits = self.classifier(sim_score)
            if isinstance(logits, (list, tuple)):
                logits, *output = logits
            else:
                output = []
            loss = self.compute_loss(logits, label, concept_label, *output)
            self.all_y[m].append(label.cpu())
            self.all_pred[m].append(logits.cpu())
            self.log(f'test_{m}_loss', loss.detach())

            for name, pathology, l, p in zip(names, pathologies, label.cpu().numpy(),
                                            self.logits_to_prob(logits).cpu().numpy()):
                self.results.append({
                    'name': name,
                    'pathology': pathology,
                    'modality': m,
                    'label': l,
                    'probability': p,
                    'pred': self.config.id_to_labels[int(p>=0.5)]
                })
            
            for name, pathology, l, p in zip(names, pathologies, concept_label.cpu().numpy(),
                                            sim_score.sigmoid().cpu().numpy()):
                self.results_concepts.append({
                    'name': name,
                    'pathology': pathology,
                    'modality': m,
                    'label': l,
                    'probability': p,
                })

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        results = pd.DataFrame(self.results_concepts)
        metrics = []
        for m in results['modality'].unique():
            df_m = results[results['modality'] == m]
            y_true = df_m['label'].astype(int).to_numpy()
            y_prob = df_m['probability'].astype(float).to_numpy()
            y_pred = (y_prob >= 0.5).astype(int)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            pre = precision_score(y_true, y_pred, zero_division=0)

            # 可选曲线类指标（当且仅当 y_true 里有两个类别时才计算）
            if np.unique(y_true).size > 1:
                auroc = roc_auc_score(y_true, y_prob)
            else:
                auroc = np.nan

            metrics.append({
                'modality': m,
                'threshold': thresh,
                'accuracy': acc,
                'f1': f1,
                'recall': rec,
                'precision': pre,
                'auroc': auroc,
                'n_samples': len(df_m),
            })
        results.to_csv(f'{self.config.output_dir}/test_concept_pred.csv', index=False)
        pd.DataFrame(metrics).to_csv(f'{self.config.output_dir}/test_pred_metrics.csv', index=False)
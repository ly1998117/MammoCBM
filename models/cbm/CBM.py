import torch
from .baseCBM import _CBM


class CBM(_CBM):
    def config_module(self):
        self.predictor.freeze()

    def config_classifier(self):
        classifier = torch.nn.Linear(len(self.concepts), self.config.num_class)
        classifier.weight.data = self.init_weight_matrix()
        classifier.bias.data.zero_()
        return classifier

    def configure_loss(self) -> dict:
        fn = super().configure_loss()
        fn.update({'concept_loss': torch.nn.BCEWithLogitsLoss()})
        return fn

    def compute_loss(self, logits, label, concept_label, *args):
        loss = self.loss.cls_loss(logits, label)
        loss += self.loss.concept_loss(args[0], concept_label)
        if self.lambda_l1 > 0:
            loss += self.lambda_l1 * self.loss.l1_norm(self.classifier.weight)
        return loss

    def forward(self, inp, modality, **kwargs):
        sim_score = self.predict_score(inp, modality)
        logits = self.classifier(sim_score)
        return logits, sim_score
